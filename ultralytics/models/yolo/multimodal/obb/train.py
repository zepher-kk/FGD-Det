# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy
from typing import Optional

from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import (
    de_parallel,
    log_multimodal_model_complexity,
    torch_distributed_zero_first,
)
from ultralytics.nn.tasks import OBBModel
from ultralytics.data.dataset import YOLOMultiModalImageDataset
from ultralytics.nn.mm.utils import normalize_modality_token
from ultralytics.engine.afss import AFSSConfig, AFSSRuntime


class MultiModalOBBTrainer(OBBTrainer):
    """
    多模态旋转框训练器（RGB+X），复用 YOLOMM 路由与 6+ 通道输入，输出旋转框预测。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

        # 与 YOLOMM 检测/分割保持一致的模态控制
        self.modality = normalize_modality_token(getattr(self.args, "modality", None))
        # 回写 args，确保训练内 validator/copy(args) 看到一致 token
        self.args.modality = self.modality
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

        if self.modality:
            LOGGER.info(f"初始化 MultiModalOBBTrainer - 单模态训练: {self.modality}-only")
        else:
            LOGGER.info("初始化 MultiModalOBBTrainer - 双模态训练")

        self._sync_afss_task_name()
        self.afss_config = AFSSConfig.from_args(self.args)
        self.afss_runtime = None
        if self.afss_config.enabled:
            self.add_callback("on_train_start", self._afss_on_train_start)
            self.add_callback("on_train_epoch_start", self._afss_on_train_epoch_start)
            self.add_callback("on_train_epoch_end", self._afss_on_train_epoch_end)
            LOGGER.info(
                "AFSS 配置已启用: task=%s warmup=%d, update_interval=%d, score_conf=%.3f, score_iou=%.3f, "
                "ema_alpha=%.3f, state_dir=%s",
                self.afss_config.task_name,
                self.afss_config.warmup_epochs,
                self.afss_config.state_update_interval,
                self.afss_config.score_conf,
                self.afss_config.score_iou,
                self.afss_config.state_ema_alpha,
                self.afss_config.state_dir,
            )

    def _sync_afss_task_name(self):
        """Keep AFSS task wiring aligned with trainer task for adapter-based runtime."""
        task_name = str(getattr(self.args, "task", "obb"))
        setattr(self.args, "afss_task_name", task_name)

    def _ensure_afss_runtime(self, dataset):
        """Initialize AFSS runtime once OBB train dataset is available."""
        if not self.afss_config.enabled or self.afss_runtime is not None:
            return self.afss_runtime
        self._sync_afss_task_name()
        self.afss_config = AFSSConfig.from_args(self.args)
        self.afss_runtime = AFSSRuntime.from_dataset(
            dataset=dataset,
            args=self.args,
            save_dir=self.save_dir,
            resume=bool(getattr(self.args, "resume", False)),
        )
        return self.afss_runtime

    def _afss_on_train_start(self, trainer):
        """Callback: announce AFSS runtime bootstrap."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_start()

    def _afss_on_train_epoch_start(self, trainer):
        """Callback: refresh AFSS epoch selection."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_start(trainer.epoch)

    def _afss_on_train_epoch_end(self, trainer):
        """Callback: persist AFSS snapshot and trigger scheduled scoring."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_end(
                trainer.epoch,
                trainer=trainer,
                validator=trainer.validator,
            )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct dataloader with optional AFSS sampler injection for OBB training."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        sampler_override = None
        if mode == "train" and self.afss_config.enabled:
            runtime = self._ensure_afss_runtime(dataset)
            sampler_override = runtime.create_sampler(rank=rank, shuffle=shuffle)
        return build_dataloader(
            dataset,
            batch_size,
            workers,
            shuffle,
            rank,
            sampler_override=sampler_override,
        )

    # -----------------
    # Dataset building
    # -----------------
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        构建多模态 OBB 数据集，开启 RGB+X 图像管线。
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
            multi_modal_image=True,
            x_modality=self._determine_x_modality_from_data(),
            x_modality_dir=self._get_x_modality_path(self._determine_x_modality_from_data()),
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
        )

    # -----------
    # Model init
    # -----------
    def get_model(self, cfg: str | dict | None = None, weights: str | None = None, verbose: bool = True):
        """
        使用 OBBModel，并按数据集的 Xch 动态确定输入通道数。
        """
        # 输入通道：双模态 3+Xch，单模态 3
        if self.is_dual_modal:
            x_channels = self.data.get("Xch", 3)
            channels = 3 + x_channels
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"多模态 OBB 模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch")
        else:
            channels = 3
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"单模态 OBB 模型初始化: {(self.modality or 'RGB')}(3ch)")

        model = OBBModel(cfg, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        if hasattr(model, "mm_router") and model.mm_router and self.modality:
            model.mm_router.set_runtime_params(
                self.modality,
                strategy=getattr(self.args, "ablation_strategy", None),
                seed=getattr(self.args, "seed", None),
            )

        if weights:
            model.load(weights)

        # Unified multimodal complexity logs
        try:
            imgsz = int(getattr(self.args, "imgsz", 640))
            log_multimodal_model_complexity(model, imgsz=imgsz, modality=self.modality)
        except Exception as e:
            LOGGER.warning(f"模型复杂度统计失败（可忽略，不影响训练）：{e}")

        return model

    def get_validator(self):
        """
        返回多模态 OBB 验证器，保持与训练时损失项一致。
        """
        from ultralytics.models.yolo.multimodal.obb.val import MultiModalOBBValidator

        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return MultiModalOBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    # -----------------
    # Helper utilities
    # -----------------
    def _determine_x_modality_from_data(self):
        """
        参考 detection trainer 的推断逻辑，解析 data.yaml 中的 X 模态名称。
        """
        data = getattr(self, "data", {}) or {}
        # modality_used 或 models 字段
        for key in ("modality_used", "models"):
            if key in data and isinstance(data[key], list):
                non_rgb = [m for m in data[key] if m != "rgb"]
                if non_rgb:
                    return non_rgb[0]
        # 兼容字段
        if "x_modality" in data:
            return data["x_modality"]
        return "depth"

    def _get_x_modality_path(self, x_modality: str):
        """
        根据 data.yaml modalities 映射获取 X 模态目录。
        """
        data = getattr(self, "data", {}) or {}
        mod_map = data.get("modalities") or data.get("modality")
        if isinstance(mod_map, dict) and x_modality in mod_map:
            return mod_map[x_modality]
        return f"images_{x_modality}"

    def final_eval(self):
        """执行最终评估并导出 LLM 友好的 JSON 结果。"""
        super().final_eval()

        # 导出 LLM 友好的 JSON 格式验证结果
        from ultralytics.utils.llm_export import export_final_val_llm_json

        try:
            export_final_val_llm_json(self)
        except Exception as e:
            LOGGER.warning(f"LLM JSON export failed: {e}")

        # 记录多模态特定信息
        x_modality = self._determine_x_modality_from_data()
        if self.modality:
            LOGGER.info(f"最终评估完成 - 单模态训练: {self.modality}-only")
        else:
            LOGGER.info(f"最终评估完成 - 双模态训练: RGB+{x_modality}")
