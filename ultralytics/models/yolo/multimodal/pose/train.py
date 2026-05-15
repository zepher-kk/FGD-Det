# Ultralytics YOLO, AGPL-3.0 license

"""
Multi-Modal Pose Trainer.

Provides MultiModalPoseTrainer for RGB+X pose estimation training.
"""

from copy import copy

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import de_parallel, log_multimodal_model_complexity, torch_distributed_zero_first
from ultralytics.nn.tasks import PoseModel
from ultralytics.nn.mm.utils import normalize_modality_token
from ultralytics.engine.afss import AFSSConfig, AFSSRuntime


class MultiModalPoseTrainer(PoseTrainer):
    """
    Multi-modal pose estimation trainer (RGB+X).

    Reuses YOLOMM routing with 6+ channel input for pose keypoint prediction.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize MultiModalPoseTrainer with multi-modal configuration."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

        # Modality control consistent with YOLOMM detect/segment/obb
        self.modality = normalize_modality_token(getattr(self.args, "modality", None))
        # 回写 args，确保训练内 validator/copy(args) 看到一致 token
        self.args.modality = self.modality
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

        if self.modality:
            LOGGER.info(f"MultiModalPoseTrainer initialized - single modal: {self.modality}-only")
        else:
            LOGGER.info("MultiModalPoseTrainer initialized - dual modal training")

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
        task_name = str(getattr(self.args, "task", "pose"))
        setattr(self.args, "afss_task_name", task_name)

    def _ensure_afss_runtime(self, dataset):
        """Initialize AFSS runtime once training dataset is available."""
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
        """Callback: refresh AFSS adaptive selection plan."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_start(trainer.epoch)

    def _afss_on_train_epoch_end(self, trainer):
        """Callback: persist AFSS state snapshot and trigger scheduled scoring."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_end(
                trainer.epoch,
                trainer=trainer,
                validator=trainer.validator,
            )

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build multi-modal pose dataset with RGB+X image pipeline.

        Args:
            img_path: Path to images directory.
            mode: Dataset mode ('train' or 'val').
            batch: Batch size.

        Returns:
            Multi-modal dataset instance.
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

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct dataloader with optional AFSS sampler injection for training."""
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

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize PoseModel with dynamic channel count based on modality.

        Args:
            cfg: Model configuration file or dict.
            weights: Path to pretrained weights.
            verbose: Whether to print model info.

        Returns:
            Initialized PoseModel instance.
        """
        # Input channels: dual modal 3+Xch, single modal 3
        if self.is_dual_modal:
            x_channels = self.data.get("Xch", 3)
            channels = 3 + x_channels
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"Multi-modal Pose model: RGB(3ch) + X({x_channels}ch) = {channels}ch")
        else:
            channels = 3
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"Single-modal Pose model: {self.modality or 'RGB'}(3ch)")

        # 检查 kpt_shape 配置，提示 kobj_loss 计算条件
        kpt_shape = self.data.get("kpt_shape", [17, 3])
        if verbose and RANK in {-1, 0}:
            kpt_dim = kpt_shape[1] if len(kpt_shape) > 1 else 2
            if kpt_dim == 3:
                LOGGER.info(f"关键点配置: {kpt_shape[0]}点 × {kpt_dim}维 (x,y,可见性) → kobj_loss: 已启用")
            else:
                LOGGER.warning(
                    f"关键点配置: {kpt_shape[0]}点 × {kpt_dim}维 (仅x,y坐标) → kobj_loss: 已禁用 (数据集无可见性标注)"
                )

        model = PoseModel(
            cfg,
            nc=self.data["nc"],
            ch=channels,
            data_kpt_shape=kpt_shape,
            verbose=verbose and RANK == -1,
        )

        # Configure multi-modal router for single modality mode
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
        """Return multi-modal pose validator with consistent loss names."""
        from .val import MultiModalPoseValidator

        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return MultiModalPoseValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def _determine_x_modality_from_data(self):
        """
        Determine X modality name from data.yaml configuration.

        Returns:
            X modality name string (e.g., 'depth', 'thermal', 'ir').
        """
        data = getattr(self, "data", {}) or {}
        # Check modality_used or models field
        for key in ("modality_used", "models"):
            if key in data and isinstance(data[key], list):
                non_rgb = [m for m in data[key] if m != "rgb"]
                if non_rgb:
                    return non_rgb[0]
        # Fallback to x_modality field
        if "x_modality" in data:
            return data["x_modality"]
        return "depth"

    def _get_x_modality_path(self, x_modality: str):
        """
        Get X modality directory path from data.yaml modalities mapping.

        Args:
            x_modality: X modality name.

        Returns:
            Directory path for X modality images.
        """
        data = getattr(self, "data", {}) or {}
        mod_map = data.get("modalities") or data.get("modality")
        if isinstance(mod_map, dict) and x_modality in mod_map:
            return mod_map[x_modality]
        return f"images_{x_modality}"

    def final_eval(self):
        """Execute final evaluation and export LLM-friendly JSON results."""
        super().final_eval()

        # Export LLM-friendly JSON format validation results
        from ultralytics.utils.llm_export import export_final_val_llm_json

        try:
            export_final_val_llm_json(self)
        except Exception as e:
            LOGGER.warning(f"LLM JSON export failed: {e}")

        # Log multi-modal specific information
        x_modality = self._determine_x_modality_from_data()
        if self.modality:
            LOGGER.info(f"Final evaluation complete - single modal: {self.modality}-only")
        else:
            LOGGER.info(f"Final evaluation complete - dual modal: RGB+{x_modality}")
