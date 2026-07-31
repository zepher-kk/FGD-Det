from __future__ import annotations
from copy import copy, deepcopy
from typing import Optional

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import de_parallel, log_multimodal_model_complexity
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.mm.utils import normalize_modality_token
from ultralytics.engine.afss import AFSSConfig, AFSSRuntime


class MultiModalSegmentationTrainer(SegmentationTrainer):







    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

        self.modality = normalize_modality_token(getattr(self.args, "modality", None))

        self.args.modality = self.modality
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

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




    def build_dataset(self, img_path, mode: str = "train", batch: Optional[int] = None):

        x_modality, x_dir = self._resolve_x_modality_and_dir()

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
            x_modality=x_modality,
            x_modality_dir=x_dir,
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
        )




    def get_model(self, cfg: str | dict | None = None, weights: str | None = None, verbose: bool = True):

        if self.is_dual_modal:
            x_channels = self.data.get("Xch", 3)
            channels = 3 + x_channels
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"多模态分割模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch")
        else:
            channels = 3
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"单模态分割模型初始化: {(self.modality or 'RGB')}(3ch)")


        cfg_dict = None
        if isinstance(cfg, str):
            try:
                from ultralytics.nn.tasks import yaml_model_load

                cfg_dict = yaml_model_load(cfg)
            except Exception:
                cfg_dict = None
        elif isinstance(cfg, dict):
            cfg_dict = deepcopy(cfg)

        if cfg_dict is not None:
            cfg_dict["dataset_config"] = dict(self.data)
            model = SegmentationModel(cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        else:
            model = SegmentationModel(cfg, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)

        if hasattr(model, "multimodal_router") and model.multimodal_router:
            model.multimodal_router.update_dataset_config(self.data)
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"已更新MultiModalRouter的数据集配置，Xch={self.data.get('Xch', 3)}")

        if hasattr(model, "mm_router") and model.mm_router and self.modality:
            model.mm_router.set_runtime_params(
                self.modality,
                strategy=getattr(self.args, "ablation_strategy", None),
                seed=getattr(self.args, "seed", None),
            )

        if weights:
            model.load(weights)


        try:
            imgsz = int(getattr(self.args, "imgsz", 640))
            log_multimodal_model_complexity(model, imgsz=imgsz, modality=self.modality)
        except Exception as e:
            LOGGER.warning(f"模型复杂度统计失败（可忽略，不影响训练）：{e}")

        return model

    def get_validator(self):
        from ultralytics.models.yolo.multimodal.segment.val import MultiModalSegmentationValidator


        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "semseg_loss"
        return MultiModalSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _sync_afss_task_name(self):

        setattr(self.args, "afss_task_name", str(getattr(self.args, "task", "segment")))

    def _ensure_afss_runtime(self, dataset):

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

        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_start()

    def _afss_on_train_epoch_start(self, trainer):

        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_start(trainer.epoch)

    def _afss_on_train_epoch_end(self, trainer):

        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_end(
                trainer.epoch,
                trainer=trainer,
                validator=trainer.validator,
            )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):

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




    def _resolve_x_modality_and_dir(self) -> tuple[str, Optional[str]]:
        x_mod = None
        x_dir = None
        data = getattr(self, "data", {}) or {}

        if "modality_used" in data and isinstance(data["modality_used"], list):
            non_rgb = [m for m in data["modality_used"] if m != "rgb"]
            if non_rgb:
                x_mod = non_rgb[0]

        if x_mod is None and "models" in data and isinstance(data["models"], list):
            non_rgb = [m for m in data["models"] if m != "rgb"]
            if non_rgb:
                x_mod = non_rgb[0]

        if x_mod is None:
            x_mod = data.get("x_modality", None)

        mod_map = data.get("modality") or data.get("modalities")
        if isinstance(mod_map, dict) and x_mod in mod_map:
            x_dir = mod_map[x_mod]
        elif x_mod:
            x_dir = f"images_{x_mod}"

        if x_mod is None:
            LOGGER.warning("无法自动确定X模态类型，使用默认值: depth")
            x_mod = "depth"
            x_dir = data.get("modality", {}).get("depth", "images_depth") if isinstance(mod_map, dict) else "images_depth"

        return x_mod, x_dir




    def plot_training_samples(self, batch, ni):







        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities,
            visualize_x_to_3ch,
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            ensure_batch_idx_long,
            resolve_x_modality,
        )

        images = batch["img"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]

        batch_idx = ensure_batch_idx_long(batch["batch_idx"]) if "batch_idx" in batch else None
        if batch_idx is None:

            import torch

            batch_idx = ensure_batch_idx_long(torch.zeros(cls.shape[0], dtype=torch.long))
            batch["batch_idx"] = batch_idx


        masks = batch.get("masks", None)


        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        rgb_images, x_images = split_modalities(images, xch)
        x_modality = resolve_x_modality(self.modality, getattr(self, 'data', None))


        if self.modality:
            if self.modality == "RGB":
                plot_images(
                    rgb_images,
                    batch_idx,
                    cls,
                    bboxes,
                    masks=masks,
                    paths=paths,
                    fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
                    on_plot=self.on_plot,
                )
            else:
                x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
                plot_images(
                    x_visual,
                    batch_idx,
                    cls,
                    bboxes,
                    masks=masks,
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
                    fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
                    on_plot=self.on_plot,
                )
            return


        plot_images(
            rgb_images,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
            on_plot=self.on_plot,
        )


        x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
        plot_images(
            x_visual,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
            fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
            on_plot=self.on_plot,
        )


        side_by_side_images = concat_side_by_side(rgb_images, x_visual)
        batch_ids_dup, cls_ids_dup, bboxes_dup, _ = duplicate_bboxes_for_side_by_side(
            batch_idx, cls, bboxes, None
        )
        plot_images(
            side_by_side_images,
            batch_ids_dup,
            cls_ids_dup,
            bboxes_dup,
            paths=[p.replace('.jpg', '_multimodal.jpg') for p in paths],
            fname=self.save_dir / f"train_batch{ni}_labels_multimodal.jpg",
            on_plot=self.on_plot,
        )




    def save_model(self):

        from ultralytics.utils.patches import torch_load
        import torch

        super().save_model()

        if hasattr(self, 'multimodal_config'):
            ckpt = torch_load(self.last, map_location='cpu')
            ckpt['multimodal_config'] = getattr(self, 'multimodal_config', None)
            ckpt['modality'] = self.modality
            torch.save(ckpt, self.last)

            if self.best.exists():
                ckpt_best = torch_load(self.best, map_location='cpu')
                ckpt_best['multimodal_config'] = getattr(self, 'multimodal_config', None)
                ckpt_best['modality'] = self.modality
                torch.save(ckpt_best, self.best)

    def final_eval(self):

        super().final_eval()


        from ultralytics.utils.llm_export import export_final_val_llm_json

        try:
            export_final_val_llm_json(self)
        except Exception as e:
            LOGGER.warning(f"LLM JSON export failed: {e}")

        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            x_modality = [m for m in self.multimodal_config['models'] if m != 'rgb'][0]
            if self.modality:
                LOGGER.info(f"最终评估完成 - 单模态训练: {self.modality}-only")
            else:
                LOGGER.info(f"最终评估完成 - 双模态训练: RGB+{x_modality}")
        else:
            LOGGER.info("最终评估完成 - 多模态分割")

