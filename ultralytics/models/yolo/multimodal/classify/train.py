from __future__ import annotations
# Ultralytics YOLO, AGPL-3.0 license

"""
多模态分类训练器

本模块提供多模态分类任务的训练器，支持 RGB+X 模态的图像分类训练。
采用 YAML 配置 + txt 标签文件的数据管理范式，与检测任务保持一致。
"""

from copy import copy
from typing import Any, Dict, Optional

import torch

from ultralytics.data import YOLOMultiModalClassifyDataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.afss import AFSSConfig, AFSSRuntime
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import is_parallel, strip_optimizer
from ultralytics.nn.mm.utils import normalize_modality_token

class MultiModalClassificationTrainer(BaseTrainer):
























    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: Optional[Dict[str, Any]] = None,
        _callbacks=None
    ):








        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"


        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224


        self.modality = normalize_modality_token(overrides.get("modality", None))
        overrides["modality"] = self.modality
        self.is_dual_modal = self.modality is None

        super().__init__(cfg, overrides, _callbacks)


        self.modality = normalize_modality_token(getattr(self.args, "modality", self.modality))

        self.args.modality = self.modality
        self.is_dual_modal = self.modality is None


        if self.modality:
            LOGGER.info(f"初始化 MultiModalClassificationTrainer - 单模态训练: {self.modality}-only")
        else:
            LOGGER.info("初始化 MultiModalClassificationTrainer - 双模态训练")

        self._sync_afss_task_name()
        self._ensure_default_afss_classify_overrides()
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

        setattr(self.args, "afss_task_name", "classify")

    def _ensure_default_afss_classify_overrides(self):

        raw_overrides = getattr(self.args, "afss_task_overrides", None)
        if raw_overrides is None:
            raw_overrides = {}
        elif hasattr(raw_overrides, "__dict__"):
            raw_overrides = vars(raw_overrides).copy()
        elif not isinstance(raw_overrides, dict):
            raw_overrides = dict(raw_overrides)

        classify_override = raw_overrides.get("classify")
        if classify_override is None:
            classify_override = {}
        elif hasattr(classify_override, "__dict__"):
            classify_override = vars(classify_override).copy()
        elif not isinstance(classify_override, dict):
            classify_override = dict(classify_override)

        classify_override.setdefault("sufficiency_mode", "top1_prob_if_correct")
        raw_overrides["classify"] = classify_override
        setattr(self.args, "afss_task_overrides", raw_overrides)

    def _ensure_afss_runtime(self, dataset):

        if not self.afss_config.enabled or self.afss_runtime is not None:
            return self.afss_runtime
        self._sync_afss_task_name()
        self._ensure_default_afss_classify_overrides()
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

    def get_dataset(self):







        self.modality = normalize_modality_token(getattr(self.args, "modality", getattr(self, "modality", None)))

        self.args.modality = self.modality
        self.is_dual_modal = self.modality is None


        self.data = check_det_dataset(self.args.data)


        if "nc" not in self.data:
            self.data["nc"] = len(self.data.get("names", {}))


        self._calculate_channels()


        return self.data

    def _calculate_channels(self):

        xch = self.data.get("Xch", 3)

        if self.modality:

            self.data["channels"] = 3
        else:

            self.data["channels"] = 3 + xch

        LOGGER.info(f"输入通道数: {self.data['channels']} (Xch={xch})")

    def set_model_attributes(self):

        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose: bool = True):











        model = ClassificationModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1
        )

        if weights:
            model.load(weights)


        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout


        for p in model.parameters():
            p.requires_grad = True

        return model

    def setup_model(self):






        import torchvision

        if str(self.model) in torchvision.models.__dict__:

            self.model = torchvision.models.__dict__[self.model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            ckpt = None
        else:
            ckpt = super().setup_model()


        ClassificationModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):











        return YOLOMultiModalClassifyDataset(
            img_path=img_path,
            data=self.data,
            args=self.args,
            augment=mode == "train",
            prefix=mode
        )

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train"
    ):












        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)

        shuffle = mode == "train"
        sampler_override = None
        if mode == "train" and self.afss_config.enabled:
            runtime = self._ensure_afss_runtime(dataset)
            sampler_override = runtime.create_sampler(rank=rank, shuffle=shuffle)

        loader = build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=shuffle,
            rank=rank,
            sampler_override=sampler_override,
        )


        if mode != "train" and hasattr(loader.dataset, "torch_transforms"):
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms

        return loader

    def preprocess_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self) -> str:

        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):

        self.loss_names = ["loss"]

        from ultralytics.models.yolo.multimodal.classify.val import (
            MultiModalClassificationValidator
        )

        return MultiModalClassificationValidator(
            self.test_loader,
            self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )

    def label_loss_items(
        self,
        loss_items: Optional[torch.Tensor] = None,
        prefix: str = "train"
    ):

        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_metrics(self):

        plot_results(file=self.csv, classify=True, on_plot=self.on_plot)

    def final_eval(self):

        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")


        from ultralytics.utils.llm_export import export_final_val_llm_json

        try:
            export_final_val_llm_json(self)
        except Exception as e:
            LOGGER.warning(f"LLM JSON export failed: {e}")

    def plot_training_samples(self, batch: Dict[str, torch.Tensor], ni: int):

        batch["batch_idx"] = torch.arange(len(batch["img"]))
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

