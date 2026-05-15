# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import numpy as np
from pathlib import Path

from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.utils import LOGGER, callbacks, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.ops import Profile
from ultralytics.utils import TQDM
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.mm.utils import normalize_modality_token


class MultiModalOBBValidator(OBBValidator):
    """
    多模态 OBB 验证器：继承 OBBValidator，加入 6+ 通道 warmup 与 runtime 模态注入。
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        # 适配父类签名
        super().__init__(dataloader, save_dir, args, _callbacks)

        # 多模态标记
        if args:
            if isinstance(args, dict):
                self.modality = args.get("modality", None)
            else:
                self.modality = getattr(args, "modality", None)
        else:
            self.modality = None

        # 仅对 rgb/x token 做归一化：rgb/RGB→RGB、x/X→X（其它模态名保持原样）
        self.modality = normalize_modality_token(self.modality)
        # 回写 args/self.args，确保训练内 copy(args) 与后续读取一致
        if args is not None:
            if isinstance(args, dict):
                args["modality"] = self.modality
            else:
                setattr(args, "modality", self.modality)
        if hasattr(self, "args") and self.args is not None:
            if isinstance(self.args, dict):
                self.args["modality"] = self.modality
            else:
                setattr(self.args, "modality", self.modality)
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

        if self.modality:
            LOGGER.info(f"初始化 MultiModalOBBValidator - 单模态验证: {self.modality}-only")
        else:
            LOGGER.info("初始化 MultiModalOBBValidator - 双模态验证")

    def build_dataset(self, img_path, mode="val", batch=None):
        """构建多模态 OBB 数据集，与 MultiModalOBBTrainer.build_dataset 对齐。"""
        stride = int(getattr(self, "stride", 32) or 32)
        x_modality = self._determine_x_modality_from_data()
        x_modality_dir = self._get_x_modality_path(x_modality)

        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=stride,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
        )

    def _determine_x_modality_from_data(self) -> str:
        """解析 data.yaml 中的 X 模态名称。"""
        data = getattr(self, "data", {}) or {}
        for key in ("modality_used", "models"):
            if key in data and isinstance(data[key], list):
                non_rgb = [m for m in data[key] if m != "rgb"]
                if non_rgb:
                    return non_rgb[0]
        if "x_modality" in data:
            return data["x_modality"]
        return "depth"

    def _get_x_modality_path(self, x_modality: str) -> str:
        """根据 data.yaml modalities 映射获取 X 模态目录。"""
        data = getattr(self, "data", {}) or {}
        mod_map = data.get("modalities") or data.get("modality")
        if isinstance(mod_map, dict) and x_modality in mod_map:
            return mod_map[x_modality]
        return f"images_{x_modality}"

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        执行验证流程，支持 6+ 通道多模态输入与旋转框评估。
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)

        if self.training:
            self.device = trainer.device
            if self.data is None:
                self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
            if hasattr(model, "mm_router") and model.mm_router and self.modality:
                model.mm_router.set_runtime_params(
                    self.modality,
                    strategy=getattr(self.args, "ablation_strategy", None),
                    seed=getattr(self.args, "seed", None),
                )
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device
            self.args.half = model.fp16
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 6, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            if not pt:
                self.args.rect = False
            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            # runtime 模态注入
            try:
                if hasattr(model, "pt") and model.pt and hasattr(model, "model") and hasattr(model.model, "mm_router") and model.model.mm_router and self.modality:
                    model.model.mm_router.set_runtime_params(
                        self.modality,
                        strategy=getattr(self.args, "ablation_strategy", None),
                        seed=getattr(self.args, "seed", None),
                    )
            except Exception:
                pass

            # 多模态 warmup
            if hasattr(self, "data") and self.data and "Xch" in self.data:
                x_channels = self.data.get("Xch", 3)
                total_channels = 3 + x_channels
                LOGGER.info(f"执行 {total_channels} 通道多模态 OBB 模型 warmup (RGB:3 + X:{x_channels})")
                model.warmup(imgsz=(1 if pt else self.args.batch, total_channels, imgsz, imgsz))
            else:
                LOGGER.info("执行 6 通道多模态 OBB 模型 warmup (默认)")
                model.warmup(imgsz=(1 if pt else self.args.batch, 6, imgsz, imgsz))

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            with dt[0]:
                batch = self.preprocess(batch)
            with dt[1]:
                preds = model(batch["img"], augment=augment)
            with dt[2]:
                if self.training:
                    orig_mode = model.training
                    try:
                        model.train()
                        self.loss += model.loss(batch, preds)[1]
                    finally:
                        if not orig_mode:
                            model.eval()
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            self.run_callbacks("on_val_batch_end")

        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
        return self.metrics.results_dict

    # ------------------------------------------------------------------
    # AFSS per-sample scoring helpers
    # ------------------------------------------------------------------

    def afss_score_sample(self, pred, batch, si):
        """Score a single OBB sample with rotated-box matching primitives."""
        if "im_file" not in batch:
            raise KeyError("AFSS sample scoring requires batch['im_file']")
        pbatch = self._prepare_batch(si, batch)
        predn = self._prepare_pred(pred, pbatch)
        result = self._process_batch(predn, pbatch)
        tp = result["tp"]
        matched = int(tp[:, 0].sum()) if len(tp) else 0
        pred_count = int(len(predn["cls"]))
        gt_count = int(len(pbatch["cls"]))

        if gt_count == 0 and pred_count == 0:
            obb_precision = 1.0
            obb_recall = 1.0
        elif gt_count == 0:
            obb_precision = 0.0
            obb_recall = 1.0
        elif pred_count == 0:
            obb_precision = 0.0
            obb_recall = 0.0
        else:
            obb_precision = matched / pred_count
            obb_recall = matched / gt_count

        precision_op = float(obb_precision)
        recall_op = float(obb_recall)
        sufficiency_raw = min(precision_op, recall_op)
        im_file = str(batch["im_file"][si])
        return {
            "sample_key": str(Path(im_file).resolve()),
            "im_file": im_file,
            "task_name": "obb",
            "precision": precision_op,
            "recall": recall_op,
            "precision_op": precision_op,
            "recall_op": recall_op,
            "sufficiency_raw": sufficiency_raw,
            "valid_for_afss": True,
            "task_metrics": {
                "obb_precision": precision_op,
                "obb_recall": recall_op,
                "matched_count": int(matched),
                "pred_count": int(pred_count),
                "gt_count": int(gt_count),
            },
            # Keep legacy counters for compatibility with current state-store expectations.
            "matched_count": int(matched),
            "pred_count": int(pred_count),
            "gt_count": int(gt_count),
        }

    def afss_score_batch(self, preds, batch):
        """Score all OBB samples in a batch for AFSS adapter reuse."""
        return [self.afss_score_sample(pred, batch, si) for si, pred in enumerate(preds)]

    def score_sample(self, pred, batch, si):
        """Backward-compatible AFSS sample helper for legacy scorer calls."""
        return self.afss_score_sample(pred, batch, si)

    def score_batch(self, preds, batch):
        """Backward-compatible AFSS batch helper for legacy scorer calls."""
        return self.afss_score_batch(preds, batch)
