from __future__ import annotations
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
RT-DETR MultiModal validator module.

This module provides the RTDETRMMValidator class for validating multi-modal RT-DETR models
with support for RGB+X modality inputs.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import LOGGER, colorstr, TQDM
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.ops import Profile
from copy import copy
from ultralytics.nn.mm.utils import normalize_modality_token
from ultralytics.engine.afss.tasks.detect import build_detect_afss_score_row


class RTDETRMMValidator(DetectionValidator):























    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):











        super().__init__(dataloader, save_dir, args, _callbacks)


        self.is_multimodal = self._detect_multimodal_mode()


        if args:
            if isinstance(args, dict):
                self.modality = args.get('modality', None)
            else:
                self.modality = getattr(args, 'modality', None)
        else:
            self.modality = None


        self.modality = normalize_modality_token(self.modality)

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


        self.multimodal_config = None


        if self.is_multimodal:
            LOGGER.info(f"🚀 {colorstr('RTDETRMMValidator')}: 多模态验证模式已启用")
            if self.modality:
                LOGGER.info(f"🎯 单模态消融验证: {colorstr(self.modality)}")
            else:
                LOGGER.info("🔄 双模态验证模式")


    def _detect_multimodal_mode(self) -> bool:









        return True

    @property
    def is_dual_modal(self) -> bool:

        return self.is_multimodal and self.modality is None

    @property
    def is_single_modal(self) -> bool:

        return self.is_multimodal and self.modality is not None

    def _get_non_distill_loss_names(self, trainer):

        _DISTILL_PREFIXES = ('distill_', 'd_out', 'd_feat')
        return [
            name
            for name in getattr(trainer, 'loss_names', ())
            if not any(str(name).startswith(p) for p in _DISTILL_PREFIXES)
        ]

    def _label_val_loss_items(self, loss_items: torch.Tensor, prefix: str = "val"):

        keys = [f"{prefix}/{x}" for x in getattr(self, '_val_loss_names', ())]
        values = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, values))

    def __call__(self, trainer=None, model=None):













        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        
        if self.training:
            self.device = trainer.device

            if self.data is None:
                self.data = trainer.data

            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self._val_loss_names = self._get_non_distill_loss_names(trainer)
            self.loss = torch.zeros(len(self._val_loss_names), device=trainer.device, dtype=trainer.loss_items.dtype)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()

            if hasattr(model, 'mm_router') and model.mm_router and self.modality:
                model.mm_router.set_runtime_params(
                    self.modality,
                    strategy=getattr(self.args, 'ablation_strategy', None),
                    seed=getattr(self.args, 'seed', None),
                )
        else:

            from ultralytics.utils import callbacks, emojis
            from ultralytics.utils.checks import check_imgsz
            from ultralytics.nn.autobackend import AutoBackend
            from ultralytics.utils.torch_utils import select_device
            from ultralytics.data.utils import check_det_dataset
            
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
                LOGGER.info(f"Setting batch={self.args.batch} for RT-DETR validation")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:

                if not hasattr(self, 'data') or self.data is None:
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

            try:
                if hasattr(model, 'pt') and model.pt and hasattr(model, 'model') and hasattr(model.model, 'mm_router') and model.model.mm_router and self.modality:
                    model.model.mm_router.set_runtime_params(
                        self.modality,
                        strategy=getattr(self.args, 'ablation_strategy', None),
                        seed=getattr(self.args, 'seed', None),
                    )
            except Exception:
                pass
            

            if hasattr(self, 'data') and self.data and 'Xch' in self.data:
                x_channels = self.data.get('Xch', 3)
                total_channels = 3 + x_channels
                LOGGER.info(f"执行{total_channels}通道多模态RT-DETR模型warmup (RGB:3 + X:{x_channels})")
                model.warmup(imgsz=(1 if pt else self.args.batch, total_channels, imgsz, imgsz))
            else:

                LOGGER.info("执行6通道多模态RT-DETR模型warmup (默认)")
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
                    loss_items = model.loss(batch, preds)[1]
                    if loss_items.numel() != len(self._val_loss_names):
                        raise RuntimeError(
                            f"Validation loss dimension mismatch: got {loss_items.numel()} items from model.loss(), "
                            f"but validator expects {len(self._val_loss_names)} non-distill items {self._val_loss_names}."
                        )
                    self.loss += loss_items


            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader) * 1E3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **self._label_val_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            LOGGER.info("Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image" %
                       tuple(self.speed.values()))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)
                stats = self.eval_json(stats)
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def build_dataset(self, img_path, mode="val", batch=None):














        if not self.is_multimodal:

            return super().build_dataset(img_path, mode, batch)


        stride = getattr(self, 'stride', 32)
        if hasattr(self, 'model') and self.model:
            stride = max(int(de_parallel(self.model).stride.max() if hasattr(self.model, 'stride') else 0), 32)


        if self.multimodal_config is None:
            self.multimodal_config = self._parse_multimodal_config()
            LOGGER.info(f"多模态验证配置解析完成 - 模态: {self.multimodal_config['models']}")


        modalities = self.multimodal_config['models']
        modalities_dict = self.multimodal_config['modalities']


        x_modalities = [m for m in modalities if m != 'rgb']
        x_modality = x_modalities[0] if x_modalities else None
        x_modality_dir = modalities_dict.get(x_modality) if x_modality else None

        LOGGER.info(f"构建多模态验证数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}")


        if self.modality:
            self._validate_modality_compatibility()
            LOGGER.info(f"启用单模态验证: {self.modality}-only，将应用智能模态填充")


        return build_yolo_dataset(
            self.args, img_path, batch, self.data,
            mode=mode,
            rect=False,
            stride=stride,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            modalities=modalities,
        )

    def postprocess(self, preds):









        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)


        imgsz_hw = getattr(self, "_imgsz_hw", None)
        if imgsz_hw is not None and len(imgsz_hw) == 2:
            h, w = int(imgsz_hw[0]), int(imgsz_hw[1])
            scale = torch.tensor([w, h, w, h], device=bboxes.device, dtype=bboxes.dtype)
            bboxes = bboxes * scale
        else:
            bboxes = bboxes * float(self.args.imgsz)


        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        from ultralytics.utils import ops as _ops

        for i, bbox in enumerate(bboxes):
            bbox = _ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred[score > self.args.conf]

        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]} for x in outputs]

    def preprocess(self, batch):













        batch = super().preprocess(batch)


        if self.is_multimodal and "img" in batch:

            try:
                self._imgsz_hw = tuple(batch["img"].shape[2:4])
            except Exception:
                self._imgsz_hw = None


            x_channels = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
            expected_channels = 3 + x_channels
            
            if batch["img"].shape[1] == expected_channels:




                pass
            elif batch["img"].shape[1] == 3:

                LOGGER.debug("接收到3通道输入，使用标准预处理")
            else:
                LOGGER.warning(f"意外的通道数: {batch['img'].shape[1]}，期望{expected_channels}或3通道")

        return batch

    def _apply_modality_ablation(self, batch):








        if not self.modality:
            return

        images = batch["img"]
        x_channels = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        total_channels = 3 + x_channels

        if images.shape[1] != total_channels:
            LOGGER.warning(f"Channel mismatch: expected {total_channels}, got {images.shape[1]}")
            return

        if self.modality == "RGB":

            images[:, 3:3+x_channels, :, :] = 0
            LOGGER.debug(f"单模态RGB验证: X模态通道({x_channels}ch)已置零")
        else:

            images[:, 0:3, :, :] = 0
            LOGGER.debug(f"单模态{self.modality}验证: RGB通道已置零")

        batch["img"] = images

    def _parse_multimodal_config(self):










        data = getattr(self, 'data', None)


        if self.modality:

            if self.modality == "RGB":

                x_modality = self._determine_x_modality_from_data()
                config = {
                    'models': ['rgb', x_modality],
                    'modalities': {
                        'rgb': 'images',
                        x_modality: f'images_{x_modality}'
                    }
                }
                LOGGER.info(f"RGB单模态验证，动态确定X模态: {x_modality}")
            else:

                if self.modality == "X":

                    actual_x_modality = self._determine_x_modality_from_data()

                    x_modality_path = self._get_x_modality_path(actual_x_modality)
                    
                    config = {
                        'models': ['rgb', actual_x_modality],
                        'modalities': {
                            'rgb': 'images',
                            actual_x_modality: x_modality_path
                        }
                    }
                    LOGGER.info(f"X模态单模态验证: {actual_x_modality}-only (从'X'解析)")
                else:

                    x_modality_path = self._get_x_modality_path(self.modality)
                    
                    config = {
                        'models': ['rgb', self.modality],
                        'modalities': {
                            'rgb': 'images',
                            self.modality: x_modality_path
                        }
                    }
                    LOGGER.info(f"X模态单模态验证: {self.modality}-only")

            return config


        return self._get_default_multimodal_config()

    def _get_default_multimodal_config(self):







        data = getattr(self, 'data', None)


        if data and 'modality_used' in data:
            modality_used = data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                config = {
                    'models': modality_used,
                    'modalities': {}
                }


                if 'modality' in data and isinstance(data['modality'], dict):
                    modality_paths = data['modality']
                    for mod in modality_used:
                        config['modalities'][mod] = modality_paths.get(
                            mod, 'images' if mod == 'rgb' else f'images_{mod}'
                        )
                else:

                    for mod in modality_used:
                        config['modalities'][mod] = 'images' if mod == 'rgb' else f'images_{mod}'

                LOGGER.info(f"Loaded multi-modal validation config: {modality_used}")
                return config


        if data and 'models' in data:
            models = data['models']
            if isinstance(models, list) and len(models) >= 2:
                config = {'models': models, 'modalities': {}}
                for modality in models:
                    config['modalities'][modality] = 'images' if modality == 'rgb' else f'images_{modality}'
                return config


        x_modality = self._determine_x_modality_from_data()
        config = {
            'models': ['rgb', x_modality],
            'modalities': {
                'rgb': 'images',
                x_modality: f'images_{x_modality}'
            }
        }
        LOGGER.info(f"Using default multi-modal validation config: rgb+{x_modality}")
        return config

    def _get_x_modality_path(self, modality_name):













        data = getattr(self, 'data', None)
        if data and 'modality' in data:
            modality_paths = data['modality']
            if isinstance(modality_paths, dict) and modality_name in modality_paths:
                return modality_paths[modality_name]
        

        if data and 'modalities' in data:
            modalities = data['modalities']
            if isinstance(modalities, dict) and modality_name in modalities:
                return modalities[modality_name]
        

        return f'images_{modality_name}'

    def _determine_x_modality_from_data(self):







        data = getattr(self, 'data', None)


        if data:

            if 'modality_used' in data:
                modality_used = data['modality_used']
                if isinstance(modality_used, list):
                    x_modalities = [m for m in modality_used if m != 'rgb']
                    if x_modalities:
                        return x_modalities[0]


            if 'models' in data:
                models = data['models']
                if isinstance(models, list):
                    x_modalities = [m for m in models if m != 'rgb']
                    if x_modalities:
                        return x_modalities[0]


            if 'modality' in data and isinstance(data['modality'], dict):
                modality_paths = data['modality']
                x_modalities = [m for m in modality_paths.keys() if m != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"从modality路径推断X模态: {x_modality}")
                    return x_modality


        LOGGER.warning("无法自动确定X模态类型，使用默认值: depth")
        return 'depth'

    def _validate_modality_compatibility(self):






        if not self.modality:
            return


        available_modalities = []
        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            available_modalities = self.multimodal_config.get('models', [])
        elif hasattr(self, 'data') and self.data and 'models' in self.data:
            available_modalities = self.data['models']


        if available_modalities:

            if self.modality == "X":

                x_modalities = [m for m in available_modalities if m != 'rgb']
                if x_modalities:
                    LOGGER.info(f"✅ 验证模态兼容性通过: '{self.modality}' 映射到 {x_modalities[0]}")
                else:
                    raise ValueError(
                        f"指定的验证modality '{self.modality}' 无法映射到有效的X模态。"
                        f"可用模态列表: {available_modalities}，但没有找到非RGB的X模态。"
                    )
            else:

                if self.modality not in available_modalities:
                    raise ValueError(
                        f"指定的验证modality '{self.modality}' 不在可用模态列表中: {available_modalities}。"
                        f"请检查数据配置或modality参数。"
                    )
                LOGGER.info(f"✅ 验证模态兼容性通过: {self.modality} 在可用模态 {available_modalities} 中")
        else:

            LOGGER.warning(f"⚠️  无法验证验证modality '{self.modality}' 的兼容性，未找到可用模态配置")

    def plot_val_samples(self, batch, ni):










        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities, visualize_x_to_3ch, concat_side_by_side,
            duplicate_bboxes_for_side_by_side, ensure_batch_idx_long, resolve_x_modality
        )
        

        multimodal_images = batch["img"]
        

        x_channels = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        

        rgb_images, x_images = split_modalities(multimodal_images, x_channels)
        

        x_modality = resolve_x_modality(self.modality, getattr(self, 'data', None))
        

        batch_idx = ensure_batch_idx_long(batch["batch_idx"])
        batch["batch_idx"] = batch_idx
        

        if self.modality:

            if self.modality == "RGB":

                plot_images(
                    rgb_images,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
            else:

                x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
                plot_images(
                    x_visual,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
        else:

            try:

                plot_images(
                    rgb_images,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                

                x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
                plot_images(
                    x_visual,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                

                side_by_side_images = concat_side_by_side(rgb_images, x_visual)

                batch_ids_dup, cls_ids_dup, bboxes_dup, _ = duplicate_bboxes_for_side_by_side(
                    batch["batch_idx"], batch["cls"].squeeze(-1), batch["bboxes"], None
                )
                plot_images(
                    side_by_side_images,
                    batch_ids_dup,
                    cls_ids_dup,
                    bboxes_dup,
                    paths=[p.replace('.jpg', '_multimodal.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_labels_multimodal.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
            except Exception as e:
                LOGGER.warning(f"绘制{x_modality}模态验证样本失败: {e}")
    
    def plot_predictions(self, batch, preds, ni):











        from ultralytics.utils.plotting import plot_images, output_to_target
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities, visualize_x_to_3ch, concat_side_by_side,
            to_norm_xywh_for_plot, duplicate_bboxes_for_side_by_side, resolve_x_modality,
            ensure_batch_idx_long, clip_boxes_norm_xywh
        )
        

        multimodal_images = batch["img"]
        

        x_channels = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        

        rgb_images, x_images = split_modalities(multimodal_images, x_channels)
        

        x_modality = resolve_x_modality(self.modality, getattr(self, 'data', None))
        

        batch_ids, cls_ids, boxes_xywh_px, confs = output_to_target(preds, max_det=self.args.max_det)
        

        _, _, H, W = rgb_images.shape
        img_hw = (H, W)
        

        batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm = to_norm_xywh_for_plot(
            batch_ids, cls_ids, boxes_xywh_px, confs, img_hw
        )
        

        batch_ids_norm = ensure_batch_idx_long(batch_ids_norm)

        if (isinstance(boxes_norm, torch.Tensor) and boxes_norm.numel() > 0) or (
            isinstance(boxes_norm, np.ndarray) and boxes_norm.size > 0
        ):
            boxes_norm = clip_boxes_norm_xywh(boxes_norm, 0.0, 1.0, 0.0, 1.0)
        

        if self.modality:

            if self.modality == "RGB":

                plot_images(
                    rgb_images,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
            else:

                x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
                plot_images(
                    x_visual,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
        else:

            try:

                plot_images(
                    rgb_images,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                

                x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
                plot_images(
                    x_visual,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                

                side_by_side_images = concat_side_by_side(rgb_images, x_visual)

                batch_ids_dup, cls_ids_dup, boxes_dup, confs_dup = duplicate_bboxes_for_side_by_side(
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm
                )
                plot_images(
                    side_by_side_images,
                    batch_ids_dup, cls_ids_dup, boxes_dup, confs_dup,
                    paths=[p.replace('.jpg', '_multimodal.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_pred_multimodal.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
            except Exception as e:
                LOGGER.warning(f"绘制{x_modality}模态预测结果失败: {e}")





    def afss_score_sample(self, pred, batch, si):

        if "im_file" not in batch:
            raise KeyError("AFSS sample scoring requires batch['im_file']")
        pbatch = self._prepare_batch(si, batch)
        predn = self._prepare_pred(pred, pbatch)
        result = self._process_batch(predn, pbatch)
        tp = result["tp"]
        matched = int(tp[:, 0].sum()) if len(tp) else 0
        return build_detect_afss_score_row(
            im_file=str(batch["im_file"][si]),
            matched=matched,
            pred_count=int(len(predn["cls"])),
            gt_count=int(len(pbatch["cls"])),
            task_name="detect",
        )

    def afss_score_batch(self, preds, batch):

        return [self.afss_score_sample(pred, batch, si) for si, pred in enumerate(preds)]

    def score_sample(self, pred, batch, si):

        return self.afss_score_sample(pred, batch, si)

    def score_batch(self, preds, batch):

        return self.afss_score_batch(preds, batch)

