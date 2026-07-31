from __future__ import annotations
# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
from pathlib import Path
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.dataset import YOLOMultiModalImageDataset
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import de_parallel
import torch
from ultralytics.utils.checks import check_imgsz
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import TQDM, callbacks, emojis
from ultralytics.utils.ops import Profile
import numpy as np
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
import json
from ultralytics.nn.mm.utils import normalize_modality_token
from ultralytics.engine.afss.tasks.detect import build_detect_afss_score_row

class MultiModalDetectionValidator(DetectionValidator):







    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):











        super().__init__(dataloader, save_dir, args, _callbacks)
        



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
        

        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        

        if self.modality:
            LOGGER.info(f"初始化MultiModalDetectionValidator - 单模态验证模式: {self.modality}-only")
        else:
            LOGGER.info("初始化MultiModalDetectionValidator - 双模态验证模式")
        

        self.multimodal_config = None

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

    @smart_inference_mode()
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
                LOGGER.info(f"执行{total_channels}通道多模态YOLO模型warmup (RGB:3 + X:{x_channels})")
                model.warmup(imgsz=(1 if pt else self.args.batch, total_channels, imgsz, imgsz))
            else:

                LOGGER.info("执行6通道多模态YOLO模型warmup (默认)")
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
                        loss_items = model.loss(batch, preds)[1]
                        if loss_items.numel() != len(self._val_loss_names):
                            raise RuntimeError(
                                f"Validation loss dimension mismatch: got {loss_items.numel()} items from model.loss(), "
                                f"but validator expects {len(self._val_loss_names)} non-distill items {self._val_loss_names}."
                            )
                        self.loss += loss_items
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
            results = {**stats, **self._label_val_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)
                stats = self.eval_json(stats)
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def _parse_multimodal_config(self):












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
        

        config = self._get_default_multimodal_config()
        
        if not self.data:
            LOGGER.warning("验证器未提供数据配置，使用默认多模态配置: rgb+depth")
            return config
        

        if 'modality_used' in self.data:
            modality_used = self.data['modality_used']


            if not isinstance(modality_used, list):
                raise ValueError(f"验证配置中'modality_used'必须是列表格式，当前为: {type(modality_used)}")

            if len(modality_used) != 2:
                raise ValueError(f"多模态验证要求恰好2个模态，当前提供: {len(modality_used)} - {modality_used}")

            if 'rgb' not in modality_used:
                raise ValueError(f"多模态验证必须包含'rgb'模态，当前: {modality_used}")

            config['models'] = modality_used
            LOGGER.info(f"验证使用配置中的模态组合: {modality_used}")
        else:
            LOGGER.info(f"验证未找到'modality_used'配置，使用默认组合: {config['models']}")
        

        if 'modality' in self.data:
            modality_paths = self.data['modality']


            if not isinstance(modality_paths, dict):
                raise ValueError(f"验证配置中'modality'必须是字典格式，当前为: {type(modality_paths)}")


            modalities = {'rgb': 'images'}


            for modality in config['models']:
                if modality == 'rgb':
                    continue
                elif modality in modality_paths:
                    modalities[modality] = modality_paths[modality]
                else:
                    modalities[modality] = f'images_{modality}'
                    LOGGER.warning(f"验证未找到'{modality}'模态路径配置，使用默认: images_{modality}")

            config['modalities'] = modalities
            LOGGER.info(f"验证使用配置中的模态路径映射: {modalities}")
        else:

            x_modality = [m for m in config['models'] if m != 'rgb'][0]
            config['modalities']['rgb'] = 'images'
            config['modalities'][x_modality] = f'images_{x_modality}'
            LOGGER.info(f"验证未找到'modality'配置，生成默认路径映射: {config['modalities']}")
        
        return config
    
    def _get_x_modality_path(self, modality_name):













        if self.data and 'modality' in self.data:
            modality_paths = self.data['modality']
            if isinstance(modality_paths, dict) and modality_name in modality_paths:
                return modality_paths[modality_name]
        

        if self.data and 'modalities' in self.data:
            modalities = self.data['modalities']
            if isinstance(modalities, dict) and modality_name in modalities:
                return modalities[modality_name]
        

        return f'images_{modality_name}'
    
    def _determine_x_modality_from_data(self):














        if self.data and 'modality_used' in self.data:
            modality_used = self.data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                x_modalities = [m for m in modality_used if m != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"验证-从data.yaml的modality_used读取X模态: {x_modality}")
                    return x_modality


        if self.data and 'models' in self.data:
            models = self.data['models']
            if isinstance(models, list) and len(models) >= 2:
                x_modalities = [m for m in models if m != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"验证-从数据配置读取X模态: {x_modality}")
                    return x_modality
        

        if self.data and 'modality' in self.data:
            modality = self.data['modality']
            if isinstance(modality, dict):
                x_modalities = [k for k in modality.keys() if k != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"验证-从data.yaml的modality配置推断X模态: {x_modality}")
                    return x_modality


        if self.data and 'modalities' in self.data:
            modalities = self.data['modalities']
            if isinstance(modalities, dict):
                x_modalities = [k for k in modalities.keys() if k != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"验证-从modalities配置推断X模态: {x_modality}")
                    return x_modality


        if self.data and 'path' in self.data:
            try:
                import os
                data_path = self.data['path']
                if os.path.exists(data_path):

                    for item in os.listdir(data_path):
                        if item.startswith('images_') and item != 'images':
                            x_modality = item.replace('images_', '')
                            LOGGER.info(f"验证-从目录结构推断X模态: {x_modality}")
                            return x_modality
            except Exception as e:
                LOGGER.debug(f"验证-目录结构推断失败: {e}")
        

        LOGGER.warning("验证-无法自动确定X模态类型，使用默认值: depth")
        return 'depth'
    
    def _get_default_multimodal_config(self):







        if self.data and 'modality_used' in self.data:
            modality_used = self.data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                LOGGER.info(f"验证-从modality_used配置读取模态组合: {modality_used}")
                config = {
                    'models': modality_used,
                    'modalities': {
                        'rgb': 'images'
                    }
                }

                for modality in modality_used:
                    if modality != 'rgb':
                        if self.data and 'modality' in self.data and modality in self.data['modality']:
                            config['modalities'][modality] = self.data['modality'][modality]
                        else:
                            config['modalities'][modality] = f'images_{modality}'
                return config


        if self.data and 'models' in self.data:
            models = self.data['models']
            if isinstance(models, list) and len(models) >= 2:
                LOGGER.info(f"验证-从models配置读取模态组合: {models}")
                config = {
                    'models': models,
                    'modalities': {
                        'rgb': 'images'
                    }
                }

                for modality in models:
                    if modality != 'rgb':
                        config['modalities'][modality] = f'images_{modality}'
                return config
        

        x_modality = self._determine_x_modality_from_data()
        config = {
            'models': ['rgb', x_modality],
            'modalities': {
                'rgb': 'images',
                x_modality: f'images_{x_modality}'
            }
        }
        LOGGER.info(f"验证-生成默认多模态配置: rgb+{x_modality}")
        return config

    def get_desc(self):

        return ("%22s" + "%11s" * 7) % ("Class", "Images", "Instances", "Box(P", "R", "F1", "mAP50", "mAP50-95)")

    def print_results(self):

        pf = "%22s" + "%11i" * 2 + "%11.3g" * 5


        f1_arr = self.metrics.box.f1
        mf = float(f1_arr.mean()) if hasattr(f1_arr, 'mean') and len(f1_arr) else 0.0


        mr = self.metrics.mean_results()
        LOGGER.info(
            pf % ("all", self.seen, self.metrics.nt_per_class.sum(), mr[0], mr[1], mf, mr[2], mr[3])
        )

        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")


        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                p, r, ap50, ap = self.metrics.box.class_result(i)
                f1_i = float(self.metrics.box.f1[i]) if i < len(self.metrics.box.f1) else 0.0
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        p, r, f1_i, ap50, ap,
                    )
                )

    def build_dataset(self, img_path, mode="val", batch=None):















        if self.multimodal_config is None:
            self.multimodal_config = self._parse_multimodal_config()
            LOGGER.info(f"多模态验证配置解析完成 - 模态: {self.multimodal_config['models']}")
        

        modalities = self.multimodal_config['models']
        modalities_dict = self.multimodal_config['modalities']


        x_modalities = [m for m in modalities if m != 'rgb']
        x_modality = x_modalities[0] if x_modalities else None
        x_modality_dir = modalities_dict.get(x_modality) if x_modality else None


        stride = self.stride if hasattr(self, 'stride') and self.stride else 32


        if self.modality:

            LOGGER.info(f"构建多模态验证数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}")
            LOGGER.info(f"启用单模态验证: {self.modality}-only，将应用智能模态填充")
        else:

            LOGGER.info(f"构建多模态验证数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}")


        return build_yolo_dataset(
            self.args, img_path, batch, self.data,
            mode=mode,
            rect=True,
            stride=stride,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            modalities=modalities,

        )

    def init_metrics(self, model):











        super().init_metrics(model)
        

        if model and not hasattr(self, 'stride'):
            self.stride = max(int(de_parallel(model).stride.max() if hasattr(model, 'stride') else 0), 32)
        

        
    def preprocess(self, batch):













        batch = super().preprocess(batch)
        

        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        expected_channels = 3 + xch
        if batch["img"].shape[1] != expected_channels:
            LOGGER.warning(f"期望{expected_channels}通道输入（RGB:3 + X:{xch}），但收到 {batch['img'].shape[1]} 通道")
            return batch
        




        
        return batch

    def _apply_modality_ablation(self, batch):








        if not self.modality:
            return
        
        images = batch["img"]
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        
        if self.modality == "RGB":

            images[:, 3:3+xch, :, :] = 0
            LOGGER.debug(f"单模态RGB验证: X模态通道(3:{3+xch})已置零")
        elif self.modality == "X":

            images[:, 0:3, :, :] = 0
            LOGGER.debug("单模态X验证: RGB通道(0:3)已置零")
        else:

            images[:, 0:3, :, :] = 0
            LOGGER.debug(f"单模态{self.modality}验证: RGB通道(0:3)已置零")
        
        batch["img"] = images
        
    def plot_val_samples(self, batch, ni):











        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities, visualize_x_to_3ch, concat_side_by_side,
            duplicate_bboxes_for_side_by_side, ensure_batch_idx_long, resolve_x_modality
        )
        

        multimodal_images = batch["img"]
        

        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        

        rgb_images, x_images = split_modalities(multimodal_images, xch)
        

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
        

        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        

        rgb_images, x_images = split_modalities(multimodal_images, xch)
        

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

