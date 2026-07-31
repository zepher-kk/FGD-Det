from __future__ import annotations
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
RT-DETR MultiModal trainer module.

This module provides the RTDETRMMTrainer class for training multi-modal RT-DETR models
with support for RGB+X modality inputs.
"""

import torch
from copy import copy
from pathlib import Path
from typing import Optional

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.dataset import YOLOMultiModalImageDataset
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel, log_multimodal_model_complexity
from ultralytics.utils.patches import torch_load
from ultralytics.nn.mm.utils import normalize_modality_token
from ultralytics.engine.afss import AFSSConfig, AFSSRuntime
from ultralytics.utils.torch_utils import torch_distributed_zero_first

from .utils import detect_mm, require_multimodal

class RTDETRMMTrainer(DetectionTrainer):





































    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):








        if overrides is None:
            overrides = {}
        overrides["task"] = "detect"
        


        cfg_model = ""
        if isinstance(cfg, dict):
            cfg_model = str(cfg.get("model", ""))
        else:
            cfg_model = str(getattr(cfg, "model", "")) if cfg is not None else ""
        model_name = str(overrides.get("model", cfg_model))


        self.mm_evidence = require_multimodal(model_name, who="RTDETRMMTrainer").to_dict()
        self.is_multimodal = True

        self.modality = normalize_modality_token(overrides.get("modality", None))
        overrides["modality"] = self.modality
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        self.multimodal_config = None
        

        super().__init__(cfg, overrides, _callbacks)
        

        self.modality = normalize_modality_token(getattr(self.args, "modality", None))

        self.args.modality = self.modality
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        

        if self.modality:
            LOGGER.info(f"Initializing RTDETRMMTrainer - Single-modal training mode: {self.modality}-only")
        else:
            LOGGER.info("Initializing RTDETRMMTrainer - Dual-modal training mode (RGB+X)")


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


        self.distill_runtime = None
        self._distill_student_collector = None
        self._distill_cfg = self._parse_distill_arg()

    def _sync_afss_task_name(self):

        task_name = str(getattr(self.args, "task", "detect"))
        setattr(self.args, "afss_task_name", task_name)

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

    def _setup_train(self, world_size):

        super()._setup_train(world_size)
        self._init_distill_runtime()
        self._distill_loss_names = ()
        self._distill_epoch_accum = {}
        self._distill_epoch_count = 0
        if self.distill_runtime is not None:
            from ultralytics.nn.mm.distill.adapters import RTDETRMMDetectDistillAdapter
            self._distill_adapter = RTDETRMMDetectDistillAdapter(
                runtime=self.distill_runtime,
                config=self.distill_runtime.config,
                student_model=de_parallel(self.model),
                trainer=self,
            )
            _, mode = self._distill_cfg
            distill_names = []
            if mode in ("output", "both"):
                distill_names.extend(["d_out", "d_out_cls", "d_out_box"])
            if mode in ("feature", "both"):
                distill_names.extend(["d_feat", "d_feat_fg", "d_feat_bg", "d_feat_cwd", "d_feat_ctx"])
            self._distill_loss_names = tuple(distill_names)
            self.add_callback("on_train_epoch_end", self._log_distill_epoch_summary)

    def compute_batch_loss(self, batch):

        if self.distill_runtime is None:
            return self.model(batch)

        from ultralytics.utils.torch_utils import de_parallel

        student_model = de_parallel(self.model)
        self._distill_student_collector.reset()
        det_loss, det_items, student_preds = student_model.distill_forward(batch)

        teacher_outputs = self.distill_runtime.run_teachers(batch)

        self._distill_adapter.set_epoch_state(self.epoch, self.epochs)
        student_features = self._distill_student_collector.features
        distill_loss, distill_items = self._distill_adapter.compute_distill_loss(
            student_preds, student_features, teacher_outputs
        )

        total_loss = det_loss + distill_loss

        _zero = 0.0
        distill_values = {}
        if "d_out" in distill_items:
            distill_values["d_out"] = float(distill_items["d_out"])
            distill_values["d_out_cls"] = float(distill_items.get("distill_output_cls", _zero))
            distill_values["d_out_box"] = float(distill_items.get("distill_output_box", _zero))
        if "distill_feature" in distill_items:
            distill_values["d_feat"] = float(distill_items["distill_feature"])
            distill_values["d_feat_fg"] = float(distill_items.get("distill_feature_fg", _zero))
            distill_values["d_feat_bg"] = float(distill_items.get("distill_feature_bg", _zero))
            distill_values["d_feat_cwd"] = float(distill_items.get("distill_feature_cwd", _zero))
            distill_values["d_feat_ctx"] = float(distill_items.get("distill_feature_ctx", _zero))
        for k, v in distill_values.items():
            self._distill_epoch_accum[k] = self._distill_epoch_accum.get(k, 0.0) + v
        self._distill_epoch_count += 1

        return total_loss, det_items

    def _log_distill_epoch_summary(self, trainer):

        if not self._distill_loss_names or self._distill_epoch_count == 0:
            return
        n = self._distill_epoch_count
        parts = [f"{name}={self._distill_epoch_accum.get(name, 0.0) / n:.4f}"
                 for name in self._distill_loss_names]
        LOGGER.info(f"  Distill Epoch {self.epoch + 1}/{self.epochs}: {'  '.join(parts)}")

    def save_metrics(self, metrics):

        if self._distill_loss_names and self._distill_epoch_count > 0:
            n = self._distill_epoch_count
            for name in self._distill_loss_names:
                key = f"train/{name}"
                if key not in metrics:
                    metrics[key] = round(self._distill_epoch_accum.get(name, 0.0) / n, 5)
        super().save_metrics(metrics)
        self._distill_epoch_accum = {}
        self._distill_epoch_count = 0

    def _check_multimodal_model(self):






        model_name = str(self.args.model)
        return detect_mm(model_name).is_multimodal
    
    def build_dataset(self, img_path, mode="train", batch=None):














        if not self.is_multimodal:
            raise ValueError(
                "RTDETRMMTrainer is designed for multi-modal RT-DETR models only. "
                "For standard RT-DETR training, use RTDETRTrainer instead. "
                "RTDETRMM 的判定基于 YAML/CKPT 的内容判据（RGB/X/Dual 路由标记或 ckpt 元信息），不依赖文件名后缀。"
            )


        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)


        if not hasattr(self, 'multimodal_config') or self.multimodal_config is None:
            self.multimodal_config = self._parse_multimodal_config()
            LOGGER.info(f"多模态配置解析完成 - 模态: {self.multimodal_config['models']}")


        modalities = self.multimodal_config['models']
        

        x_modality = [m for m in self.multimodal_config['models'] if m != 'rgb'][0]
        x_modality_dir = self.multimodal_config['modalities'][x_modality]

        LOGGER.info(f"构建多模态数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}")


        if self.modality:
            self._validate_modality_compatibility()
            LOGGER.info(f"启用单模态训练: {self.modality}-only，将应用智能模态填充")


        return build_yolo_dataset(
            self.args, img_path, batch, self.data,
            mode=mode,
            rect=False,
            stride=gs,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=getattr(self.args, 'enable_self_modal_generation', False)
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

    def get_model(self, cfg: Optional[dict] = None, weights: Optional[str] = None, verbose: bool = True):














        from ultralytics.utils import RANK
        

        if self.is_dual_modal:

            x_channels = self.data.get('Xch', 3)
            channels = 3 + x_channels
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"多模态RT-DETR模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch总输入")
        else:

            channels = 3
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"单模态RT-DETR模型初始化: {self.modality or 'RGB'}(3ch)")
        

        self.data["channels"] = channels
        if verbose and RANK in {-1, 0}:
            LOGGER.info(f"已更新data[\"channels\"]为{channels}，确保验证器兼容性")
        

        cfg_in = cfg or self.args.model
        cfg_dict = None
        if isinstance(cfg_in, str):
            try:
                from ultralytics.nn.tasks import yaml_model_load
                cfg_dict = yaml_model_load(cfg_in)
            except Exception:
                cfg_dict = None
        elif isinstance(cfg_in, dict):
            from copy import deepcopy
            cfg_dict = deepcopy(cfg_in)

        if cfg_dict is not None:
            cfg_dict['dataset_config'] = dict(self.data)
            model = RTDETRDetectionModel(
                cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1
            )
        else:
            model = RTDETRDetectionModel(
                cfg_in, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1
            )
        

        if weights:
            model.load(weights)

        try:
            if hasattr(model, 'multimodal_router') and model.multimodal_router:
                model.multimodal_router.update_dataset_config(self.data)
        except Exception as e:
            LOGGER.warning(f"RTDETRMMTrainer: update_dataset_config failed: {e}")


        try:
            imgsz = int(getattr(self.args, 'imgsz', 640))
            log_multimodal_model_complexity(model, imgsz=imgsz, modality=self.modality)
        except Exception as e:
            LOGGER.warning(f"模型复杂度统计失败（可忽略，不影响训练）：{e}")

        return model
    
    def preprocess_batch(self, batch):













        batch = super().preprocess_batch(batch)
        
        if not self.is_multimodal:
            return batch
        



        try:
            if self.modality and hasattr(self, 'model') and self.model is not None:

                if hasattr(self.model, 'mm_router') and self.model.mm_router:
                    self.model.mm_router.set_runtime_params(
                        self.modality,
                        strategy=getattr(self.args, 'ablation_strategy', None),
                        seed=getattr(self.args, 'seed', None),
                    )
        except Exception:

            pass

        return batch
    
    def get_validator(self):









        self.loss_names = ("giou_loss", "cls_loss", "l1_loss")

        from .val import RTDETRMMValidator


        validator = RTDETRMMValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args)
        )



        validator.data = self.data

        return validator
    
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
                LOGGER.info(f"RGB single-modal training, X modality: {x_modality}")
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
                    LOGGER.info(f"X模态单模态训练: {actual_x_modality}-only (从'X'解析)")
                else:

                    x_modality_path = self._get_x_modality_path(self.modality)
                    
                    config = {
                        'models': ['rgb', self.modality],
                        'modalities': {
                            'rgb': 'images',
                            self.modality: x_modality_path
                        }
                    }
                    LOGGER.info(f"X模态单模态训练: {self.modality}-only")
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

                LOGGER.info(f"Loaded multi-modal config: {modality_used}")
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
        LOGGER.info(f"Using default multi-modal config: rgb+{x_modality}")
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
        

        LOGGER.warning("Cannot determine X modality type, using default: depth")
        return 'depth'
    
    def _validate_modality_compatibility(self):






        if not self.modality:
            return
        

        available_modalities = []
        if self.multimodal_config:
            available_modalities = self.multimodal_config.get('models', [])
        

        if available_modalities:

            if self.modality == "X":

                x_modalities = [m for m in available_modalities if m != 'rgb']
                if x_modalities:
                    LOGGER.info(f"✅ 模态兼容性验证通过: '{self.modality}' 映射到 {x_modalities[0]}")
                else:
                    raise ValueError(
                        f"指定的modality '{self.modality}' 无法映射到有效的X模态。"
                        f"可用模态列表: {available_modalities}，但没有找到非RGB的X模态。"
                    )
            else:

                if self.modality not in available_modalities:
                    raise ValueError(
                        f"Specified modality '{self.modality}' not in available modalities: {available_modalities}. "
                        f"Please check data configuration or modality parameter."
                    )
                LOGGER.info(f"✅ Modality compatibility validated: {self.modality} in {available_modalities}")
        else:

            LOGGER.warning(f"⚠️  无法验证modality '{self.modality}' 的兼容性，未找到可用模态配置")
    
    def save_model(self):






        super().save_model()
        

        if self.is_multimodal and hasattr(self, 'multimodal_config'):

            ckpt = torch_load(self.last, map_location='cpu')
            ckpt['multimodal_config'] = self.multimodal_config
            ckpt['modality'] = self.modality
            ckpt['is_multimodal'] = True
            torch.save(ckpt, self.last)
            

            if self.best.exists():
                ckpt_best = torch_load(self.best, map_location='cpu')
                ckpt_best['multimodal_config'] = self.multimodal_config
                ckpt_best['modality'] = self.modality
                ckpt_best['is_multimodal'] = True
                torch.save(ckpt_best, self.best)
    
    def final_eval(self):




        super().final_eval()


        from ultralytics.utils.llm_export import export_final_val_llm_json

        try:
            export_final_val_llm_json(self)
        except Exception as e:
            LOGGER.warning(f"LLM JSON export failed: {e}")


        if self.is_multimodal and hasattr(self, 'multimodal_config') and self.multimodal_config:
            x_modality = [m for m in self.multimodal_config['models'] if m != 'rgb'][0]
            if self.modality:
                LOGGER.info(f"Final evaluation complete - Single-modal training: {self.modality}-only")
            else:
                LOGGER.info(f"Final evaluation complete - Dual-modal training: RGB+{x_modality}")
        else:
            LOGGER.info("Final evaluation complete - Standard RT-DETR training")



    def _parse_distill_arg(self):

        distill = getattr(self.args, "distill", None)
        if distill is None:
            return None
        if not isinstance(distill, (list, tuple)) or len(distill) != 2:
            raise ValueError(
                "distill must be [yaml_path, mode] (list/tuple of length 2), "
                f"got {type(distill).__name__}: {distill}"
            )
        yaml_path, mode = str(distill[0]), str(distill[1]).lower()
        if mode not in ("output", "feature", "both"):
            raise ValueError(f"distill mode must be output/feature/both, got '{mode}'")
        LOGGER.info(f"Distillation enabled: yaml={yaml_path}, mode={mode}")
        return yaml_path, mode

    def _init_distill_runtime(self):

        if self._distill_cfg is None:
            return
        yaml_path, mode = self._distill_cfg
        from ultralytics.nn.mm.distill.schema import load_distill_config
        from ultralytics.nn.mm.distill.runtime import DistillRuntime
        from ultralytics.utils.torch_utils import de_parallel

        config = load_distill_config(yaml_path)
        student = de_parallel(self.model)
        self.distill_runtime = DistillRuntime(
            config=config,
            mode=mode,
            family="rtdetrmm",
            student_model=student,
            device=student.device if hasattr(student, 'device') else next(student.parameters()).device,
        )
        self._distill_student_collector = self.distill_runtime.register_student_hooks(student)

    def plot_metrics(self):

        from ultralytics.utils.plotting import plot_distill_results


        super().plot_metrics()


        if self._distill_cfg is not None:
            plot_distill_results(file=self.csv, family="rtdetrmm", on_plot=self.on_plot)

