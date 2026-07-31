from __future__ import annotations
# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
from copy import copy

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.nn.mm.pruning.trainability import (
    find_frozen_floating_parameters,
    restore_parameter_trainability,
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLOMultiModalImageDataset
from ultralytics.nn.mm.complexity import (
    compute_default_multimodal_complexity_report,
    log_default_complexity,
)
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.patches import torch_load
from ultralytics.nn.mm.utils import normalize_modality_token
from ultralytics.engine.afss import AFSSConfig, AFSSRuntime


class MultiModalDetectionTrainer(DetectionTrainer):













    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):








        if overrides is None:
            overrides = {}
        overrides["task"] = "detect"
        super().__init__(cfg, overrides, _callbacks)
        


        self.modality = normalize_modality_token(getattr(self.args, "modality", None))

        self.args.modality = self.modality
        

        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None


        self._multimodal_config_logged = False


        if self.modality:
            LOGGER.info(f"初始化MultiModalDetectionTrainer - 单模态训练模式: {self.modality}-only")
        else:
            LOGGER.info("初始化MultiModalDetectionTrainer - 双模态训练模式")


        self.distill_runtime = None
        self._distill_student_collector = None
        self._distill_cfg = self._parse_distill_arg()
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

        task_name = str(getattr(self.args, "task", "detect"))
        setattr(self.args, "afss_task_name", task_name)

    def _setup_train(self, world_size):

        super()._setup_train(world_size)
        self._init_distill_runtime()

        self._distill_loss_names = ()
        self._distill_epoch_accum = {}
        self._distill_epoch_count = 0
        if self.distill_runtime is not None:
            from ultralytics.nn.mm.distill.adapters import YOLOMMDetectDistillAdapter
            self._distill_adapter = YOLOMMDetectDistillAdapter(
                runtime=self.distill_runtime,
                config=self.distill_runtime.config,
                student_model=de_parallel(self.model),
                trainer=self,
            )


            _, mode = self._distill_cfg
            distill_names = []
            if mode in ("output", "both"):
                distill_names.extend(["d_out", "d_out_cls", "d_out_loc"])
            if mode in ("feature", "both"):
                distill_names.extend(["d_feat", "d_feat_fg", "d_feat_bg", "d_feat_cwd", "d_feat_ctx"])
            self._distill_loss_names = tuple(distill_names)
            self.add_callback("on_train_epoch_end", self._log_distill_epoch_summary)

    def compute_batch_loss(self, batch):





        if self.distill_runtime is None:
            return self.model(batch)


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
            distill_values["d_out_loc"] = float(distill_items.get("distill_output_loc", _zero))
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
                LOGGER.info(f"RGB单模态训练，动态确定X模态: {x_modality}")
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
        

        config = self._get_default_multimodal_config()
        
        if not self.data:
            LOGGER.warning("训练器未提供数据配置，使用默认多模态配置: rgb+depth")
            return config
        

        if 'modality_used' in self.data:
            models = self.data['modality_used']


            if not isinstance(models, list):
                raise ValueError(f"data.yaml中的'modality_used'必须是列表格式，当前为: {type(models)}")

            if len(models) != 2:
                raise ValueError(f"多模态检测要求恰好2个模态，当前提供: {len(models)} - {models}")

            if 'rgb' not in models:
                raise ValueError(f"多模态组合必须包含'rgb'模态，当前: {models}")

            config['models'] = models
            LOGGER.info(f"从data.yaml的modality_used读取模态组合: {models}")
        elif 'models' in self.data:

            models = self.data['models']


            if not isinstance(models, list):
                raise ValueError(f"data.yaml中的'models'必须是列表格式，当前为: {type(models)}")

            if len(models) != 2:
                raise ValueError(f"多模态检测要求恰好2个模态，当前提供: {len(models)} - {models}")

            if 'rgb' not in models:
                raise ValueError(f"多模态组合必须包含'rgb'模态，当前: {models}")

            config['models'] = models
            LOGGER.info(f"使用配置中的模态组合: {models}")
        else:
            LOGGER.debug(f"未找到'modality_used'或'models'配置，使用默认组合: {config['models']}")
        

        if 'modality' in self.data:
            modalities = self.data['modality']


            if not isinstance(modalities, dict):
                raise ValueError(f"data.yaml中的'modality'必须是字典格式，当前为: {type(modalities)}")


            for modality in config['models']:
                if modality not in modalities:
                    if modality == 'rgb':
                        modalities[modality] = 'images'
                        LOGGER.debug(f"'{modality}'模态路径未配置，使用默认: images")
                    else:
                        modalities[modality] = f'images_{modality}'
                        LOGGER.debug(f"'{modality}'模态路径未配置，使用默认: images_{modality}")

            config['modalities'] = modalities
            LOGGER.info(f"从data.yaml的modality读取路径映射: {modalities}")
        elif 'modalities' in self.data:

            modalities = self.data['modalities']


            if not isinstance(modalities, dict):
                raise ValueError(f"data.yaml中的'modalities'必须是字典格式，当前为: {type(modalities)}")


            for modality in config['models']:
                if modality not in modalities:
                    if modality == 'rgb':
                        modalities[modality] = 'images'
                        LOGGER.debug(f"'{modality}'模态路径未配置，使用默认: images")
                    else:
                        modalities[modality] = f'images_{modality}'
                        LOGGER.debug(f"'{modality}'模态路径未配置，使用默认: images_{modality}")

            config['modalities'] = modalities
            LOGGER.info(f"使用配置中的模态路径映射: {modalities}")
        else:

            x_modality = [m for m in config['models'] if m != 'rgb'][0]
            config['modalities']['rgb'] = 'images'
            config['modalities'][x_modality] = f'images_{x_modality}'
            LOGGER.debug(f"未找到'modality'或'modalities'配置，生成默认路径映射: {config['modalities']}")
        


        x_modality = [m for m in config['models'] if m != 'rgb'][0]
        LOGGER.info(f"✅ 使用用户配置的X模态: {x_modality} (配置驱动，支持任意模态类型)")
        
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
                    LOGGER.info(f"从data.yaml的modality_used读取X模态: {x_modality}")
                    return x_modality


        if self.data and 'models' in self.data:
            models = self.data['models']
            if isinstance(models, list) and len(models) >= 2:
                x_modalities = [m for m in models if m != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"从数据配置读取X模态: {x_modality}")
                    return x_modality
        

        if self.data and 'modality' in self.data:
            modality = self.data['modality']
            if isinstance(modality, dict):
                x_modalities = [k for k in modality.keys() if k != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"从data.yaml的modality配置推断X模态: {x_modality}")
                    return x_modality


        if self.data and 'modalities' in self.data:
            modalities = self.data['modalities']
            if isinstance(modalities, dict):
                x_modalities = [k for k in modalities.keys() if k != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"从modalities配置推断X模态: {x_modality}")
                    return x_modality


        if self.data and 'path' in self.data:
            try:
                import os
                data_path = self.data['path']
                if os.path.exists(data_path):

                    for item in os.listdir(data_path):
                        if item.startswith('images_') and item != 'images':
                            x_modality = item.replace('images_', '')
                            LOGGER.info(f"从目录结构推断X模态: {x_modality}")
                            return x_modality
            except Exception as e:
                LOGGER.debug(f"目录结构推断失败: {e}")
        

        LOGGER.warning("无法自动确定X模态类型，使用默认值: depth")
        return 'depth'
    
    def _get_default_multimodal_config(self):







        if self.data and 'modality_used' in self.data:
            modality_used = self.data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                LOGGER.info(f"从data.yaml读取模态组合: {modality_used}")
                config = {
                    'models': modality_used,
                    'modalities': {}
                }


                if 'modality' in self.data and isinstance(self.data['modality'], dict):
                    modality_paths = self.data['modality']
                    for mod in modality_used:
                        if mod in modality_paths:
                            config['modalities'][mod] = modality_paths[mod]
                        else:

                            config['modalities'][mod] = 'images' if mod == 'rgb' else f'images_{mod}'
                    LOGGER.info(f"从data.yaml读取路径映射: {config['modalities']}")
                else:

                    for mod in modality_used:
                        config['modalities'][mod] = 'images' if mod == 'rgb' else f'images_{mod}'
                    LOGGER.info(f"生成默认路径映射: {config['modalities']}")

                return config


        if self.data and 'models' in self.data:
            models = self.data['models']
            if isinstance(models, list) and len(models) >= 2:
                LOGGER.info(f"从数据配置读取模态组合: {models}")
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
        LOGGER.info(f"生成默认多模态配置: rgb+{x_modality}")
        return config
    
    def _validate_modality_compatibility(self):






        if not self.modality:
            return
        

        available_modalities = []
        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            available_modalities = self.multimodal_config.get('models', [])
        elif self.data and 'models' in self.data:
            available_modalities = self.data['models']
        

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
                        f"指定的modality '{self.modality}' 不在可用模态列表中: {available_modalities}。"
                        f"请检查数据配置或modality参数。"
                    )
                LOGGER.info(f"✅ 模态兼容性验证通过: {self.modality} 在可用模态 {available_modalities} 中")
        else:

            LOGGER.warning(f"⚠️  无法验证modality '{self.modality}' 的兼容性，未找到可用模态配置")

    def build_dataset(self, img_path, mode="train", batch=None):












        self.multimodal_config = self._parse_multimodal_config()


        self._validate_modality_compatibility()


        x_modality = [m for m in self.multimodal_config['models'] if m != 'rgb'][0]
        x_modality_dir = self.multimodal_config['modalities'][x_modality]


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
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=getattr(self.args, 'enable_self_modal_generation', False)
        )

    def get_validator(self):









        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")


        from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator

        return MultiModalDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )

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


        batch_idx = ensure_batch_idx_long(batch["batch_idx"])
        batch["batch_idx"] = batch_idx


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

    def plot_metrics(self):





        from ultralytics.utils.plotting import plot_distill_results


        super().plot_metrics()


        if self._distill_cfg is not None:
            plot_distill_results(file=self.csv, family="yolomm", on_plot=self.on_plot)

        LOGGER.info("多模态训练指标绘制完成")
    
    def get_model(self, cfg=None, weights=None, verbose=True):













        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        

        if self.is_dual_modal:

            x_channels = self.data.get('Xch', 3)
            channels = 3 + x_channels
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"多模态模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch总输入")
        else:

            channels = 3
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"单模态模型初始化: {self.modality or 'RGB'}(3ch)")
        

        cfg_dict = None
        if isinstance(cfg, str):
            try:
                from ultralytics.nn.tasks import yaml_model_load
                cfg_dict = yaml_model_load(cfg)
            except Exception:
                cfg_dict = None
        elif isinstance(cfg, dict):
            from copy import deepcopy
            cfg_dict = deepcopy(cfg)

        if cfg_dict is not None:

            cfg_dict['dataset_config'] = dict(self.data)
            model = DetectionModel(cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        else:
            model = DetectionModel(cfg, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        

        if hasattr(model, 'multimodal_router') and model.multimodal_router:
            model.multimodal_router.update_dataset_config(self.data)
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"已更新MultiModalRouter的数据集配置，Xch={self.data.get('Xch', 3)}")

        if hasattr(model, 'mm_router') and model.mm_router and self.modality:

            model.mm_router.set_runtime_params(
                self.modality,
                strategy=getattr(self.args, 'ablation_strategy', None),
                seed=getattr(self.args, 'seed', None),
            )
        
        if weights:
            model.load(weights)

        try:
            imgsz = int(getattr(self.args, "imgsz", 640))
            report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
            log_default_complexity(model, report, LOGGER)
        except Exception as e:
            LOGGER.warning(f"模型复杂度统计失败（可忽略，不影响训练）：{e}")
        
        return model

    def save_model(self):






        super().save_model()


        if hasattr(self, 'multimodal_config'):
            ckpt = torch_load(self.last, map_location='cpu')
            ckpt['multimodal_config'] = self.multimodal_config
            ckpt['modality'] = self.modality
            torch.save(ckpt, self.last)


            if self.best.exists():
                ckpt_best = torch_load(self.best, map_location='cpu')
                ckpt_best['multimodal_config'] = self.multimodal_config
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
            LOGGER.info("最终评估完成 - 多模态训练")



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

        config = load_distill_config(yaml_path)
        student = de_parallel(self.model)
        self.distill_runtime = DistillRuntime(
            config=config,
            mode=mode,
            family="yolomm",
            student_model=student,
            device=student.device if hasattr(student, 'device') else next(student.parameters()).device,
        )
        self._distill_student_collector = self.distill_runtime.register_student_hooks(student)






def require_pruned_checkpoint(ckpt: dict) -> dict:











    prune_info = ckpt.get("prune_info")
    if not isinstance(prune_info, dict) or prune_info.get("is_pruned") is not True:
        raise ValueError(
            'finetrain 仅接受带 `prune_info={"is_pruned": True}` 的剪枝权重。'
        )
    return prune_info

class PrunedMultiModalDetectionTrainer(MultiModalDetectionTrainer):









    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.prune_info = None


        if getattr(self.args, "model_scale", None):
            raise ValueError(
                "finetrain 不支持 model_scale/scale; 剪枝后结构必须直接来自 checkpoint。"
            )

    def get_model(self, cfg=None, weights=None, verbose=True):

        if isinstance(self.model, torch.nn.Module):

            if not isinstance(self.prune_info, dict) or self.prune_info.get("is_pruned") is not True:
                raise ValueError("剪枝后训练器收到内存模型，但缺少合法 prune_info。")
            return self.model


        if not str(self.model).endswith(".pt"):
            raise ValueError("finetrain 仅接受带 prune_info 的剪枝后 .pt 权重。")

        from ultralytics.nn.tasks import attempt_load_one_weight

        weights_obj, ckpt = attempt_load_one_weight(self.model)
        self.prune_info = require_pruned_checkpoint(ckpt)
        self.model = weights_obj
        return self.model

    def setup_model(self):





        if not isinstance(self.model, torch.nn.Module):
            self.model = self.get_model()

        if not isinstance(self.prune_info, dict) or self.prune_info.get("is_pruned") is not True:
            raise ValueError("剪枝后训练器收到模型，但缺少合法 prune_info。")



        restored = restore_parameter_trainability(self.model)
        remaining = find_frozen_floating_parameters(self.model)
        if remaining:
            raise RuntimeError(
                "Legacy pruned checkpoint still contains frozen floating parameters "
                f"after finetrain normalization: {remaining[:20]}"
            )
        if restored:
            LOGGER.info(
                f"[PrunedFinetrain] Restored trainability for {len(restored)} parameters "
                "from legacy pruned checkpoint before trainer freeze policy is applied."
            )

        try:
            imgsz = int(getattr(self.args, "imgsz", 640))
            report = compute_default_multimodal_complexity_report(self.model, imgsz=imgsz)
            log_default_complexity(self.model, report, LOGGER)
        except Exception as e:
            LOGGER.warning(f"剪枝后训练模型复杂度统计失败（可忽略，不影响训练）：{e}")

    def save_model(self):

        super().save_model()

        if not isinstance(self.prune_info, dict) or self.prune_info.get("is_pruned") is not True:
            raise ValueError("PrunedMultiModalDetectionTrainer.save_model() 缺少合法 prune_info。")

        targets = [self.last]
        if self.best.exists():
            targets.append(self.best)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            epoch_pt = self.wdir / f"epoch{self.epoch}.pt"
            if epoch_pt.exists():
                targets.append(epoch_pt)

        for path in targets:
            if path.exists():
                ckpt = torch_load(path, map_location="cpu")

                ckpt["prune_info"] = self.prune_info
                torch.save(ckpt, path)

