from __future__ import annotations





from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.multimodal import MultiModalPredictor, MultiModalSegmentPredictor
from ultralytics.engine.multimodal import MultiModalOBBPredictor, MultiModalPosePredictor, MultiModalClassifyPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER

class YOLOMMPredictor(BasePredictor):














    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):








        super().__init__(cfg, overrides, _callbacks)
        self.predictor = None

    def setup_model(self, model, verbose=True, debug=False):








        super().setup_model(model, verbose)


        self.predictor = MultiModalPredictor(
            model=self.model,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            device=str(self.device),
            verbose=verbose,
            debug=debug,
            font_size=getattr(self.args, 'font_size', None),
            show_filename=getattr(self.args, 'show_filename', False)
        )

        if verbose:
            LOGGER.info("YOLOMMPredictor: 新推理引擎已初始化")

    def __call__(
        self,
        rgb_source: Union[str, Path, List[Union[str, Path]]] = None,
        x_source: Union[str, Path, List[Union[str, Path]]] = None,
        stream: bool = False,
        strict_match: bool = True,
        **kwargs: Any
    ):














        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")


        if x_source is None and isinstance(rgb_source, (list, tuple)) and len(rgb_source) == 2:
            rgb_source, x_source = rgb_source[0], rgb_source[1]


        save = kwargs.get('save', getattr(self.args, 'save', False))
        save_txt = kwargs.get('save_txt', getattr(self.args, 'save_txt', False))
        save_json = kwargs.get('save_json', getattr(self.args, 'save_json', False))
        save_dir = kwargs.get('save_dir', getattr(self.args, 'save_dir', None))


        conf = kwargs.get('conf', None)
        iou = kwargs.get('iou', None)
        max_det = kwargs.get('max_det', None)
        crop = kwargs.get('crop', getattr(self.args, 'crop', False))
        font_size = kwargs.pop('font_size', None)
        show_filename = kwargs.pop('show_filename', None)

        if save_dir is None:
            save_dir = getattr(self, 'save_dir', None)


        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=stream,
            strict_match=strict_match,
            save=save,
            save_txt=save_txt,
            save_json=save_json,
            save_dir=save_dir,
            conf=conf,
            iou=iou,
            max_det=max_det,
            crop=crop,
            font_size=font_size,
            show_filename=show_filename,
            **kwargs
        )

    def predict_cli(self, rgb_source=None, x_source=None, strict_match: bool = True):











        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")


        if x_source is None and isinstance(rgb_source, (list, tuple)) and len(rgb_source) == 2:
            rgb_source, x_source = rgb_source[0], rgb_source[1]


        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            strict_match=strict_match,
            stream=False,
            save=True,
            save_txt=getattr(self.args, 'save_txt', False),
            save_json=getattr(self.args, 'save_json', False),
            save_dir=getattr(self, 'save_dir', None)
        )

class YOLOMMSegPredictor(BasePredictor):









    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.predictor = None

    def setup_model(self, model, verbose=True, debug=False):
        super().setup_model(model, verbose)

        self.predictor = MultiModalSegmentPredictor(
            model=self.model,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            device=str(self.device),
            verbose=verbose,
            debug=debug,
            font_size=getattr(self.args, 'font_size', None),
            show_filename=getattr(self.args, 'show_filename', False)
        )

        if verbose:
            LOGGER.info("YOLOMMSegPredictor: 分割推理引擎已初始化")

    def __call__(
        self,
        rgb_source: Union[str, Path, List[Union[str, Path]]] = None,
        x_source: Union[str, Path, List[Union[str, Path]]] = None,
        stream: bool = False,
        **kwargs: Any
    ):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        save = kwargs.get('save', getattr(self.args, 'save', False))
        save_txt = kwargs.get('save_txt', getattr(self.args, 'save_txt', False))
        save_dir = kwargs.get('save_dir', getattr(self.args, 'save_dir', None))

        conf = kwargs.get('conf', None)
        iou = kwargs.get('iou', None)
        max_det = kwargs.get('max_det', None)
        crop = kwargs.get('crop', getattr(self.args, 'crop', False))
        font_size = kwargs.pop('font_size', None)
        show_filename = kwargs.pop('show_filename', None)

        if save_dir is None:
            save_dir = getattr(self, 'save_dir', None)

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=stream,
            save=save,
            save_txt=save_txt,
            save_dir=save_dir,
            conf=conf,
            iou=iou,
            max_det=max_det,
            crop=crop,
            font_size=font_size,
            show_filename=show_filename,
            **kwargs
        )

    def predict_cli(self, rgb_source=None, x_source=None):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=False,
            save=True,
            save_txt=getattr(self.args, 'save_txt', False),
            save_dir=getattr(self, 'save_dir', None)
        )

class YOLOMMOBBPredictor(BasePredictor):




    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.predictor = None

    def setup_model(self, model, verbose=True, debug=False):
        super().setup_model(model, verbose)

        self.predictor = MultiModalOBBPredictor(
            model=self.model,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            device=str(self.device),
            verbose=verbose,
            debug=debug,
            font_size=getattr(self.args, 'font_size', None),
            show_filename=getattr(self.args, 'show_filename', False)
        )

        if verbose:
            LOGGER.info("YOLOMMOBBPredictor: OBB推理引擎已初始化")

    def __call__(
        self,
        rgb_source=None,
        x_source=None,
        stream: bool = False,
        **kwargs
    ):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        save = kwargs.get('save', getattr(self.args, 'save', False))
        save_txt = kwargs.get('save_txt', getattr(self.args, 'save_txt', False))
        save_dir = kwargs.get('save_dir', getattr(self.args, 'save_dir', None))
        conf = kwargs.get('conf', None)
        iou = kwargs.get('iou', None)
        max_det = kwargs.get('max_det', None)
        crop = kwargs.get('crop', getattr(self.args, 'crop', False))
        font_size = kwargs.pop('font_size', None)
        show_filename = kwargs.pop('show_filename', None)

        if save_dir is None:
            save_dir = getattr(self, 'save_dir', None)

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=stream,
            save=save,
            save_txt=save_txt,
            save_dir=save_dir,
            conf=conf,
            iou=iou,
            max_det=max_det,
            crop=crop,
            font_size=font_size,
            show_filename=show_filename,
            **kwargs
        )

    def predict_cli(self, rgb_source=None, x_source=None):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=False,
            save=True,
            save_txt=getattr(self.args, 'save_txt', False),
            save_dir=getattr(self, 'save_dir', None)
        )

class YOLOMMPosePredictor(BasePredictor):




    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.predictor = None

    def setup_model(self, model, verbose=True, debug=False):
        super().setup_model(model, verbose)

        self.predictor = MultiModalPosePredictor(
            model=self.model,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            device=str(self.device),
            verbose=verbose,
            debug=debug,
            font_size=getattr(self.args, 'font_size', None),
            show_filename=getattr(self.args, 'show_filename', False)
        )

        if verbose:
            LOGGER.info("YOLOMMPosePredictor: Pose推理引擎已初始化")

    def __call__(
        self,
        rgb_source=None,
        x_source=None,
        stream: bool = False,
        **kwargs
    ):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        save = kwargs.get('save', getattr(self.args, 'save', False))
        save_txt = kwargs.get('save_txt', getattr(self.args, 'save_txt', False))
        save_dir = kwargs.get('save_dir', getattr(self.args, 'save_dir', None))
        conf = kwargs.get('conf', None)
        iou = kwargs.get('iou', None)
        max_det = kwargs.get('max_det', None)
        crop = kwargs.get('crop', getattr(self.args, 'crop', False))
        font_size = kwargs.pop('font_size', None)
        show_filename = kwargs.pop('show_filename', None)

        if save_dir is None:
            save_dir = getattr(self, 'save_dir', None)

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=stream,
            save=save,
            save_txt=save_txt,
            save_dir=save_dir,
            conf=conf,
            iou=iou,
            max_det=max_det,
            crop=crop,
            font_size=font_size,
            show_filename=show_filename,
            **kwargs
        )

    def predict_cli(self, rgb_source=None, x_source=None):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=False,
            save=True,
            save_txt=getattr(self.args, 'save_txt', False),
            save_dir=getattr(self, 'save_dir', None)
        )

class YOLOMMClassifyPredictor(BasePredictor):




    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.predictor = None

    def setup_model(self, model, verbose=True, debug=False):
        super().setup_model(model, verbose)

        self.predictor = MultiModalClassifyPredictor(
            model=self.model,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            device=str(self.device),
            verbose=verbose,
            debug=debug,
            font_size=getattr(self.args, 'font_size', None),
            show_filename=getattr(self.args, 'show_filename', False)
        )

        if verbose:
            LOGGER.info("YOLOMMClassifyPredictor: 分类推理引擎已初始化")

    def __call__(
        self,
        rgb_source=None,
        x_source=None,
        stream: bool = False,
        **kwargs
    ):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        save = kwargs.get('save', getattr(self.args, 'save', False))
        save_txt = kwargs.get('save_txt', getattr(self.args, 'save_txt', False))
        save_dir = kwargs.get('save_dir', getattr(self.args, 'save_dir', None))
        crop = kwargs.get('crop', getattr(self.args, 'crop', False))
        font_size = kwargs.pop('font_size', None)
        show_filename = kwargs.pop('show_filename', None)

        if save_dir is None:
            save_dir = getattr(self, 'save_dir', None)

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=stream,
            save=save,
            save_txt=save_txt,
            save_dir=save_dir,
            crop=crop,
            font_size=font_size,
            show_filename=show_filename,
            **kwargs
        )

    def predict_cli(self, rgb_source=None, x_source=None):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=False,
            save=True,
            save_txt=getattr(self.args, 'save_txt', False),
            save_dir=getattr(self, 'save_dir', None)
        )

__all__ = [
    'YOLOMMPredictor',
    'YOLOMMSegPredictor',
    'YOLOMMOBBPredictor',
    'YOLOMMPosePredictor',
    'YOLOMMClassifyPredictor',
]

