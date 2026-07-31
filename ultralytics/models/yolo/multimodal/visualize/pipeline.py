from __future__ import annotations
"""
Visualization Pipeline (Refactor Skeleton)

This module introduces a componentized pipeline for multimodal visualization that
orchestrates input resolution, router coordination, preprocessing, layer resolution,
and method execution via existing VisualizationManager. It is designed to be extended
with plugin-style visualizers in future iterations while preserving current behavior.

Current scope:
- Provide a drop-in `VisualizationPipeline.run(...)` entry that standardizes inputs
  and delegates to `VisualizationManager` with correct layer naming conventions.
- Introduce RouterAdapter, InputResolver, Preprocessor (minimal), LayerResolver, and
  a lightweight logger to prepare for progressive migration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ultralytics.utils import LOGGER





class RouterAdapter:


    def __init__(self, model: Any) -> None:
        self._model = model
        self._router = self._resolve_router(model)

    @staticmethod
    def _resolve_router(model: Any):

        for attr_chain in (
            ("mm_router",),
            ("multimodal_router",),
            ("model", "mm_router"),
            ("model", "multimodal_router"),
        ):
            m = model
            ok = True
            for a in attr_chain:
                if hasattr(m, a):
                    m = getattr(m, a)
                else:
                    ok = False
                    break
            if ok and m is not None:
                return m
        return None

    @property
    def available(self) -> bool:
        return self._router is not None

    def set_runtime_params(self, modality: Optional[str] = None, strategy: Optional[str] = None) -> None:
        if not self.available:
            return
        try:
            if strategy is None:
                self._router.set_runtime_params(modality)
            else:
                self._router.set_runtime_params(modality, strategy=strategy)
        except Exception as e:
            LOGGER.warning(f"RouterAdapter: set_runtime_params failed: {e}")

    def update_dataset_config(self, data: Optional[dict]) -> None:
        if not self.available or not data:
            return
        try:
            self._router.update_dataset_config(data)
        except Exception:

            pass

    def restore(self) -> None:
        if not self.available:
            return
        try:

            if hasattr(self._router, "restore"):
                self._router.restore()
        except Exception as e:
            LOGGER.warning(f"RouterAdapter: restore failed: {e}")

class InputResolver:


    VALID_KEYS = ("rgb", "x", "thermal", "depth", "ir", "infrared")

    @staticmethod
    def resolve(
        source: Union[str, Path, np.ndarray, Dict[str, Any], List[str], List[np.ndarray]],
        modality: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:










        from .utils import load_image


        if isinstance(source, list) and len(source) == 2 and all(isinstance(s, (str, Path)) for s in source):
            return {"rgb": load_image(source[0]), "x": load_image(source[1])}


        if isinstance(source, (str, Path)) and modality in {"rgb", "x"}:
            return {modality: load_image(source)}


        if isinstance(source, dict):
            out: Dict[str, np.ndarray] = {}
            for k, v in source.items():
                k_low = str(k).lower()
                if k_low not in InputResolver.VALID_KEYS:
                    continue
                if isinstance(v, (str, Path)):
                    out["rgb" if k_low == "rgb" else ("x" if k_low in {"x", "thermal", "ir", "infrared", "depth"} else k_low)] = load_image(v)
                elif isinstance(v, np.ndarray):
                    out["rgb" if k_low == "rgb" else ("x" if k_low in {"x", "thermal", "ir", "infrared", "depth"} else k_low)] = v
            if out:
                return out


        if isinstance(source, np.ndarray) or (
            isinstance(source, list) and source and isinstance(source[0], np.ndarray)
        ):
            return {"__passthrough__": source}

        raise ValueError(
            "Unsupported source format. Provide two paths [rgb,x], a single path with modality='rgb'|'x',\n"
            "a dict with keys among {'rgb','x','thermal','ir','infrared','depth'}, or a numpy array."
        )

class Preprocessor:


    @staticmethod
    def ensure_numpy_dict(data: Dict[str, Any]) -> Dict[str, Any]:

        return data

class LayerResolver:


    @staticmethod
    def to_manager_layers(layers: List[int], method: str) -> List[str]:
        if method == "feature_map":

            return [f"model.{i}" for i in layers]

        return [str(i) for i in layers]

@dataclass
class PipelineContext:
    router: RouterAdapter
    input_dict: Dict[str, Any]
    method: str
    layers: List[int]
    alg: Optional[str]
    save: bool
    project: str
    name: str
    modality: Optional[str]
    data_cfg: Optional[dict]

class PipelineLogger:
    @staticmethod
    def summary(ctx: PipelineContext, out_dir: Optional[Path] = None) -> None:
        try:
            msg = (
                f"vis: method={ctx.method}, layers={ctx.layers}, alg={ctx.alg}, save={ctx.save}, "
                f"project={ctx.project}/{ctx.name}, modality={ctx.modality}, router={'on' if ctx.router.available else 'off'}"
            )
            if out_dir is not None:
                msg += f", out={out_dir}"
            LOGGER.info(msg)
        except Exception:
            pass





class VisualizationPipeline:


    def __init__(self, model: Any) -> None:
        self.model = model

    def run(
        self,
        source: Union[str, Path, np.ndarray, Dict[str, Any], List[str], List[np.ndarray]],
        method: str = "heatmap",
        layers: Optional[List[int]] = None,
        alg: Optional[str] = None,
        modality: Optional[str] = None,
        save: bool = True,
        project: str = "runs/visualize",
        name: str = "exp",
        data_cfg: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Any]:

        resolved = InputResolver.resolve(source, modality=modality)


        router = RouterAdapter(self.model)
        if router.available:
            router.update_dataset_config(data_cfg)
            router.set_runtime_params(modality)


        resolved = Preprocessor.ensure_numpy_dict(resolved)


        if not layers or not isinstance(layers, list) or not all(isinstance(i, int) for i in layers):
            raise ValueError("layers must be a non-empty List[int] for pipeline visualization")
        manager_layers = LayerResolver.to_manager_layers(layers, method=method)


        ctx = PipelineContext(
            router=router,
            input_dict=resolved,
            method=method,
            layers=layers,
            alg=alg,
            save=save,
            project=project,
            name=name,
            modality=modality,
            data_cfg=data_cfg,
        )


        from .manager import VisualizationManager

        vm = VisualizationManager(model=self.model, project=project, name=name)


        if "__passthrough__" in resolved:
            vm_source = resolved["__passthrough__"]
        else:
            vm_source = resolved

        try:
            results = vm.visualize(
                source=vm_source,
                method=method,
                layers=[m for m in manager_layers],
                save=save,
                alg=alg or "gradcam",
                **kwargs,
            )

            PipelineLogger.summary(ctx, out_dir=vm.output_dir)
            return results
        finally:

            router.restore()

