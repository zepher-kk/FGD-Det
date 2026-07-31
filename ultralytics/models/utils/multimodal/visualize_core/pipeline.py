from __future__ import annotations
"""Core visualization pipeline (family-agnostic, Fail-Fast)."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .exceptions import (
    DeviceMismatchError,
    MethodNotRegisteredError,
    VisualizationError,
)
from .input_resolver import InputResolver
from .layer_resolver import LayerResolver
from .registry import REGISTRY
from .router_adapter import RouterAdapter
from .cache import Cache
from .saver import Saver
from ultralytics.utils import LOGGER
from .types import PipelineContext, CoreVisualizationResult

class Pipeline:











    def __init__(self, model: Any, family: str) -> None:
        self.model = model
        self.family = str(family)
        self._cache = Cache()

    def _ensure_out_dir(
        self,
        *,
        project: Optional[str],
        name: Optional[str],
        out_dir: Optional[str],
    ) -> Path:











        if out_dir:
            try:
                LOGGER.warning("[visualize_core] 参数 out_dir 已废弃，请使用 project/name 代替。")
            except Exception:
                pass
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p


        if project and name:
            p = Path(project) / str(name)
            p.mkdir(parents=True, exist_ok=True)
            return p


        if project and not name:
            base = Path(project)
            base.mkdir(parents=True, exist_ok=True)
            i = 1
            while True:
                nm = "exp" if i == 1 else f"exp{i}"
                p = base / nm
                if not p.exists():
                    p.mkdir(parents=True, exist_ok=True)
                    return p
                i += 1


        if name and not project:
            base = Path("runs") / "visualize" / self.family
            base.mkdir(parents=True, exist_ok=True)
            p = base / str(name)
            p.mkdir(parents=True, exist_ok=True)
            return p


        base = Path("runs") / "visualize" / self.family
        base.mkdir(parents=True, exist_ok=True)
        i = 1
        while True:
            nm = "exp" if i == 1 else f"exp{i}"
            p = base / nm
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                return p
            i += 1

    def run(
        self,
        *,
        rgb_source: Optional[Any],
        x_source: Optional[Any],
        method: str,
        layers: List[int],
        modality: Optional[str],
        save: bool,
        overlay: Optional[str] = None,

        project: Optional[str] = None,
        name: Optional[str] = None,

        out_dir: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> List[CoreVisualizationResult]:

        try:
            current_device = next(self.model.parameters()).device.type
        except Exception:
            current_device = "cuda" if hasattr(self.model, "to") else "cpu"
        if device is not None and str(device) != current_device:
            raise DeviceMismatchError(current=current_device, requested=str(device))


        inputs = InputResolver.resolve(rgb_source, x_source, modality)


        valid_layers = LayerResolver.validate_indices(self.model, layers)
        layer_names = LayerResolver.to_names(valid_layers)


        out_path = self._ensure_out_dir(project=project, name=name, out_dir=out_dir)


        method_key = {
            "heat": "heat",
            "heatmap": "heat",
            "feature": "feature",
            "feature_map": "feature",
        }.get(str(method).lower(), None)

        if not method_key:
            raise MethodNotRegisteredError(str(method), REGISTRY.list())

        plugin = REGISTRY.get(method_key)


        router = RouterAdapter(self.model)
        if router.available:
            router.update_dataset_config(kwargs.get("data_cfg"))
            router.set_runtime_params(modality)
            try:
                router.log_summary(prefix=f"vis[{self.family}]")
            except Exception:
                pass


        if isinstance(inputs, list):
            all_results: list[CoreVisualizationResult] = []
            for sample in inputs:
                try:
                    ikey = sample.get('img_key', None) if isinstance(sample, dict) else None
                    res = plugin.run(
                        model=self.model,
                        inputs=sample,
                        layers=valid_layers,
                        layer_names=layer_names,
                        save=save,
                        out_dir=out_path,
                        modality=modality,
                        family=self.family,
                        overlay=overlay,
                        img_key=ikey,
                        **kwargs,
                    )
                    if save and res:
                        saved = Saver.save(results=res, out_dir=out_path, method=method_key)
                        try:
                            LOGGER.info(f"可视化结果已保存到: {str(out_path)}（{len(saved)}个文件）")
                        except Exception:
                            pass
                    all_results.extend(res)
                finally:
                    router.restore()
            return all_results


        cache_key = self._cache.make_key(
            family=self.family,
            method=method_key,
            layers=valid_layers,
            modality=modality,
            extra={**{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}, **({"overlay": overlay} if overlay is not None else {})},
        )

        try:
            cached = self._cache.get(cache_key)
            if cached is not None:
                if save:
                    saved = Saver.save(results=cached, out_dir=out_path, method=method_key)
                    try:
                        LOGGER.info(f"可视化结果已保存到: {str(out_path)}（{len(saved)}个文件）")
                    except Exception:
                        pass
                return cached

            results = plugin.run(
                model=self.model,
                inputs=inputs,
                layers=valid_layers,
                layer_names=layer_names,
                save=save,
                out_dir=out_path,
                modality=modality,
                family=self.family,
                overlay=overlay,
                **kwargs,
            )

            if save and results:
                saved = Saver.save(results=results, out_dir=out_path, method=method_key)
                try:
                    LOGGER.info(f"可视化结果已保存到: {str(out_path)}（{len(saved)}个文件）")
                except Exception:
                    pass
            self._cache.set(cache_key, results)
            return results
        finally:
            router.restore()

