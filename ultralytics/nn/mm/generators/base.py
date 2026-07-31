from __future__ import annotations
"""
离线模态生成基础类与注册表。

设计目标：
- 完全离线：仅负责读取源数据、生成目标模态并保存，不与训练/推理管线耦合。
- 不自动降级：缺权重/设备/依赖即抛错，由调用方显式处理。
- 可插拔：不同生成方法通过注册表选择，统一 run 接口。
"""

import abc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device





@dataclass
class SaveOptions:


    enable_save: bool = True
    save_dir: Optional[Path | str] = None
    keep_structure: bool = True
    overwrite: bool = False

@dataclass
class GeneratorRunStats:


    total: int = 0
    success: int = 0
    failed: int = 0
    failures: List[Tuple[str, str]] = field(default_factory=list)





class GeneratorRegistry:


    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, generator_cls: type):
        if not issubclass(generator_cls, ModalGeneratorBase):
            raise TypeError(f"{generator_cls} 不是 ModalGeneratorBase 的子类")
        cls._registry[name] = generator_cls

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise KeyError(f"未注册的生成方法: {name}")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_methods(cls) -> List[str]:
        return sorted(cls._registry.keys())





class ModalGeneratorBase(abc.ABC):











    def __init__(
        self,
        method: str,
        device: str | torch.device | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        save_options: Optional[SaveOptions | Dict[str, Any]] = None,
        method_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.method = method
        self.device = select_device(device or "cpu", verbose=False)
        if batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")
        self.batch_size = batch_size
        self.num_workers = max(0, num_workers)
        self.method_cfg = method_cfg or {}

        if isinstance(save_options, dict):
            self.save_options = SaveOptions(**save_options)
        elif save_options is None:
            self.save_options = SaveOptions()
        else:
            self.save_options = save_options

        self.model = None
        self._loaded = False



    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def preprocess(self, item: str):
        pass

    @abc.abstractmethod
    def infer(self, batch_inputs: List[Any]) -> List[Any]:
        pass

    @abc.abstractmethod
    def postprocess(self, outputs: List[Any], metas: List[Dict[str, Any]]) -> List[Any]:
        pass

    @abc.abstractmethod
    def save(self, outputs: List[Any], metas: List[Dict[str, Any]]) -> List[str]:
        pass



    def _ensure_loaded(self):
        if not self._loaded:
            self.load_model()
            if self.model is None:
                raise RuntimeError("模型未加载成功")
            self._loaded = True

    def _gather_sources(self, source: str | Path | Iterable[str | Path]) -> List[str]:





        paths: List[str] = []
        if isinstance(source, (str, Path)):
            src = Path(source)
            if src.is_dir():
                paths = [str(p) for p in src.rglob("*") if p.is_file()]
            elif src.is_file():
                paths = [str(src)]
            else:
                raise FileNotFoundError(f"找不到输入: {source}")
        else:
            for item in source:
                p = Path(item)
                if p.exists():
                    paths.append(str(p))
                else:
                    raise FileNotFoundError(f"找不到输入: {item}")

        if not paths:
            raise RuntimeError("未找到任何可处理的输入文件")
        return sorted(paths)

    def run(self, source: str | Path | Iterable[str | Path]) -> GeneratorRunStats:



        self._ensure_loaded()
        files = self._gather_sources(source)
        self._source_root = os.path.commonpath(files) if files else None

        stats = GeneratorRunStats(total=len(files))
        LOGGER.info(f"[{self.method}] 离线生成开始，样本数: {len(files)}, 设备: {self.device}")


        for start in range(0, len(files), self.batch_size):
            batch_paths = files[start : start + self.batch_size]
            try:
                batch_inputs, metas = [], []
                for p in batch_paths:
                    inp, meta = self.preprocess(p)
                    batch_inputs.append(inp)
                    metas.append(meta)

                outputs = self.infer(batch_inputs)
                outputs = self.postprocess(outputs, metas)
                saved = self.save(outputs, metas) if self.save_options.enable_save else []

                stats.success += len(batch_paths)
                processed = stats.success + stats.failed
                if processed % 100 == 0 or processed == stats.total:
                    LOGGER.info(
                        f"[{self.method}] 进度 {processed}/{stats.total} | 成功 {stats.success} | 失败 {stats.failed}"
                    )
                if saved and processed % 500 == 0:
                    LOGGER.debug(f"[{self.method}] 示例已保存: {saved[:1]}")
            except Exception as e:
                for p in batch_paths:
                    stats.failures.append((p, str(e)))
                stats.failed += len(batch_paths)
                LOGGER.error(f"[{self.method}] 处理失败 ({batch_paths}): {e}")

        LOGGER.info(
            f"[{self.method}] 结束: 成功 {stats.success}/{stats.total}, 失败 {stats.failed}"
        )
        return stats

__all__ = [
    "ModalGeneratorBase",
    "GeneratorRegistry",
    "SaveOptions",
    "GeneratorRunStats",
]

