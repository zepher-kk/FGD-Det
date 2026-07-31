from __future__ import annotations
"""
RD modules for YOLO-RD style dictionary injection.

Exports:
- DConv: Dictionary-injection block (CG -> GIE -> PONO -> D) with residual mix.
- RepNCSPELAND: ELAN-style block followed by DConv (composed on RepNCSPELAN4).

Purpose:
- Provide the minimal necessary components to bring YOLO-RD's核心“检索-字典”思想
  到 Ultralytics 代码库，便于在 YAML 中被引用或在多模态结构中复用。

Notes:
- 依赖现有模块 Conv 与 RepNCSPELAN4，无外部依赖。
- 若需在 YAML 中直接使用，请确保在 ultralytics.nn.modules/__init__.py 和
  ultralytics.nn.tasks 的构建映射中完成注册（本仓库已按 FFN 模块模式注册）。
"""

import torch
from torch import nn


from ..conv import Conv
from ..block import RepNCSPELAN4

__all__ = ("DConv", "RepNCSPELAND")

class DConv(nn.Module):




































    def __init__(self, c1: int, alpha: float = 0.8, atoms: int = 512) -> None:
        super().__init__()
        self.alpha = float(alpha)


        self.CG = Conv(c1, atoms, 1)

        self.GIE = Conv(atoms, atoms, 5, g=atoms, act=False)

        self.D = Conv(atoms, c1, 1, act=False)

    @staticmethod
    def _pono(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + eps)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = self.CG(r)
        x = self.GIE(x)
        x = self._pono(x)
        x = self.D(x)
        return self.alpha * x + (1.0 - self.alpha) * r

class RepNCSPELAND(RepNCSPELAN4):






































    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        n: int = 1,
        *,
        atoms: int = 512,
        alpha: float = 0.8,
    ) -> None:
        super().__init__(c1, c2, c3, c4, n)
        self.dconv = DConv(c2, alpha=alpha, atoms=atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return self.dconv(x)

