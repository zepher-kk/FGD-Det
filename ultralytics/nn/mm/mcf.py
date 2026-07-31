from __future__ import annotations
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MCF-Gated Fusion module for multimodal main/aux branches.

设计目标：
- 以输入列表第 1 个张量为主模态，第 2 个张量为副模态（main_idx=0，aux_idx=1）。
- 副模态先经过零初始化的卷积门（ZeroConv 风格），再与主模态按 add 或 concat 融合。
- 不改 YAML 顶层 input/ch 配置，直接在融合节点替换 Add/Concat 即可。
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, autopad

class MCFGatedFusion(nn.Module):



















    def __init__(
        self,
        c_main: int,
        c_aux: int | None = None,
        out_channels: int | None = None,
        mode: str = "add",
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        main_idx: int = 0,
        aux_idx: int = 1,
        zero_init: bool = True,
        use_bn: bool = False,
        act: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.main_idx = main_idx
        self.aux_idx = aux_idx
        c_aux = c_aux or c_main
        out_channels = out_channels or c_main


        self.gate = nn.Conv2d(
            c_aux,
            out_channels if mode == "concat" else c_main,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p),
            groups=g,
            bias=not use_bn,
        )
        if zero_init:
            nn.init.zeros_(self.gate.weight)
            if self.gate.bias is not None:
                nn.init.zeros_(self.gate.bias)

        self.bn = nn.BatchNorm2d(self.gate.out_channels) if use_bn else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()


        self.post = (
            Conv(out_channels + c_main, c_main, k=1, s=1, p=0, g=1, act=True)
            if mode == "concat"
            else None
        )

    def forward(self, xs):
        x_main = xs[self.main_idx]
        x_aux = xs[self.aux_idx]
        x_aux = self.act(self.bn(self.gate(x_aux)))

        if self.mode == "add":
            return x_main + x_aux

        return self.post(torch.cat((x_main, x_aux), 1))

__all__ = ["MCFGatedFusion"]

