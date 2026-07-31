from __future__ import annotations
"""
Dynamic Tanh (DyT) Normalization Module.

论文: UMIS-YOLO: Underwater Multimodal Images Instance Segmentation With YOLO
来源: IEEE Transactions on Geoscience and Remote Sensing, Vol.63, 2025
参考: Zhu et al., "Transformers without normalization" (CVPR 2025)

DyT 是一种无需传统归一化层的动态变换模块，通过元素级操作实现与 BatchNorm/LayerNorm 类似的效果。
"""

import torch
import torch.nn as nn

__all__ = ['DyT']

class DyT(nn.Module):















    def __init__(self, channels: int, alpha_init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:









        out = torch.tanh(self.alpha * x)


        if x.dim() == 4:

            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        elif x.dim() == 2:

            out = out * self.weight.view(1, -1) + self.bias.view(1, -1)
        else:

            shape = [1] * x.dim()
            shape[1] = -1
            out = out * self.weight.view(*shape) + self.bias.view(*shape)

        return out

