from __future__ import annotations
"""
Residual Feature Fusion (RFF) Module.

论文: UMIS-YOLO: Underwater Multimodal Images Instance Segmentation With YOLO
来源: IEEE Transactions on Geoscience and Remote Sensing, Vol.63, 2025

RFF 模块用于融合低级（P1级）特征和高级特征，以保留像素级信息:
1. 通道对齐：通过 1×1 卷积对齐通道数
2. 空间对齐：通过双线性插值对齐空间尺寸
3. 双分支融合：主分支(Concat+增强) + 辅助分支(Add+Sigmoid门控)
4. 多尺度融合：使用分组卷积融合不同尺度特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dyt import DyT

__all__ = ['RFF']

class FBlock(nn.Module):







    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.dyt = DyT(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dyt(self.conv(x)))

class RFF(nn.Module):















    def __init__(
        self,
        low_channels: int | None = None,
        high_channels: int | None = None,
        groups: int = 4,
    ):
        super().__init__()
        self.low_channels = low_channels
        self.high_channels = high_channels
        self.groups = groups
        self._built = False


        self.low_align: nn.Module | None = None
        self.high_align: nn.Module | None = None
        self.fblock1: FBlock | None = None
        self.fblock2: FBlock | None = None
        self.fblock3: nn.Module | None = None
        self.branch_conv_low: nn.Module | None = None
        self.branch_conv_high: nn.Module | None = None
        self.residual_block: FBlock | None = None
        self.out_dyt: DyT | None = None

        if low_channels is not None and high_channels is not None:
            self._build(low_channels, high_channels)

    def _build(self, low_channels: int, high_channels: int):

        if self._built and self.low_channels == low_channels and self.high_channels == high_channels:
            return

        self.low_channels = low_channels
        self.high_channels = high_channels
        dim = low_channels


        self.low_align = nn.Conv2d(low_channels, dim, 1, bias=False) if low_channels != dim else nn.Identity()
        self.high_align = nn.Conv2d(high_channels, dim, 1, bias=False) if high_channels != dim else nn.Identity()


        self.fblock1 = FBlock(dim * 2, dim)
        self.fblock2 = FBlock(dim, dim)


        self.branch_conv_low = nn.Conv2d(dim, dim, 1, bias=False)
        self.branch_conv_high = nn.Conv2d(dim, dim, 1, bias=False)


        groups = min(self.groups, dim)
        if dim % groups != 0:
            groups = 1
        self.fblock3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=groups, bias=False),
            DyT(dim),
            nn.ReLU(inplace=True),
        )


        self.residual_block = FBlock(dim * 2, dim)


        self.out_dyt = DyT(dim)

        self._built = True

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor = None) -> torch.Tensor:










        if x_high is None:
            if isinstance(x_low, (list, tuple)) and len(x_low) == 2:
                x_low, x_high = x_low
            else:
                raise ValueError("RFF 需要两路输入 (低级特征, 高级特征)")


        if not self._built:
            self._build(x_low.shape[1], x_high.shape[1])
            self.to(x_low.device)


        x_low_adj = self.low_align(x_low)
        x_high_adj = self.high_align(x_high)


        if x_low_adj.shape[-2:] != x_high_adj.shape[-2:]:
            x_low_aligned = F.interpolate(
                x_low_adj,
                size=x_high_adj.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        else:
            x_low_aligned = x_low_adj


        concat_feat = torch.cat([x_low_aligned, x_high_adj], dim=1)
        ce = self.fblock1(concat_feat)
        f_fused = self.fblock2(ce)


        branch_low = self.branch_conv_low(x_low_aligned)
        branch_high = self.branch_conv_high(x_high_adj)
        a = torch.sigmoid(branch_low + branch_high)


        f_multi = self.fblock3(f_fused * a)


        r = self.residual_block(concat_feat)


        out = self.out_dyt(f_multi + r)
        return out

