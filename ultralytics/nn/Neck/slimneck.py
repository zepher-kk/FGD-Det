"""
SlimNeck - 轻量化 Neck 模块集
论文: Slim-neck by GSConv: A Better Design Paradigm of Detector Architectures for Autonomous Vehicles
论文链接: https://arxiv.org/pdf/2206.02424

通过 GSConv（分组混洗卷积）替代标准卷积，在保持精度的同时大幅降低计算量。
包含:
  - GSConv: 分组混洗卷积，核心构建单元
  - GSBottleneck: GS 瓶颈块
  - GSBottleneckC: 廉价 GS 瓶颈块（使用深度可分离卷积捷径）
  - VoVGSCSP: 一次聚合的 VoV-GSCSP 模块
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv


class GSConv(nn.Module):
    """GSConv 分组混洗卷积（SlimNeck 核心构建单元）。

    将标准卷积分解为常规卷积和深度可分离卷积，然后通过通道混洗合并。

    Args:
        c1: 输入通道数
        c2: 输出通道数
        k: 卷积核大小，默认 1
        s: 步幅，默认 1
        p: 填充
        g: 分组数
        d: 膨胀率
        act: 是否使用激活函数
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    """GS 瓶颈块，由两个 GSConv 组成。

    Args:
        c1: 输入通道数
        c2: 输出通道数
        k: 卷积核大小，默认 3
        s: 步幅，默认 1
        e: 通道扩展比例，默认 0.5
    """

    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv_lighting = nn.Sequential(GSConv(c1, c_, 1, 1), GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class GSBottleneckC(GSBottleneck):
    """廉价 GS 瓶颈块变体，使用深度可分离卷积作为捷径连接。

    Args:
        c1: 输入通道数
        c2: 输出通道数
        k: 卷积核大小，默认 3
        s: 步幅，默认 1
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, k, s, act=False)


class VoVGSCSP(nn.Module):
    """VoV-GSCSP 模块，一次聚合多个 GS 瓶颈块。

    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: GSBottleneck 堆叠数量，默认 1
        shortcut: 是否使用捷径连接
        g: 分组数
        e: 通道扩展比例，默认 0.5
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


__all__ = ("GSConv", "GSBottleneck", "GSBottleneckC", "VoVGSCSP")
