from __future__ import annotations
"""
Frequency-Domain Feature Enhancement Fusion (FDFEF) Module.

论文: UMIS-YOLO: Underwater Multimodal Images Instance Segmentation With YOLO
来源: IEEE Transactions on Geoscience and Remote Sensing, Vol.63, 2025

FDFEF 模块通过傅里叶变换在频域内增强和融合多模态特征：
1. 频域特征增强：FFT -> 可学习权重调制 -> IFFT + 残差
2. 幅度谱和相位谱融合：通过可学习权重融合两个模态的频域信息
"""

import torch
import torch.nn as nn
import torch.fft as fft

from .dyt import DyT

__all__ = ['FDFEF']

class FrequencyEnhancement(nn.Module):











    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels


        self.weight_real = nn.Parameter(torch.empty(channels, 1, 1))
        self.weight_imag = nn.Parameter(torch.zeros(channels, 1, 1))
        nn.init.xavier_uniform_(self.weight_real.view(channels, 1))

        self.dyt = DyT(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:









        x_fft = fft.rfft2(x, norm='ortho')


        weight_complex = torch.complex(self.weight_real, self.weight_imag)


        x_fft_enhanced = x_fft * weight_complex


        x_enhanced = fft.irfft2(x_fft_enhanced, s=x.shape[-2:], norm='ortho')


        out = self.dyt(x_enhanced + x)
        return out

class FDFEF(nn.Module):










    def __init__(self, dim: int | None = None):
        super().__init__()
        self.dim = dim
        self._built = False


        self.enhance1: FrequencyEnhancement | None = None
        self.enhance2: FrequencyEnhancement | None = None
        self.alpha1: nn.Parameter | None = None
        self.alpha2: nn.Parameter | None = None
        self.beta1: nn.Parameter | None = None
        self.beta2: nn.Parameter | None = None
        self.out_dyt: DyT | None = None

        if dim is not None:
            self._build(dim)

    def _build(self, dim: int):

        if self._built and self.dim == dim:
            return
        self.dim = dim


        self.enhance1 = FrequencyEnhancement(dim)
        self.enhance2 = FrequencyEnhancement(dim)


        self.alpha1 = nn.Parameter(torch.ones(dim, 1, 1) * 0.5)
        self.alpha2 = nn.Parameter(torch.ones(dim, 1, 1) * 0.5)


        self.beta1 = nn.Parameter(torch.ones(dim, 1, 1) * 0.5)
        self.beta2 = nn.Parameter(torch.ones(dim, 1, 1) * 0.5)


        self.out_dyt = DyT(dim)

        self._built = True

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> torch.Tensor:










        if x2 is None:
            if isinstance(x1, (list, tuple)) and len(x1) == 2:
                x1, x2 = x1
            else:
                raise ValueError("FDFEF 需要两路输入")

        if x1.shape != x2.shape:
            raise ValueError(f"FDFEF 要求两路输入形状一致，got {x1.shape} vs {x2.shape}")


        if not self._built:
            self._build(x1.shape[1])

            device = x1.device
            self.to(device)


        x1_enhanced = self.enhance1(x1)
        x2_enhanced = self.enhance2(x2)



        f1 = fft.rfft2(x1_enhanced, norm='ortho')
        f2 = fft.rfft2(x2_enhanced, norm='ortho')


        amp1 = torch.abs(f1)
        amp2 = torch.abs(f2)
        phase1 = torch.angle(f1)
        phase2 = torch.angle(f2)


        amp_fused = self.alpha1 * amp1 + self.alpha2 * amp2


        phase_fused = self.beta1 * phase1 + self.beta2 * phase2


        f_recon = amp_fused * torch.exp(1j * phase_fused)


        x_fused = fft.irfft2(f_recon, s=x1.shape[-2:], norm='ortho')


        out = self.out_dyt(x_fused)
        return out

