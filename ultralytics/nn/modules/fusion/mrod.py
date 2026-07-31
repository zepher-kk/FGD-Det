from __future__ import annotations
"""
MROD-YOLO Fusion Modules
论文: MROD-YOLO: Multimodal Joint Representation for Small Object Detection
      in Remote Sensing Imagery via Multiscale Iterative Aggregation
来源: IEEE TGRS 2025

包含模块:
- GCB (Global Context Block): 全局上下文注意力块
- MJRNet (Multimodal Joint Representation Network): 多模态联合表示网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GCB', 'MJRNet']

class GCB(nn.Module):
















    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        mid_channels = max(in_channels // reduction, 8)


        self.conv_attn = nn.Conv2d(in_channels, 1, kernel_size=1)


        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.LayerNorm([mid_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):






        B, C, H, W = x.shape



        attn = self.conv_attn(x).view(B, 1, -1)
        attn = F.softmax(attn, dim=-1)


        x_flat = x.view(B, C, -1)
        context = torch.bmm(x_flat, attn.transpose(1, 2))
        context = context.view(B, C, 1, 1)


        context = self.bottleneck(context)


        out = x + context

        return out

class MJRNet(nn.Module):





















    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels


        self.gcb_rgb = GCB(in_channels, reduction)
        self.gcb_ir = GCB(in_channels, reduction)


        self.mask_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.mask_ir = nn.Conv2d(in_channels, in_channels, kernel_size=1)


        self.refine_rgb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.refine_ir = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )


        self.gcb_fusion = GCB(in_channels * 2, reduction)


        self.compress = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):






        rgb, ir = x[0], x[1]


        g_rgb = self.gcb_rgb(rgb)
        g_ir = self.gcb_ir(ir)


        m_rgb = self.mask_rgb(rgb)
        m_ir = self.mask_ir(ir)


        rgb_weighted = g_rgb * torch.sigmoid(m_rgb)
        ir_weighted = g_ir * torch.sigmoid(m_ir)


        rgb_out = self.refine_rgb(rgb + rgb_weighted)
        ir_out = self.refine_ir(ir + ir_weighted)


        fused = torch.cat([rgb_out, ir_out], dim=1)
        fused = self.gcb_fusion(fused)


        out = self.compress(fused)

        return out

