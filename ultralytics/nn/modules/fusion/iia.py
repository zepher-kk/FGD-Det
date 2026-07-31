from __future__ import annotations




import torch
import torch.nn as nn

class IIA(nn.Module):
    def __init__(self, channel: int | None = None, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel = channel
        self.kernel_size = int(kernel_size)
        self._built = False
        self._c = None
        self.conv_h: nn.Module | None = None
        self.conv_w: nn.Module | None = None

    def _build_if_needed(self, c: int) -> None:
        if self._built and self._c == c:
            return
        k = self.kernel_size
        p = k // 2
        self.conv_h = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, k), padding=(0, p), bias=False),
            nn.Sigmoid(),
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(k, 1), padding=(p, 0), bias=False),
            nn.Sigmoid(),
        )
        self._built = True
        self._c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise TypeError("IIA 期望输入 [B, C, H, W]")
        B, C, H, W = x.shape
        self._build_if_needed(C)

        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg, maxv], dim=1)
        attn_h = self.conv_h(pooled)
        x_h = x * attn_h
        attn_w = self.conv_w(pooled)
        x_w = x * attn_w
        return x + x_h + x_w

