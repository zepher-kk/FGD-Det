from __future__ import annotations
"""UAMFDet plug-and-play fusion modules (PyTorch-only, no torchvision).

This file implements a production-oriented, fail-fast adaptation of the core ideas from:
UAMFDet: Acoustic-Optical Fusion for Underwater Multi-Modal Object Detection.

Design goals (per project plan):
- Plug-and-play via YAML (registered in tasks.py:parse_model()).
- Works for YOLOMM and RTDETRMM at P3/P4/P5.
- Instance-level fusion (MIFM) implemented with grid_sample (no roi_align/torchvision).
- Fail-fast: invalid params or incompatible inputs raise ValueError with key parameter names.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MDSFF", "MDF", "MIFM", "UAMFDetFusion"]

def _unpack_two_inputs(x1: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | list, x2: torch.Tensor | None):

    if x2 is not None:
        return x1, x2
    if isinstance(x1, (list, tuple)) and len(x1) == 2:
        return x1[0], x1[1]
    raise ValueError("expects 2 inputs")

def _check_4d(x: torch.Tensor, name: str):
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError(f"{name} must be a 4D torch.Tensor [B,C,H,W], got {type(x)} with ndim={getattr(x, 'ndim', None)}")

def _make_base_grid(h: int, w: int, device: torch.device) -> torch.Tensor:





    if h <= 0 or w <= 0:
        raise ValueError(f"H and W must be >0, got H={h}, W={w}")

    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * (2.0 / w) - 1.0
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * (2.0 / h) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)

class MDSFF(nn.Module):







    def __init__(
        self,
        c_main: int,
        c_aux: int,
        K: int = 8,
        n_heads: int = 4,
        offset_scale: float = 1.0,
        auto_channel_align: bool = False,
    ):
        super().__init__()
        if K <= 0:
            raise ValueError("K must be >0")
        if n_heads <= 0:
            raise ValueError("n_heads must be >0")
        if offset_scale <= 0:
            raise ValueError("offset_scale must be >0")
        if c_main <= 0 or c_aux <= 0:
            raise ValueError(f"channels must be >0, got c_main={c_main}, c_aux={c_aux}")
        if c_main % n_heads != 0:
            raise ValueError(f"c_main must be divisible by n_heads, got c_main={c_main}, n_heads={n_heads}")

        self.c_main = int(c_main)
        self.c_aux = int(c_aux)
        self.K = int(K)
        self.n_heads = int(n_heads)
        self.offset_scale = float(offset_scale)
        self.auto_channel_align = bool(auto_channel_align)

        if self.c_main != self.c_aux and not self.auto_channel_align:
            raise ValueError(
                f"MDSFF expects equal channels unless auto_channel_align=True, got c_main={self.c_main}, c_aux={self.c_aux}"
            )

        self.aux_in_proj = nn.Identity() if self.c_main == self.c_aux else nn.Conv2d(self.c_aux, self.c_main, 1, 1, 0, bias=False)


        self.offset_conv = nn.Conv2d(self.c_main, 2 * self.K, kernel_size=3, stride=1, padding=1)


        self.q_proj = nn.Conv2d(self.c_main, self.c_main, kernel_size=1, stride=1, padding=0, bias=False)
        self.k_proj = nn.Conv2d(self.c_main, self.c_main, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_proj = nn.Conv2d(self.c_main, self.c_main, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_main: torch.Tensor, x_aux: torch.Tensor | None = None) -> torch.Tensor:
        x_main, x_aux = _unpack_two_inputs(x_main, x_aux)
        _check_4d(x_main, "x_main")
        _check_4d(x_aux, "x_aux")
        if x_main.shape[0] != x_aux.shape[0]:
            raise ValueError(f"batch size mismatch: {x_main.shape[0]} vs {x_aux.shape[0]}")
        if x_main.shape[-2:] != x_aux.shape[-2:]:
            raise ValueError(f"spatial size mismatch: {x_main.shape[-2:]} vs {x_aux.shape[-2:]}")

        b, _, h, w = x_main.shape
        x_aux = self.aux_in_proj(x_aux)


        offsets = torch.tanh(self.offset_conv(x_main))
        offsets = offsets.view(b, self.K, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()



        scale = torch.tensor([2.0 / w, 2.0 / h], device=x_main.device, dtype=torch.float32).view(1, 1, 1, 1, 2)
        grid_base = _make_base_grid(h, w, x_main.device).unsqueeze(1)
        grid = grid_base + (offsets.to(torch.float32) * self.offset_scale) * scale
        grid = grid.clamp(-1.0, 1.0)


        x_aux_rep = x_aux.unsqueeze(1).expand(b, self.K, self.c_main, h, w).reshape(b * self.K, self.c_main, h, w)
        grid_rep = grid.reshape(b * self.K, h, w, 2)
        sampled = F.grid_sample(x_aux_rep, grid_rep, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.view(b, self.K, self.c_main, h, w)


        d = self.c_main // self.n_heads
        q = self.q_proj(x_main).view(b, self.n_heads, d, h, w)
        k = self.k_proj(sampled.reshape(b * self.K, self.c_main, h, w)).view(b, self.K, self.n_heads, d, h, w)
        sim = (q.unsqueeze(1) * k).sum(dim=3) / math.sqrt(d)
        attn = torch.softmax(sim, dim=1)


        w_k = attn.mean(dim=2)
        out = (sampled * w_k.unsqueeze(2)).sum(dim=1)
        return self.out_proj(out)

class MDF(nn.Module):


    def __init__(
        self,
        c_main: int,
        c_aux: int,
        reduction: int = 16,
        auto_channel_align: bool = False,
    ):
        super().__init__()
        if c_main <= 0 or c_aux <= 0:
            raise ValueError(f"channels must be >0, got c_main={c_main}, c_aux={c_aux}")
        if reduction <= 0:
            raise ValueError("reduction must be >0")

        self.c_main = int(c_main)
        self.c_aux = int(c_aux)
        self.reduction = int(reduction)
        self.auto_channel_align = bool(auto_channel_align)

        if self.c_main != self.c_aux and not self.auto_channel_align:
            raise ValueError(
                f"MDF expects equal channels unless auto_channel_align=True, got c_main={self.c_main}, c_aux={self.c_aux}"
            )
        self.aux_in_proj = nn.Identity() if self.c_main == self.c_aux else nn.Conv2d(self.c_aux, self.c_main, 1, 1, 0, bias=False)

        hidden = max(self.c_main // self.reduction, 1)
        self.fc1 = nn.Linear(2 * self.c_main, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, self.c_main, bias=True)

    def forward(self, x_main: torch.Tensor, x_aux_aligned: torch.Tensor | None = None) -> torch.Tensor:
        x_main, x_aux_aligned = _unpack_two_inputs(x_main, x_aux_aligned)
        _check_4d(x_main, "x_main")
        _check_4d(x_aux_aligned, "x_aux")
        if x_main.shape[0] != x_aux_aligned.shape[0]:
            raise ValueError(f"batch size mismatch: {x_main.shape[0]} vs {x_aux_aligned.shape[0]}")
        if x_main.shape[-2:] != x_aux_aligned.shape[-2:]:
            raise ValueError(f"spatial size mismatch: {x_main.shape[-2:]} vs {x_aux_aligned.shape[-2:]}")

        x_aux_aligned = self.aux_in_proj(x_aux_aligned)
        diff = x_main - x_aux_aligned


        avg = F.adaptive_avg_pool2d(diff, 1).flatten(1)
        mx = F.adaptive_max_pool2d(diff, 1).flatten(1)
        s = torch.cat([avg, mx], dim=1)
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True))).view(-1, self.c_main, 1, 1)
        return x_main + gate * diff

class MIFM(nn.Module):


    def __init__(
        self,
        c_main: int,
        c_aux: int,
        patch_size: int = 7,
        n_train: int = 256,
        n_infer: int = 500,
        reduction: int = 16,
        auto_channel_align: bool = False,
    ):
        super().__init__()
        if c_main <= 0 or c_aux <= 0:
            raise ValueError(f"channels must be >0, got c_main={c_main}, c_aux={c_aux}")
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd positive integer")
        if n_train <= 0 or n_infer <= 0:
            raise ValueError("n_train and n_infer must be >0")
        if reduction <= 0:
            raise ValueError("reduction must be >0")

        self.c_main = int(c_main)
        self.c_aux = int(c_aux)
        self.patch_size = int(patch_size)
        self.n_train = int(n_train)
        self.n_infer = int(n_infer)
        self.reduction = int(reduction)
        self.auto_channel_align = bool(auto_channel_align)

        if self.c_main != self.c_aux and not self.auto_channel_align:
            raise ValueError(
                f"MIFM expects equal channels unless auto_channel_align=True, got c_main={self.c_main}, c_aux={self.c_aux}"
            )
        self.aux_in_proj = nn.Identity() if self.c_main == self.c_aux else nn.Conv2d(self.c_aux, self.c_main, 1, 1, 0, bias=False)


        self.scorer = nn.Conv2d(self.c_main, 1, kernel_size=1, stride=1, padding=0)

        hidden = max(self.c_main // self.reduction, 1)
        self.inst_fc1 = nn.Linear(3 * self.c_main, hidden, bias=True)
        self.inst_fc2 = nn.Linear(hidden, self.c_main, bias=True)
        self.inject_proj = nn.Linear(self.c_main, self.c_main, bias=True)

    def forward(self, x_main: torch.Tensor, x_aux: torch.Tensor | None = None) -> torch.Tensor:
        x_main, x_aux = _unpack_two_inputs(x_main, x_aux)
        _check_4d(x_main, "x_main")
        _check_4d(x_aux, "x_aux")
        if x_main.shape[0] != x_aux.shape[0]:
            raise ValueError(f"batch size mismatch: {x_main.shape[0]} vs {x_aux.shape[0]}")
        if x_main.shape[-2:] != x_aux.shape[-2:]:
            raise ValueError(f"spatial size mismatch: {x_main.shape[-2:]} vs {x_aux.shape[-2:]}")

        b, _, h, w = x_main.shape
        x_aux = self.aux_in_proj(x_aux)

        n = self.n_train if self.training else self.n_infer
        if n <= 0:
            raise ValueError("n_train/n_infer must be >0")
        if n > h * w:
            raise ValueError(f"N_total exceeds feature map size: N_total={n} > H*W={h*w}")


        score = torch.sigmoid(self.scorer(x_main)).flatten(1)
        _, idx = torch.topk(score, k=n, dim=1, largest=True, sorted=False)


        idx_y = idx // w
        idx_x = idx - idx_y * w


        r = self.patch_size // 2
        dx = torch.arange(-r, r + 1, device=x_main.device, dtype=torch.float32)
        dy = torch.arange(-r, r + 1, device=x_main.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(dy, dx, indexing="ij")

        off_x = xx * (2.0 / w)
        off_y = yy * (2.0 / h)
        off = torch.stack([off_x, off_y], dim=-1)


        cx = (idx_x.to(torch.float32) + 0.5) * (2.0 / w) - 1.0
        cy = (idx_y.to(torch.float32) + 0.5) * (2.0 / h) - 1.0
        center = torch.stack([cx, cy], dim=-1)

        grid = center[:, :, None, None, :].to(torch.float32) + off[None, None, :, :, :]
        grid = grid.clamp(-1.0, 1.0).reshape(b * n, self.patch_size, self.patch_size, 2)


        x_main_rep = x_main.unsqueeze(1).expand(b, n, self.c_main, h, w).reshape(b * n, self.c_main, h, w)
        x_aux_rep = x_aux.unsqueeze(1).expand(b, n, self.c_main, h, w).reshape(b * n, self.c_main, h, w)

        p_main = F.grid_sample(x_main_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        p_aux = F.grid_sample(x_aux_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


        v_main = 0.5 * (p_main.mean(dim=(2, 3)) + p_main.amax(dim=(2, 3)))
        v_aux = 0.5 * (p_aux.mean(dim=(2, 3)) + p_aux.amax(dim=(2, 3)))
        v_main = v_main.view(b, n, self.c_main)
        v_aux = v_aux.view(b, n, self.c_main)


        sim = torch.matmul(v_main, v_aux.transpose(1, 2)) / math.sqrt(self.c_main)
        w_mat = torch.softmax(sim, dim=-1)
        v_aux_hat = torch.matmul(w_mat, v_aux)


        z = torch.cat([v_main, v_aux_hat, v_main - v_aux_hat], dim=-1)
        gate = torch.sigmoid(self.inst_fc2(F.relu(self.inst_fc1(z), inplace=True)))
        v_fused = v_main + gate * (v_main - v_aux_hat)

        v_inj = self.inject_proj(v_fused).transpose(1, 2)
        idx_exp = idx.unsqueeze(1).expand(b, self.c_main, n)
        inject = torch.zeros((b, self.c_main, h * w), device=x_main.device, dtype=x_main.dtype)
        inject.scatter_add_(2, idx_exp, v_inj.to(dtype=x_main.dtype))
        inject = inject.view(b, self.c_main, h, w)

        return x_main + inject

class UAMFDetFusion(nn.Module):


    def __init__(
        self,
        c_main: int,
        c_aux: int,
        K: int = 8,
        n_heads: int = 4,
        offset_scale: float = 1.0,
        patch_size: int = 7,
        n_train: int = 256,
        n_infer: int = 500,
        enable_mdsff: bool = True,
        enable_mdf: bool = True,
        enable_mifm: bool = True,
        auto_channel_align: bool = False,
    ):
        super().__init__()
        self.c_main = int(c_main)
        self.c_aux = int(c_aux)
        self.enable_mdsff = bool(enable_mdsff)
        self.enable_mdf = bool(enable_mdf)
        self.enable_mifm = bool(enable_mifm)
        self.auto_channel_align = bool(auto_channel_align)


        self.mdsff = MDSFF(self.c_main, self.c_aux, K=K, n_heads=n_heads, offset_scale=offset_scale, auto_channel_align=self.auto_channel_align)
        self.mdf = MDF(self.c_main, self.c_aux, reduction=16, auto_channel_align=self.auto_channel_align)
        self.mifm = MIFM(
            self.c_main,
            self.c_aux,
            patch_size=patch_size,
            n_train=n_train,
            n_infer=n_infer,
            reduction=16,
            auto_channel_align=self.auto_channel_align,
        )

    def forward(self, x_main: torch.Tensor, x_aux: torch.Tensor | None = None) -> torch.Tensor:
        x_main, x_aux = _unpack_two_inputs(x_main, x_aux)
        _check_4d(x_main, "x_main")
        _check_4d(x_aux, "x_aux")
        if x_main.shape[-2:] != x_aux.shape[-2:]:
            raise ValueError(f"spatial size mismatch: {x_main.shape[-2:]} vs {x_aux.shape[-2:]}")

        x_aux_aligned = self.mdsff(x_main, x_aux) if self.enable_mdsff else self.mdsff.aux_in_proj(x_aux)
        x_fine = self.mdf(x_main, x_aux_aligned) if self.enable_mdf else x_main
        x_out = self.mifm(x_fine, x_aux_aligned) if self.enable_mifm else x_fine
        return x_out

