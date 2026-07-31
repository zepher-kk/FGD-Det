from __future__ import annotations



"""
RTDETRMM family feature-level distillation guidance.

Provides query-aware foreground-guided feature mask generation using teacher
DETR decoder output: high-confidence queries are projected onto feature grids
to produce soft foreground masks.

Design:
- Reuses RTDETROutputBundle from output_rtdetrmm.py (no duplicate decoding).
- Batch-vectorised mask generation (no Python per-image-per-query loops).
- Mask semantics unified with YOLOMM: B x 1 x H x W, score/sqrt(area),
  clamped [0,1], so shared feature loss can be used directly.
"""

from dataclasses import dataclass
from typing import List

import torch

from .output_rtdetrmm import (
    RTDETROutputBundle,
    build_rtdetr_output_bundle,
    select_rtdetr_kd_queries,
)





_RTDETR_FEATURE_TOPK = 100
_FEATURE_CONF_THR = 0.05





@dataclass
class RTDETRMMFeatureGuidance:












    boxes_xyxy: List[torch.Tensor]
    scores: List[torch.Tensor]





def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:








    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)





def build_rtdetr_feature_guidance(
    teacher_preds,
    conf_thr: float = _FEATURE_CONF_THR,
    topk: int = _RTDETR_FEATURE_TOPK,
    input_size: tuple[int, int] | None = None,
) -> RTDETRMMFeatureGuidance:



















    bundle: RTDETROutputBundle = build_rtdetr_output_bundle(teacher_preds)


    query_indices = select_rtdetr_kd_queries(
        bundle, conf_thr=conf_thr, topk_queries=topk
    )


    raw_boxes = bundle.boxes
    raw_scores = bundle.scores
    B = raw_boxes.shape[0]


    max_scores = raw_scores.sigmoid().max(dim=-1).values


    boxes_xyxy_all = _cxcywh_to_xyxy(raw_boxes)


    if input_size is not None:
        img_h, img_w = input_size
        scale = torch.tensor(
            [img_w, img_h, img_w, img_h],
            dtype=boxes_xyxy_all.dtype,
            device=boxes_xyxy_all.device,
        )
        boxes_xyxy_all = boxes_xyxy_all * scale


    boxes_list: List[torch.Tensor] = []
    scores_list: List[torch.Tensor] = []

    for b in range(B):
        idx = query_indices[b]
        if len(idx) == 0:
            boxes_list.append(
                torch.empty(0, 4, dtype=raw_boxes.dtype, device=raw_boxes.device)
            )
            scores_list.append(
                torch.empty(0, dtype=raw_scores.dtype, device=raw_scores.device)
            )
        else:
            boxes_list.append(boxes_xyxy_all[b, idx].detach())
            scores_list.append(max_scores[b, idx].detach())

    return RTDETRMMFeatureGuidance(boxes_xyxy=boxes_list, scores=scores_list)





def build_rtdetr_feature_mask(
    guidance: RTDETRMMFeatureGuidance,
    feature_shape: tuple[int, int, int, int],
    input_size: tuple[int, int] | None = None,
) -> torch.Tensor:


























    B, _, H, W = feature_shape


    if len(guidance.boxes_xyxy) > 0 and len(guidance.boxes_xyxy[0]) > 0:
        device = guidance.boxes_xyxy[0].device
        dtype = guidance.boxes_xyxy[0].dtype
    else:

        device = torch.device("cpu")
        dtype = torch.float32
        for boxes_t in guidance.boxes_xyxy:
            if boxes_t.numel() > 0:
                device = boxes_t.device
                dtype = boxes_t.dtype
                break


    counts = [boxes.shape[0] for boxes in guidance.boxes_xyxy]
    N_max = max(counts) if counts else 0

    if N_max == 0:

        return torch.zeros(B, 1, H, W, dtype=dtype, device=device)


    padded_boxes = torch.zeros(B, N_max, 4, dtype=dtype, device=device)
    padded_scores = torch.zeros(B, N_max, dtype=dtype, device=device)
    valid_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

    for b in range(B):
        n = counts[b]
        if n > 0:
            padded_boxes[b, :n] = guidance.boxes_xyxy[b]
            padded_scores[b, :n] = guidance.scores[b]
            valid_mask[b, :n] = True


    if input_size is not None:
        img_h, img_w = input_size
        scale_x = W / img_w
        scale_y = H / img_h
    else:

        scale_x = 1.0
        scale_y = 1.0


    scaled_boxes = padded_boxes.clone()
    scaled_boxes[..., 0] *= scale_x
    scaled_boxes[..., 1] *= scale_y
    scaled_boxes[..., 2] *= scale_x
    scaled_boxes[..., 3] *= scale_y


    scaled_boxes[..., 0].clamp_(min=0.0, max=float(W))
    scaled_boxes[..., 1].clamp_(min=0.0, max=float(H))
    scaled_boxes[..., 2].clamp_(min=0.0, max=float(W))
    scaled_boxes[..., 3].clamp_(min=0.0, max=float(H))



    grid_x = torch.arange(W, dtype=dtype, device=device).view(1, 1, 1, W) + 0.5
    grid_y = torch.arange(H, dtype=dtype, device=device).view(1, 1, H, 1) + 0.5


    x1 = scaled_boxes[..., 0].unsqueeze(-1).unsqueeze(-1)
    y1 = scaled_boxes[..., 1].unsqueeze(-1).unsqueeze(-1)
    x2 = scaled_boxes[..., 2].unsqueeze(-1).unsqueeze(-1)
    y2 = scaled_boxes[..., 3].unsqueeze(-1).unsqueeze(-1)


    inside = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y < y2)
    inside = inside.float()


    box_w = (scaled_boxes[..., 2] - scaled_boxes[..., 0]).clamp(min=0.0)
    box_h = (scaled_boxes[..., 3] - scaled_boxes[..., 1]).clamp(min=0.0)
    area = box_w * box_h
    weights = padded_scores / (area + 1e-6).sqrt()


    weights = weights * valid_mask.float()


    weights = weights.unsqueeze(-1).unsqueeze(-1)



    mask = (inside * weights).sum(dim=1)


    mask = mask.clamp(0.0, 1.0).unsqueeze(1)

    return mask

