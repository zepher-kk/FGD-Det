from __future__ import annotations



"""
YOLOMM family feature-level distillation guidance.

Provides foreground-guided feature mask generation using teacher detection
output: high-confidence boxes are projected onto feature grids to produce
soft foreground masks for detection-friendly feature distillation.

Design:
- Reuses YOLOMMOutputBundle from output_yolomm.py (no duplicate decoding).
- Batch-vectorised mask generation (no Python per-image-per-box loops).
- Mask semantics: B x 1 x H x W, score/sqrt(area) accumulation, clamped [0,1].
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import torch

from .output_yolomm import YOLOMMOutputBundle, build_yolomm_output_bundle





_YOLOMM_FEATURE_TOPK = 100
_FEATURE_CONF_THR = 0.05





@dataclass
class YOLOMMFeatureGuidance:









    boxes_xyxy: List[torch.Tensor]
    scores: List[torch.Tensor]





def build_yolomm_feature_guidance(
    teacher_preds,
    teacher_model,
    conf_thr: float = _FEATURE_CONF_THR,
    topk: int = _YOLOMM_FEATURE_TOPK,
) -> YOLOMMFeatureGuidance:















    bundle: YOLOMMOutputBundle = build_yolomm_output_bundle(
        teacher_preds, teacher_model
    )


    cls_logits = bundle.cls_logits
    box_decoded = bundle.box_decoded


    max_scores = cls_logits.sigmoid().max(dim=-1).values

    B = max_scores.shape[0]
    device = max_scores.device

    boxes_list: List[torch.Tensor] = []
    scores_list: List[torch.Tensor] = []



    for b in range(B):
        img_scores = max_scores[b]
        img_boxes = box_decoded[b]


        mask = img_scores >= conf_thr
        n_valid = mask.sum().item()

        if n_valid == 0:

            boxes_list.append(
                torch.zeros(0, 4, device=device, dtype=img_boxes.dtype)
            )
            scores_list.append(
                torch.zeros(0, device=device, dtype=img_scores.dtype)
            )
            continue

        valid_scores = img_scores[mask]
        valid_boxes = img_boxes[mask]


        if n_valid > topk:
            topk_vals, topk_idx = valid_scores.topk(topk)
            valid_scores = topk_vals
            valid_boxes = valid_boxes[topk_idx]


        boxes_list.append(valid_boxes.detach())
        scores_list.append(valid_scores.detach())

    return YOLOMMFeatureGuidance(boxes_xyxy=boxes_list, scores=scores_list)





def build_yolomm_feature_mask(
    guidance: YOLOMMFeatureGuidance,
    feature_shape: Union[Tuple[int, int, int, int], torch.Size],
    input_size: Union[Tuple[int, int], None] = None,
) -> torch.Tensor:
















    B, _, H, W = feature_shape
    device = guidance.boxes_xyxy[0].device if guidance.boxes_xyxy else torch.device("cpu")


    dtype = torch.float32
    if guidance.boxes_xyxy and guidance.boxes_xyxy[0].numel() > 0:
        dtype = guidance.boxes_xyxy[0].dtype




    n_boxes_per_image = [boxes.shape[0] for boxes in guidance.boxes_xyxy]
    N_max = max(n_boxes_per_image) if n_boxes_per_image else 0


    if N_max == 0:
        return torch.zeros(B, 1, H, W, device=device, dtype=dtype)


    boxes_padded = torch.zeros(B, N_max, 4, device=device, dtype=dtype)
    scores_padded = torch.zeros(B, N_max, device=device, dtype=dtype)
    valid_mask = torch.zeros(B, N_max, device=device, dtype=torch.bool)

    for b in range(B):
        n_b = n_boxes_per_image[b]
        if n_b > 0:
            boxes_padded[b, :n_b] = guidance.boxes_xyxy[b]
            scores_padded[b, :n_b] = guidance.scores[b]
            valid_mask[b, :n_b] = True




    if input_size is not None:
        img_h, img_w = input_size
        scale_w = W / img_w
        scale_h = H / img_h
        scale = torch.tensor(
            [scale_w, scale_h, scale_w, scale_h],
            device=device, dtype=dtype,
        )
        boxes_scaled = boxes_padded * scale.view(1, 1, 4)
    else:

        boxes_scaled = boxes_padded





    grid_x = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)
    grid_y = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)


    x1 = boxes_scaled[:, :, 0].view(B, N_max, 1, 1)
    y1 = boxes_scaled[:, :, 1].view(B, N_max, 1, 1)
    x2 = boxes_scaled[:, :, 2].view(B, N_max, 1, 1)
    y2 = boxes_scaled[:, :, 3].view(B, N_max, 1, 1)


    in_box = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y < y2)





    area = (boxes_scaled[:, :, 2] - boxes_scaled[:, :, 0]) * (
        boxes_scaled[:, :, 3] - boxes_scaled[:, :, 1]
    )

    weights = scores_padded / (area + 1e-6).sqrt()


    weights = weights * valid_mask.float()



    weighted_coverage = in_box.float() * weights.view(B, N_max, 1, 1)
    mask = weighted_coverage.sum(dim=1, keepdim=False)


    mask = mask.unsqueeze(1).clamp(0.0, 1.0)

    return mask

