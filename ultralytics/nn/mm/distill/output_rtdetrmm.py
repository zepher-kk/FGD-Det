from __future__ import annotations



"""
RTDETRMM family output distillation.

Implements query-aware output distillation designed for DETR-style detection
heads, where the output is a set of learned queries rather than dense anchors.

Key design:
- **Query selection**: only distill "effective queries" -- those with high
  teacher confidence -- rather than all 300 queries uniformly.
- **Final decoder layer only**: distillation operates on the last decoder layer
  output (no multi-layer decoder guidance in this version).
- **Classification distillation**: soft classification on selected queries.
- **Box distillation**: L1 + GIoU on selected query boxes.

References:
- Chang et al., DETRDistill, ICCV 2023
- Wang et al., KD-DETR, CVPR 2024
- D3ETR, IJCAI 2024
- OD-DETR, IJCAI 2024
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER





@dataclass
class RTDETROutputBundle:









    scores: torch.Tensor
    boxes: torch.Tensor

def build_rtdetr_output_bundle(preds) -> RTDETROutputBundle:


















    if isinstance(preds, (tuple, list)) and len(preds) == 5:
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds




        if dec_bboxes.dim() == 4:
            final_boxes = dec_bboxes[-1]
            final_scores = dec_scores[-1]
        elif dec_bboxes.dim() == 3:
            final_boxes = dec_bboxes
            final_scores = dec_scores
        else:
            raise ValueError(
                f"Unexpected dec_bboxes shape: {dec_bboxes.shape}. "
                f"Expected 3D or 4D tensor."
            )


        if dn_meta is not None and "dn_num_split" in dn_meta:
            dn_split = dn_meta["dn_num_split"]

            _, final_boxes = torch.split(final_boxes, dn_split, dim=1)
            _, final_scores = torch.split(final_scores, dn_split, dim=1)

        return RTDETROutputBundle(scores=final_scores, boxes=final_boxes)
    else:
        raise ValueError(
            f"Expected 5-tuple (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta), "
            f"got {type(preds)} with length {len(preds) if isinstance(preds, (tuple, list)) else 'N/A'}"
        )





def select_rtdetr_kd_queries(
    teacher_bundle: RTDETROutputBundle,
    conf_thr: float = 0.05,
    topk_queries: int = 100,
) -> List[torch.Tensor]:
















    scores = teacher_bundle.scores
    B, Q, _ = scores.shape

    max_scores = scores.sigmoid().max(dim=-1).values

    query_indices = []
    for b in range(B):
        above_thr = max_scores[b] >= conf_thr
        n_valid = above_thr.sum().item()

        if n_valid == 0:

            query_indices.append(torch.empty(0, dtype=torch.long, device=scores.device))
        elif n_valid > topk_queries:
            _, topk_idx = max_scores[b].topk(topk_queries)
            query_indices.append(topk_idx)
        else:
            query_indices.append(above_thr.nonzero(as_tuple=False).squeeze(-1))

    return query_indices





def compute_rtdetr_cls_kd(
    student_bundle: RTDETROutputBundle,
    teacher_bundle: RTDETROutputBundle,
    query_indices: List[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:











    s_scores = student_bundle.scores
    t_scores = teacher_bundle.scores
    B = s_scores.shape[0]
    device = s_scores.device

    losses = []
    for b in range(B):
        idx = query_indices[b]
        if len(idx) == 0:
            continue
        s_q = s_scores[b, idx]
        t_q = t_scores[b, idx]


        t_soft = (t_q / temperature).sigmoid().detach()
        loss = F.binary_cross_entropy_with_logits(
            s_q / temperature, t_soft, reduction='mean'
        )
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=device)
    return sum(losses) / len(losses)





def compute_rtdetr_box_kd(
    student_bundle: RTDETROutputBundle,
    teacher_bundle: RTDETROutputBundle,
    query_indices: List[torch.Tensor],
) -> torch.Tensor:












    s_boxes = student_bundle.boxes
    t_boxes = teacher_bundle.boxes
    B = s_boxes.shape[0]
    device = s_boxes.device

    all_s = []
    all_t = []
    for b in range(B):
        idx = query_indices[b]
        if len(idx) == 0:
            continue
        all_s.append(s_boxes[b, idx])
        all_t.append(t_boxes[b, idx])

    if not all_s:
        return torch.tensor(0.0, device=device)

    s_cat = torch.cat(all_s, dim=0)
    t_cat = torch.cat(all_t, dim=0).detach()


    l1_loss = F.l1_loss(s_cat, t_cat, reduction='mean')


    s_xyxy = _cxcywh_to_xyxy(s_cat)
    t_xyxy = _cxcywh_to_xyxy(t_cat)
    giou_loss = _giou_loss(s_xyxy, t_xyxy)

    return l1_loss + giou_loss






_OUTPUT_DISTILL_WEIGHT = 0.5
_OUTPUT_CLS_WEIGHT = 1.0
_OUTPUT_BOX_WEIGHT = 1.0
_OUTPUT_WARMUP_EPOCHS = 5
_QUERY_CONF_THR = 0.05
_QUERY_TOPK = 100
_CLS_TEMPERATURE = 1.0

def compute_rtdetr_output_kd(
    student_preds,
    teacher_preds,
    current_epoch: int = 0,
    total_epochs: int = 100,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
















    device = _get_device(student_preds)


    s_bundle = build_rtdetr_output_bundle(student_preds)
    t_bundle = build_rtdetr_output_bundle(teacher_preds)


    query_indices = select_rtdetr_kd_queries(
        t_bundle,
        conf_thr=_QUERY_CONF_THR,
        topk_queries=_QUERY_TOPK,
    )


    cls_kd = compute_rtdetr_cls_kd(s_bundle, t_bundle, query_indices, temperature=_CLS_TEMPERATURE)


    box_kd = compute_rtdetr_box_kd(s_bundle, t_bundle, query_indices)


    raw_total = _OUTPUT_CLS_WEIGHT * cls_kd + _OUTPUT_BOX_WEIGHT * box_kd


    warmup_factor = _compute_warmup_factor(current_epoch, _OUTPUT_WARMUP_EPOCHS)
    total_output_kd = _OUTPUT_DISTILL_WEIGHT * warmup_factor * raw_total

    items = {
        "distill_output_cls": cls_kd.detach(),
        "distill_output_box": box_kd.detach(),
        "distill_output_total": total_output_kd.detach(),
    }

    return total_output_kd, items





def _compute_warmup_factor(current_epoch: int, warmup_epochs: int) -> float:

    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, current_epoch / warmup_epochs)

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:





    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

def _giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:










    inter_x1 = torch.max(pred[:, 0], target[:, 0])
    inter_y1 = torch.max(pred[:, 1], target[:, 1])
    inter_x2 = torch.min(pred[:, 2], target[:, 2])
    inter_y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)


    area_pred = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_target = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = area_pred + area_target - inter + 1e-7

    iou = inter / union


    enc_x1 = torch.min(pred[:, 0], target[:, 0])
    enc_y1 = torch.min(pred[:, 1], target[:, 1])
    enc_x2 = torch.max(pred[:, 2], target[:, 2])
    enc_y2 = torch.max(pred[:, 3], target[:, 3])
    area_enc = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-7

    giou = iou - (area_enc - union) / area_enc
    return (1.0 - giou).mean()

def _get_device(preds) -> torch.device:
    if isinstance(preds, torch.Tensor):
        return preds.device
    if isinstance(preds, (tuple, list)):
        for p in preds:
            if isinstance(p, torch.Tensor):
                return p.device
    return torch.device("cpu")

