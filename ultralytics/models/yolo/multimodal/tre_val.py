from __future__ import annotations
"""
TRE (Triple-Reference Evaluation) Validator for multi-modal object detection.

Evaluates fusion robustness by running model inference at multiple IR offsets,
then classifying each prediction's behavior against three reference frames:
  - RGB-GT: original annotation (fixed)
  - X-GT:   GT shifted by (dx, dy) — "where IR says the object is"
  - Union-GT: bounding-box union of RGB-GT and X-GT

Outputs four TRE metrics per offset:
  AP_robust, MRR, SDR, RFS
Plus integrated RFS_total over the offset range.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.multimodal.cocoval import MultiModalCOCOValidator
from ultralytics.utils import LOGGER, TQDM, callbacks
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils import emojis





def _box_iou_np(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:

    lt = np.maximum(box1[:, None, :2], box2[None, :, :2])
    rb = np.minimum(box1[:, None, 2:], box2[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1[:, None] + area2[None, :] - inter + eps)

def _greedy_match_per_image(
    pred_boxes_xyxy: np.ndarray,
    gt_boxes_xyxy: np.ndarray,
    pred_cls: np.ndarray,
    gt_cls: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:






    D = len(pred_boxes_xyxy)
    G = len(gt_boxes_xyxy)
    if D == 0 or G == 0:
        return np.zeros(D, dtype=bool)

    iou = _box_iou_np(pred_boxes_xyxy, gt_boxes_xyxy)
    class_ok = pred_cls[:, None] == gt_cls[None, :]
    iou = iou * class_ok

    matched = np.zeros(D, dtype=bool)
    gt_used = np.zeros(G, dtype=bool)
    for d in np.argsort(-iou.max(axis=1)):
        valid = (iou[d] >= iou_threshold) & (~gt_used)
        if not valid.any():
            continue
        g = int(np.argmax(iou[d] * valid.astype(float)))
        matched[d] = True
        gt_used[g] = True
    return matched

def _behaviour_classify(
    matched_rgb: np.ndarray,
    matched_x: np.ndarray,
    matched_union: np.ndarray,
) -> np.ndarray:



    N = len(matched_rgb)
    labels = np.full(N, 3, dtype=int)
    labels[matched_rgb] = 0
    labels[~matched_rgb & matched_x] = 1
    labels[~matched_rgb & ~matched_x & matched_union] = 2
    return labels

def _compute_ap(recall, precision):

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))

def _compute_ap_robust(tp, conf, pred_cls, num_classes):

    if len(tp) == 0:
        return 0.0, 0.0
    order = np.argsort(-conf)
    tp, pred_cls = tp[order], pred_cls[order]
    ap50s, aps = [], []
    for c in range(num_classes):
        mask = pred_cls == c
        if not mask.any():
            continue
        tpc = tp[mask].astype(float).cumsum()
        fpc = (1.0 - tp[mask].astype(float)).cumsum()
        n_gt = int(tp.sum())
        if tpc[-1] == 0:
            continue
        recall = tpc / max(tpc[-1], 1)
        precision = tpc / np.maximum(tpc + fpc, 1e-16)
        ap50s.append(_compute_ap(recall, precision))
    return float(np.mean(ap50s)) if ap50s else 0.0, 0.0





class TREValidator(MultiModalCOCOValidator):












    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self._tre_dx = 0
        self._tre_dy = 0





    def _shift_ir_in_batch(self, batch: dict) -> dict:









        dx, dy = self._tre_dx, self._tre_dy
        if dx == 0 and dy == 0:
            return batch

        img = batch["img"]
        rgb = img[:, :3, :, :]
        ir = img[:, 3:, :, :]

        B, C_ir, H, W = ir.shape
        device = ir.device
        gray = 114.0 / 255.0





        theta = torch.tensor(
            [[1.0, 0.0, -2.0 * dx / W],
             [0.0, 1.0, -2.0 * dy / H]],
            device=device, dtype=torch.float32,
        ).unsqueeze(0).repeat(B, 1, 1)

        grid = F.affine_grid(theta, ir.shape, align_corners=False)


        ir_shifted = F.grid_sample(
            ir - gray, grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        ) + gray

        batch["img"] = torch.cat([rgb, ir_shifted], dim=1)
        return batch





    @smart_inference_mode()
    def validate_tre(
        self,
        offsets=None,
        trainer=None,
        model=None,
    ):













        if offsets is None:
            offsets = [(s, s) for s in range(0, 16, 3)]


        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            if self.data is None:
                self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device
            self.args.half = model.fp16
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = 1
            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' not found"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            if not pt:
                self.args.rect = False
            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(
                self.data.get(self.args.split), self.args.batch
            )
            model.eval()

            if hasattr(self, "data") and self.data and "Xch" in self.data:
                xch = self.data.get("Xch", 3)
                total_ch = 3 + xch
            else:
                total_ch = 6
            model.warmup(imgsz=(1 if pt else self.args.batch, total_ch, imgsz, imgsz))

        self.run_callbacks("on_val_start")
        self.model = model


        self.nc = self.nc or getattr(model, 'nc', None) or self.data.get('nc', 80)
        self.names = getattr(model, 'names', {}) or self.data.get('names', {})


        all_gt_boxes_xyxy_norm = []
        all_gt_cls = []
        all_img_sizes = []
        image_ids = []

        LOGGER.info("Collecting GT annotations ...")
        for batch_i, batch in enumerate(self.dataloader):
            batch = self.preprocess(batch)
            for si in range(len(batch["img"])):
                pbatch = self._prepare_batch(si, batch)
                ori_h, ori_w = pbatch["ori_shape"]
                all_img_sizes.append((int(ori_h), int(ori_w)))
                image_ids.append(batch_i * self.args.batch + si)

                if pbatch["bboxes"] is not None and pbatch["bboxes"].numel() > 0:
                    scale = torch.tensor(
                        [ori_w, ori_h, ori_w, ori_h],
                        device=pbatch["bboxes"].device,
                        dtype=pbatch["bboxes"].dtype,
                    )
                    gt_norm = (pbatch["bboxes"] / scale).clamp_(0, 1)
                    all_gt_boxes_xyxy_norm.append(gt_norm.cpu().numpy())
                    all_gt_cls.append(pbatch["cls"].cpu().numpy().astype(int))
                else:
                    all_gt_boxes_xyxy_norm.append(np.zeros((0, 4)))
                    all_gt_cls.append(np.zeros(0, dtype=int))

        num_images = len(all_img_sizes)
        num_classes = int(self.nc)
        LOGGER.info(f"Collected GT for {num_images} images, {num_classes} classes.")


        all_results = []
        for dx, dy in offsets:
            self._tre_dx = dx
            self._tre_dy = dy
            LOGGER.info(f"TRE offset ({dx}, {dy}) — running inference ...")
            t_start = time.time()


            self.init_metrics(de_parallel(model))
            self.jdict = []



            dt = (
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device),
            )
            bar = TQDM(self.dataloader, desc=f"TRE offset=({dx},{dy})",
                        total=len(self.dataloader))


            all_pred_boxes_norm = [[] for _ in range(num_images)]
            all_pred_cls_list = [[] for _ in range(num_images)]
            all_pred_scores = [[] for _ in range(num_images)]

            for batch_i, batch in enumerate(bar):

                with dt[0]:
                    batch = self.preprocess(batch)
                    batch = self._shift_ir_in_batch(batch)


                with dt[1]:
                    preds = model(batch["img"], augment=self.args.augment and (not self.training))


                with dt[2]:
                    preds = self.postprocess(preds)


                for si, pred in enumerate(preds):
                    img_idx = batch_i * self.args.batch + si
                    if img_idx >= num_images:
                        break
                    if pred is not None:
                        pbatch = self._prepare_batch(si, batch)
                        predn = self._prepare_pred(pred, pbatch) if isinstance(pred, dict) else pred
                        if isinstance(predn, dict):

                            if predn.get("bboxes") is None or predn["bboxes"].numel() == 0:
                                continue
                            bboxes_t = predn["bboxes"]
                            cls_t = predn["cls"]
                            scores_t = predn.get("conf", predn.get("scores", bboxes_t[:, 4] if bboxes_t.shape[1] > 4 else torch.zeros(bboxes_t.shape[0])))
                        else:

                            if predn.numel() == 0:
                                continue
                            bboxes_t = predn[:, :4]
                            cls_t = predn[:, 5]
                            scores_t = predn[:, 4]

                        ori_h, ori_w = all_img_sizes[img_idx]
                        scale = torch.tensor(
                            [ori_w, ori_h, ori_w, ori_h],
                            device=bboxes_t.device,
                            dtype=bboxes_t.dtype,
                        )
                        bboxes_norm = (bboxes_t / scale).clamp_(0, 1)
                        all_pred_boxes_norm[img_idx].append(bboxes_norm.cpu().numpy())
                        all_pred_cls_list[img_idx].append(
                            cls_t.cpu().numpy().astype(int)
                        )
                        all_pred_scores[img_idx].append(scores_t.cpu().numpy())


                self.update_metrics(preds, batch)



            pred_boxes_flat = []
            pred_cls_flat = []
            pred_scores_flat = []
            pred_img_idx = []
            for i in range(num_images):
                if all_pred_boxes_norm[i]:
                    boxes_i = np.vstack(all_pred_boxes_norm[i])
                    cls_i = np.concatenate(all_pred_cls_list[i])
                    scores_i = np.concatenate(all_pred_scores[i])
                else:
                    boxes_i = np.zeros((0, 4))
                    cls_i = np.zeros(0, dtype=int)
                    scores_i = np.zeros(0)
                pred_boxes_flat.append(boxes_i)
                pred_cls_flat.append(cls_i)
                pred_scores_flat.append(scores_i)
                pred_img_idx.extend([i] * len(boxes_i))




            conf_threshold = getattr(self.args, 'conf', 0.25)
            if conf_threshold is None or conf_threshold < 0.01:
                conf_threshold = 0.25
            n_rgb = 0
            n_x = 0
            n_fused = 0
            n_spurious = 0
            total_preds = 0

            for i in range(num_images):
                pred_boxes = pred_boxes_flat[i]
                gt_boxes = all_gt_boxes_xyxy_norm[i]
                gt_cls = all_gt_cls[i]
                pred_cls = pred_cls_flat[i]
                pred_scores = pred_scores_flat[i]

                if len(pred_boxes) == 0:
                    continue


                keep = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[keep]
                pred_cls = pred_cls[keep]

                D = len(pred_boxes)
                total_preds += D

                ori_h, ori_w = all_img_sizes[i]


                dx_norm = dx / ori_w
                dy_norm = dy / ori_h
                x_gt_boxes = gt_boxes.copy()
                x_gt_boxes[:, 0] += dx_norm
                x_gt_boxes[:, 1] += dy_norm
                x_gt_boxes[:, 2] += dx_norm
                x_gt_boxes[:, 3] += dy_norm
                x_gt_boxes = np.clip(x_gt_boxes, 0.0, 1.0)


                union_gt_boxes = np.zeros_like(gt_boxes)
                union_gt_boxes[:, 0] = np.minimum(gt_boxes[:, 0], x_gt_boxes[:, 0])
                union_gt_boxes[:, 1] = np.minimum(gt_boxes[:, 1], x_gt_boxes[:, 1])
                union_gt_boxes[:, 2] = np.maximum(gt_boxes[:, 2], x_gt_boxes[:, 2])
                union_gt_boxes[:, 3] = np.maximum(gt_boxes[:, 3], x_gt_boxes[:, 3])


                matched_rgb = _greedy_match_per_image(pred_boxes, gt_boxes, pred_cls, gt_cls)
                matched_x = _greedy_match_per_image(pred_boxes, x_gt_boxes, pred_cls, gt_cls)
                matched_union = _greedy_match_per_image(pred_boxes, union_gt_boxes, pred_cls, gt_cls)


                labels = _behaviour_classify(matched_rgb, matched_x, matched_union)
                n_rgb += int((labels == 0).sum())
                n_x += int((labels == 1).sum())
                n_fused += int((labels == 2).sum())
                n_spurious += int((labels == 3).sum())


            stats = self.get_stats()
            ap_robust_50 = float(stats.get("metrics/mAP50(B)", 0.0))
            ap_robust_50_95 = float(stats.get("metrics/mAP50-95(B)", 0.0))


            mrr = n_rgb / max(n_x, 1) if n_x > 0 else float("inf")
            sdr = n_spurious / max(total_preds, 1)
            if np.isinf(mrr):
                log_mrr = np.log(1e6 + 1)
                rfs_50 = ap_robust_50 * log_mrr / max(sdr, 1e-7)
                rfs_50_95 = ap_robust_50_95 * log_mrr / max(sdr, 1e-7)
            else:
                rfs_50 = ap_robust_50 * math.log(mrr + 1) / max(sdr, 1e-7)
                rfs_50_95 = ap_robust_50_95 * math.log(mrr + 1) / max(sdr, 1e-7)

            elapsed = time.time() - t_start

            tre_result = {
                "offset_dx": dx,
                "offset_dy": dy,
                "AP_robust_mAP50": round(ap_robust_50, 6),
                "AP_robust_mAP50_95": round(ap_robust_50_95, 6),
                "MRR": round(mrr, 4) if not np.isinf(mrr) else "inf",
                "SDR": round(sdr, 6),
                "RFS_mAP50": round(rfs_50, 6),
                "RFS_mAP50_95": round(rfs_50_95, 6),
                "behavior_counts": {
                    "RGB_Grounded": n_rgb,
                    "X_Grounded": n_x,
                    "Fused": n_fused,
                    "Spurious": n_spurious,
                    "Total": total_preds,
                    "conf_threshold": conf_threshold,
                },
                "runtime_s": round(elapsed, 1),
            }
            all_results.append(tre_result)
            self._log_tre_result(tre_result)


        rfs_values_50 = [r["RFS_mAP50"] for r in all_results]
        rfs_values_95 = [r["RFS_mAP50_95"] for r in all_results]
        offset_mags = [math.sqrt(r["offset_dx"] ** 2 + r["offset_dy"] ** 2) for r in all_results]
        rfs_total_50 = float(np.trapz(rfs_values_50, offset_mags)) if len(offset_mags) >= 2 else rfs_values_50[0]
        rfs_total_95 = float(np.trapz(rfs_values_95, offset_mags)) if len(offset_mags) >= 2 else rfs_values_95[0]

        self.run_callbacks("on_val_end")

        result = {
            "offsets": [(r["offset_dx"], r["offset_dy"]) for r in all_results],
            "per_offset": all_results,
            "RFS_total_mAP50": round(rfs_total_50, 6),
            "RFS_total_mAP50_95": round(rfs_total_95, 6),
            "num_images": num_images,
            "num_classes": num_classes,
        }

        self._print_tre_summary(result)
        return result





    def _log_tre_result(self, r):
        bc = r["behavior_counts"]
        ct = bc.get("conf_threshold", 0.25)
        LOGGER.info(
            f"  offset=({r['offset_dx']},{r['offset_dy']}) | "
            f"AP50={r['AP_robust_mAP50']:.4f} | "
            f"AP={r['AP_robust_mAP50_95']:.4f} | "
            f"MRR={r['MRR']} | SDR(conf≥{ct})={r['SDR']:.4f} | "
            f"RFS={r['RFS_mAP50_95']:.4f} | "
            f"RGB={bc['RGB_Grounded']} X={bc['X_Grounded']} "
            f"Fused={bc['Fused']} Spur={bc['Spurious']} | "
            f"{r['runtime_s']}s"
        )

    def _print_tre_summary(self, result):
        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info("TRE Evaluation Summary")
        LOGGER.info("=" * 60)
        LOGGER.info(f"  RFS_total (mAP50):     {result['RFS_total_mAP50']:.4f}")
        LOGGER.info(f"  RFS_total (mAP50-95):  {result['RFS_total_mAP50_95']:.4f}")
        LOGGER.info("-" * 60)
        LOGGER.info(f"  {'Offset':>12s}  {'AP50':>8s}  {'AP':>8s}  {'MRR':>8s}  {'SDR':>8s}  {'RFS':>8s}")
        LOGGER.info("  " + "-" * 52)
        for r in result["per_offset"]:
            mrr_str = f"{r['MRR']:.1f}" if r['MRR'] != "inf" else "  inf"
            LOGGER.info(
                f"  ({r['offset_dx']:>2d},{r['offset_dy']:>2d})     "
                f"{r['AP_robust_mAP50']:>8.4f}  "
                f"{r['AP_robust_mAP50_95']:>8.4f}  "
                f"{mrr_str:>8s}  "
                f"{r['SDR']:>8.4f}  "
                f"{r['RFS_mAP50_95']:>8.4f}"
            )
        LOGGER.info("=" * 60)
        LOGGER.info("")

