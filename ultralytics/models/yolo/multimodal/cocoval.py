from __future__ import annotations
# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator
from ultralytics.nn.mm.complexity import (
    build_default_complexity_summary,
    compute_default_multimodal_complexity_report,
)
from ultralytics.utils.coco_metrics import COCOMetrics, COCO_AREA_SMALL, COCO_AREA_MEDIUM
from ultralytics.utils.coco_eval_bbox_mm import COCOevalBBoxMM
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.ops import Profile
from ultralytics.utils import TQDM, callbacks
import torch
import numpy as np
import json
import csv
import time
from pathlib import Path
from tqdm import tqdm

class MultiModalCOCOValidator(MultiModalDetectionValidator):






















    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):











        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        

        self.coco_metrics = None
        

        self.coco_stats = []
        

        self.total_batches = len(dataloader) if dataloader else 0
        self.current_batch = 0
        self.progress_bar = None
        

        self.num_images_processed = 0
        

        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}
        self.times = []
        

        if self.modality:
            LOGGER.info(f"初始化MultiModalCOCOValidator - 单模态COCO验证: {self.modality}-only")
        else:
            LOGGER.info("初始化MultiModalCOCOValidator - 双模态COCO验证")


        self._coco_computed = False

    def init_metrics(self, model):










        self.model = model
        


        super().init_metrics(model)
        

        self.image_ori_shapes = []
        self.all_pred_boxes = []
        self.all_target_boxes = []
        self.all_pred_cls = []
        self.all_target_cls = []
        self.pred_to_img = []
        self.target_to_img = []
        

        self.coco_metrics = COCOMetrics(
            save_dir=self.save_dir,
            names=getattr(model, 'names', {}),
            plot=self.args.plots if hasattr(self.args, 'plots') else False,
            on_plot=getattr(self, 'on_plot', None)
        )
        

        self.complexity_report = None
        self.gflops_total = None
        self.stage_gflops = {}
        try:
            imgsz = int(getattr(self.args, 'imgsz', 640))
            self.complexity_report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
            summary = build_default_complexity_summary(model, self.complexity_report)
            self.gflops_total = summary["gflops_total"]
            self.stage_gflops = summary["stage_gflops"]
        except Exception:
            pass
        

        self.coco_stats = []

        self.gt_size_counts = {"small": 0, "medium": 0, "large": 0}

        self._coco_computed = False
        

        if not hasattr(self, 'nc'):
            self.nc = getattr(model, 'nc', len(getattr(model, 'names', {})))
        if not hasattr(self, 'end2end'):
            self.end2end = getattr(model, "end2end", False)
        if not hasattr(self, 'names'):
            self.names = getattr(model, 'names', {})
        if not hasattr(self, 'seen'):
            self.seen = 0
        if not hasattr(self, 'jdict'):
            self.jdict = []
        
        LOGGER.debug(f"初始化COCO评估指标 - 类别数: {self.nc}")

    def get_desc(self):








        return ("%22s" + "%11s" * 7) % ("Class", "Images", "Instances", "Box(P", "R", "F1", "mAP50", "mAP50-95)")

    def update_metrics(self, preds, batch):











        self.current_batch += 1
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            self.progress_bar.set_description(f"验证批次 {self.current_batch}/{self.total_batches}")
        

        if not hasattr(self, 'image_ori_shapes'):
            self.image_ori_shapes = []
            self.all_pred_boxes = []
            self.all_target_boxes = []
            self.all_pred_cls = []
            self.all_target_cls = []
            self.pred_to_img = []
            self.target_to_img = []


        super().update_metrics(preds, batch)


        for si, pred in enumerate(preds):

            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred, pbatch)


            ori_shape = pbatch["ori_shape"]
            self.image_ori_shapes.append(ori_shape)
            img_idx = len(self.image_ori_shapes) - 1


            h, w = int(ori_shape[0]), int(ori_shape[1])
            if w == 0 or h == 0:
                continue
            scale = torch.tensor([w, h, w, h], device=predn["bboxes"].device, dtype=predn["bboxes"].dtype)


            if predn["bboxes"] is not None and predn["bboxes"].numel() > 0:
                bboxes_norm = (predn["bboxes"] / scale).clamp_(0, 1)
                for i in range(bboxes_norm.shape[0]):
                    self.all_pred_boxes.append(bboxes_norm[i].detach().cpu().numpy())
                    self.all_pred_cls.append(predn["cls"][i].detach().cpu().numpy())
                    self.pred_to_img.append(img_idx)


            if pbatch["bboxes"] is not None and pbatch["bboxes"].numel() > 0:
                gt_norm = (pbatch["bboxes"] / scale).clamp_(0, 1)
                gt_cls = pbatch["cls"]
                for i in range(gt_norm.shape[0]):
                    self.all_target_boxes.append(gt_norm[i].detach().cpu().numpy())
                    self.all_target_cls.append(gt_cls[i].detach().cpu().numpy())
                    self.target_to_img.append(img_idx)
    
    def update_speed_stats(self, preprocess_time=None, inference_time=None, postprocess_time=None, total_time=None):









        if preprocess_time is not None:
            self.speed['preprocess'] = preprocess_time
        if inference_time is not None:
            self.speed['inference'] = inference_time
        if postprocess_time is not None:
            self.speed['postprocess'] = postprocess_time
        if total_time is not None:
            self.times.append(total_time)

    def print_results(self):



        if self.coco_metrics is None:
            LOGGER.warning("COCO指标尚未初始化，无法输出结果")
            return


        super().print_results()


        LOGGER.info("")


        self._print_class_metrics()


        self._print_overall_metrics()


        self._save_csv_results()
    
    def _print_table(self, table_data):






        if not table_data:
            return

        ncols = len(table_data[0])

        fmt = "%22s" + "%11s" * (ncols - 1)
        for row in table_data:
            LOGGER.info(fmt % tuple(str(c) for c in row))
    
    def _print_speed_stats(self):

        pre = self.speed.get('preprocess', 0.0)
        inf = self.speed.get('inference', 0.0)
        post = self.speed.get('postprocess', 0.0)
        total = pre + inf + post
        fps = 1000.0 / total if total > 0 else 0.0
        LOGGER.info(
            "COCO Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image (%.1f FPS)"
            % (pre, inf, post, fps)
        )
    
    def _print_class_metrics(self):

        hdr = ("%22s" + "%11s" * 4) % ("Class", "AP", "AP50", "AP75", "F1")
        row_fmt = "%22s" + "%11.3g" * 4


        f1_map = {}
        if hasattr(self.metrics, 'box') and hasattr(self.metrics.box, 'f1') and len(self.metrics.box.f1):
            for i, c in enumerate(self.metrics.box.ap_class_index):
                f1_map[int(c)] = float(self.metrics.box.f1[i])

        rows = []

        if hasattr(self.coco_metrics, 'class_stats') and self.coco_metrics.class_stats and 'ap' in self.coco_metrics.class_stats:
            ap_array = self.coco_metrics.class_stats['ap']
            unique_classes = self.coco_metrics.class_stats['unique_classes']
            for ci, class_idx in enumerate(unique_classes):
                class_idx = int(class_idx)
                name = self.names[class_idx] if class_idx < len(self.names) else f"class_{class_idx}"
                if ci < ap_array.shape[0]:
                    ap = float(ap_array[ci].mean())
                    ap50 = float(ap_array[ci, 0]) if 0 < ap_array.shape[1] else 0.0
                    ap75 = float(ap_array[ci, 5]) if 5 < ap_array.shape[1] else 0.0
                else:
                    ap = ap50 = ap75 = 0.0
                rows.append((name, ap, ap50, ap75, f1_map.get(class_idx, 0.0)))
        elif hasattr(self.coco_metrics, 'per_class_metrics') and isinstance(self.coco_metrics.per_class_metrics, dict) and self.coco_metrics.per_class_metrics:
            for class_id in sorted(self.coco_metrics.per_class_metrics.keys()):
                ci = int(class_id)
                name = self.names[ci] if ci < len(self.names) else f"class_{ci}"
                m = self.coco_metrics.per_class_metrics[class_id]
                rows.append((name, float(m.get('AP', 0.0)), float(m.get('AP50', 0.0)), float(m.get('AP75', 0.0)), f1_map.get(ci, 0.0)))
        elif hasattr(self.metrics, 'box') and hasattr(self.metrics.box, 'ap_class_index'):
            for i, c in enumerate(self.metrics.box.ap_class_index):
                name = self.names[c] if c < len(self.names) else f"class_{c}"
                p, r, ap50, ap = self.metrics.box.class_result(i)
                f1_i = float(self.metrics.box.f1[i]) if i < len(self.metrics.box.f1) else 0.0
                rows.append((name, ap, ap50, 0.0, f1_i))

        if rows:
            LOGGER.info("COCO Per-Class Metrics:")
            LOGGER.info(hdr)

            ap_all = np.mean([r[1] for r in rows])
            ap50_all = np.mean([r[2] for r in rows])
            ap75_all = np.mean([r[3] for r in rows])
            f1_all = np.mean([r[4] for r in rows])
            LOGGER.info(row_fmt % ("all", ap_all, ap50_all, ap75_all, f1_all))
            for name, ap, ap50, ap75, f1 in rows:
                LOGGER.info(row_fmt % (name, ap, ap50, ap75, f1))
        else:
            LOGGER.info("COCO Per-Class Metrics: N/A")
    
    def _print_overall_metrics(self):

        row_fmt_s7 = "%22s" + "%11s" * 7
        row_fmt_g7 = "%22s" + "%11.3g" * 7
        row_fmt_s6 = "%22s" + "%11s" * 6
        row_fmt_g6 = "%22s" + "%11.3g" * 6


        f1_arr = self.metrics.box.f1 if hasattr(self.metrics, 'box') and hasattr(self.metrics.box, 'f1') else []
        mf = float(np.mean(f1_arr)) if hasattr(f1_arr, '__len__') and len(f1_arr) else 0.0


        LOGGER.info("COCO AP Summary:")
        LOGGER.info(row_fmt_s7 % ("Overall", "AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "mF1"))
        LOGGER.info(row_fmt_g7 % (
            "all",
            getattr(self.coco_metrics, 'AP', 0.0),
            getattr(self.coco_metrics, 'AP50', 0.0),
            getattr(self.coco_metrics, 'AP75', 0.0),
            getattr(self.coco_metrics, 'APsmall', 0.0),
            getattr(self.coco_metrics, 'APmedium', 0.0),
            getattr(self.coco_metrics, 'APlarge', 0.0),
            mf,
        ))


        LOGGER.info("COCO AR Summary:")
        LOGGER.info(row_fmt_s6 % ("Overall", "AR@1", "AR@10", "AR@100", "ARsmall", "ARmedium", "ARlarge"))
        LOGGER.info(row_fmt_g6 % (
            "all",
            getattr(self.coco_metrics, 'AR1', 0.0),
            getattr(self.coco_metrics, 'AR10', 0.0),
            getattr(self.coco_metrics, 'AR100', 0.0),
            getattr(self.coco_metrics, 'ARsmall', 0.0),
            getattr(self.coco_metrics, 'ARmedium', 0.0),
            getattr(self.coco_metrics, 'ARlarge', 0.0),
        ))


        LOGGER.info("COCO Size Breakdown:")
        size_hdr = ("%22s" + "%11s" * 5) % ("Size", "AP", "AP50", "AP75", "AR@100", "GTs")
        size_row = "%22s" + "%11.3g" * 4 + "%11d"
        LOGGER.info(size_hdr)
        for tag in ("small", "medium", "large"):
            LOGGER.info(size_row % (
                tag.capitalize(),
                getattr(self.coco_metrics, f'AP{tag}', 0.0),
                getattr(self.coco_metrics, f'AP{tag}50', 0.0),
                getattr(self.coco_metrics, f'AP{tag}75', 0.0),
                getattr(self.coco_metrics, f'AR{tag}', 0.0),
                self.gt_size_counts.get(tag, 0),
            ))


        try:
            if hasattr(self.coco_metrics, 'class_stats') and self.coco_metrics.class_stats and 'ap' in self.coco_metrics.class_stats:
                ap_array = self.coco_metrics.class_stats['ap']
                if ap_array.size > 0:
                    ap_iou_mean = ap_array.mean(axis=0)
                    ious = np.linspace(0.50, 0.95, 10)
                    iou_labels = [f"@{t:.2f}" for t in ious]
                    LOGGER.info("COCO AP@IoU Slices:")
                    hdr = ("%22s" + "%9s" * 10) % tuple(["IoU-Slice"] + iou_labels)
                    LOGGER.info(hdr)
                    vals = tuple(["all"] + [float(v) for v in ap_iou_mean])
                    LOGGER.info(("%22s" + "%9.3f" * 10) % vals)
        except Exception:
            pass


        params = 0
        if hasattr(self, 'model') and self.model is not None:
            try:
                params = build_default_complexity_summary(self.model, self.complexity_report)["params"] if self.complexity_report else 0
            except Exception:
                params = 0
        total_g = f"{self.gflops_total:.2f}" if isinstance(self.gflops_total, (int, float)) and self.gflops_total else "N/A"
        stage = self.stage_gflops or {}
        LOGGER.info(
            "Model: Params=%s | GFLOPs(total[default])=%s | Stages(rgb/x/fusion/head)=%s/%s/%s/%s"
            % (
                f"{params:,}" if params > 0 else "N/A",
                total_g,
                f"{stage.get('rgb_branch', 0.0):.2f}",
                f"{stage.get('x_branch', 0.0):.2f}",
                f"{stage.get('fusion', 0.0):.2f}",
                f"{stage.get('head', 0.0):.2f}",
            )
        )
    
    def _save_csv_results(self):








        try:

            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            

            f1_map = {}
            if hasattr(self.metrics, 'box') and hasattr(self.metrics.box, 'f1') and len(self.metrics.box.f1):
                for i, c in enumerate(self.metrics.box.ap_class_index):
                    f1_map[int(c)] = float(self.metrics.box.f1[i])


            class_csv_path = save_dir / "coco_metrics_by_class.csv"
            with open(class_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Class', 'AP', 'AP50', 'AP75', 'F1']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()


                if hasattr(self.coco_metrics, 'class_stats') and self.coco_metrics.class_stats and 'ap' in self.coco_metrics.class_stats:
                    ap_array = self.coco_metrics.class_stats['ap']
                    unique_classes = self.coco_metrics.class_stats['unique_classes']

                    iou_50_idx = 0
                    iou_75_idx = 5

                    for ci, class_idx in enumerate(unique_classes):
                        class_idx = int(class_idx)
                        class_name = self.names[class_idx] if class_idx < len(self.names) else f"class_{class_idx}"

                        if ci < ap_array.shape[0]:
                            ap = float(ap_array[ci].mean())
                            ap50 = float(ap_array[ci, iou_50_idx]) if iou_50_idx < ap_array.shape[1] else 0
                            ap75 = float(ap_array[ci, iou_75_idx]) if iou_75_idx < ap_array.shape[1] else 0
                        else:
                            ap = ap50 = ap75 = 0

                        writer.writerow({
                            'Class': class_name,
                            'AP': f"{ap:.3f}",
                            'AP50': f"{ap50:.3f}",
                            'AP75': f"{ap75:.3f}",
                            'F1': f"{f1_map.get(class_idx, 0.0):.3f}"
                        })
                elif hasattr(self.metrics, 'box') and hasattr(self.metrics.box, 'ap_class_index'):

                    ap_class_index = self.metrics.box.ap_class_index
                    for i, c in enumerate(ap_class_index):
                        class_name = self.names[c] if c < len(self.names) else f"class_{c}"
                        p, r, ap50, ap = self.metrics.box.class_result(i)
                        f1_i = float(self.metrics.box.f1[i]) if i < len(self.metrics.box.f1) else 0.0
                        writer.writerow({
                            'Class': class_name,
                            'AP': f"{ap:.3f}",
                            'AP50': f"{ap50:.3f}",
                            'AP75': "0.000",
                            'F1': f"{f1_i:.3f}"
                        })
            

            size_csv_path = save_dir / "coco_metrics_by_size.csv"
            with open(size_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Size', 'AP', 'AP50', 'AP75']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                sizes = ['Small', 'Medium', 'Large']
                for size in sizes:
                    ap = getattr(self.coco_metrics, f'AP{size.lower()}', 0.0)
                    ap50 = getattr(self.coco_metrics, f'AP{size.lower()}50', 0.0)
                    ap75 = getattr(self.coco_metrics, f'AP{size.lower()}75', 0.0)
                    
                    writer.writerow({
                        'Size': size,
                        'AP': f"{ap:.3f}",
                        'AP50': f"{ap50:.3f}",
                        'AP75': f"{ap75:.3f}"
                    })
            

            overall_csv_path = save_dir / "coco_metrics_overall.csv"
            with open(overall_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Metric', 'Value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                

                fps = 0.0
                if any(self.speed.values()):
                    total_time = self.speed['preprocess'] + self.speed['inference'] + self.speed['postprocess']
                    if total_time > 0:
                        fps = 1000.0 / total_time
                
                params = 0
                if hasattr(self, 'model') and self.model is not None:
                    try:
                        if hasattr(self.model, 'parameters'):
                            params = sum(p.numel() for p in self.model.parameters())
                        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'parameters'):
                            params = sum(p.numel() for p in self.model.model.parameters())
                    except:
                        params = 0
                

                f1_arr_csv = self.metrics.box.f1 if hasattr(self.metrics, 'box') and hasattr(self.metrics.box, 'f1') else []
                mf_csv = float(np.mean(f1_arr_csv)) if hasattr(f1_arr_csv, '__len__') and len(f1_arr_csv) else 0.0


                metrics_data = [
                    ('AP', f"{self.coco_metrics.AP:.3f}"),
                    ('AP50', f"{self.coco_metrics.AP50:.3f}"),
                    ('AP75', f"{self.coco_metrics.AP75:.3f}"),
                    ('APsmall', f"{getattr(self.coco_metrics, 'APsmall', 0.0):.3f}"),
                    ('APmedium', f"{getattr(self.coco_metrics, 'APmedium', 0.0):.3f}"),
                    ('APlarge', f"{getattr(self.coco_metrics, 'APlarge', 0.0):.3f}"),
                    ('mF1', f"{mf_csv:.3f}"),
                    ('AR1', f"{getattr(self.coco_metrics, 'AR1', 0.0):.3f}"),
                    ('AR10', f"{getattr(self.coco_metrics, 'AR10', 0.0):.3f}"),
                    ('AR100', f"{getattr(self.coco_metrics, 'AR100', 0.0):.3f}"),
                    ('ARsmall', f"{getattr(self.coco_metrics, 'ARsmall', 0.0):.3f}"),
                    ('ARmedium', f"{getattr(self.coco_metrics, 'ARmedium', 0.0):.3f}"),
                    ('ARlarge', f"{getattr(self.coco_metrics, 'ARlarge', 0.0):.3f}"),
                    ('FPS', f"{fps:.1f}"),
                    ('Parameters', str(params) if params > 0 else "N/A"),
                    ('GFLOPs(total[default])', (f"{self.gflops_total:.2f}" if self.gflops_total else "N/A")),
                    ('GFLOPs(rgb_branch)', (f"{self.stage_gflops.get('rgb_branch', 0.0):.2f}" if self.stage_gflops else "N/A")),
                    ('GFLOPs(x_branch)', (f"{self.stage_gflops.get('x_branch', 0.0):.2f}" if self.stage_gflops else "N/A")),
                    ('GFLOPs(fusion)', (f"{self.stage_gflops.get('fusion', 0.0):.2f}" if self.stage_gflops else "N/A")),
                    ('GFLOPs(head)', (f"{self.stage_gflops.get('head', 0.0):.2f}" if self.stage_gflops else "N/A")),
                    ('Images', str(self.num_images_processed)),
                    ('Modality', self.modality if self.modality else 'multimodal')
                ]
                
                for metric, value in metrics_data:
                    writer.writerow({'Metric': metric, 'Value': value})
            

            comprehensive_csv_path = save_dir / "coco_metrics_comprehensive.csv"
            with open(comprehensive_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Category', 'Type', 'Metric', 'Value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                

                if hasattr(self.coco_metrics, 'class_stats') and self.coco_metrics.class_stats and 'ap' in self.coco_metrics.class_stats:
                    ap_array = self.coco_metrics.class_stats['ap']
                    unique_classes = self.coco_metrics.class_stats['unique_classes']
                    
                    iou_50_idx = 0
                    iou_75_idx = 5
                    
                    for ci, class_idx in enumerate(unique_classes):
                        class_idx = int(class_idx)
                        class_name = self.names[class_idx] if class_idx < len(self.names) else f"class_{class_idx}"
                        
                        if ci < ap_array.shape[0]:
                            ap = float(ap_array[ci].mean())
                            ap50 = float(ap_array[ci, iou_50_idx]) if iou_50_idx < ap_array.shape[1] else 0
                            ap75 = float(ap_array[ci, iou_75_idx]) if iou_75_idx < ap_array.shape[1] else 0
                        else:
                            ap = ap50 = ap75 = 0
                        
                        writer.writerow({'Category': class_name, 'Type': 'Class', 'Metric': 'AP', 'Value': f"{ap:.3f}"})
                        writer.writerow({'Category': class_name, 'Type': 'Class', 'Metric': 'AP50', 'Value': f"{ap50:.3f}"})
                        writer.writerow({'Category': class_name, 'Type': 'Class', 'Metric': 'AP75', 'Value': f"{ap75:.3f}"})
                        writer.writerow({'Category': class_name, 'Type': 'Class', 'Metric': 'F1', 'Value': f"{f1_map.get(class_idx, 0.0):.3f}"})
                

                sizes = ['Small', 'Medium', 'Large']
                for size in sizes:
                    ap = getattr(self.coco_metrics, f'AP{size.lower()}', 0.0)
                    ap50 = getattr(self.coco_metrics, f'AP{size.lower()}50', 0.0)
                    ap75 = getattr(self.coco_metrics, f'AP{size.lower()}75', 0.0)
                    
                    writer.writerow({'Category': size, 'Type': 'Size', 'Metric': 'AP', 'Value': f"{ap:.3f}"})
                    writer.writerow({'Category': size, 'Type': 'Size', 'Metric': 'AP50', 'Value': f"{ap50:.3f}"})
                    writer.writerow({'Category': size, 'Type': 'Size', 'Metric': 'AP75', 'Value': f"{ap75:.3f}"})
                

                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'AP', 'Value': f"{self.coco_metrics.AP:.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'AP50', 'Value': f"{self.coco_metrics.AP50:.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'AP75', 'Value': f"{self.coco_metrics.AP75:.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'mF1', 'Value': f"{mf_csv:.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'AR1', 'Value': f"{getattr(self.coco_metrics, 'AR1', 0.0):.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'AR10', 'Value': f"{getattr(self.coco_metrics, 'AR10', 0.0):.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'AR100', 'Value': f"{getattr(self.coco_metrics, 'AR100', 0.0):.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'GFLOPs(total[default])', 'Value': (f"{self.gflops_total:.2f}" if self.gflops_total else "N/A")})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'GFLOPs(rgb_branch)', 'Value': (f"{self.stage_gflops.get('rgb_branch', 0.0):.2f}" if self.stage_gflops else "N/A")})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'GFLOPs(x_branch)', 'Value': (f"{self.stage_gflops.get('x_branch', 0.0):.2f}" if self.stage_gflops else "N/A")})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'GFLOPs(fusion)', 'Value': (f"{self.stage_gflops.get('fusion', 0.0):.2f}" if self.stage_gflops else "N/A")})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'GFLOPs(head)', 'Value': (f"{self.stage_gflops.get('head', 0.0):.2f}" if self.stage_gflops else "N/A")})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'APsmall', 'Value': f"{getattr(self.coco_metrics, 'APsmall', 0.0):.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'APmedium', 'Value': f"{getattr(self.coco_metrics, 'APmedium', 0.0):.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'APlarge', 'Value': f"{getattr(self.coco_metrics, 'APlarge', 0.0):.3f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'FPS', 'Value': f"{fps:.1f}"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'Parameters', 'Value': str(params) if params > 0 else "N/A"})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'Images', 'Value': str(self.num_images_processed)})
                writer.writerow({'Category': 'Overall', 'Type': 'Summary', 'Metric': 'Modality', 'Value': self.modality if self.modality else 'multimodal'})
            
            LOGGER.info(f"CSV结果已保存到: {save_dir}")
            LOGGER.info(f"  - {class_csv_path.name}")
            LOGGER.info(f"  - {size_csv_path.name}")
            LOGGER.info(f"  - {overall_csv_path.name}")
            LOGGER.info(f"  - {comprehensive_csv_path.name}")
            
        except Exception as e:
            LOGGER.warning(f"保存CSV文件时出错: {e}")

    def get_stats(self):











        if self.coco_metrics is None:
            LOGGER.warning("COCO指标尚未初始化，返回空字典")
            return {}




        if not self._coco_computed:
            if hasattr(self, "metrics") and hasattr(self.metrics, "stats") and self.metrics.stats:
                self._process_coco_stats_from_metrics()
            else:
                self._set_default_coco_stats()
            self._coco_computed = True


        base_stats = super().get_stats()


        stats = {

            'metrics/precision(B)': getattr(self.coco_metrics, 'precision', 0.0),
            'metrics/recall(B)': getattr(self.coco_metrics, 'recall', 0.0),
            'metrics/mAP50(B)': getattr(self.coco_metrics, 'AP50', 0.0),
            'metrics/mAP50-95(B)': getattr(self.coco_metrics, 'AP', 0.0),
            

            'metrics/coco/AP': getattr(self.coco_metrics, 'AP', 0.0),
            'metrics/coco/AP50': getattr(self.coco_metrics, 'AP50', 0.0),
            'metrics/coco/AP75': getattr(self.coco_metrics, 'AP75', 0.0),
            'metrics/coco/APsmall': getattr(self.coco_metrics, 'APsmall', 0.0),
            'metrics/coco/APmedium': getattr(self.coco_metrics, 'APmedium', 0.0),
            'metrics/coco/APlarge': getattr(self.coco_metrics, 'APlarge', 0.0),
            'metrics/coco/AR1': getattr(self.coco_metrics, 'AR1', 0.0),
            'metrics/coco/AR10': getattr(self.coco_metrics, 'AR10', 0.0),
            'metrics/coco/AR100': getattr(self.coco_metrics, 'AR100', 0.0),
            'metrics/coco/ARsmall': getattr(self.coco_metrics, 'ARsmall', 0.0),
            'metrics/coco/ARmedium': getattr(self.coco_metrics, 'ARmedium', 0.0),
            'metrics/coco/ARlarge': getattr(self.coco_metrics, 'ARlarge', 0.0),
            

            'fitness': getattr(self.coco_metrics, 'AP', 0.0),
            

            'val/speed_preprocess': self.speed.get('preprocess', 0.0),
            'val/speed_inference': self.speed.get('inference', 0.0),
            'val/speed_postprocess': self.speed.get('postprocess', 0.0),
            

            'val/images': len(self.coco_stats),
            'val/instances': sum(len(stat.get('ground_truth_labels', [])) for stat in self.coco_stats),
        }
        

        if hasattr(self, 'gflops_total') and self.gflops_total:
            stats['model/GFLOPs_default_total'] = float(self.gflops_total)
        if hasattr(self, 'stage_gflops') and self.stage_gflops:
            stats['model/GFLOPs_rgb_branch'] = float(self.stage_gflops.get('rgb_branch', 0.0))
            stats['model/GFLOPs_x_branch'] = float(self.stage_gflops.get('x_branch', 0.0))
            stats['model/GFLOPs_fusion'] = float(self.stage_gflops.get('fusion', 0.0))
            stats['model/GFLOPs_head'] = float(self.stage_gflops.get('head', 0.0))
        

        if self.modality:
            stats[f'val/modality'] = self.modality
            stats[f'metrics/coco/modality'] = self.modality
        else:
            stats[f'val/modality'] = 'multimodal'
            stats[f'metrics/coco/modality'] = 'RGB+X'
        

        if hasattr(self.coco_metrics, 'per_class_metrics'):
            for class_id, class_metrics in self.coco_metrics.per_class_metrics.items():
                class_name = getattr(self.coco_metrics, 'names', {}).get(class_id, f'class_{class_id}')
                stats[f'metrics/coco/class_{class_name}_AP'] = class_metrics.get('AP', 0.0)
                stats[f'metrics/coco/class_{class_name}_AP50'] = class_metrics.get('AP50', 0.0)
                stats[f'metrics/coco/class_{class_name}_AP75'] = class_metrics.get('AP75', 0.0)
        

        try:
            if isinstance(base_stats, dict):
                base_stats.update(stats)
                return base_stats
        except Exception:
            pass
        return stats


    
    def _set_default_coco_stats(self):



        default_stats = {
            'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
            'APsmall': 0.0, 'APmedium': 0.0, 'APlarge': 0.0,
            'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0,
            'ARsmall': 0.0, 'ARmedium': 0.0, 'ARlarge': 0.0
        }
        self.coco_metrics.update(default_stats)
    
    def _process_coco_stats_from_metrics(self):






        try:

            stats = self.metrics.stats
            

            tp = np.concatenate(stats['tp'], axis=0) if stats['tp'] else np.array([])
            conf = np.concatenate(stats['conf'], axis=0) if stats['conf'] else np.array([])
            pred_cls = np.concatenate(stats['pred_cls'], axis=0) if stats['pred_cls'] else np.array([])
            target_cls = np.concatenate(stats['target_cls'], axis=0) if stats['target_cls'] else np.array([])
            

            

            pred_boxes = None
            target_boxes = None
            ori_shapes = None
            

            pred_to_img = None
            target_to_img = None
            
            if hasattr(self, 'all_pred_boxes') and self.all_pred_boxes:
                pred_boxes = np.array(self.all_pred_boxes)
                pred_to_img = np.array(self.pred_to_img)
                if len(conf) != len(pred_boxes):
                    raise ValueError(
                        f"预测数量与框数量不一致：conf={len(conf)}, pred_boxes={len(pred_boxes)}。请检查采集逻辑。"
                    )
                    
            if hasattr(self, 'all_target_boxes') and self.all_target_boxes:
                target_boxes = np.array(self.all_target_boxes)
                target_to_img = np.array(self.target_to_img)
                if len(target_cls) != len(target_boxes):
                    raise ValueError(
                        f"GT数量与GT框数量不一致：target_cls={len(target_cls)}, target_boxes={len(target_boxes)}。请检查采集逻辑。"
                    )
                
            if hasattr(self, 'image_ori_shapes') and self.image_ori_shapes:

                ori_shapes = self.image_ori_shapes
            

            try:
                self.gt_size_counts = {"small": 0, "medium": 0, "large": 0}
                if target_boxes is not None and target_to_img is not None and ori_shapes is not None and len(target_boxes) == len(target_to_img):
                    for i in range(len(target_boxes)):
                        img_idx = int(target_to_img[i])
                        area = COCOMetrics.calculate_bbox_area(target_boxes[i], ori_shapes[img_idx], from_format='xyxy', normalized=True)
                        if area < COCO_AREA_SMALL:
                            self.gt_size_counts["small"] += 1
                        elif area < COCO_AREA_MEDIUM:
                            self.gt_size_counts["medium"] += 1
                        else:
                            self.gt_size_counts["large"] += 1
            except Exception:
                pass



            if conf.size == 0 or len(target_cls) == 0:
                self._set_default_coco_stats()
                return
            if pred_boxes is None or target_boxes is None or ori_shapes is None:
                self._set_default_coco_stats()
                return


            def xyxy01_to_xywh_px(box_xyxy01, shape_hw):
                h, w = int(shape_hw[0]), int(shape_hw[1])
                x1 = float(box_xyxy01[0]) * w
                y1 = float(box_xyxy01[1]) * h
                x2 = float(box_xyxy01[2]) * w
                y2 = float(box_xyxy01[3]) * h
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                return [x1, y1, bw, bh]

            img_count = len(ori_shapes) if isinstance(ori_shapes, (list, tuple)) else int(np.max(pred_to_img)) + 1
            imgIds = list(range(img_count))


            def to_cat_id(c):
                try:
                    return int(c)
                except Exception:
                    return int(c) if hasattr(c, 'item') else 0


            dts = []
            for i in range(len(conf)):
                img_id = int(pred_to_img[i])
                bbox_xywh = xyxy01_to_xywh_px(pred_boxes[i], ori_shapes[img_id])
                dts.append({
                    'image_id': img_id,
                    'category_id': to_cat_id(pred_cls[i]),
                    'bbox': bbox_xywh,
                    'score': float(conf[i]),
                    'id': i + 1,
                    'area': max(0.0, bbox_xywh[2]) * max(0.0, bbox_xywh[3]),
                })


            gts = []
            for j in range(len(target_cls)):
                img_id = int(target_to_img[j])
                bbox_xywh = xyxy01_to_xywh_px(target_boxes[j], ori_shapes[img_id])
                gts.append({
                    'image_id': img_id,
                    'category_id': to_cat_id(target_cls[j]),
                    'bbox': bbox_xywh,
                    'iscrowd': 0,
                    'ignore': 0,
                    'id': j + 1,
                    'area': max(0.0, bbox_xywh[2]) * max(0.0, bbox_xywh[3]),
                })

            catIds = sorted(list({d['category_id'] for d in dts} | {g['category_id'] for g in gts}))

            evaluator = COCOevalBBoxMM()
            evaluator.set_data(gts=gts, dts=dts, imgIds=imgIds, catIds=catIds)
            evaluator.evaluate()
            evaluator.accumulate()
            coco_stats = evaluator.summarize()

            for k, v in list(coco_stats.items()):
                if v == -1:
                    coco_stats[k] = 0.0


            self.coco_metrics.update({
                'AP': coco_stats.get('AP', 0.0),
                'AP50': coco_stats.get('AP50', 0.0),
                'AP75': coco_stats.get('AP75', 0.0),
                'APsmall': coco_stats.get('APsmall', 0.0),
                'APmedium': coco_stats.get('APmedium', 0.0),
                'APlarge': coco_stats.get('APlarge', 0.0),
                'APsmall50': coco_stats.get('APsmall50', 0.0),
                'APsmall75': coco_stats.get('APsmall75', 0.0),
                'APmedium50': coco_stats.get('APmedium50', 0.0),
                'APmedium75': coco_stats.get('APmedium75', 0.0),
                'APlarge50': coco_stats.get('APlarge50', 0.0),
                'APlarge75': coco_stats.get('APlarge75', 0.0),
                'AR1': coco_stats.get('AR1', 0.0),
                'AR10': coco_stats.get('AR10', 0.0),
                'AR100': coco_stats.get('AR100', 0.0),
                'ARsmall': coco_stats.get('ARsmall', 0.0),
                'ARmedium': coco_stats.get('ARmedium', 0.0),
                'ARlarge': coco_stats.get('ARlarge', 0.0),
            })


            try:
                per_class = evaluator.compute_per_class_metrics()

                self.coco_metrics.per_class_metrics = per_class
            except Exception:
                pass

            LOGGER.debug(
                f"COCO BBox 评估完成（ported）：AP={self.coco_metrics.AP:.3f}, AP50={self.coco_metrics.AP50:.3f}, "
                f"AP75={self.coco_metrics.AP75:.3f}"
            )
            
        except Exception as e:
            LOGGER.error(f"计算COCO指标时出错: {e}")
            import traceback
            traceback.print_exc()
            self._set_default_coco_stats()
    
    def _preprocess_coco_data(self):






        all_predictions = []
        all_ground_truths = []
        

        for stats in tqdm(self.coco_stats, desc="处理图像数据", unit="图像", leave=False):
            image_id = stats['image_id']
            preds = stats['predictions']
            gt_labels = stats['ground_truth_labels']
            gt_bboxes = stats['ground_truth_bboxes']
            orig_shape = stats['original_shape']
            

            if isinstance(preds, torch.Tensor) and len(preds) > 0:
                pred_data = self._process_predictions_batch(preds, image_id, orig_shape)
                all_predictions.extend(pred_data)
            

            if isinstance(gt_labels, torch.Tensor) and len(gt_labels) > 0:
                gt_data = self._process_ground_truths_batch(gt_labels, gt_bboxes, image_id, orig_shape)
                all_ground_truths.extend(gt_data)
        
        return all_predictions, all_ground_truths
    
    def _process_predictions_batch(self, preds, image_id, orig_shape):











        predictions = []
        

        if len(preds) > 0 and preds.shape[-1] >= 6:

            bboxes = preds[:, :4].cpu().numpy()
            confs = preds[:, 4].cpu().numpy()
            classes = preds[:, 5].cpu().numpy().astype(int)
            

            areas = np.array([COCOMetrics.calculate_bbox_area(bbox, orig_shape) for bbox in bboxes])
            

            for i in range(len(preds)):
                predictions.append({
                    'image_id': image_id,
                    'bbox': bboxes[i],
                    'confidence': float(confs[i]),
                    'class': int(classes[i]),
                    'area': areas[i],
                    'original_shape': orig_shape
                })
        
        return predictions
    
    def _process_ground_truths_batch(self, gt_labels, gt_bboxes, image_id, orig_shape):












        ground_truths = []
        
        if len(gt_labels) > 0 and len(gt_bboxes) > 0:

            labels = gt_labels.cpu().numpy().astype(int)
            bboxes = gt_bboxes.cpu().numpy()
            

            areas = np.array([COCOMetrics.calculate_bbox_area(bbox, orig_shape) for bbox in bboxes])
            

            for i in range(len(labels)):
                ground_truths.append({
                    'image_id': image_id,
                    'bbox': bboxes[i],
                    'class': int(labels[i]),
                    'area': areas[i],
                    'original_shape': orig_shape
                })
        
        return ground_truths
    
    def _compute_coco_metrics_optimized(self, predictions, ground_truths):












        if len(predictions) == 0 or len(ground_truths) == 0:
            return {
                'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                'APsmall': 0.0, 'APmedium': 0.0, 'APlarge': 0.0,
                'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0,
                'ARsmall': 0.0, 'ARmedium': 0.0, 'ARlarge': 0.0
            }
        
        try:

            tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, ori_shapes = self._convert_to_coco_format(
                predictions, ground_truths
            )
            

            temp_metrics = COCOMetrics(save_dir=self.save_dir, names=self.coco_metrics.names)
            temp_metrics.process(
                tp, conf, pred_cls, target_cls,
                pred_boxes=pred_boxes, 
                target_boxes=target_boxes, 
                ori_shapes=ori_shapes,
                show_progress=True
            )
            

            return temp_metrics.get_summary_dict()
            
        except Exception as e:
            LOGGER.error(f"COCO指标计算内部错误: {e}")
            return {
                'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                'APsmall': 0.0, 'APmedium': 0.0, 'APlarge': 0.0,
                'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0,
                'ARsmall': 0.0, 'ARmedium': 0.0, 'ARlarge': 0.0
            }
    
    def _convert_to_coco_format(self, predictions, ground_truths):











        if not predictions or not ground_truths:
            return (np.array([]), np.array([]), np.array([]), np.array([]), 
                   np.array([]).reshape(0, 4), np.array([]).reshape(0, 4), [])
        

        pred_confs = np.array([p['confidence'] for p in predictions])
        pred_classes = np.array([p['class'] for p in predictions])
        pred_boxes = np.array([p['bbox'] for p in predictions])
        

        target_classes = np.array([gt['class'] for gt in ground_truths])
        target_boxes = np.array([gt['bbox'] for gt in ground_truths])
        

        ori_shapes = list(set([tuple(p['original_shape']) for p in predictions + ground_truths]))
        


        tp = np.ones((len(predictions), 10))
        
        return tp, pred_confs, pred_classes, target_classes, pred_boxes, target_boxes, ori_shapes

    def run_validation(self):






        self.init_progress_bar()
        
        try:



            


            
            LOGGER.info("多模态COCO验证流程完成")
            
        finally:

            self.close_progress_bar()

    def _extract_ori_shapes(self, batch, batch_size):
















        orig_shapes = batch.get("ori_shape", None)
        
        if orig_shapes is None:

            LOGGER.warning(f"批次缺少原始尺寸信息，使用默认值 (640, 640)")
            return [(640, 640)] * batch_size
        

        if isinstance(orig_shapes, torch.Tensor):
            orig_shapes = orig_shapes.cpu().numpy().tolist()
        

        if isinstance(orig_shapes, (tuple, list)) and len(orig_shapes) == 2 and isinstance(orig_shapes[0], (int, float)):

            return [tuple(orig_shapes)] * batch_size
        

        if isinstance(orig_shapes, (list, tuple)):
            result = []
            for i in range(batch_size):
                if i < len(orig_shapes):
                    shape = orig_shapes[i]
                    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                        result.append(tuple(shape[:2]))
                    else:
                        result.append((640, 640))
                else:
                    result.append((640, 640))
            return result
        

        LOGGER.warning(f"无法解析原始尺寸格式: {type(orig_shapes)}，使用默认值")
        return [(640, 640)] * batch_size
    
    def _filter_labels_for_image(self, labels, bboxes, batch_idx, image_idx):














        if len(batch_idx) > 0 and len(labels) > 0:

            mask = (batch_idx == image_idx)
            current_labels = labels[mask] if mask.any() else torch.tensor([])
            current_bboxes = bboxes[mask] if mask.any() and len(bboxes) > 0 else torch.tensor([]).reshape(0, 4)
        else:
            current_labels = labels if len(labels) > 0 else torch.tensor([])
            current_bboxes = bboxes if len(bboxes) > 0 else torch.tensor([]).reshape(0, 4)
        

        if isinstance(current_labels, torch.Tensor) and current_labels.numel() > 0:
            current_labels = current_labels.clone()
        if isinstance(current_bboxes, torch.Tensor) and current_bboxes.numel() > 0:
            current_bboxes = current_bboxes.clone()
            
        return current_labels, current_bboxes
    
    def init_progress_bar(self):







        if self.total_batches > 0:
            self.progress_bar = tqdm(
                total=self.total_batches,
                desc="多模态COCO验证",
                unit="batch",
                leave=True,
                ncols=100
            )
            if self.modality:
                self.progress_bar.set_description(f"{self.modality.upper()}模态COCO验证")
            else:
                self.progress_bar.set_description("RGB+X多模态COCO验证")
    
    def close_progress_bar(self):



        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
    
    def finalize_metrics(self):







        self.close_progress_bar()
        


        super().finalize_metrics()
        

        if not self._coco_computed:
            if hasattr(self, 'metrics') and hasattr(self.metrics, 'stats') and self.metrics.stats:
                with tqdm(total=1, desc="计算COCO指标", unit="stage") as pbar:
                    self._process_coco_stats_from_metrics()
                    pbar.update(1)
            else:
                LOGGER.warning("没有可用的统计数据进行COCO评估")
                self._set_default_coco_stats()
            self._coco_computed = True
        

        if hasattr(self, 'metrics') and hasattr(self.metrics, 'seen'):
            self.num_images_processed = self.metrics.seen
        else:
            self.num_images_processed = self.seen
        

    
    def save_json(self, save_dir=None, filename=None):












        if self.coco_metrics is None:
            LOGGER.warning("COCO指标尚未计算，无法保存结果")
            return None
        

        save_dir = Path(save_dir or self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            modality_suffix = f"_{self.modality}" if self.modality else "_multimodal" 
            filename = f"coco_results{modality_suffix}.json"
        
        save_path = save_dir / filename
        

        results_data = {
            "evaluation_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "validator_type": "MultiModalCOCOValidator",
                "modality": self.modality if self.modality else "RGB+X",
                "dataset": getattr(self.args, 'data', 'N/A'),
                "num_classes": self.nc,
                "num_images": len(self.coco_stats),
                "num_instances": sum(len(stat.get('ground_truth_labels', [])) for stat in self.coco_stats)
            },
            
            "coco_metrics": {

                "AP": getattr(self.coco_metrics, 'AP', 0.0),
                "AP50": getattr(self.coco_metrics, 'AP50', 0.0),
                "AP75": getattr(self.coco_metrics, 'AP75', 0.0),
                "APsmall": getattr(self.coco_metrics, 'APsmall', 0.0),
                "APmedium": getattr(self.coco_metrics, 'APmedium', 0.0),
                "APlarge": getattr(self.coco_metrics, 'APlarge', 0.0),
                

                "AR1": getattr(self.coco_metrics, 'AR1', 0.0),
                "AR10": getattr(self.coco_metrics, 'AR10', 0.0),
                "AR100": getattr(self.coco_metrics, 'AR100', 0.0),
                "ARsmall": getattr(self.coco_metrics, 'ARsmall', 0.0),
                "ARmedium": getattr(self.coco_metrics, 'ARmedium', 0.0),
                "ARlarge": getattr(self.coco_metrics, 'ARlarge', 0.0),
                

                "precision": getattr(self.coco_metrics, 'precision', 0.0),
                "recall": getattr(self.coco_metrics, 'recall', 0.0)
            },
            
            "speed_statistics": {
                "preprocess_ms": self.speed.get('preprocess', 0.0),
                "inference_ms": self.speed.get('inference', 0.0),
                "postprocess_ms": self.speed.get('postprocess', 0.0),
                "total_ms": sum(self.speed.values()),
                "fps": 1000 / np.mean(self.times) if self.times else 0.0,
                "avg_time_per_image_ms": np.mean(self.times) if self.times else 0.0
            },
            
            "detailed_stats": self._get_detailed_stats_for_json(),
            
            "configuration": {
                "args": vars(self.args) if self.args else {},
                "model_info": {
                    "stride": getattr(self, 'stride', None),
                    "nc": self.nc
                }
            }
        }
        

        if hasattr(self.coco_metrics, 'per_class_metrics') and self.coco_metrics.per_class_metrics:
            class_names = getattr(self.coco_metrics, 'names', {})
            results_data["per_class_metrics"] = {}
            
            for class_id, class_metrics in self.coco_metrics.per_class_metrics.items():
                class_name = class_names.get(class_id, f"class_{class_id}")
                results_data["per_class_metrics"][class_name] = {
                    "AP": class_metrics.get('AP', 0.0),
                    "AP50": class_metrics.get('AP50', 0.0),
                    "AP75": class_metrics.get('AP75', 0.0),
                    "AR100": class_metrics.get('AR100', 0.0)
                }
        

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            LOGGER.info(f"COCO评估结果已保存到: {save_path}")
            return save_path
            
        except Exception as e:
            LOGGER.error(f"保存COCO结果时出错: {e}")
            return None
    
    def _get_detailed_stats_for_json(self):






        detailed_stats = {
            "total_predictions": 0,
            "total_ground_truths": 0,
            "images_processed": len(self.coco_stats),
            "processing_details": []
        }
        
        for i, stat in enumerate(self.coco_stats):
            image_detail = {
                "image_id": stat.get('image_id', i),
                "num_predictions": len(stat.get('predictions', [])) if isinstance(stat.get('predictions'), (list, torch.Tensor)) else 0,
                "num_ground_truths": len(stat.get('ground_truth_labels', [])) if isinstance(stat.get('ground_truth_labels'), (list, torch.Tensor)) else 0,
                "original_shape": stat.get('original_shape', (0, 0))
            }
            
            detailed_stats["total_predictions"] += image_detail["num_predictions"]
            detailed_stats["total_ground_truths"] += image_detail["num_ground_truths"]
            detailed_stats["processing_details"].append(image_detail)
        
        return detailed_stats
    
    def save_results(self, save_conf=True, save_json_results=True, plots=True):













        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info(f"开始保存COCO验证结果到: {save_dir}")
        
        results_saved = []
        

        if save_json_results:
            try:
                json_path = self.save_json(save_dir)
                if json_path:
                    results_saved.append(f"JSON结果: {json_path}")
            except Exception as e:
                LOGGER.error(f"保存JSON结果失败: {e}")
        

        if save_conf and hasattr(self.coco_metrics, 'confusion_matrix'):
            try:
                conf_path = save_dir / "confusion_matrix.png"
                if hasattr(self.coco_metrics.confusion_matrix, 'plot'):
                    self.coco_metrics.confusion_matrix.plot(save_dir=save_dir, names=getattr(self.coco_metrics, 'names', {}))
                    results_saved.append(f"混淆矩阵: {conf_path}")
            except Exception as e:
                LOGGER.warning(f"保存混淆矩阵失败: {e}")
        

        if plots:
            try:
                self._save_visualization_plots(save_dir)
                results_saved.append(f"可视化图表: {save_dir / 'plots'}")
            except Exception as e:
                LOGGER.warning(f"生成可视化图表失败: {e}")
        

        try:
            self._save_summary_report(save_dir)
            results_saved.append(f"汇总报告: {save_dir / 'coco_summary.txt'}")
        except Exception as e:
            LOGGER.warning(f"保存汇总报告失败: {e}")
        

        try:
            if hasattr(super(), 'save_results'):

                super_args = {}
                if 'save_conf' in super().save_results.__code__.co_varnames:
                    super_args['save_conf'] = save_conf
                if 'plots' in super().save_results.__code__.co_varnames:
                    super_args['plots'] = plots
                
                super().save_results(**super_args)
        except Exception as e:
            LOGGER.warning(f"调用父类保存方法失败: {e}")
        

        if results_saved:
            LOGGER.info("保存的结果文件:")
            for result in results_saved:
                LOGGER.info(f"  - {result}")
        else:
            LOGGER.warning("未保存任何结果文件")
        
        return save_dir
    
    def _save_visualization_plots(self, save_dir):






        plots_dir = save_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        

        self._plot_coco_metrics_comparison(plots_dir)
        

        self._plot_speed_statistics(plots_dir)
        

        if hasattr(self.coco_metrics, 'plot') and callable(self.coco_metrics.plot):
            try:
                self.coco_metrics.plot(save_dir=plots_dir)
            except Exception as e:
                LOGGER.warning(f"COCOMetrics绘图失败: {e}")
    
    def _plot_coco_metrics_comparison(self, save_dir):



        try:
            import matplotlib.pyplot as plt
            

            ap_metrics = ['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge']
            ap_values = [getattr(self.coco_metrics, metric, 0.0) for metric in ap_metrics]
            

            ar_metrics = ['AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge']
            ar_values = [getattr(self.coco_metrics, metric, 0.0) for metric in ar_metrics]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            

            ax1.bar(ap_metrics, ap_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
            ax1.set_title('COCO Average Precision (AP) Metrics')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            

            ax2.bar(ar_metrics, ar_values, color=['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e'])
            ax2.set_title('COCO Average Recall (AR) Metrics')
            ax2.set_ylabel('Score')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'coco_metrics_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            LOGGER.warning("matplotlib未安装，跳过COCO指标对比图生成")
        except Exception as e:
            LOGGER.warning(f"生成COCO指标对比图失败: {e}")
    
    def _plot_speed_statistics(self, save_dir):



        try:
            import matplotlib.pyplot as plt
            
            if not any(self.speed.values()):
                return
            
            stages = list(self.speed.keys())
            times = list(self.speed.values())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(stages, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            plt.title('Processing Speed Statistics')
            plt.ylabel('Time (ms)')
            plt.xlabel('Processing Stage')
            

            for bar, time_val in zip(bars, times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{time_val:.1f}ms', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'speed_statistics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            LOGGER.warning("matplotlib未安装，跳过速度统计图生成")
        except Exception as e:
            LOGGER.warning(f"生成速度统计图失败: {e}")
    
    def _save_summary_report(self, save_dir):






        report_path = save_dir / "coco_summary.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("多模态COCO评估汇总报告\n")
                f.write("=" * 80 + "\n\n")
                

                f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                f.write(f"验证器类型: MultiModalCOCOValidator\n")
                f.write(f"验证模式: {self.modality if self.modality else 'RGB+X'}\n")
                f.write(f"数据集: {getattr(self.args, 'data', 'N/A')}\n")
                f.write(f"类别数: {self.nc}\n")
                f.write(f"验证图像数: {len(self.coco_stats)}\n\n")
                

                f.write("主要COCO指标:\n")
                f.write("-" * 40 + "\n")
                f.write(f"mAP@0.5:0.95:  {getattr(self.coco_metrics, 'AP', 0.0):.3f}\n")
                f.write(f"mAP@0.5:      {getattr(self.coco_metrics, 'AP50', 0.0):.3f}\n")
                f.write(f"mAP@0.75:     {getattr(self.coco_metrics, 'AP75', 0.0):.3f}\n\n")
                

                f.write("不同尺寸目标指标:\n")
                f.write("-" * 40 + "\n")
                f.write(f"APsmall:      {getattr(self.coco_metrics, 'APsmall', 0.0):.3f}\n")
                f.write(f"APmedium:     {getattr(self.coco_metrics, 'APmedium', 0.0):.3f}\n")
                f.write(f"APlarge:      {getattr(self.coco_metrics, 'APlarge', 0.0):.3f}\n\n")
                

                f.write("召回指标:\n")
                f.write("-" * 40 + "\n")
                f.write(f"AR1:          {getattr(self.coco_metrics, 'AR1', 0.0):.3f}\n")
                f.write(f"AR10:         {getattr(self.coco_metrics, 'AR10', 0.0):.3f}\n")
                f.write(f"AR100:        {getattr(self.coco_metrics, 'AR100', 0.0):.3f}\n\n")
                

                if any(self.speed.values()):
                    f.write("速度统计:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"预处理:       {self.speed.get('preprocess', 0.0):.1f}ms\n")
                    f.write(f"推理:         {self.speed.get('inference', 0.0):.1f}ms\n")
                    f.write(f"后处理:       {self.speed.get('postprocess', 0.0):.1f}ms\n")
                    if self.times:
                        avg_time = np.mean(self.times)
                        f.write(f"平均处理时间: {avg_time:.1f}ms/图像\n")
                        f.write(f"处理速度:     {1000/avg_time:.1f} FPS\n")
                
                f.write("\n" + "=" * 80 + "\n")
                
        except Exception as e:
            LOGGER.error(f"保存汇总报告失败: {e}")
    
    def __enter__(self):



        self.init_progress_bar()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):



        self.close_progress_bar()
        return False

