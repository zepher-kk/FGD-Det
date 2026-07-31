from __future__ import annotations
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import smart_inference_mode, compute_model_gflops
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import callbacks
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.mm.utils import normalize_modality_token


class MultiModalSegmentationValidator(SegmentationValidator):







    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)

        if args:
            if isinstance(args, dict):
                self.modality = args.get("modality", None)
            else:
                self.modality = getattr(args, "modality", None)
        else:
            self.modality = None


        self.modality = normalize_modality_token(self.modality)

        if args is not None:
            if isinstance(args, dict):
                args["modality"] = self.modality
            else:
                setattr(args, "modality", self.modality)
        if hasattr(self, "args") and self.args is not None:
            if isinstance(self.args, dict):
                self.args["modality"] = self.modality
            else:
                setattr(self.args, "modality", self.modality)

        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        if self.training:

            self.device = trainer.device
            if self.data is None:
                self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            model.eval()

            try:
                if hasattr(model, "mm_router") and model.mm_router and self.modality:
                    model.mm_router.set_runtime_params(self.modality)
            except Exception:
                pass
        else:


            pass


        return super().__call__(trainer=trainer, model=model)




    def build_dataset(self, img_path, mode: str = "val", batch=None):








        data = getattr(self, "data", {}) or {}
        modalities = None
        mod_map = None
        if "modality_used" in data and isinstance(data["modality_used"], list):
            modalities = data["modality_used"]
        elif "models" in data and isinstance(data["models"], list):
            modalities = data["models"]
        mod_map = data.get("modality") or data.get("modalities")


        x_modality = None
        if isinstance(modalities, list):
            non_rgb = [m for m in modalities if m != "rgb"]
            if non_rgb:
                x_modality = non_rgb[0]
        if x_modality is None:
            x_modality = data.get("x_modality", None)

        x_modality_dir = None
        if isinstance(mod_map, dict) and x_modality in mod_map:
            x_modality_dir = mod_map[x_modality]
        elif x_modality:
            x_modality_dir = f"images_{x_modality}"


        stride = getattr(self, "stride", 32) or 32

        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=True,
            stride=stride,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=False,
        )




    def plot_val_samples(self, batch, ni):

        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities,
            visualize_x_to_3ch,
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            ensure_batch_idx_long,
            resolve_x_modality,
        )

        images = batch["img"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        masks = batch.get("masks", None)

        batch_idx = ensure_batch_idx_long(batch.get("batch_idx")) if "batch_idx" in batch else None
        if batch_idx is None:
            import torch
            batch_idx = ensure_batch_idx_long(torch.zeros(cls.shape[0], dtype=torch.long))

        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        rgb_images, x_images = split_modalities(images, xch)
        x_modality = resolve_x_modality(getattr(self, 'modality', None), getattr(self, 'data', None))


        plot_images(
            rgb_images,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=paths,
            fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
            on_plot=self.on_plot,
        )

        x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
        plot_images(
            x_visual,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
            fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
            on_plot=self.on_plot,
        )

        side = concat_side_by_side(rgb_images, x_visual)
        bidx2, cls2, bb2, _ = duplicate_bboxes_for_side_by_side(batch_idx, cls, bboxes, None)
        plot_images(
            side,
            bidx2,
            cls2,
            bb2,
            paths=[p.replace('.jpg', '_multimodal.jpg') for p in paths],
            fname=self.save_dir / f"val_batch{ni}_labels_multimodal.jpg",
            on_plot=self.on_plot,
        )




    def plot_predictions(self, batch, preds, ni):








        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities,
            visualize_x_to_3ch,
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            resolve_x_modality,
        )
        import torch
        from ultralytics.utils import ops

        CAP = 100

        images = batch["img"]
        paths = batch["im_file"]
        B, C, H, W = images.shape


        for i, p in enumerate(preds):
            p["batch_idx"] = torch.ones_like(p["cls"]) * i

        keys = preds[0].keys()
        batched_preds = {}


        for k in keys:
            if k == "masks":

                masks_list = []
                for p in preds:
                    m = p["masks"][:CAP] if p["masks"].numel() else p["masks"]
                    masks_list.append(m.to(torch.uint8).cpu())
                batched_preds[k] = torch.cat(masks_list, dim=0) if masks_list and masks_list[0].numel() else torch.zeros((0, H, W), dtype=torch.uint8)
            else:
                batched_preds[k] = torch.cat([p[k][:CAP] for p in preds], dim=0)


        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])
        batched_preds["bboxes"][:, 0] /= W
        batched_preds["bboxes"][:, 1] /= H
        batched_preds["bboxes"][:, 2] /= W
        batched_preds["bboxes"][:, 3] /= H


        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        rgb_images, x_images = split_modalities(images, xch)
        x_modality = resolve_x_modality(getattr(self, 'modality', None), getattr(self, 'data', None))



        plot_images(
            rgb_images,
            batched_preds["batch_idx"].long(),
            batched_preds["cls"],
            batched_preds["bboxes"],
            confs=batched_preds.get("conf"),
            masks=batched_preds.get("masks"),
            paths=paths,
            fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )


        x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
        plot_images(
            x_visual,
            batched_preds["batch_idx"].long(),
            batched_preds["cls"],
            batched_preds["bboxes"],
            confs=batched_preds.get("conf"),
            masks=batched_preds.get("masks"),
            paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
            fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )


        side = concat_side_by_side(rgb_images, x_visual)
        bidx2, cls2, bb2, conf2 = duplicate_bboxes_for_side_by_side(
            batched_preds["batch_idx"].long(),
            batched_preds["cls"],
            batched_preds["bboxes"],
            batched_preds.get("conf")
        )
        masks_side = torch.cat([batched_preds["masks"], batched_preds["masks"]], dim=2) if batched_preds["masks"].numel() else batched_preds["masks"]
        plot_images(
            side,
            bidx2,
            cls2,
            bb2,
            confs=conf2,
            masks=masks_side,
            paths=[p.replace('.jpg', '_multimodal.jpg') for p in paths],
            fname=self.save_dir / f"val_batch{ni}_pred_multimodal.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )




    def afss_score_sample(self, pred, batch, si):

        if "im_file" not in batch:
            raise KeyError("AFSS sample scoring requires batch['im_file']")

        pbatch = self._prepare_batch(si, batch)
        predn = self._prepare_pred(pred, pbatch)
        result = self._process_batch(predn, pbatch)

        tp_box = result.get("tp")
        tp_mask = result.get("tp_m")
        matched_box = int(tp_box[:, 0].sum()) if tp_box is not None and len(tp_box) else 0
        matched_mask = int(tp_mask[:, 0].sum()) if tp_mask is not None and len(tp_mask) else 0
        pred_count = int(len(predn["cls"]))
        gt_count = int(len(pbatch["cls"]))
        pred_mask_count = int(predn["masks"].shape[0]) if "masks" in predn else 0
        gt_mask_count = int(pbatch["masks"].shape[0]) if "masks" in pbatch else 0

        if gt_count == 0 and pred_count == 0:
            box_precision = 1.0
            box_recall = 1.0
            mask_precision = 1.0
            mask_recall = 1.0
            empty_case = "no_pred_no_label"
        elif gt_count == 0:
            box_precision = 0.0
            box_recall = 1.0
            mask_precision = 0.0
            mask_recall = 1.0
            empty_case = "pred_without_label"
        elif pred_count == 0:
            box_precision = 0.0
            box_recall = 0.0
            mask_precision = 0.0
            mask_recall = 0.0
            empty_case = "label_without_pred"
        else:
            box_precision = matched_box / pred_count
            box_recall = matched_box / gt_count
            mask_precision = matched_mask / pred_count
            mask_recall = matched_mask / gt_count
            empty_case = "normal"

        box_sufficiency = min(box_precision, box_recall)
        mask_sufficiency = min(mask_precision, mask_recall)
        sufficiency_raw = min(box_sufficiency, mask_sufficiency)

        return {
            "im_file": batch["im_file"][si],
            "task_name": "segment",
            "precision_op": box_precision,
            "recall_op": box_recall,
            "sufficiency_raw": sufficiency_raw,
            "valid_for_afss": True,
            "task_metrics": {
                "box_precision": float(box_precision),
                "box_recall": float(box_recall),
                "mask_precision": float(mask_precision),
                "mask_recall": float(mask_recall),
                "box_sufficiency": float(box_sufficiency),
                "mask_sufficiency": float(mask_sufficiency),
                "matched_box_count": int(matched_box),
                "matched_mask_count": int(matched_mask),
                "pred_count": int(pred_count),
                "gt_count": int(gt_count),
                "pred_mask_count": int(pred_mask_count),
                "gt_mask_count": int(gt_mask_count),
                "overlap_mask": int(bool(getattr(self.args, "overlap_mask", False))),
                "empty_case": empty_case,
            },

            "precision": float(box_precision),
            "recall": float(box_recall),
            "matched_count": int(matched_box),
            "pred_count": int(pred_count),
            "gt_count": int(gt_count),
        }

    def afss_score_batch(self, preds, batch):

        return [self.afss_score_sample(pred, batch, si) for si, pred in enumerate(preds)]

    def score_sample(self, pred, batch, si):

        return self.afss_score_sample(pred, batch, si)

    def score_batch(self, preds, batch):

        return self.afss_score_batch(preds, batch)

