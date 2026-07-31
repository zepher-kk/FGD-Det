from __future__ import annotations
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.models.yolo.multimodal.predict import MultiModalDetectionPredictor

class MultiModalSegmentationPredictor(MultiModalDetectionPredictor, SegmentationPredictor):











    def write_results(self, i, p, im, s) -> str:







        orig_save = getattr(self.args, 'save', False)
        try:
            self.args.save = False
            string = super().write_results(i, p, im, s)
        finally:
            self.args.save = orig_save

        if not orig_save:
            return string

        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            visualize_x_to_3ch,
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            ensure_batch_idx_long,
        )
        import torch
        from pathlib import Path


        result = self.results[i]
        base = p.stem


        if im.dim() == 4:
            im_single = im[i:i+1]
        else:
            im_single = im.unsqueeze(0) if im.dim() == 3 else im


        ch = im_single.shape[1] if im_single.dim() == 4 else im_single.shape[0]


        is_dual_modal = ch == 6
        is_single_modal = ch == 3

        if not (is_dual_modal or is_single_modal):

            return string


        if is_dual_modal:

            rgb_tensor = im_single[:, :3]
            x_tensor = im_single[:, 3:]
        else:

            x_tensor = None
            rgb_tensor = None
            if hasattr(self, 'modality') and self.modality:
                if str(self.modality).lower() == 'rgb':
                    rgb_tensor = im_single
                else:
                    x_tensor = im_single
            else:

                rgb_tensor = im_single


        n_boxes = 0 if result.boxes is None else len(result.boxes)
        if n_boxes:
            cls_ids = result.boxes.cls.long()
            boxes_norm = result.boxes.xywhn if hasattr(result.boxes, 'xywhn') else None
            confs = getattr(result.boxes, 'conf', None)
            if not isinstance(cls_ids, torch.Tensor):
                cls_ids = torch.as_tensor(cls_ids, dtype=torch.long)
            if boxes_norm is not None and not isinstance(boxes_norm, torch.Tensor):
                boxes_norm = torch.as_tensor(boxes_norm, dtype=torch.float32)
            if confs is not None and not isinstance(confs, torch.Tensor):
                confs = torch.as_tensor(confs, dtype=torch.float32)
            batch_idx = ensure_batch_idx_long(torch.zeros(cls_ids.shape[0]))
        else:
            cls_ids = torch.zeros((0,), dtype=torch.long)
            boxes_norm = torch.zeros((0, 4), dtype=torch.float32)
            confs = torch.zeros((0,), dtype=torch.float32)
            batch_idx = ensure_batch_idx_long(torch.zeros((0,), dtype=torch.long))

        names = getattr(self.model, 'names', {})


        if getattr(result, 'masks', None) is not None and getattr(result.masks, 'data', None) is not None:


            masks_data = result.masks.data
            if masks_data.dtype == torch.float32 or masks_data.dtype == torch.float64:


                masks = (masks_data > 0.5).to(torch.uint8) * 255
            else:

                masks = torch.as_tensor(masks_data, dtype=torch.uint8)
        else:

            H = im_single.shape[-2]
            W = im_single.shape[-1]
            masks = torch.zeros((0, H, W), dtype=torch.uint8)


        if is_single_modal:
            if rgb_tensor is not None:

                fname_rgb = self.save_dir / f"pred_{base}_labels_rgb.jpg"
                plot_images(rgb_tensor, batch_idx, cls_ids, boxes_norm, masks=masks, confs=confs,
                            paths=[str(p)], fname=fname_rgb, names=names)
            elif x_tensor is not None:

                x_visual = visualize_x_to_3ch(x_tensor, colorize=False, x_modality='x')
                fname_x = self.save_dir / f"pred_{base}_labels_x.jpg"
                plot_images(x_visual, batch_idx, cls_ids, boxes_norm, masks=masks, confs=confs,
                            paths=[str(p.with_name(f"{base}_x{p.suffix}"))], fname=fname_x, names=names)
            return string



        fname_rgb = self.save_dir / f"pred_{base}_labels_rgb.jpg"
        plot_images(rgb_tensor, batch_idx, cls_ids, boxes_norm, masks=masks, confs=confs,
                    paths=[str(p)], fname=fname_rgb, names=names)



        x_visual = visualize_x_to_3ch(x_tensor, colorize=False, x_modality='x')
        fname_x = self.save_dir / f"pred_{base}_labels_x.jpg"
        plot_images(x_visual, batch_idx, cls_ids, boxes_norm, masks=masks, confs=confs,
                    paths=[str(p.with_name(f"{base}_x{p.suffix}"))], fname=fname_x, names=names)


        side = concat_side_by_side(rgb_tensor, x_visual)
        batch_dup, cls_dup, boxes_dup, confs_dup = duplicate_bboxes_for_side_by_side(batch_idx, cls_ids, boxes_norm, confs)
        masks_side = torch.cat([masks, masks], dim=2) if masks.numel() else masks
        fname_mm = self.save_dir / f"pred_{base}_labels_multimodal.jpg"
        plot_images(side, batch_dup, cls_dup, boxes_dup, masks=masks_side, confs=confs_dup,
                    paths=[str(p.with_name(f"{base}_multimodal{p.suffix}"))], fname=fname_mm, names=names)

        return string

