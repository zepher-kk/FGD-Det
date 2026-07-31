from __future__ import annotations
"""
Heatmap visualization for YOLO multi-modal models.

This module provides heatmap visualization capabilities using various algorithms,
with primary support for Grad-CAM. It integrates with the VisualizationManager
to provide insights into model decision-making processes.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, List, Dict, Union, Any

from .manager import HeatmapResult
from .utils import HookManager, load_image
from ultralytics.utils.ops import non_max_suppression
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM, EigenGradCAM,
    XGradCAM, LayerCAM, FullGrad
)

def _resolve_cv2_colormap(colormap: Union[str, int, None]) -> Optional[int]:












    if colormap is None:
        return None
    if isinstance(colormap, int):
        return colormap
    name = str(colormap).lower().strip()
    if name in {"none", "gray", "grey", "grayscale"}:
        return None
    cmap_map = {
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "jet": cv2.COLORMAP_JET,
        "parula": cv2.COLORMAP_PARULA,
        "hot": cv2.COLORMAP_HOT,
    }
    if name not in cmap_map:
        raise ValueError(f"Unknown colormap: {colormap}. Supported: {', '.join(sorted(cmap_map.keys()))} (or 'gray'/'none').")
    return cmap_map[name]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):


















    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)


    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)


    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (top, bottom, left, right)

class DetectionTarget(torch.nn.Module):







    def __init__(self, conf_threshold: float = 0.2, ratio: float = 1.0, output_type: str = 'all'):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.output_type = output_type

    def __call__(self, model_output):
        post_result, pre_post_boxes = model_output
        result = []
        num_detections = int(post_result.size(0) * self.ratio)
        for i in range(num_detections):
            if float(post_result[i].max()) < self.conf_threshold:
                break
            if self.output_type == 'class' or self.output_type == 'all':
                result.append(post_result[i].max())
            if self.output_type == 'box' or self.output_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        if not result:
            return post_result[0].max()
        return sum(result)

class MultiModalWrapper(torch.nn.Module):







    
    def __init__(self, model):






        super().__init__()
        self.model = model
        
    def forward(self, x):









        return self.model(x)

class DetectionActivationsAndGradients:








    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):


        pred = result
        while isinstance(pred, (list, tuple)):
            pred = pred[0]

        if pred.ndim == 2:
            pred = pred.unsqueeze(0)

        outputs = []

        for bi in range(pred.shape[0]):
            logits = pred[bi, :, 4:]
            boxes = pred[bi, :, :4]
            _, indices = torch.sort(logits.max(1)[0], descending=True)
            outputs.append([logits[indices], boxes[indices]])
        return outputs

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        return self.post_process(model_output)

    def release(self):
        for handle in self.handles:
            handle.remove()

class HeatmapVisualizer:












    
    def __init__(self, model):






        self.model = model
        self.hook_manager = HookManager(model)
        self.padding_info = None
        
    def visualize(self, images: Union[torch.Tensor, Dict[str, torch.Tensor], str, Dict[str, str]], 
                  layers: List[str], 
                  targets: Optional[torch.Tensor] = None,
                  alg: str = 'gradcam',
                  batch_mode: bool = False,
                  renormalize: bool = False,
                  preprocessed_input: Optional[torch.Tensor] = None,
                  original_images: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
                  **kwargs) -> List[HeatmapResult]:



















        colormap = kwargs.get("colormap", "jet")
        blend_alpha = kwargs.get("blend_alpha", kwargs.get("alpha", 0.5))
        if not isinstance(blend_alpha, (int, float)) or not (0.0 <= float(blend_alpha) <= 1.0):
            raise ValueError(f"blend_alpha must be in [0,1], got: {blend_alpha}")
        cmap_cv2 = _resolve_cv2_colormap(colormap)

        def _as_bool(v: Any, default: bool) -> bool:
            if v is None:
                return bool(default)
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(int(v))
            if isinstance(v, str):
                s = v.strip().lower()
                if s in {"1", "true", "yes", "y", "on"}:
                    return True
                if s in {"0", "false", "no", "n", "off"}:
                    return False
            raise ValueError(f"cam_smooth 必须为 bool（或可解析为 bool 的字符串），收到: {v!r}")

        cam_smooth = _as_bool(kwargs.get("cam_smooth", True), True)
        cam_smooth_sigma = kwargs.get("cam_smooth_sigma", 1.2)
        try:
            cam_smooth_sigma = float(cam_smooth_sigma)
        except Exception:
            raise ValueError(f"cam_smooth_sigma 必须为数字，收到: {kwargs.get('cam_smooth_sigma')!r}")
        if not np.isfinite(cam_smooth_sigma) or cam_smooth_sigma < 0:
            raise ValueError(f"cam_smooth_sigma 必须为非负数，收到: {cam_smooth_sigma}")

        cam_resize_interp = str(kwargs.get("cam_resize_interp", "cubic")).lower().strip()

        alg_map = {
            'gradcam': GradCAM,
            'gradcam++': GradCAMPlusPlus,
            'scorecam': ScoreCAM,
            'eigencam': EigenCAM,
            'eigengradcam': EigenGradCAM,
            'xgradcam': XGradCAM,
            'layercam': LayerCAM,
            'fullgrad': FullGrad,
        }
        

        if not batch_mode:
            if isinstance(images, str):
                images = load_image(images)
            elif isinstance(images, dict) and all(isinstance(v, str) for v in images.values()):
                images = {k: load_image(v) if isinstance(v, str) else v 
                         for k, v in images.items()}
        

        is_batch = False
        batch_size = 1
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            is_batch = True
            batch_size = images.shape[0]
        elif isinstance(images, dict):

            first_modal = next(iter(images.values()))
            if isinstance(first_modal, torch.Tensor) and first_modal.dim() == 4:
                is_batch = True
                batch_size = first_modal.shape[0]
                

        if not layers:
            raise ValueError("layers parameter must be provided for visualization")
        

        if preprocessed_input is not None:
            processed_images = preprocessed_input
        else:
            processed_images = self._preprocess_image(images)
        target_batch = (
            int(processed_images.shape[0])
            if isinstance(processed_images, torch.Tensor) and processed_images.dim() == 4
            else batch_size
        )
        

        for p in self.model.parameters():
            p.requires_grad_(True)
        

        all_results = []
        
        for layer in layers:

            target_layer = self._get_layer_by_name(layer)
            

            layer_type = target_layer.__class__.__name__
            if layer_type in ['BatchNorm2d', 'Dropout', 'Upsample']:
                print(f"Warning: Layer {layer} ({layer_type}) may not produce meaningful heatmaps")
            

            cam_algorithm = alg_map.get(alg, GradCAM)
            wrapper = MultiModalWrapper(self.model)
            cam = cam_algorithm(model=wrapper, target_layers=[target_layer])

            cam.activations_and_grads = DetectionActivationsAndGradients(
                wrapper, [target_layer], None
            )


            if targets is None:

                conf_threshold = kwargs.get('conf_threshold', 0.2)
                ratio = kwargs.get('ratio', 1.0)
                output_type = kwargs.get('output_type', 'all')
                targets = [
                    DetectionTarget(conf_threshold=conf_threshold, ratio=ratio, output_type=output_type)
                    for _ in range(max(1, target_batch))
                ]
            

            grayscale_cam = cam(input_tensor=processed_images, targets=targets)
            

            if renormalize:

                with torch.no_grad():
                    detections = self.model(processed_images)
                    if isinstance(detections, (list, tuple)):
                        detections = detections[0]
                    

                    conf_threshold = kwargs.get('conf_threshold', 0.25)
                    nms_results = non_max_suppression(detections, conf_thres=conf_threshold, iou_thres=0.45)
                    

                    for i in range(grayscale_cam.shape[0]):
                        if nms_results[i] is not None and len(nms_results[i]) > 0:

                            boxes = nms_results[i][:, :4].cpu().numpy().astype(int)

                            grayscale_cam[i] = self.renormalize_cam_in_bounding_boxes(
                                boxes, processed_images[i].cpu().numpy(), grayscale_cam[i]
                            )
            

            if batch_mode:
                results = []
                for i in range(batch_size):
                    if isinstance(images, dict):
                        orig_imgs = {k: v[i] for k, v in images.items()}
                        heatmaps = {k: grayscale_cam[i] for k in images.keys()}
                    else:
                        orig_imgs = images[i]
                        heatmaps = grayscale_cam[i]
                        
                    processed = self._postprocess(
                        heatmaps,
                        orig_imgs,
                        batch=False,
                        colormap=cmap_cv2,
                        blend_alpha=float(blend_alpha),
                        cam_smooth=cam_smooth,
                        cam_smooth_sigma=float(cam_smooth_sigma),
                        cam_resize_interp=cam_resize_interp,
                    )
                    result = HeatmapResult(
                        original_image=processed['original'],
                        heatmap=processed['heatmap'], 
                        overlay=processed['overlay'],
                        metadata={
                            'layer': layer,
                            'algorithm': alg
                        }
                    )
                    results.append(result)
                all_results.extend(results)
            else:

                if original_images is not None:
                    imgs_for_post = original_images
                else:
                    imgs_for_post = images
                if isinstance(imgs_for_post, dict):

                    heatmaps = {k: grayscale_cam for k in imgs_for_post.keys()}
                    processed = self._postprocess(
                        heatmaps,
                        imgs_for_post,
                        batch=False,
                        colormap=cmap_cv2,
                        blend_alpha=float(blend_alpha),
                        cam_smooth=cam_smooth,
                        cam_smooth_sigma=float(cam_smooth_sigma),
                        cam_resize_interp=cam_resize_interp,
                    )
                else:
                    processed = self._postprocess(
                        grayscale_cam,
                        imgs_for_post,
                        batch=False,
                        colormap=cmap_cv2,
                        blend_alpha=float(blend_alpha),
                        cam_smooth=cam_smooth,
                        cam_smooth_sigma=float(cam_smooth_sigma),
                        cam_resize_interp=cam_resize_interp,
                    )
                
                result = HeatmapResult(
                    original_image=processed['original'],
                    heatmap=processed['heatmap'],
                    overlay=processed['overlay'],
                    metadata={
                        'layer': layer,
                        'algorithm': alg
                    }
                )
                all_results.append(result)
        
        return all_results
    
    def _preprocess_image(self, images: Union[torch.Tensor, Dict[str, torch.Tensor], str, Dict[str, str]]) -> torch.Tensor:










        self.padding_info = None
        

        input_size = 640
        if hasattr(self.model, 'args') and hasattr(self.model.args, 'imgsz'):
            input_size = self.model.args.imgsz
        

        if isinstance(images, str):
            image = load_image(images)

            image, ratio, padding = letterbox(image, new_shape=input_size)
            self.padding_info = padding

            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(images, dict):

            processed = {}
            first_ratio = None
            first_padding = None
            

            for idx, (key, img) in enumerate(images.items()):
                if idx == 0:

                    if isinstance(img, str):
                        img_temp = load_image(img)
                    elif isinstance(img, np.ndarray):
                        img_temp = img
                    elif isinstance(img, torch.Tensor):
                        img_temp = img.numpy() if not img.is_cuda else img.cpu().numpy()
                        if img_temp.ndim == 4:
                            img_temp = img_temp[0]
                        if img_temp.shape[0] in [3, 6]:
                            img_temp = img_temp.transpose(1, 2, 0)
                    else:
                        img_temp = img
                    

                    _, first_ratio, first_padding = letterbox(img_temp, new_shape=input_size)
                    self.padding_info = first_padding
                    break
            

            for key, img in images.items():
                if isinstance(img, str):
                    img = load_image(img)
                elif isinstance(img, torch.Tensor):

                    img_np = img.numpy() if not img.is_cuda else img.cpu().numpy()
                    if img_np.ndim == 4:
                        img_np = img_np[0]
                    if img_np.shape[0] in [3, 6]:
                        img_np = img_np.transpose(1, 2, 0)
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    img = img_np
                

                img, _, _ = letterbox(img, new_shape=input_size)
                

                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                processed[key] = img
                

            image = torch.cat(list(processed.values()), dim=1)
        elif isinstance(images, np.ndarray):

            if images.ndim == 3:
                images, ratio, padding = letterbox(images, new_shape=input_size)
                self.padding_info = padding
            elif images.ndim == 4:

                img_first = images[0]
                img_first, ratio, padding = letterbox(img_first, new_shape=input_size)
                self.padding_info = padding

                processed_batch = []
                for img in images:
                    img_processed, _, _ = letterbox(img, new_shape=input_size)
                    processed_batch.append(img_processed)
                images = np.stack(processed_batch)
            

            image = images.astype(np.float32) / 255.0
            if image.ndim == 3:
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            else:
                image = torch.from_numpy(image).permute(0, 3, 1, 2)
        elif isinstance(images, torch.Tensor):

            if images.dim() == 3 or (images.dim() == 4 and images.shape[0] == 1):
                img_np = images.squeeze(0) if images.dim() == 4 else images
                img_np = img_np.numpy() if not img_np.is_cuda else img_np.cpu().numpy()
                if img_np.shape[0] in [3, 6]:
                    img_np = img_np.transpose(1, 2, 0)
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                img_np, ratio, padding = letterbox(img_np, new_shape=input_size)
                self.padding_info = padding
                image = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            elif images.dim() == 4:

                batch_size = images.shape[0]
                processed_batch = []
                for i in range(batch_size):
                    img_np = images[i].numpy() if not images.is_cuda else images[i].cpu().numpy()
                    if img_np.shape[0] in [3, 6]:
                        img_np = img_np.transpose(1, 2, 0)
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    
                    if i == 0:

                        img_np, ratio, padding = letterbox(img_np, new_shape=input_size)
                        self.padding_info = padding
                    else:

                        img_np, _, _ = letterbox(img_np, new_shape=input_size)
                    
                    img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1)
                    processed_batch.append(img_tensor)
                
                image = torch.stack(processed_batch)
            else:

                if images.max() > 1.0:
                    image = images / 255.0
                else:
                    image = images
                if image.dim() == 3:
                    image = image.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")
            
        return image
        
    def _get_layer_by_name(self, layer_name: str):












        parts = layer_name.split('.')
        module = self.model.model
        
        try:
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
        except (AttributeError, IndexError, KeyError) as e:

            available_layers = self._get_available_layers()
            raise ValueError(
                f"Layer '{layer_name}' not found in model.\n"
                f"Error: {str(e)}\n\n"
                f"Available top-level layers:\n" + 
                "\n".join([f"  - {name}" for name in available_layers[:10]]) +
                (f"\n  ... and {len(available_layers) - 10} more layers" if len(available_layers) > 10 else "") +
                f"\n\nTip: Use one of the above layer names or check your model architecture."
            )
                
        return module
    
    def _get_available_layers(self) -> List[str]:






        layers = []
        try:

            if hasattr(self.model, 'model'):
                for idx, module in enumerate(self.model.model):
                    layer_name = f"{idx}"
                    layer_type = module.__class__.__name__
                    layers.append(f"{layer_name} ({layer_type})")
                    

            for name, module in self.model.named_modules():
                if name and name.count('.') <= 2:
                    layers.append(f"{name} ({module.__class__.__name__})")
                    
        except Exception:

            layers = ["model.0", "model.4", "model.6", "model.8", "model.10"]
            
        return layers[:50]
        
            
    def _postprocess(
        self,
        heatmaps: Union[torch.Tensor, Dict[str, torch.Tensor]],
        original_images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        batch: bool = False,
        colormap: Optional[int] = None,
        blend_alpha: float = 0.5,
        cam_smooth: bool = True,
        cam_smooth_sigma: float = 1.2,
        cam_resize_interp: str = "cubic",
    ) -> Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:











        if isinstance(heatmaps, dict):

            if batch and isinstance(next(iter(heatmaps.values())), torch.Tensor) and next(iter(heatmaps.values())).dim() == 4:

                batch_size = next(iter(heatmaps.values())).shape[0]
                batch_results = []
                for i in range(batch_size):
                    result = {}
                    for modality, heatmap in heatmaps.items():
                        orig_img = original_images[modality]
                        single_processed = self._postprocess_single(
                            heatmap[i],
                            orig_img[i],
                            colormap=colormap,
                            blend_alpha=blend_alpha,
                            cam_smooth=cam_smooth,
                            cam_smooth_sigma=cam_smooth_sigma,
                            cam_resize_interp=cam_resize_interp,
                        )

                        for key, value in single_processed.items():
                            result[f"{modality}_{key}"] = value
                    batch_results.append(result)
                return batch_results
            else:

                processed = {'original': {}, 'heatmap': {}, 'overlay': {}}
                for modality, heatmap in heatmaps.items():

                    if modality in original_images:
                        orig_img = original_images[modality]
                        single_processed = self._postprocess_single(
                            heatmap,
                            orig_img,
                            colormap=colormap,
                            blend_alpha=blend_alpha,
                            cam_smooth=cam_smooth,
                            cam_smooth_sigma=cam_smooth_sigma,
                            cam_resize_interp=cam_resize_interp,
                        )
                        

                        processed['original'][modality] = single_processed['original']
                        processed['heatmap'][modality] = single_processed['heatmap']
                        processed['overlay'][modality] = single_processed['overlay']
                return processed
        else:

            if batch and isinstance(heatmaps, torch.Tensor) and heatmaps.dim() == 4:

                batch_processed = []
                for i in range(heatmaps.shape[0]):
                    single_processed = self._postprocess_single(
                        heatmaps[i],
                        original_images[i],
                        colormap=colormap,
                        blend_alpha=blend_alpha,
                        cam_smooth=cam_smooth,
                        cam_smooth_sigma=cam_smooth_sigma,
                        cam_resize_interp=cam_resize_interp,
                    )
                    batch_processed.append(single_processed)
                return batch_processed
            else:

                return self._postprocess_single(
                    heatmaps,
                    original_images,
                    colormap=colormap,
                    blend_alpha=blend_alpha,
                    cam_smooth=cam_smooth,
                    cam_smooth_sigma=cam_smooth_sigma,
                    cam_resize_interp=cam_resize_interp,
                )
            
    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):












        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)

            box_cam = grayscale_cam[y1:y2, x1:x2].copy()
            if box_cam.size > 0 and box_cam.max() > 0:
                box_cam = (box_cam - box_cam.min()) / (box_cam.max() - box_cam.min())
            renormalized_cam[y1:y2, x1:x2] = box_cam
        return renormalized_cam
    
    def _postprocess_single(
        self,
        heatmap: torch.Tensor,
        original_image: torch.Tensor,
        colormap: Optional[int] = None,
        blend_alpha: float = 0.5,
        cam_smooth: bool = True,
        cam_smooth_sigma: float = 1.2,
        cam_resize_interp: str = "cubic",
    ) -> Dict[str, np.ndarray]:











        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
            
        if isinstance(original_image, torch.Tensor):
            orig_np = original_image.cpu().numpy()
            if orig_np.shape[0] in [3, 6]:
                orig_np = orig_np.transpose(1, 2, 0)

            if orig_np.max() <= 1.0:
                orig_np = (orig_np * 255).astype(np.uint8)
        else:
            orig_np = original_image
            

        if orig_np.dtype != np.uint8:
            if orig_np.max() <= 1.0:
                orig_np = (orig_np * 255).clip(0, 255).astype(np.uint8)
            else:
                orig_np = orig_np.clip(0, 255).astype(np.uint8)


        if orig_np.ndim == 2:
            orig_np = orig_np[:, :, None]
            

        if heatmap.ndim > 2:
            heatmap = heatmap.squeeze()
            

        target_size = (orig_np.shape[1], orig_np.shape[0])
        interp_name = str(cam_resize_interp).lower().strip()
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "bilinear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "lanczos4": cv2.INTER_LANCZOS4,
            "area": cv2.INTER_AREA,
        }
        if interp_name not in interp_map:
            raise ValueError(f"不支持的 cam_resize_interp: {cam_resize_interp}，可选：{', '.join(sorted(interp_map.keys()))}")
        heatmap_resized = cv2.resize(heatmap, target_size, interpolation=interp_map[interp_name])
        heatmap_resized = heatmap_resized.astype(np.float32)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)


        if cam_smooth:
            sigma = float(cam_smooth_sigma)
            if sigma > 0:
                heatmap_resized = cv2.GaussianBlur(heatmap_resized, (0, 0), sigmaX=sigma, sigmaY=sigma)
                heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
        

        heatmap_normalized = (np.clip(heatmap_resized, 0.0, 1.0) * 255).astype(np.uint8)
        

        if colormap is None:
            heatmap_colored = cv2.cvtColor(heatmap_normalized, cv2.COLOR_GRAY2RGB)
        else:
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        

        if orig_np.shape[2] == 6:
            orig_rgb = orig_np[:, :, :3]
        else:
            orig_rgb = orig_np
            

        if orig_rgb.shape[2] == 1:
            orig_rgb = cv2.cvtColor(orig_rgb, cv2.COLOR_GRAY2RGB)
            

        overlayed = cv2.addWeighted(orig_rgb, 1 - float(blend_alpha), heatmap_colored, float(blend_alpha), 0)
        

        
        return {

            'original': orig_rgb,
            'heatmap': heatmap_colored,
            'overlay': overlayed
        }

