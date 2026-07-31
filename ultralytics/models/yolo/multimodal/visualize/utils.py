from __future__ import annotations

"""
多模态可视化工具函数和类。

包含图像加载、预处理、保存等基础功能，以及用于特征提取的HookManager类。
"""

import os
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

def load_image(
    image: Union[str, Path, np.ndarray, torch.Tensor, Image.Image],
    mode: str = 'RGB'
) -> np.ndarray:













    if isinstance(image, (str, Path)):

        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            

        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
            

        if mode == 'RGB' and len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    elif isinstance(image, np.ndarray):

        img = image.copy()
        
    elif isinstance(image, torch.Tensor):

        img = image.cpu().numpy()
        

        if img.ndim == 3 and img.shape[0] in [1, 3, 6]:
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 4:

            img = img[0]
            if img.shape[0] in [1, 3, 6]:
                img = np.transpose(img, (1, 2, 0))
                
    elif isinstance(image, Image.Image):

        if mode == 'RGB':
            img = np.array(image.convert('RGB'))
        else:
            img = np.array(image)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
    else:
        raise ValueError(f"不支持的图像类型: {type(image)}")
        
    return img.astype(np.float32)

def preprocess_image(
    image: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    to_tensor: bool = True
) -> Union[np.ndarray, torch.Tensor]:












    img = image.copy()
    

    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    

    if normalize:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.0:
            img = img / img.max()
    

    if to_tensor:

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        
    return img

def normalize_image(
    image: Union[np.ndarray, torch.Tensor],
    mean: Optional[Union[float, List[float]]] = None,
    std: Optional[Union[float, List[float]]] = None
) -> Union[np.ndarray, torch.Tensor]:











    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
        
    if isinstance(image, np.ndarray):

        img = image.copy()
        if isinstance(mean, (list, tuple)):
            mean = np.array(mean).reshape(1, 1, -1)
        if isinstance(std, (list, tuple)):
            std = np.array(std).reshape(1, 1, -1)
        return (img - mean) / std
        
    else:

        img = image.clone()
        if isinstance(mean, (list, tuple)):
            mean = torch.tensor(mean).view(-1, 1, 1)
        if isinstance(std, (list, tuple)):
            std = torch.tensor(std).view(-1, 1, 1)
            

        if img.device != mean.device:
            mean = mean.to(img.device)
            std = std.to(img.device)
            
        return (img - mean) / std

def save_image(
    image: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    denormalize: bool = False,
    mean: Optional[Union[float, List[float]]] = None,
    std: Optional[Union[float, List[float]]] = None
) -> None:











    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        

    if image.ndim == 3:
        if image.shape[0] in [1, 3, 6]:

            image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 2:

        image = image[:, :, np.newaxis]
        

    if denormalize:
        if mean is not None and std is not None:
            if isinstance(mean, (list, tuple)):
                mean = np.array(mean).reshape(1, 1, -1)
            if isinstance(std, (list, tuple)):
                std = np.array(std).reshape(1, 1, -1)
            image = image * std + mean
    

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255)
        else:
            image = image.clip(0, 255)
        image = image.astype(np.uint8)
    

    if image.shape[2] == 1:

        cv2.imwrite(str(path), image[:, :, 0])
    elif image.shape[2] == 3:

        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:

        cv2.imwrite(str(path), image)

class HookManager:






    
    def __init__(self, model: nn.Module):






        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.features: Dict[int, torch.Tensor] = {}
        self.gradients: Dict[int, torch.Tensor] = {}
        self._vishook_modules: List[nn.Module] = []
        
    def register_forward_hook(
        self,
        layer_or_module: Union[int, nn.Module],
        hook_fn: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> torch.utils.hooks.RemovableHandle:













        module: nn.Module
        if isinstance(layer_or_module, int):

            if not hasattr(self.model, 'model'):
                raise AttributeError("模型没有'model'属性")
            layer_idx = layer_or_module
            if layer_idx < 0 or layer_idx >= len(self.model.model):
                raise IndexError(f"层索引 {layer_idx} 超出范围 [0, {len(self.model.model)-1}]")
            module = self.model.model[layer_idx]
            tag = name or f"vishook_fwd_layer_{layer_idx}"
        elif isinstance(layer_or_module, nn.Module):
            module = layer_or_module
            tag = name or f"vishook_fwd_mod_{id(module)}"
        else:
            raise TypeError(f"不支持的layer_or_module类型: {type(layer_or_module)}，期望 int 或 nn.Module")
        

        def default_forward_hook(mod, input, output):

            try:
                key = getattr(mod, "_vishook_key", None)
                if key is None:
                    key = id(mod)
                self.features[int(key) if isinstance(key, int) else key] = output.detach()
            except Exception:

                self.features[id(mod)] = output.detach()
            

        if hook_fn is None:
            hook_fn = default_forward_hook
            

        if not hasattr(module, "_vishook_handles"):
            module._vishook_handles = []
        if not hasattr(module, "_vishook_tags"):
            module._vishook_tags = set()
        module._vishook_tags.add(tag)

        module._vishook_key = (name if name is not None else id(module))

        handle = module.register_forward_hook(hook_fn)
        module._vishook_handles.append(handle)
        self.hooks.append(handle)
        if module not in self._vishook_modules:
            self._vishook_modules.append(module)
        
        return handle
        
    def register_backward_hook(
        self,
        layer_or_module: Union[int, nn.Module],
        hook_fn: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> torch.utils.hooks.RemovableHandle:













        module: nn.Module
        if isinstance(layer_or_module, int):
            if not hasattr(self.model, 'model'):
                raise AttributeError("模型没有'model'属性")
            layer_idx = layer_or_module
            if layer_idx < 0 or layer_idx >= len(self.model.model):
                raise IndexError(f"层索引 {layer_idx} 超出范围 [0, {len(self.model.model)-1}]")
            module = self.model.model[layer_idx]
            tag = name or f"vishook_bwd_layer_{layer_idx}"
        elif isinstance(layer_or_module, nn.Module):
            module = layer_or_module
            tag = name or f"vishook_bwd_mod_{id(module)}"
        else:
            raise TypeError(f"不支持的layer_or_module类型: {type(layer_or_module)}，期望 int 或 nn.Module")
        

        def default_backward_hook(mod, grad_input, grad_output):
            key = getattr(mod, "_vishook_key", id(mod))
            if isinstance(grad_output, tuple):
                self.gradients[int(key) if isinstance(key, int) else key] = grad_output[0].detach()
            else:
                self.gradients[int(key) if isinstance(key, int) else key] = grad_output.detach()
                

        if hook_fn is None:
            hook_fn = default_backward_hook
            
        if not hasattr(module, "_vishook_handles"):
            module._vishook_handles = []
        if not hasattr(module, "_vishook_tags"):
            module._vishook_tags = set()
        module._vishook_tags.add(tag)
        module._vishook_key = id(module)

        handle = module.register_backward_hook(hook_fn)
        module._vishook_handles.append(handle)
        self.hooks.append(handle)
        if module not in self._vishook_modules:
            self._vishook_modules.append(module)
        
        return handle
        
    def clear_hooks(self) -> None:







        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks.clear()


        for mod in self._vishook_modules:
            try:
                handles = getattr(mod, "_vishook_handles", [])
                for h in list(handles):
                    try:
                        h.remove()
                    except Exception:
                        pass
                if hasattr(mod, "_vishook_handles"):
                    mod._vishook_handles = []
                if hasattr(mod, "_vishook_tags"):
                    mod._vishook_tags = set()
                if hasattr(mod, "_vishook_key"):
                    delattr(mod, "_vishook_key")
            except Exception:
                pass
        self._vishook_modules.clear()
        

        self.features.clear()
        self.gradients.clear()
        
    def get_features(self, layer_idx: Optional[int] = None) -> Union[Dict[int, torch.Tensor], torch.Tensor]:









        if layer_idx is not None:
            return self.features.get(layer_idx)
        return self.features.copy()
        
    def get_gradients(self, layer_idx: Optional[int] = None) -> Union[Dict[int, torch.Tensor], torch.Tensor]:









        if layer_idx is not None:
            return self.gradients.get(layer_idx)
        return self.gradients.copy()
        
    def __del__(self):





        self.clear_hooks()


    def remove_all_hooks(self) -> None:
        self.clear_hooks()

