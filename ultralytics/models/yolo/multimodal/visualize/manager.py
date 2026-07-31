from __future__ import annotations
"""
Visualization Manager for YOLOMM multimodal object detection.

This module provides:
- Data models for visualization results
- VisualizationManager as the main entry point for all visualization tasks
- Support for heatmap and feature map visualization
- Automatic output directory management
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import numpy as np
import cv2
import torch

from .utils import HookManager

class VisualizationResult:

    
    def __init__(self, 
                 vis_type: str,
                 data: Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]],
                 metadata: Optional[Dict[str, Any]] = None):








        self.type = vis_type
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:

        result = {
            'type': self.type,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
        

        if isinstance(self.data, dict):
            result['data_keys'] = list(self.data.keys())
        elif isinstance(self.data, list):
            result['data_count'] = len(self.data)
        else:
            result['data_shape'] = self.data.shape if hasattr(self.data, 'shape') else None
            
        return result
    
    def save(self, output_dir: Union[str, Path], prefix: str = "") -> List[str]:










        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        
        timestamp = self.timestamp.strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{self.type}_{timestamp}" if prefix else f"{self.type}_{timestamp}"
        
        def _to_uint8(img: np.ndarray) -> np.ndarray:
            if img.dtype == np.uint8:
                return img
            x = img
            try:
                if x.max() <= 1.0:
                    x = (x * 255.0).clip(0, 255)
                else:
                    x = x.clip(0, 255)
            except Exception:
                x = np.nan_to_num(x).clip(0, 255)
            return x.astype(np.uint8)

        def _imwrite_rgb(path: Path, img: np.ndarray) -> None:
            arr = _to_uint8(img)

            if arr.ndim == 3 and arr.shape[2] == 3:
                cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(path), arr)


        if isinstance(self.data, dict):

            for key, value in self.data.items():
                if isinstance(value, np.ndarray):
                    filename = f"{base_name}_{key}.png"
                    filepath = output_dir / filename
                    _imwrite_rgb(filepath, value)
                    saved_files.append(str(filepath))
                    
        elif isinstance(self.data, list):

            for idx, value in enumerate(self.data):
                if isinstance(value, np.ndarray):
                    filename = f"{base_name}_{idx:03d}.png"
                    filepath = output_dir / filename
                    _imwrite_rgb(filepath, value)
                    saved_files.append(str(filepath))
                    
        elif isinstance(self.data, np.ndarray):

            filename = f"{base_name}.png"
            filepath = output_dir / filename
            _imwrite_rgb(filepath, self.data)
            saved_files.append(str(filepath))
            
        return saved_files

class HeatmapResult(VisualizationResult):

    
    def __init__(self,
                 original_image: Union[np.ndarray, Dict[str, np.ndarray]],
                 heatmap: Union[np.ndarray, Dict[str, np.ndarray]],
                 overlay: Union[np.ndarray, Dict[str, np.ndarray]],
                 metadata: Optional[Dict[str, Any]] = None):










        self.original_image = original_image
        self.heatmap = heatmap
        self.overlay = overlay
        

        if isinstance(original_image, dict):

            data = {}
            for modal in original_image.keys():
                if modal in heatmap and modal in overlay:
                    data[f"{modal}_original"] = original_image[modal]
                    data[f"{modal}_heatmap"] = heatmap[modal]
                    data[f"{modal}_overlay"] = overlay[modal]
        else:

            data = {
                'original': original_image,
                'heatmap': heatmap,
                'overlay': overlay
            }
            

        super().__init__(vis_type='heatmap', data=data, metadata=metadata)
        
    @property
    def rgb_heatmap(self) -> Optional[np.ndarray]:

        if isinstance(self.heatmap, dict):
            return self.heatmap.get('rgb')
        return self.heatmap if not isinstance(self.original_image, dict) else None
    
    @property
    def x_heatmap(self) -> Optional[np.ndarray]:

        if isinstance(self.heatmap, dict):
            return self.heatmap.get('x')
        return None

class FeatureMapResult(VisualizationResult):

    
    def __init__(self,
                 layer_idx: Union[int, List[int]],
                 feature_maps: List[np.ndarray],
                 metadata: Optional[Dict[str, Any]] = None):








        self.layer_idx = layer_idx
        self.feature_maps = feature_maps
        

        if metadata is None:
            metadata = {}
        metadata['layer_idx'] = layer_idx
        metadata['num_maps'] = len(feature_maps)
        
        super().__init__(vis_type='feature_map', data=feature_maps, metadata=metadata)
        
    def get_feature_map(self, idx: int) -> Optional[np.ndarray]:

        if 0 <= idx < len(self.feature_maps):
            return self.feature_maps[idx]
        return None

class VisualizationManager:

    

    SUPPORTED_METHODS = ['heatmap', 'feature_map']
    
    def __init__(self, model, project: str = "runs/visualize", name: str = "exp"):









        self.model = model
        

        self.output_dir = self._setup_output_dir(project, name)
        

        self._visualizers = {}
        

        self.cache = {}
        

        self.hook_manager = HookManager(model)
    
    def __call__(self, 
                 source: Union[np.ndarray, Dict[str, np.ndarray]],
                 method: str = "heatmap",
                 **kwargs) -> VisualizationResult:
















        return self.visualize(source, method=method, **kwargs)
        
    def _setup_output_dir(self, project: str, name: str) -> Path:










        project_path = Path(project)
        

        i = 1
        while True:
            exp_name = name if i == 1 else f"{name}{i}"
            output_dir = project_path / exp_name
            if not output_dir.exists():
                break
            i += 1
            

        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Visualization output directory: {output_dir}")
        return output_dir
    
    def _load_images(self, source: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:











        return source
    
    def _validate_method(self, method: str) -> None:









        if method not in self.SUPPORTED_METHODS:

            method_descriptions = {
                'heatmap': 'Heatmap visualization (supports multiple algorithms: gradcam, gradcam++, ablationcam)',
                'feature_map': 'Feature map extraction and visualization'
            }
            
            supported_list = "\n".join([
                f"  - {m}: {method_descriptions.get(m, 'Method description not available')}"
                for m in self.SUPPORTED_METHODS
            ])
            
            raise ValueError(
                f"Unsupported visualization method: '{method}'.\n"
                f"\nSupported methods are:\n{supported_list}\n\n"
                f"Please choose one of the above methods."
            )
    
    def _get_visualizer(self, method: str):











        if method not in self._visualizers:
            try:
                if method == 'heatmap':
                    from .heatmap import HeatmapVisualizer
                    self._visualizers[method] = HeatmapVisualizer(self.model)
                elif method == 'feature_map':
                    from .feature import FeatureMapVisualizer
                    self._visualizers[method] = FeatureMapVisualizer(self.model)
                else:
                    raise ValueError(f"Visualizer not implemented for method: {method}")
            except ImportError as e:

                available_methods = ['heatmap', 'feature_map']
                raise ImportError(
                    f"Failed to import visualizer for method '{method}'.\n"
                    f"Please check if the module exists and all dependencies are installed.\n"
                    f"Available methods: {', '.join(available_methods)}\n"
                    f"Original error: {e}"
                )
                
        return self._visualizers[method]
    
    def _validate_input(self, source: Any) -> None:










        if source is None:
            raise ValueError("Input source cannot be None")
            
        if isinstance(source, dict):
            if len(source) == 0:
                raise ValueError("Input dictionary cannot be empty")

            valid_keys = {'rgb', 'x', 'thermal', 'depth', 'ir', 'infrared'}
            if not any(k in valid_keys for k in source.keys()):
                raise ValueError(
                    f"Dictionary must contain at least one valid modality key: {valid_keys}. "
                    f"Found keys: {list(source.keys())}"
                )

            for key, array in source.items():
                if not isinstance(array, np.ndarray):
                    raise TypeError(f"Dictionary value for '{key}' must be a NumPy array, got {type(array)}")
                self._validate_numpy_array(array, f"Dict['{key}']")
                
        elif isinstance(source, np.ndarray):
            self._validate_numpy_array(source, "Input")
        else:
            raise TypeError(
                f"Unsupported input type: {type(source)}. "
                "Expected: numpy array or dict of numpy arrays"
            )
    
    def _validate_numpy_array(self, array: np.ndarray, name: str) -> None:











        if array.size == 0:
            raise ValueError(f"{name} numpy array cannot be empty")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} numpy array must have numeric dtype, got {array.dtype}")
        if array.ndim < 2 or array.ndim > 4:
            raise ValueError(
                f"{name} numpy array must be 2D (HW), 3D (HWC/CHW), or 4D (NCHW), got shape {array.shape}. "
                f"Dimensions: {array.ndim}D"
            )

        if array.ndim >= 3:

            channel_dim = array.shape[2] if array.ndim == 3 else array.shape[1] if array.ndim == 4 else None
            if channel_dim is not None and channel_dim not in [1, 3, 4, 6]:
                raise ValueError(
                    f"{name} numpy array has unusual channel count: {channel_dim}. "
                    f"Expected 1 (grayscale), 3 (RGB), 4 (RGBA), or 6 (multi-modal)"
                )

        if np.any(np.isnan(array)):
            raise ValueError(f"{name} numpy array contains NaN values")
        if np.any(np.isinf(array)):
            raise ValueError(f"{name} numpy array contains infinite values")
    
    def _generate_cache_key(self, source: Any, method: str, layers: Optional[List[str]], alg: Optional[str] = None, **kwargs) -> str:














        key_parts = [method]
        

        if alg is not None:
            key_parts.append(f"alg:{alg}")
        

        if isinstance(source, np.ndarray):

            key_parts.append(f"array_{source.shape}_{source.dtype}")

            flat = source.flatten()
            indices = np.linspace(0, len(flat)-1, min(100, len(flat)), dtype=int)
            key_parts.append(str(flat[indices].tolist()))
        elif isinstance(source, dict):

            for k, v in sorted(source.items()):
                key_parts.append(f"{k}:{self._generate_cache_key(v, '', None)}")
            

        if layers is not None:
            key_parts.append(str(layers))
            

        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
            

        key_string = "_".join(str(part) for part in key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def visualize(self, 
                  source: Union[np.ndarray, Dict[str, np.ndarray]],
                  method: str = "heatmap",
                  layers: Optional[List[str]] = None,
                  save: bool = True,
                  alg: str = 'gradcam',
                  **kwargs) -> List[VisualizationResult]:

























        try:

            self._validate_input(source)
            self._validate_method(method)
            

            if not layers:
                raise ValueError("layers parameter must be provided")
            

            all_results = []
            total_layers = len(layers)
            
            for idx, layer in enumerate(layers):

                if total_layers > 1:
                    print(f"Processing layer {layer} ({idx+1}/{total_layers})...")
                

                cache_key = self._generate_cache_key(source, method, [layer], alg, **kwargs)
                

                if cache_key in self.cache:
                    print(f"Using cached result for {method} visualization of layer {layer}")
                    cached_result = self.cache[cache_key]
                    

                    if save and cached_result:

                        layer_idx = layer.split('.')[-1]
                        saved_files = self._save_with_layer_info(cached_result, layer_idx)
                        print(f"Saved cached visualization results for layer {layer}: {len(saved_files)} files")
                    
                    all_results.append(cached_result)
                    continue
                

                try:
                    visualizer = self._get_visualizer(method)
                except Exception as e:

                    raise RuntimeError(
                        f"可视化方法 '{method}' 未就绪或依赖缺失。请确认方法实现与依赖安装，原始错误: {e}"
                    )
                

                try:

                    if method == 'heatmap':
                        result = visualizer.visualize(source, layers=[layer], alg=alg, **kwargs)
                    else:
                        result = visualizer.visualize(source, layers=[layer], **kwargs)
                    

                    if isinstance(result, list) and len(result) == 1:
                        result = result[0]
                    

                    if result and hasattr(result, 'metadata'):
                        result.metadata['layer'] = layer
                        layer_idx = layer.split('.')[-1]
                        result.metadata['layer_idx'] = int(layer_idx)
                    

                    self.cache[cache_key] = result
                    

                    if save and result:
                        layer_idx = layer.split('.')[-1]
                        saved_files = self._save_with_layer_info(result, layer_idx)
                        print(f"Saved visualization results for layer {layer}: {len(saved_files)} files")
                    
                    all_results.append(result)
                
                except Exception as e:
                    raise RuntimeError(f"Visualization failed for layer {layer}: {e}")
            
            return all_results
        except Exception as e:

            print(f"Error in visualization: {e}")
            

            try:
                if hasattr(self, 'hook_manager'):
                    self.hook_manager.remove_all_hooks()
            except:
                pass
                
            raise
    
    
    def _placeholder_visualization(self,
                                 images: Union[np.ndarray, Dict[str, np.ndarray]],
                                 method: str,
                                 layers: Optional[Union[int, List[int]]],
                                 **kwargs) -> VisualizationResult:












        print(f"Creating placeholder visualization for method: {method}")
        print(f"Target layers: {layers}")
        if kwargs:
            print(f"Additional args: {kwargs}")
        
        if isinstance(images, dict):

            placeholder_heatmap = {}
            placeholder_overlay = {}
            
            for modal, img in images.items():

                h, w = img.shape[:2]
                gradient = np.linspace(0, 255, h*w, dtype=np.uint8).reshape(h, w)
                

                heatmap = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                

                if len(img.shape) == 2:
                    img_color = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 1:
                    img_color = cv2.cvtColor(img[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
                else:
                    img_color = img[:, :, :3].astype(np.uint8) if img.shape[2] > 3 else img.astype(np.uint8)
                    if img_color.shape[2] < 3:
                        img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2RGB)
                

                img_color = img_color.astype(np.uint8)
                heatmap = heatmap.astype(np.uint8)
                    
                overlay = cv2.addWeighted(img_color, 0.7, heatmap, 0.3, 0)
                
                placeholder_heatmap[modal] = heatmap
                placeholder_overlay[modal] = overlay
                
            return HeatmapResult(
                original_image=images,
                heatmap=placeholder_heatmap,
                overlay=placeholder_overlay,
                metadata={
                    'method': method,
                    'layers': layers,
                    'note': 'This is a placeholder visualization'
                }
            )
        else:

            h, w = images.shape[:2]
            gradient = np.linspace(0, 255, h*w, dtype=np.uint8).reshape(h, w)
            

            heatmap = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            

            if len(images.shape) == 2:
                img_color = cv2.cvtColor(images, cv2.COLOR_GRAY2RGB)
            else:
                img_color = images[:, :, :3] if images.shape[2] > 3 else images
            

            if img_color.dtype != heatmap.dtype:
                img_color = img_color.astype(np.uint8)
                heatmap = heatmap.astype(np.uint8)
                
            overlay = cv2.addWeighted(img_color, 0.7, heatmap, 0.3, 0)
            
            return HeatmapResult(
                original_image=images,
                heatmap=heatmap,
                overlay=overlay,
                metadata={
                    'method': method,
                    'layers': layers,
                    'note': 'This is a placeholder visualization'
                }
            )
    
    def _save_with_layer_info(self, result: VisualizationResult, layer_idx: str) -> List[str]:










        saved_files = []
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        

        base_name = f"{result.type}_layer{layer_idx}"

        def _to_uint8(img: np.ndarray) -> np.ndarray:
            if img.dtype == np.uint8:
                return img
            x = img
            try:
                if x.max() <= 1.0:
                    x = (x * 255.0).clip(0, 255)
                else:
                    x = x.clip(0, 255)
            except Exception:
                x = np.nan_to_num(x).clip(0, 255)
            return x.astype(np.uint8)

        def _imwrite_rgb(path: Path, img: np.ndarray) -> None:
            arr = _to_uint8(img)
            if arr.ndim == 3 and arr.shape[2] == 3:
                cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(path), arr)
        

        if isinstance(result.data, dict):

            for key, value in result.data.items():
                if isinstance(value, np.ndarray):
                    filename = f"{base_name}_{key}.png"
                    filepath = self.output_dir / filename
                    _imwrite_rgb(filepath, value)
                    saved_files.append(str(filepath))
                    
        elif isinstance(result.data, list):

            for idx, value in enumerate(result.data):
                if isinstance(value, np.ndarray):
                    filename = f"{base_name}_{idx:03d}.png"
                    filepath = self.output_dir / filename
                    _imwrite_rgb(filepath, value)
                    saved_files.append(str(filepath))
                    
        elif isinstance(result.data, np.ndarray):

            filename = f"{base_name}.png"
            filepath = self.output_dir / filename
            _imwrite_rgb(filepath, result.data)
            saved_files.append(str(filepath))
            
        return saved_files
    
    def clear_cache(self):

        self._visualizers.clear()
        self.cache.clear()
        print("Cleared all cached visualizers and results")
    
    def get_supported_methods(self) -> List[str]:






        return self.SUPPORTED_METHODS.copy()
    
    def save_config(self, config_file: Optional[str] = None):






        if config_file is None:
            config_file = self.output_dir / "config.json"
        else:
            config_file = Path(config_file)
            
        config = {
            'model_name': self.model.__class__.__name__,
            'output_dir': str(self.output_dir),
            'supported_methods': self.SUPPORTED_METHODS,
            'cached_visualizers': list(self._visualizers.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"Saved configuration to: {config_file}")
    
    def __enter__(self):

        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):

        try:

            if hasattr(self, 'hook_manager'):
                self.hook_manager.remove_all_hooks()
            

            self.clear_cache()
            

            self.save_config()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            
        return False
    
    def __del__(self):

        try:

            if hasattr(self, 'hook_manager'):

                pass
            

            if hasattr(self, '_visualizers'):
                self._visualizers.clear()
                
        except Exception:

            pass

