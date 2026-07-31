from __future__ import annotations
"""
Feature map visualization module for YOLOMM multi-modal detection.

This module provides the FeatureMapVisualizer class for extracting and visualizing
intermediate feature maps from model layers during forward pass.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import cv2

from .manager import FeatureMapResult
from .utils import HookManager, load_image

class FeatureMapVisualizer:











    
    def __init__(self, model: torch.nn.Module):






        self.model = model
        self.hook_manager = HookManager(model)
    
    def visualize(
        self,
        images: Union[str, np.ndarray, torch.Tensor, Dict[str, Any]],
        layers: Optional[List[Union[str, int]]] = None,
        top_k: int = 8,
        selection_method: str = 'sum',
        **kwargs
    ) -> FeatureMapResult:














        if isinstance(images, dict):
            results = {}
            for modality, img in images.items():
                results[modality] = self._visualize_single_modality(
                    img, layers, top_k, selection_method, **kwargs
                )

            return self._combine_multimodal_results(results)
        else:

            return self._visualize_single_modality(
                images, layers, top_k, selection_method, **kwargs
            )
    
    def _visualize_single_modality(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        layers: Optional[List[Union[str, int]]],
        top_k: int,
        selection_method: str,
        **kwargs
    ) -> FeatureMapResult:














        img_tensor = self._prepare_input(image)
        

        if layers is None:
            layers = self._get_default_layers()
        

        feature_maps = self._extract_features(img_tensor, layers)
        

        selected_features = self._select_channels(
            feature_maps, top_k, selection_method
        )
        

        grid_image = self._render_grid(selected_features, **kwargs)
        

        metadata = {
            'layers': layers,
            'top_k': top_k,
            'selection_method': selection_method,
            'num_features': len(selected_features),
            'feature_stats': self._compute_feature_stats(selected_features)
        }
        

        if layers is None:
            layer_idx_val = 'auto'
        else:
            layer_idx_val = 'multi' if len(layers) > 1 else (layers[0] if len(layers) == 1 else 'auto')

        return FeatureMapResult(
            layer_idx=layer_idx_val,
            feature_maps=grid_image,
            metadata=metadata
        )
    
    def _prepare_input(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:










        if isinstance(image, str):
            image = load_image(image)
        

        if isinstance(image, np.ndarray):

            if image.ndim == 3:
                image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()
        

        if image.ndim == 3:
            image = image.unsqueeze(0)
        

        device = next(self.model.parameters()).device
        image = image.to(device)
        
        return image
    
    def _get_default_layers(self) -> List[str]:







        default_layers = []
        

        if hasattr(self.model, 'model'):
            model = self.model.model

            for i, layer in enumerate(model):
                layer_name = layer.__class__.__name__
                if any(name in layer_name for name in ['Conv', 'C2f', 'C3', 'SPPF']):
                    if i in [4, 6, 8, 10]:
                        default_layers.append(f'model.{i}')
        

        if not default_layers:
            default_layers = ['model.4', 'model.6', 'model.8', 'model.10']
        
        return default_layers
    
    def _extract_features(
        self,
        input_tensor: torch.Tensor,
        layers: List[Union[str, int]]
    ) -> Dict[str, torch.Tensor]:











        original_device = next(self.model.parameters()).device
        

        for layer in layers:
            if isinstance(layer, int):

                modules = list(self.model.modules())
                if 0 <= layer < len(modules):
                    self.hook_manager.register_forward_hook(
                        modules[layer], name=f'layer_{layer}'
                    )
                else:

                    raise ValueError(
                        f"Layer index {layer} is out of range.\n"
                        f"Model has {len(modules)} modules (indices 0 to {len(modules)-1})."
                    )
            else:

                try:
                    module = self._get_module_by_name(layer)
                    if module is not None:
                        self.hook_manager.register_forward_hook(module, name=layer)
                    else:
                        raise ValueError(f"Could not find module: {layer}")
                except ValueError as ve:

                    available_layers = self._get_default_layers()
                    raise ValueError(
                        f"Failed to register hook for layer '{layer}':\n{str(ve)}\n\n"
                        f"Suggested layers to try:\n" + 
                        "\n".join([f"  - {l}" for l in available_layers[:5]])
                    )
        
        try:

            input_tensor = input_tensor.to(original_device)
            with torch.no_grad():
                _ = self.model(input_tensor)
            features = self.hook_manager.get_features()
        except RuntimeError as e:

            if 'out of memory' in str(e).lower():
                raise RuntimeError(
                    "GPU 显存不足（OOM）。请尝试以下方案后重试：\n"
                    "- 使用更小的输入图（或更小批量）\n"
                    "- 减小 top_k 或选择更少的层\n"
                    "- 显式将模型与数据迁移到 CPU：model.to('cpu') 并重试"
                ) from e
            raise
        finally:

            self.hook_manager.remove_all_hooks()
        
        return features
    
    def _get_module_by_name(self, name: str) -> Optional[torch.nn.Module]:









        parts = name.split('.')
        module = self.model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit() and hasattr(module, '__getitem__'):
                module = module[int(part)]
            else:
                return None
        
        return module
    
    def _select_channels(
        self,
        feature_maps: Dict[str, torch.Tensor],
        top_k: int,
        method: str
    ) -> List[Tuple[str, int, torch.Tensor]]:












        selected = []
        
        for layer_name, features in feature_maps.items():

            if features.ndim == 4:
                batch_size = features.shape[0]
                num_channels = features.shape[1]
                

                for batch_idx in range(batch_size):
                    batch_features = features[batch_idx]
                    

                    if method == 'sum':

                        channel_scores = batch_features.abs().sum(dim=(1, 2))
                    elif method == 'var':

                        channel_scores = batch_features.var(dim=(1, 2))
                    else:
                        raise ValueError(f"Unknown selection method: {method}")
                    

                    k = min(top_k, num_channels)
                    top_indices = torch.topk(channel_scores, k).indices
                    


                    for idx in top_indices:
                        selected.append((
                            layer_name,
                            idx.item(),
                            batch_features[idx].cpu().numpy(),
                            batch_idx
                        ))
            elif features.ndim == 3:

                num_channels = features.shape[0]
                

                if method == 'sum':
                    channel_scores = features.abs().sum(dim=(1, 2))
                elif method == 'var':
                    channel_scores = features.var(dim=(1, 2))
                else:
                    raise ValueError(f"Unknown selection method: {method}")
                

                k = min(top_k, num_channels)
                top_indices = torch.topk(channel_scores, k).indices
                

                for idx in top_indices:
                    selected.append((
                        layer_name,
                        idx.item(),
                        features[idx].cpu().numpy(),
                        0
                    ))
        
        return selected
    
    def _render_grid(
        self,
        selected_features: List[Tuple[str, int, np.ndarray]],
        grid_size: Optional[Tuple[int, int]] = None,
        feature_size: Tuple[int, int] = (128, 128),
        show_stats: bool = True,

        colormap: str = 'gray',
        **kwargs
    ) -> np.ndarray:














        if not selected_features:

            return np.zeros((256, 256, 3), dtype=np.uint8)
        
        num_features = len(selected_features)
        

        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_features)))
            rows = int(np.ceil(num_features / cols))
        else:
            rows, cols = grid_size
        

        cell_h, cell_w = feature_size
        padding = 5
        text_height = 30 if show_stats else 20
        
        grid_h = rows * (cell_h + text_height + padding) + padding
        grid_w = cols * (cell_w + padding) + padding
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
        

        cmap_name = str(colormap).lower() if isinstance(colormap, str) else 'viridis'
        if cmap_name in {'none', 'gray', 'grey', 'grayscale'}:
            cmap = None
        else:
            cmap = getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_VIRIDIS)
        

        for idx, item in enumerate(selected_features):

            if len(item) == 4:
                layer_name, channel_idx, feature_map, batch_idx = item
            else:
                layer_name, channel_idx, feature_map = item
                batch_idx = 0
                
            row = idx // cols
            col = idx % cols
            

            y = row * (cell_h + text_height + padding) + padding
            x = col * (cell_w + padding) + padding
            

            feat_norm = self._normalize_feature(feature_map)
            

            feat_resized = cv2.resize(feat_norm, (cell_w, cell_h))
            

            if cmap is None:

                feat_colored = cv2.cvtColor(feat_resized, cv2.COLOR_GRAY2RGB)
            else:
                feat_colored = cv2.applyColorMap(feat_resized, cmap)
                feat_colored = cv2.cvtColor(feat_colored, cv2.COLOR_BGR2RGB)
            

            grid[y:y+cell_h, x:x+cell_w] = feat_colored
            

            text_y = y + cell_h + 15
            

            label = f"{layer_name.split('.')[-1]} ch:{channel_idx}"
            cv2.putText(grid, label, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            

            if show_stats:
                stats_text = f"μ:{feature_map.mean():.1f} σ:{feature_map.std():.1f}"
                cv2.putText(grid, stats_text, (x, text_y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        return grid
    
    def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:










        if feature.size == 0:
            return np.zeros_like(feature, dtype=np.uint8)
        

        f_min, f_max = feature.min(), feature.max()
        if f_max > f_min:
            feature_norm = (feature - f_min) / (f_max - f_min)
        else:
            feature_norm = np.zeros_like(feature)
        

        feature_uint8 = (feature_norm * 255).astype(np.uint8)
        
        return feature_uint8
    
    def _compute_feature_stats(
        self,
        selected_features: List[Tuple[str, int, np.ndarray]]
    ) -> Dict[str, Any]:









        stats = {
            'num_features': len(selected_features),
            'layers': {},
        }
        

        for item in selected_features:

            if len(item) == 4:
                layer_name, channel_idx, feature_map, batch_idx = item
            else:
                layer_name, channel_idx, feature_map = item
                
            if layer_name not in stats['layers']:
                stats['layers'][layer_name] = {
                    'channels': [],
                    'mean_activation': 0,
                    'max_activation': -float('inf'),
                    'min_activation': float('inf'),
                }
            
            layer_stats = stats['layers'][layer_name]
            layer_stats['channels'].append(channel_idx)
            layer_stats['mean_activation'] += feature_map.mean()
            layer_stats['max_activation'] = max(
                layer_stats['max_activation'], feature_map.max()
            )
            layer_stats['min_activation'] = min(
                layer_stats['min_activation'], feature_map.min()
            )
        

        for layer_stats in stats['layers'].values():
            num_channels = len(layer_stats['channels'])
            if num_channels > 0:
                layer_stats['mean_activation'] /= num_channels
        
        return stats
    
    def _combine_multimodal_results(
        self,
        results: Dict[str, FeatureMapResult]
    ) -> FeatureMapResult:










        grids = []
        combined_metadata = {
            'modalities': {},
            'combined': True
        }
        
        for modality, result in results.items():
            grids.append(result.feature_maps)
            combined_metadata['modalities'][modality] = result.metadata
        

        combined_grid = self._stack_grids_with_labels(grids, list(results.keys()))
        
        return FeatureMapResult(
            layer_idx='multi',
            feature_maps=combined_grid,
            metadata=combined_metadata
        )
    
    def _stack_grids_with_labels(
        self,
        grids: List[np.ndarray],
        labels: List[str]
    ) -> np.ndarray:










        if not grids:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        

        max_width = max(g.shape[1] for g in grids)
        label_height = 30
        

        labeled_grids = []
        for grid, label in zip(grids, labels):
            h, w = grid.shape[:2]
            

            labeled = np.ones((h + label_height, max_width, 3), dtype=np.uint8) * 255
            

            x_offset = (max_width - w) // 2
            labeled[label_height:, x_offset:x_offset+w] = grid
            

            cv2.putText(labeled, f"Modality: {label}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            labeled_grids.append(labeled)
        

        combined = np.vstack(labeled_grids)
        
        return combined

