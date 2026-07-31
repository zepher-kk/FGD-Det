from __future__ import annotations
# Ultralytics YOLO 🚀, AGPL-3.0 license

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from ultralytics.utils import LOGGER
from .self_modal_generator import SelfModalGenerator

class ModalityFiller:






    

    DEFAULT_STRATEGY_WEIGHTS = {
        'copy': 0.3,
        'noise': 0.25,
        'channel_repeat': 0.2,
        'edge_blur': 0.15,
        'mixed': 0.1
    }
    
    def __init__(self, strategy_weights: Optional[Dict[str, float]] = None, 
                 noise_std: float = 0.1, blur_kernel_size: int = 5):








        self.strategy_weights = strategy_weights or self.DEFAULT_STRATEGY_WEIGHTS
        self.noise_std = noise_std
        self.blur_kernel_size = blur_kernel_size
        

        if abs(sum(self.strategy_weights.values()) - 1.0) > 1e-6:
            LOGGER.warning(f"策略权重总和不为1.0: {sum(self.strategy_weights.values())}")
    
    def generate_filling(self, source_tensor: torch.Tensor, 
                        source_modality: str, 
                        target_modality: str,
                        strategy: Optional[str] = None) -> torch.Tensor:












        if strategy is None:
            strategy = self._select_random_strategy()
        
        if strategy == 'copy':
            return self._create_copy_fill(source_tensor)
        elif strategy == 'noise':
            return self._create_noise_fill(source_tensor)
        elif strategy == 'channel_repeat':
            return self._create_channel_repeat_fill(source_tensor)
        elif strategy == 'edge_blur':
            return self._create_edge_blur_fill(source_tensor)
        elif strategy == 'mixed':
            return self._create_mixed_fill(source_tensor)
        else:
            LOGGER.warning(f"未知填充策略: {strategy}, 使用复制策略")
            return self._create_copy_fill(source_tensor)
    
    def _select_random_strategy(self) -> str:

        strategies = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())
        return random.choices(strategies, weights=weights)[0]
    
    def _create_copy_fill(self, tensor: torch.Tensor) -> torch.Tensor:









        return tensor.clone()
    
    def _create_noise_fill(self, tensor: torch.Tensor) -> torch.Tensor:









        noise = torch.randn_like(tensor) * self.noise_std
        noisy_tensor = tensor + noise

        return torch.clamp(noisy_tensor, 0.0, 1.0)
    
    def _create_channel_repeat_fill(self, tensor: torch.Tensor) -> torch.Tensor:










        if tensor.shape[1] == 3:
            grayscale = tensor.mean(dim=1, keepdim=True)
            repeated = grayscale.repeat(1, 3, 1, 1)
            return repeated
        else:

            return tensor.repeat(1, 3, 1, 1)
    
    def _create_edge_blur_fill(self, tensor: torch.Tensor) -> torch.Tensor:










        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=tensor.dtype, device=tensor.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=tensor.dtype, device=tensor.device).unsqueeze(0).unsqueeze(0)
        

        edges = []
        for c in range(tensor.shape[1]):
            channel = tensor[:, c:c+1, :, :]
            edge_x = F.conv2d(channel, sobel_x, padding=1)
            edge_y = F.conv2d(channel, sobel_y, padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            edges.append(edge_magnitude)
        
        edge_tensor = torch.cat(edges, dim=1)
        

        return self._apply_gaussian_blur(edge_tensor)
    
    def _create_mixed_fill(self, tensor: torch.Tensor) -> torch.Tensor:










        available_strategies = ['copy', 'noise', 'channel_repeat', 'edge_blur']
        selected_strategies = random.sample(available_strategies, 
                                          random.randint(2, min(3, len(available_strategies))))
        
        results = []
        for strategy in selected_strategies:
            if strategy == 'copy':
                results.append(self._create_copy_fill(tensor))
            elif strategy == 'noise':
                results.append(self._create_noise_fill(tensor))
            elif strategy == 'channel_repeat':
                results.append(self._create_channel_repeat_fill(tensor))
            elif strategy == 'edge_blur':
                results.append(self._create_edge_blur_fill(tensor))
        

        weights = torch.softmax(torch.rand(len(results)), dim=0)
        mixed_result = torch.zeros_like(tensor)
        for i, result in enumerate(results):
            mixed_result += weights[i] * result
        
        return mixed_result
    
    def _apply_gaussian_blur(self, tensor: torch.Tensor) -> torch.Tensor:










        kernel_size = self.blur_kernel_size
        sigma = kernel_size / 3.0
        

        x = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device) - kernel_size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        

        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
        

        blurred_channels = []
        for c in range(tensor.shape[1]):
            channel = tensor[:, c:c+1, :, :]
            blurred = F.conv2d(channel, gaussian_2d, padding=kernel_size//2)
            blurred_channels.append(blurred)
        
        return torch.cat(blurred_channels, dim=1)
    
    def get_statistics(self, tensor: torch.Tensor) -> Dict[str, float]:









        return {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'max': tensor.max().item(),
            'min': tensor.min().item(),
            'shape': list(tensor.shape)
        }


default_modality_filler = ModalityFiller()
default_self_modal_generator = SelfModalGenerator()

def generate_modality_filling(source_tensor: torch.Tensor,
                            source_modality: str,
                            target_modality: str,
                            strategy: Optional[str] = None,
                            filler: Optional[ModalityFiller] = None) -> torch.Tensor:













    if filler is None:
        filler = default_modality_filler

    return filler.generate_filling(source_tensor, source_modality, target_modality, strategy)

def generate_self_modality(rgb_tensor: torch.Tensor, modal_type: str = 'edge',
                          algorithm: str = 'auto',
                          generator: Optional[SelfModalGenerator] = None) -> torch.Tensor:












    if generator is None:
        generator = default_self_modal_generator

    return generator.generate_self_modality(rgb_tensor, modal_type, algorithm)

