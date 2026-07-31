from __future__ import annotations
# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Union
from ultralytics.utils import LOGGER

class SelfModalGenerator:






    

    ALGORITHM_CONFIGS = {
        'edge': {
            'sobel_weight': 0.8,
            'canny_threshold': [100, 200],
            'blur_kernel_size': 5,
            'edge_enhancement': 1.2,
            'normalize_output': True
        },
        'texture': {
            'lbp_radius': 3,
            'lbp_points': 24,
            'gabor_frequency': 0.6,
            'gabor_orientations': 8,
            'texture_contrast': 1.5
        },
        'gradient': {
            'magnitude_weight': 0.7,
            'direction_weight': 0.3,
            'gradient_threshold': 0.1,
            'smooth_factor': 0.8
        }
    }
    
    def __init__(self, device: Optional[torch.device] = None, cache_enabled: bool = True):







        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_enabled = cache_enabled
        self._cache = {} if cache_enabled else None
        

        self._sobel_kernels = self._create_sobel_kernels()
        self._gabor_kernels = self._create_gabor_kernels()
        

    
    def generate_self_modality(self, rgb_tensor: torch.Tensor, 
                              modal_type: str = 'edge',
                              algorithm: str = 'auto') -> torch.Tensor:












        if not isinstance(rgb_tensor, torch.Tensor):
            raise TypeError(f"输入必须是torch.Tensor，实际类型: {type(rgb_tensor)}")
        
        if rgb_tensor.dim() != 4 or rgb_tensor.shape[1] != 3:
            raise ValueError(f"输入形状必须是[B, 3, H, W]，实际形状: {rgb_tensor.shape}")
        

        cache_key = self._get_cache_key(rgb_tensor, modal_type, algorithm)
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        

        rgb_tensor = rgb_tensor.to(self.device)
        

        if modal_type == 'edge':
            result = self._generate_edge_modality(rgb_tensor, algorithm)
        elif modal_type == 'texture':
            result = self._generate_texture_modality(rgb_tensor, algorithm)
        elif modal_type == 'gradient':
            result = self._generate_gradient_modality(rgb_tensor, algorithm)
        else:
            raise ValueError(f"不支持的模态类型: {modal_type}，支持的类型: ['edge', 'texture', 'gradient']")
        

        if self.cache_enabled:
            self._cache[cache_key] = result.clone()
        
        return result
    
    def _generate_edge_modality(self, rgb_tensor: torch.Tensor, algorithm: str = 'auto') -> torch.Tensor:










        config = self.ALGORITHM_CONFIGS['edge']
        
        if algorithm == 'auto' or algorithm == 'sobel':

            edge_tensor = self._apply_sobel_edge_detection(rgb_tensor)
        elif algorithm == 'canny':

            edge_tensor = self._apply_canny_edge_detection(rgb_tensor, config['canny_threshold'])
        elif algorithm == 'scharr':

            edge_tensor = self._apply_scharr_edge_detection(rgb_tensor)
        else:
            LOGGER.warning(f"未知边缘检测算法: {algorithm}，使用Sobel算法")
            edge_tensor = self._apply_sobel_edge_detection(rgb_tensor)
        

        if config['edge_enhancement'] != 1.0:
            edge_tensor = edge_tensor * config['edge_enhancement']
        

        if config['normalize_output']:
            edge_tensor = self._normalize_tensor(edge_tensor)
        

        if config['blur_kernel_size'] > 1:
            edge_tensor = self._apply_gaussian_blur(edge_tensor, config['blur_kernel_size'])
        
        return edge_tensor
    
    def _generate_texture_modality(self, rgb_tensor: torch.Tensor, algorithm: str = 'auto') -> torch.Tensor:










        config = self.ALGORITHM_CONFIGS['texture']
        
        if algorithm == 'auto' or algorithm == 'lbp':

            texture_tensor = self._apply_lbp_texture(rgb_tensor, config['lbp_radius'], config['lbp_points'])
        elif algorithm == 'gabor':

            texture_tensor = self._apply_gabor_texture(rgb_tensor, config['gabor_frequency'], config['gabor_orientations'])
        elif algorithm == 'variance':

            texture_tensor = self._apply_variance_texture(rgb_tensor)
        else:
            LOGGER.warning(f"未知纹理算法: {algorithm}，使用LBP算法")
            texture_tensor = self._apply_lbp_texture(rgb_tensor, config['lbp_radius'], config['lbp_points'])
        

        if config['texture_contrast'] != 1.0:
            texture_tensor = texture_tensor * config['texture_contrast']
        

        texture_tensor = self._normalize_tensor(texture_tensor)
        
        return texture_tensor
    
    def _generate_gradient_modality(self, rgb_tensor: torch.Tensor, algorithm: str = 'auto') -> torch.Tensor:










        config = self.ALGORITHM_CONFIGS['gradient']
        

        grad_x, grad_y = self._compute_gradients(rgb_tensor)
        
        if algorithm == 'auto' or algorithm == 'combined':

            magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            direction = torch.atan2(grad_y, grad_x)
            

            direction = (direction + np.pi) / (2 * np.pi)
            

            gradient_tensor = (config['magnitude_weight'] * magnitude + 
                             config['direction_weight'] * direction)
        elif algorithm == 'magnitude':

            gradient_tensor = torch.sqrt(grad_x**2 + grad_y**2)
        elif algorithm == 'direction':

            direction = torch.atan2(grad_y, grad_x)
            gradient_tensor = (direction + np.pi) / (2 * np.pi)
        else:
            LOGGER.warning(f"未知梯度算法: {algorithm}，使用组合算法")
            magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            direction = torch.atan2(grad_y, grad_x)
            direction = (direction + np.pi) / (2 * np.pi)
            gradient_tensor = (config['magnitude_weight'] * magnitude + 
                             config['direction_weight'] * direction)
        

        if config['gradient_threshold'] > 0:
            gradient_tensor = torch.where(gradient_tensor > config['gradient_threshold'], 
                                        gradient_tensor, 
                                        torch.zeros_like(gradient_tensor))
        

        if config['smooth_factor'] < 1.0:
            gradient_tensor = self._apply_gaussian_blur(gradient_tensor, kernel_size=3)
            gradient_tensor = config['smooth_factor'] * gradient_tensor + (1 - config['smooth_factor']) * rgb_tensor
        

        gradient_tensor = self._normalize_tensor(gradient_tensor)
        
        return gradient_tensor

    def _apply_sobel_edge_detection(self, tensor: torch.Tensor) -> torch.Tensor:

        sobel_x, sobel_y = self._sobel_kernels

        edges = []
        for c in range(tensor.shape[1]):
            channel = tensor[:, c:c+1, :, :]
            edge_x = F.conv2d(channel, sobel_x, padding=1)
            edge_y = F.conv2d(channel, sobel_y, padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            edges.append(edge_magnitude)

        return torch.cat(edges, dim=1)

    def _apply_canny_edge_detection(self, tensor: torch.Tensor, thresholds: list) -> torch.Tensor:


        blurred = self._apply_gaussian_blur(tensor, kernel_size=5)


        grad_x, grad_y = self._compute_gradients(blurred)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)


        low_thresh, high_thresh = thresholds
        low_thresh = low_thresh / 255.0
        high_thresh = high_thresh / 255.0


        strong_edges = magnitude > high_thresh
        weak_edges = (magnitude > low_thresh) & (magnitude <= high_thresh)


        edges = strong_edges.float()
        edges = edges + 0.5 * weak_edges.float()


        edges = torch.clamp(edges, 0.0, 1.0)

        return edges

    def _apply_scharr_edge_detection(self, tensor: torch.Tensor) -> torch.Tensor:


        scharr_x = torch.tensor([[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]],
                               dtype=tensor.dtype, device=tensor.device).unsqueeze(0)
        scharr_y = torch.tensor([[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]],
                               dtype=tensor.dtype, device=tensor.device).unsqueeze(0)

        edges = []
        for c in range(tensor.shape[1]):
            channel = tensor[:, c:c+1, :, :]
            edge_x = F.conv2d(channel, scharr_x, padding=1)
            edge_y = F.conv2d(channel, scharr_y, padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            edges.append(edge_magnitude)

        return torch.cat(edges, dim=1)

    def _apply_lbp_texture(self, tensor: torch.Tensor, radius: int, points: int) -> torch.Tensor:


        gray = tensor.mean(dim=1, keepdim=True)


        kernel = torch.ones(1, 1, 3, 3, device=tensor.device) / 9.0
        kernel[0, 0, 1, 1] = 0


        neighbor_avg = F.conv2d(gray, kernel, padding=1)


        lbp = (gray > neighbor_avg).float()


        lbp_3ch = lbp.repeat(1, 3, 1, 1)

        return lbp_3ch

    def _apply_gabor_texture(self, tensor: torch.Tensor, frequency: float, orientations: int) -> torch.Tensor:


        gabor_responses = []


        for gabor_kernel in self._gabor_kernels[:orientations]:
            channel_responses = []
            for c in range(tensor.shape[1]):
                channel = tensor[:, c:c+1, :, :]
                response = F.conv2d(channel, gabor_kernel, padding=gabor_kernel.shape[-1]//2)
                channel_responses.append(response)


            combined_channel_response = torch.cat(channel_responses, dim=1)
            gabor_responses.append(combined_channel_response)


        if gabor_responses:
            combined_response = torch.stack(gabor_responses, dim=0).mean(dim=0)
        else:
            combined_response = tensor

        return combined_response

    def _apply_variance_texture(self, tensor: torch.Tensor, window_size: int = 5) -> torch.Tensor:


        kernel = torch.ones(1, 1, window_size, window_size, device=tensor.device) / (window_size**2)

        texture_channels = []
        for c in range(tensor.shape[1]):
            channel = tensor[:, c:c+1, :, :]


            local_mean = F.conv2d(channel, kernel, padding=window_size//2)


            local_var = F.conv2d(channel**2, kernel, padding=window_size//2) - local_mean**2

            texture_channels.append(local_var)

        return torch.cat(texture_channels, dim=1)

    def _compute_gradients(self, tensor: torch.Tensor) -> tuple:


        grad_x_kernel = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                                   dtype=tensor.dtype, device=tensor.device).unsqueeze(0)
        grad_y_kernel = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                                   dtype=tensor.dtype, device=tensor.device).unsqueeze(0)

        grad_x_channels = []
        grad_y_channels = []

        for c in range(tensor.shape[1]):
            channel = tensor[:, c:c+1, :, :]
            grad_x = F.conv2d(channel, grad_x_kernel, padding=1)
            grad_y = F.conv2d(channel, grad_y_kernel, padding=1)
            grad_x_channels.append(grad_x)
            grad_y_channels.append(grad_y)

        grad_x_tensor = torch.cat(grad_x_channels, dim=1)
        grad_y_tensor = torch.cat(grad_y_channels, dim=1)

        return grad_x_tensor, grad_y_tensor

    def _apply_gaussian_blur(self, tensor: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:

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

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:

        min_val = tensor.min()
        max_val = tensor.max()

        if max_val > min_val:
            normalized = (tensor - min_val) / (max_val - min_val)
        else:
            normalized = tensor


        return torch.clamp(normalized, 0.0, 1.0 - 1e-7)

    def _create_sobel_kernels(self) -> tuple:

        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                              dtype=torch.float32, device=self.device).unsqueeze(0)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                              dtype=torch.float32, device=self.device).unsqueeze(0)
        return sobel_x, sobel_y

    def _create_gabor_kernels(self, kernel_size: int = 15, num_orientations: int = 8) -> list:

        kernels = []
        frequency = self.ALGORITHM_CONFIGS['texture']['gabor_frequency']

        for i in range(num_orientations):
            theta = i * np.pi / num_orientations


            x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
            y = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
            X, Y = torch.meshgrid(x, y, indexing='ij')


            x_rot = X * np.cos(theta) + Y * np.sin(theta)
            y_rot = -X * np.sin(theta) + Y * np.cos(theta)


            sigma = kernel_size / 4.0
            gabor = torch.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2)) * torch.cos(2 * np.pi * frequency * x_rot)


            gabor = gabor / gabor.abs().sum()
            gabor = gabor.unsqueeze(0).unsqueeze(0)

            kernels.append(gabor)

        return kernels

    def _get_cache_key(self, tensor: torch.Tensor, modal_type: str, algorithm: str) -> str:


        shape_str = 'x'.join(map(str, tensor.shape))
        data_hash = hash(tensor.flatten()[:100].sum().item())
        return f"{modal_type}_{algorithm}_{shape_str}_{data_hash}"

    def clear_cache(self):

        if self.cache_enabled and self._cache:
            self._cache.clear()
            LOGGER.info("🔧 SelfModalGenerator: 缓存已清空")

    def get_cache_info(self) -> Dict[str, Any]:

        if not self.cache_enabled:
            return {'enabled': False}

        return {
            'enabled': True,
            'size': len(self._cache),
            'keys': list(self._cache.keys())
        }


default_self_modal_generator = SelfModalGenerator()

def generate_self_modality(rgb_tensor: torch.Tensor,
                          modal_type: str = 'edge',
                          algorithm: str = 'auto',
                          generator: Optional[SelfModalGenerator] = None) -> torch.Tensor:












    if generator is None:
        generator = default_self_modal_generator

    return generator.generate_self_modality(rgb_tensor, modal_type, algorithm)

