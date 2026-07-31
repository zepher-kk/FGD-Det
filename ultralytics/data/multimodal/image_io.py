from __future__ import annotations
# Ultralytics YOLO, AGPL-3.0 license

"""
多模态图像I/O复用层

本模块提供多模态图像加载、路径查找、对齐等功能的Mixin类，
可被不同任务的数据集类（如检测、分类、分割等）共享复用。

核心功能:
- X模态图像路径查找（同名不同扩展枚举）
- X模态图像加载（支持 npy/npz/tif/标准图像）
- RGB+X 图像空间对齐
- 通道数校验与转换
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from ultralytics.utils import LOGGER

class MultiModalImageIOMixin:



















    SUPPORTED_EXTENSIONS = ['.npy', '.npz', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']

    def find_corresponding_x_image(
        self,
        rgb_path: Union[str, Path],
        x_modality_dir: str,
        x_modality_suffix: Optional[str] = None
    ) -> str:























        rgb_path = Path(rgb_path)



        dataset_root = rgb_path.parent.parent.parent
        split_dir = rgb_path.parent.name
        x_dir = dataset_root / x_modality_dir / split_dir


        if x_modality_suffix:
            x_filename = rgb_path.stem + x_modality_suffix + rgb_path.suffix
        else:
            x_filename = rgb_path.name

        x_path = x_dir / x_filename


        if not x_path.exists():
            for ext in self.SUPPORTED_EXTENSIONS:
                test_path = x_dir / (rgb_path.stem + ext)
                if test_path.exists():
                    return str(test_path)

        return str(x_path)

    def load_x_modality(self, x_path: Union[str, Path]) -> np.ndarray:



















        x_path = Path(x_path)

        if not x_path.exists():
            raise FileNotFoundError(f"X模态图像不存在: {x_path}")

        suffix = x_path.suffix.lower()

        if suffix == '.npy':

            x_img = np.load(x_path)

        elif suffix == '.npz':

            with np.load(x_path) as npz_file:
                preferred_keys = ('image', 'arr_0', 'array', 'data')
                selected_key = next((k for k in preferred_keys if k in npz_file.files), None)

                if selected_key is None:
                    if len(npz_file.files) == 1:
                        selected_key = npz_file.files[0]
                    else:
                        raise ValueError(
                            f"npz文件 {x_path} 含多个数组 {npz_file.files}，"
                            f"无法确定默认键，请使用标准键(image/arr_0)。"
                        )
                x_img = npz_file[selected_key]

        elif suffix in ['.tiff', '.tif']:

            x_img = cv2.imread(str(x_path), cv2.IMREAD_UNCHANGED)

        else:

            x_img = cv2.imread(str(x_path))

        if x_img is None:
            raise ValueError(f"无法读取X模态图像: {x_path}")

        return x_img

    def align_x_to_rgb(
        self,
        x_img: np.ndarray,
        rgb_shape: Tuple[int, int]
    ) -> np.ndarray:










        target_h, target_w = rgb_shape
        x_h, x_w = x_img.shape[:2]

        if (x_h, x_w) != (target_h, target_w):
            x_img = cv2.resize(x_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return x_img

    def validate_x_channels(
        self,
        x_img: np.ndarray,
        expected_xch: int,
        strict: bool = False,
        x_path_hint: Optional[str] = None
    ) -> np.ndarray:





















        if len(x_img.shape) == 2:
            actual_xch = 1
        elif len(x_img.shape) == 3:
            actual_xch = x_img.shape[2]
        else:
            raise ValueError(f"X模态图像维度异常: {x_img.shape}")


        if strict and actual_xch != expected_xch:
            raise ValueError(
                f"X通道不一致: 期望={expected_xch}, 实际={actual_xch}。"
                f" 文件: {x_path_hint or 'unknown'}"
            )


        if len(x_img.shape) == 2:

            if expected_xch == 1:
                x_img = x_img[:, :, np.newaxis]
            else:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)

        elif x_img.shape[2] == 1:

            if expected_xch != 1:
                x_img = np.repeat(x_img, 3, axis=2)

        elif x_img.shape[2] == 4:

            x_img = x_img[:, :, :3]
            if expected_xch == 1:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        elif x_img.shape[2] == 3:

            if expected_xch == 1:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]


        final_ch = x_img.shape[2] if len(x_img.shape) == 3 else 1
        if final_ch != expected_xch:
            LOGGER.warning(
                f"X模态通道数({final_ch})与期望({expected_xch})不匹配，"
                f"文件: {x_path_hint or 'unknown'}"
            )

        return x_img

    def load_and_align_x_image(
        self,
        rgb_path: Union[str, Path],
        rgb_img: np.ndarray,
        x_modality_dir: str,
        expected_xch: int,
        x_modality_suffix: Optional[str] = None,
        strict: bool = False
    ) -> np.ndarray:

















        x_path = self.find_corresponding_x_image(rgb_path, x_modality_dir, x_modality_suffix)


        x_img = self.load_x_modality(x_path)


        x_img = self.align_x_to_rgb(x_img, rgb_img.shape[:2])


        x_img = self.validate_x_channels(x_img, expected_xch, strict, x_path)

        return x_img

    def concatenate_multimodal(
        self,
        rgb_img: np.ndarray,
        x_img: np.ndarray
    ) -> np.ndarray:










        return np.concatenate([rgb_img, x_img], axis=2)

