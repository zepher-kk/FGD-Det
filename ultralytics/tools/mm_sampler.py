from __future__ import annotations
"""
MultiModal Dataset Sampler - YOLOMM 多模态数据集图像采样工具

提供从多模态数据集 YAML 配置中随机采样匹配图像对的功能，
输出 RGB 和 X 模态图像路径供 YOLOMM.predict() 方法使用。

Usage:
    from mm_sampler import MultiModalSampler

    # 初始化采样器
    sampler = MultiModalSampler('path/to/dataset.yaml')

    # 随机采样一对图像
    rgb_path, x_path = sampler.sample_one()

    # 采样多对图像
    pairs = sampler.sample(n=5)  # [(rgb1, x1), (rgb2, x2), ...]

    # 采样多对图像（以两个列表形式输出，按图片对顺序对齐）
    rgb_list, x_list = sampler.sample_source_list(n=5)  # (['rgb1', 'rgb2', ...], ['x1', 'x2', ...])

    # 直接供 predict 方法使用
    from ultralytics import YOLOMM
    model = YOLOMM('yolo11n-mm.yaml')
    model.predict(source=[rgb_path, x_path])

    # 或使用便捷方法
    sources = sampler.get_predict_sources(n=3)
    for source in sources:
        model.predict(source=source)

Author: YOLOMM Team
"""

import random
import yaml
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Any

class MultiModalSampler:



    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(
        self,
        yaml_path: Union[str, Path],
        split: str = 'val',
        seed: Optional[int] = None
    ):








        self.yaml_path = Path(yaml_path)
        self.split = split

        if seed is not None:
            random.seed(seed)


        self._parse_yaml()


        self._scan_image_pairs()

    def _parse_yaml(self) -> None:

        if not self.yaml_path.exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {self.yaml_path}")

        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)


        self.root_path = Path(self.config.get('path', ''))
        if not self.root_path.is_absolute():

            self.root_path = self.yaml_path.parent / self.root_path


        split_path = self.config.get(self.split)
        if split_path is None:
            available_splits = [k for k in ['train', 'val', 'test'] if k in self.config]
            raise ValueError(
                f"数据集中未找到 '{self.split}' 分割。"
                f"可用分割: {available_splits}"
            )

        self.split_path = Path(split_path)
        if not self.split_path.is_absolute():
            self.split_path = self.root_path / self.split_path


        self.modality_config = self.config.get('modality') or self.config.get('modalities', {})


        self.modality_used = self.config.get('modality_used', ['rgb'])
        if isinstance(self.modality_used, str):
            self.modality_used = [self.modality_used]


        self._resolve_modalities()


        self.x_channels = self.config.get('Xch', 3)


        self.names = self.config.get('names', {})

    def _resolve_modalities(self) -> None:

        if len(self.modality_used) < 2:
            raise ValueError(
                f"多模态采样需要至少两个模态，但配置中只有: {self.modality_used}"
            )


        rgb_key = self.modality_used[0].lower()
        x_key = self.modality_used[1].lower()


        self.rgb_modality_name = rgb_key
        self.x_modality_name = x_key


        rgb_rel_path = self.modality_config.get(rgb_key, 'images')
        self.rgb_base_path = self.root_path / rgb_rel_path / self.split_path.name


        if not self.rgb_base_path.exists():
            self.rgb_base_path = self.split_path


        x_rel_path = self.modality_config.get(x_key)
        if x_rel_path is None:

            possible_x_paths = [
                f'images_{x_key}',
                f'{x_key}',
                f'images-{x_key}'
            ]
            for p in possible_x_paths:
                test_path = self.root_path / p / self.split_path.name
                if test_path.exists():
                    x_rel_path = p
                    break
            if x_rel_path is None:
                x_rel_path = f'images_{x_key}'

        self.x_base_path = self.root_path / x_rel_path / self.split_path.name


        if not self.x_base_path.exists():

            alt_x_path = self.root_path / x_rel_path
            if alt_x_path.exists():
                self.x_base_path = alt_x_path

    def _scan_image_pairs(self) -> None:

        self.image_pairs: List[Tuple[Path, Path]] = []

        if not self.rgb_base_path.exists():
            raise FileNotFoundError(f"RGB 模态路径不存在: {self.rgb_base_path}")

        if not self.x_base_path.exists():
            raise FileNotFoundError(f"X 模态路径不存在: {self.x_base_path}")


        rgb_images = self._scan_images(self.rgb_base_path)


        x_images = self._scan_images(self.x_base_path)


        x_image_dict: Dict[str, Path] = {}
        for x_img in x_images:
            x_image_dict[x_img.stem] = x_img


        for rgb_img in rgb_images:
            stem = rgb_img.stem
            if stem in x_image_dict:
                self.image_pairs.append((rgb_img, x_image_dict[stem]))

        if not self.image_pairs:
            raise ValueError(
                f"未找到匹配的图像对。\n"
                f"RGB 路径: {self.rgb_base_path} ({len(rgb_images)} 张图像)\n"
                f"X 模态路径: {self.x_base_path} ({len(x_images)} 张图像)"
            )

    def _scan_images(self, directory: Path) -> List[Path]:

        images = []
        for ext in self.IMAGE_EXTENSIONS:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
        return sorted(images)

    def sample_one(self) -> Tuple[str, str]:






        if not self.image_pairs:
            raise RuntimeError("没有可用的图像对")

        rgb_path, x_path = random.choice(self.image_pairs)
        return str(rgb_path), str(x_path)

    def sample(self, n: int = 1, replace: bool = False) -> List[Tuple[str, str]]:










        if not self.image_pairs:
            raise RuntimeError("没有可用的图像对")

        if not replace and n > len(self.image_pairs):
            raise ValueError(
                f"请求采样 {n} 对，但只有 {len(self.image_pairs)} 对可用。"
                f"设置 replace=True 允许重复采样。"
            )

        if replace:
            selected = [random.choice(self.image_pairs) for _ in range(n)]
        else:
            selected = random.sample(self.image_pairs, n)

        return [(str(rgb), str(x)) for rgb, x in selected]

    def sample_source_list(self, n: int = 1, replace: bool = False) -> tuple[list[str], list[str]]:






        pairs = self.sample(n=n, replace=replace)
        if not pairs:
            return [], []
        rgb_list, x_list = zip(*pairs)
        return list(rgb_list), list(x_list)

    def sample_by_index(self, index: int) -> Tuple[str, str]:









        if index < 0 or index >= len(self.image_pairs):
            raise IndexError(
                f"索引 {index} 超出范围 [0, {len(self.image_pairs)})"
            )
        rgb_path, x_path = self.image_pairs[index]
        return str(rgb_path), str(x_path)

    def get_predict_source(self) -> List[str]:






        rgb_path, x_path = self.sample_one()
        return [rgb_path, x_path]

    def get_predict_sources(self, n: int = 1) -> List[List[str]]:









        pairs = self.sample(n)
        return [list(pair) for pair in pairs]

    def get_source_dirs(self) -> Tuple[str, str]:











        return str(self.rgb_base_path), str(self.x_base_path)

    def get_all_pairs(self) -> List[Tuple[str, str]]:






        return [(str(rgb), str(x)) for rgb, x in self.image_pairs]

    def __len__(self) -> int:

        return len(self.image_pairs)

    def __iter__(self):

        for rgb_path, x_path in self.image_pairs:
            yield str(rgb_path), str(x_path)

    def info(self) -> Dict[str, Any]:






        return {
            'yaml_path': str(self.yaml_path),
            'root_path': str(self.root_path),
            'split': self.split,
            'rgb_modality': self.rgb_modality_name,
            'x_modality': self.x_modality_name,
            'rgb_base_path': str(self.rgb_base_path),
            'x_base_path': str(self.x_base_path),
            'x_channels': self.x_channels,
            'num_pairs': len(self.image_pairs),
            'modality_used': self.modality_used,
            'class_names': self.names
        }

    def __repr__(self) -> str:
        return (
            f"MultiModalSampler(\n"
            f"  yaml='{self.yaml_path.name}',\n"
            f"  split='{self.split}',\n"
            f"  modalities=['{self.rgb_modality_name}', '{self.x_modality_name}'],\n"
            f"  num_pairs={len(self.image_pairs)}\n"
            f")"
        )

def sample_from_yaml(
    yaml_path: Union[str, Path],
    n: int = 1,
    split: str = 'val',
    seed: Optional[int] = None
) -> List[List[str]]:

















    sampler = MultiModalSampler(yaml_path, split=split, seed=seed)
    return sampler.get_predict_sources(n)

def quick_sample(yaml_path: Union[str, Path], split: str = 'val') -> List[str]:
















    sampler = MultiModalSampler(yaml_path, split=split)
    return sampler.get_predict_source()

def sample_source(
    dataset_yaml: str,
    split: str = "val",
    seed: int | None = None,
    index: int | None = None,
) -> tuple[str, str]:












    sampler = MultiModalSampler(dataset_yaml, split=split, seed=seed)
    if index is not None:
        return sampler.sample_by_index(index)
    return sampler.sample_one()

def sample_source_list(
    dataset_yaml: str,
    n: int = 1,
    split: str = "val",
    seed: int | None = None,
    replace: bool = False,
) -> tuple[list[str], list[str]]:













    sampler = MultiModalSampler(dataset_yaml, split=split, seed=seed)
    return sampler.sample_source_list(n=n, replace=replace)

