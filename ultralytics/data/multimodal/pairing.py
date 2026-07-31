from __future__ import annotations





from pathlib import Path
from typing import List, Dict, Union
from ultralytics.utils import LOGGER
from ultralytics.nn.mm import MultiModalSourceMatcher

class PairingResolver:














    def __init__(self, x_modality: str = "unknown", verbose: bool = True):







        self.x_modality = x_modality
        self.verbose = verbose

    def resolve(
        self,
        rgb_source: Union[str, Path, List[Union[str, Path]], None] = None,
        x_source: Union[str, Path, List[Union[str, Path]], None] = None,
        strict_match: bool = True
    ) -> List[Dict[str, Union[str, Path, None]]]:


































        if rgb_source is None and x_source is not None:
            return self._resolve_single_modality(x_source, modality="x")

        if rgb_source is not None and x_source is None:
            return self._resolve_single_modality(rgb_source, modality="rgb")



        if isinstance(rgb_source, (str, Path)) and isinstance(x_source, (str, Path)):
            rgb_p, x_p = Path(rgb_source), Path(x_source)
            if rgb_p.is_dir() and x_p.is_dir():
                matcher = MultiModalSourceMatcher(rgb_source, x_source, strict_match=strict_match)
                pairs = matcher.match()
                samples = [
                    self._create_sample_spec(rgb_path=rp, x_path=xp, sample_idx=i)
                    for i, (rp, xp) in enumerate(pairs)
                ]
                if self.verbose:
                    LOGGER.info(f"目录批量配对完成: {len(samples)} 对有效样本")
                return samples


        if isinstance(rgb_source, list) and isinstance(x_source, list):
            pairs = MultiModalSourceMatcher.match_lists(rgb_source, x_source, strict_match=strict_match)
            samples = [
                self._create_sample_spec(rgb_path=rp, x_path=xp, sample_idx=i)
                for i, (rp, xp) in enumerate(pairs)
            ]
            if self.verbose:
                LOGGER.info(f"列表批量配对完成: {len(samples)} 对有效样本")
            return samples


        rgb_list = self._normalize_to_list(rgb_source, "rgb_source")
        x_list = self._normalize_to_list(x_source, "x_source")


        if len(rgb_list) != len(x_list):
            raise ValueError(
                f"RGB和X模态数量不匹配：\n"
                f"  rgb_source: {len(rgb_list)} 张\n"
                f"  x_source: {len(x_list)} 张\n"
                f"请确保两者数量相同。"
            )


        samples = []
        for idx, (rgb_path, x_path) in enumerate(zip(rgb_list, x_list)):
            samples.append(self._create_sample_spec(
                rgb_path=Path(rgb_path),
                x_path=Path(x_path),
                sample_idx=idx
            ))

        if self.verbose:
            LOGGER.info(f"双模态配对完成: {len(samples)} 对有效样本")

        return samples

    def _resolve_single_modality(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        modality: str
    ) -> List[Dict[str, Union[str, Path, None]]]:








        if isinstance(source, (str, Path)) and Path(source).is_dir():
            src_dir = Path(source)
            files = sorted([
                p for p in src_dir.iterdir()
                if p.is_file() and p.suffix.lower() in MultiModalSourceMatcher.SUPPORTED_FORMATS
            ])
            if not files:
                modality_name = "RGB" if modality == "rgb" else self.x_modality
                raise ValueError(f"{modality_name}目录中未找到受支持的图像文件: {src_dir}")
        else:
            files = self._normalize_to_list(source, f"{modality}_source")

        samples = []
        for idx, path in enumerate(files):
            if modality == "rgb":
                samples.append(self._create_sample_spec(rgb_path=Path(path), x_path=None, sample_idx=idx))
            else:
                samples.append(self._create_sample_spec(rgb_path=None, x_path=Path(path), sample_idx=idx))

        if self.verbose:
            if modality == "rgb":
                LOGGER.info(f"单RGB模态推理: {len(samples)} 个样本（{self.x_modality}将使用零填充）")
            else:
                LOGGER.info(f"单{self.x_modality}模态推理: {len(samples)} 个样本（RGB将使用零填充）")

        return samples

    def _normalize_to_list(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        param_name: str
    ) -> List[Path]:










        if isinstance(source, (str, Path)):
            return [Path(source)]
        elif isinstance(source, list):
            if not source:
                raise ValueError(f"{param_name} 不能为空列表")
            return [Path(item) for item in source]
        else:
            raise ValueError(
                f"{param_name} 类型不支持: {type(source)}\n"
                f"支持类型: str, Path, List[str], List[Path]"
            )

    def _create_sample_spec(
        self,
        rgb_path: Union[Path, None],
        x_path: Union[Path, None],
        sample_idx: int
    ) -> Dict[str, Union[str, Path, None]]:
















        if rgb_path is not None:
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB文件不存在: {rgb_path}")
            if not rgb_path.is_file():
                raise ValueError(f"RGB路径不是文件: {rgb_path}")


        if x_path is not None:
            if not x_path.exists():
                raise FileNotFoundError(f"X模态文件不存在: {x_path}")
            if not x_path.is_file():
                raise ValueError(f"X模态路径不是文件: {x_path}")


        if rgb_path is not None:
            sample_id = rgb_path.stem
        elif x_path is not None:
            sample_id = x_path.stem
        else:
            sample_id = f"sample_{sample_idx:03d}"

        return {
            "id": sample_id,
            "rgb_path": rgb_path,
            "x_path": x_path,
            "x_modality": self.x_modality
        }

