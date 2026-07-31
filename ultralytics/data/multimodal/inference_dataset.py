from __future__ import annotations





import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Union
from ultralytics.utils import LOGGER
from ultralytics.data.augment import LetterBox
from .utils import align_and_validate_x, letterbox_with_ratio_pad, to_tensor_rgb, to_tensor_x

class MultiModalInferenceDataset:















    def __init__(
        self,
        samples: List[Dict[str, Union[str, Path]]],
        imgsz: Union[int, tuple],
        dataset_config: Dict,
        stride: int = 32,
        pad: float = 0.5,
        verbose: bool = True
    ):











        self.samples = samples
        self.ni = len(samples)
        self.verbose = verbose


        self.Xch = int(dataset_config.get('Xch', 3))
        self.x_modality = str(dataset_config.get('x_modality', 'unknown'))


        if isinstance(imgsz, int):
            self.imgsz = (imgsz, imgsz)
        else:
            self.imgsz = tuple(imgsz)

        self.stride = stride
        self.pad = pad


        self.letterbox = LetterBox(
            new_shape=self.imgsz,
            auto=False,
            scale_fill=False,
            scaleup=False,
            center=True,
            stride=self.stride
        )


        self.rgb_files = [str(s['rgb_path']) for s in samples]
        self.x_files = [str(s['x_path']) for s in samples]

        if self.verbose:
            LOGGER.info(f"MultiModalInferenceDataset 初始化完成:")
            LOGGER.info(f"  样本数: {self.ni}")
            LOGGER.info(f"  推理尺寸: {self.imgsz}")
            LOGGER.info(f"  X模态类型: {self.x_modality}")
            LOGGER.info(f"  X模态通道数: {self.Xch}")

    def __len__(self) -> int:

        return self.ni

    def __getitem__(self, index: int) -> Dict:














        sample_spec = self.samples[index]


        rgb_path = sample_spec['rgb_path']
        if rgb_path is not None:

            rgb0 = cv2.imread(str(rgb_path))
            if rgb0 is None:
                raise ValueError(f"无法读取RGB图像: {rgb_path}")


            rgb_lb, ratio_pad = letterbox_with_ratio_pad(self.letterbox, rgb0)

            rgb_t = to_tensor_rgb(rgb_lb)
            ori_shape = rgb0.shape[:2]
        else:

            H, W = self.imgsz
            rgb_t = torch.zeros([3, H, W], dtype=torch.float32)
            rgb0 = None
            ori_shape = (H, W)
            ratio_pad = (1.0, (0.0, 0.0))


        x_path = sample_spec['x_path']
        if x_path is not None:

            x0 = self._load_x_modality(x_path)


            if rgb0 is not None:
                x0 = align_and_validate_x(x0, rgb0, expected_xch=self.Xch)
                x_lb, _ = letterbox_with_ratio_pad(self.letterbox, x0)
            else:

                if x0.ndim == 2:
                    x0 = np.expand_dims(x0, axis=2)


                actual_xch = x0.shape[2]
                if actual_xch != self.Xch:
                    if self.Xch in {1, 3} and actual_xch in {1, 3}:

                        if actual_xch == 3 and self.Xch == 1:

                            x0 = cv2.cvtColor(x0, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                        elif actual_xch == 1 and self.Xch == 3:

                            x0 = np.repeat(x0, 3, axis=2)
                    else:

                        raise ValueError(
                            f"X模态通道数不匹配: 期望{self.Xch}, 实际{actual_xch}。"
                            f"仅当Xch在{{1,3}}时允许1<->3显式转换。"
                        )

                x_lb, ratio_pad = letterbox_with_ratio_pad(self.letterbox, x0)
                ori_shape = x0.shape[:2]


            x_t = to_tensor_x(x_lb)
        else:

            H, W = self.imgsz
            x_t = torch.zeros([self.Xch, H, W], dtype=torch.float32)
            x0 = None


        im = torch.cat([rgb_t, x_t], dim=0).unsqueeze(0)


        imgsz_hw = tuple(im.shape[2:4])

        return {
            "id": sample_spec['id'],
            "paths": {
                "rgb": rgb_path,
                "x": x_path
            },
            "orig_imgs": {
                "rgb": rgb0,
                "x": x0
            },
            "meta": {
                "x_modality": self.x_modality,
                "xch": self.Xch,
                "ori_shape": ori_shape,
                "imgsz": imgsz_hw,
                "ratio_pad": ratio_pad
            },
            "im": im
        }

    def _load_x_modality(self, x_path: Path) -> np.ndarray:













        if not x_path.exists():
            raise FileNotFoundError(f"X模态文件不存在: {x_path}")


        suffix = x_path.suffix.lower()

        try:
            if suffix == '.npy':

                x_img = np.load(x_path)

            elif suffix == '.npz':

                with np.load(x_path) as npz_file:

                    preferred_keys = ('image', 'arr_0', 'array', 'data')
                    selected_key = next(
                        (k for k in preferred_keys if k in npz_file.files),
                        None
                    )

                    if selected_key is None:
                        if len(npz_file.files) == 1:
                            selected_key = npz_file.files[0]
                        else:
                            raise ValueError(
                                f"npz文件含多个数组 {npz_file.files}，"
                                f"无法确定默认键，请使用标准键(image/arr_0)"
                            )

                    x_img = npz_file[selected_key]

            elif suffix in {'.tiff', '.tif'}:

                x_img = cv2.imread(str(x_path), cv2.IMREAD_UNCHANGED)

            else:

                x_img = cv2.imread(str(x_path), cv2.IMREAD_UNCHANGED)

            if x_img is None:
                raise ValueError(f"无法读取X模态图像: {x_path}")

            return x_img

        except Exception as e:
            raise ValueError(f"加载X模态图像失败: {x_path}\n错误: {e}")

    def __iter__(self):

        for i in range(len(self)):
            yield self[i]

