from __future__ import annotations





import cv2
import numpy as np
import torch
from typing import Tuple
from ultralytics.utils import LOGGER

def align_and_validate_x(
    x_img: np.ndarray,
    rgb_img: np.ndarray,
    expected_xch: int
) -> np.ndarray:















    rgb_h, rgb_w = rgb_img.shape[:2]
    x_h, x_w = x_img.shape[:2]

    if (x_h, x_w) != (rgb_h, rgb_w):
        if len(x_img.shape) == 2:

            x_img = cv2.resize(x_img, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)
        else:

            x_img = cv2.resize(x_img, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)



    if len(x_img.shape) == 2:
        actual_xch = 1
        x_img = x_img[:, :, np.newaxis]
    elif len(x_img.shape) == 3:
        actual_xch = x_img.shape[2]
    else:
        raise ValueError(f"X模态图像维度异常: {x_img.shape}（应为 H×W 或 H×W×C）")


    if expected_xch in {1, 3} and actual_xch in {1, 3}:

        if actual_xch == 1 and expected_xch == 3:

            x_img = np.repeat(x_img, 3, axis=2)
        elif actual_xch == 3 and expected_xch == 1:

            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]


    elif actual_xch == expected_xch:

        pass

    else:

        raise ValueError(
            f"X模态通道数不匹配：期望 {expected_xch} 通道，实际 {actual_xch} 通道。\n"
            f"仅当 Xch∈{{1,3}} 时允许 1↔3 显式转换。\n"
            f"对于 Xch>3，请提供严格匹配通道数的多通道文件（推荐 .tif/.npy/.npz）。"
        )


    if x_img.shape[2] != expected_xch:
        raise ValueError(
            f"X模态通道数校验失败：期望 {expected_xch}，实际 {x_img.shape[2]}"
        )

    return x_img

def letterbox_with_ratio_pad(
    letterbox_func,
    img: np.ndarray
) -> Tuple[np.ndarray, Tuple[float, Tuple[float, float]]]:

















    h0, w0 = img.shape[:2]


    result = letterbox_func(labels={}, image=img)


    letterboxed_img = result
    h, w = letterboxed_img.shape[:2]



    gain = min(h / h0, w / w0) if h0 > 0 and w0 > 0 else 1.0



    padw = (w - w0 * gain) / 2
    padh = (h - h0 * gain) / 2

    ratio_pad = (gain, (padw, padh))

    return letterboxed_img, ratio_pad

def to_tensor_rgb(img: np.ndarray) -> torch.Tensor:















    img_t = img.transpose(2, 0, 1)


    img_t = torch.from_numpy(np.ascontiguousarray(img_t)).float()


    if img_t.max() > 1.0:
        img_t /= 255.0

    return img_t

def to_tensor_x(img: np.ndarray) -> torch.Tensor:










    img_t = img.transpose(2, 0, 1)


    img_t = torch.from_numpy(np.ascontiguousarray(img_t)).float()


    if img.dtype == np.uint8:
        img_t /= 255.0
    elif img.dtype == np.uint16:
        img_t /= 65535.0


    return img_t

