# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""RTDETRMM 独立模型家族（不依附 RTDETR）。

说明：
- 该包提供 RTDETRMM 的 model/train/val/predict/visualize 等组件。
- 设计目标：RTDETRMM 不继承 RTDETR，不依赖 ultralytics.models.rtdetr.*。
"""

from .model import RTDETRMM

__all__ = ["RTDETRMM"]
