from __future__ import annotations
# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
多模态YOLO模型架构组件

本模块包含完整的多模态目标检测系统，支持RGB+X模态的训练、验证、推理和数据生成。

主要组件:
- MultiModalDetectionTrainer: 多模态训练器
- MultiModalDetectionValidator: 多模态验证器  
- MultiModalDetectionPredictor: 多模态预测器
- ModalityFiller: 模态填充器
- SelfModalGenerator: 自体模态生成器

# 可视化复用组件定位（简要说明）
# ultralytics/models/utils/multimodal/vis.py：统一多模态绘图与几何处理（拆分、灰度、并排、坐标裁剪），
# 为 train/val/cocoval/predict 提供一致的可视化输出风格与坐标逻辑。

使用示例:
    >>> from ultralytics.models.yolo.multimodal import MultiModalDetectionTrainer
    >>> trainer = MultiModalDetectionTrainer(model='yolo11n-mm.yaml', data='multimodal_data.yaml')
    >>> trainer.train()
    
    >>> from ultralytics.models.yolo.multimodal import MultiModalDetectionPredictor
    >>> predictor = MultiModalDetectionPredictor()
    >>> results = predictor.predict(['rgb_image.jpg', 'x_image.jpg'])
    
    >>> from ultralytics.models.yolo.multimodal import generate_modality_filling
    >>> filled_tensor = generate_modality_filling(rgb_tensor, 'rgb', 'depth')
"""


from .train import MultiModalDetectionTrainer
from .val import MultiModalDetectionValidator
from .cocoval import MultiModalCOCOValidator
from .predict import MultiModalDetectionPredictor
from .mm_predictor import (
    YOLOMMPredictor,
    YOLOMMSegPredictor,
    YOLOMMOBBPredictor,
    YOLOMMPosePredictor,
    YOLOMMClassifyPredictor,
)
from .obb import (
    MultiModalOBBTrainer,
    MultiModalOBBValidator,
)
from .pose import (
    MultiModalPoseTrainer,
    MultiModalPoseValidator,
)
from .classify import (
    MultiModalClassificationTrainer,
    MultiModalClassificationValidator,
)


from ultralytics.nn.mm.filling import (
    ModalityFiller,
    generate_modality_filling,
)


from .self_modal_generator import (
    SelfModalGenerator,
    generate_self_modality as generate_self_modality_direct,
    default_self_modal_generator as default_self_modal_generator_direct,
)


__all__ = [

    'MultiModalDetectionTrainer',
    'MultiModalDetectionValidator',
    'MultiModalCOCOValidator',
    'MultiModalDetectionPredictor',
    'YOLOMMPredictor',
    'YOLOMMSegPredictor',
    'YOLOMMOBBPredictor',
    'YOLOMMPosePredictor',
    'YOLOMMClassifyPredictor',
    'MultiModalOBBTrainer',
    'MultiModalOBBValidator',
    'MultiModalPoseTrainer',
    'MultiModalPoseValidator',
    'MultiModalClassificationTrainer',
    'MultiModalClassificationValidator',
    

    'ModalityFiller',
    'generate_modality_filling',
    

    'SelfModalGenerator',
    'generate_self_modality',
    'default_self_modal_generator',
]


__version__ = '1.0.0'
__author__ = 'Ultralytics MultiModal Team'
__description__ = 'Ultralytics YOLO MultiModal Detection Components'



generate_self_modality = generate_self_modality_direct
default_self_modal_generator = default_self_modal_generator_direct


SUPPORTED_MODALITIES = ['rgb', 'depth', 'thermal', 'ir', 'nir', 'lidar', 'sar']
SUPPORTED_FUSION_TYPES = ['early', 'mid', 'late']
DEFAULT_CHANNELS = 6

def get_multimodal_info():






    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'supported_modalities': SUPPORTED_MODALITIES,
        'supported_fusion_types': SUPPORTED_FUSION_TYPES,
        'default_channels': DEFAULT_CHANNELS,
        'components': {
            'trainer': 'MultiModalDetectionTrainer',
            'validator': 'MultiModalDetectionValidator',
            'coco_validator': 'MultiModalCOCOValidator',
            'predictor': 'MultiModalDetectionPredictor',
            'modality_filler': 'ModalityFiller',
            'self_modal_generator': 'SelfModalGenerator',
        }
    }

def validate_multimodal_setup():






    try:

        assert MultiModalDetectionTrainer is not None
        assert MultiModalDetectionValidator is not None
        assert MultiModalCOCOValidator is not None
        assert MultiModalDetectionPredictor is not None
        

        assert ModalityFiller is not None
        assert generate_modality_filling is not None

        assert SelfModalGenerator is not None
        assert generate_self_modality is not None
        assert default_self_modal_generator is not None
        
        return True
    except (AssertionError, ImportError) as e:
        print(f"多模态组件验证失败: {e}")
        return False


if __name__ != '__main__':

    _validation_result = validate_multimodal_setup()
    if not _validation_result:
        import warnings
        warnings.warn("多模态组件验证失败，某些功能可能不可用", UserWarning)

