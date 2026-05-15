"""Fusion modules package aggregating fusion-related blocks.
This subpackage hosts fusion modules for multimodal/RGBT feature interaction,
including CFFormer-style FCM/FFN blocks, ICAFusion variants, CTF, and
DEYOLO's DEA/BiFocus family (DEA, DECA, DEPA, BiFocus, C2f_BiFocus).
"""

# Export only the public fusion blocks needed by outer code
from .FCM_FFN import (
    FeatureFusion,
    FeatureInteraction,
    ChannelEmbed,
    CrossAttention,
    FCM,
    FCMFeatureFusion,  # 二创：FCM→FFM 串联封装
    ConvFFN_GLU,       # 二创：卷积版 FFN（GLU 门控）
)
from .CAM import CAM  # Cross-Modal Attention Mechanism

# 拆分后的独立模块文件导入（保持对外类名不变）
from .ssa import SequenceShuffleAttention
from .fcm_comp import FeatureComplementaryMapping
from .tsa import TokenSelectiveAttention
from .sefn import SEFN
from .edffn import EDFFN
from .msaa import FusionConvMSAA
from .iia import IIA
from .hfp import HighFrequencyPerception
from .sdfm import SpatialDependencyPerception
from .msc import MSC
from .pst import PST
from .icafusion import NiNfusion  # public
from .ctf import CrossTransformerFusion, MultiHeadCrossAttention  # public
from .deyolo import DEA, DECA, DEPA, BiFocus, C2f_BiFocus  # public
# RD 模块导出（YOLO-RD 核心模块）
from .RD import DConv, RepNCSPELAND  # public (YOLO-RD core modules)
from .UniRGB_IR import (
    SpatialPriorModuleLite,
    ConvMixFusion,
    ScalarGate,
    ChannelGate,
    ncc,
)  # public
# MROD-YOLO 模块（早期融合）
from .mrod import GCB, MJRNet
# UMIS-YOLO 模块（频域增强融合 + 残差特征融合）
from .dyt import DyT
from .fdfef import FDFEF
from .rff import RFF
# 跨模态融合模块（CIDAF/CGAFusion/DAF/WDAF）
from .cidaf import CIDAF
from .cgafusion import CGAFusion
from .daf import DAF, DynamicAlignFusion
from .wdaf import WDAF
from .mine import (SymmetricFreqGuidedFusion,DecoupledFreqGuidedFusion,DecoupledFreqGuidedFusion_Pro_Safe,DecoupledFreqGuidedFusion_BiFocus,DecoupledFreqGuidedFusion_FDFEF,FrequencyFocusedDownSampling2,
                   DecoupledFreqGuidedFusion_HFP,DecoupledFreqGuidedFusion_GCB,DecoupledFreqGuidedFusion_RD,DecoupledFreqGuidedFusion_IIA
                  ,SymmetricFreqGuidedFusion_new,DySample,FrequencyFocusedDownSampling,DecoupledFreqGuidedFusion_HFBypass,LAGFusion,HeavyDFGF,
                  DFGF_DWconv_CA,DFGF_BiFocus,Deep_CFFM,SymmetricFreqGuidedFusion_attn,DecoupledFreqGuidedFusion_attn,DecoupledFreqGuidedFusion_trans,
                  GetIndex,ContextGuideFusionModuleV2,DecoupledFreqGuidedFusion_re)
from .wtconv2d_imp import (WTConv2dMaxPool,WTConv2d_imp,FocusWNC,SPPFCSPC)
__all__ = (
    'FeatureFusion', 'FeatureInteraction', 'ChannelEmbed', 'CrossAttention', 'FCM', 'FCMFeatureFusion', 'CAM',
    # Advanced fusion/attention blocks
    'SequenceShuffleAttention', 'FeatureComplementaryMapping', 'TokenSelectiveAttention', 'SEFN', 'EDFFN',
    'FusionConvMSAA', 'IIA', 'HighFrequencyPerception', 'SpatialDependencyPerception', 'MSC', 'PST',
    'ConvFFN_GLU', 'NiNfusion', 'CrossTransformerFusion', 'MultiHeadCrossAttention',
    'DEA', 'DECA', 'DEPA', 'BiFocus', 'C2f_BiFocus',
    'DConv', 'RepNCSPELAND',  # RD 模块
    'SpatialPriorModuleLite', 'ConvMixFusion', 'ScalarGate', 'ChannelGate', 'ncc',
    # MROD-YOLO 模块（早期融合）
    'GCB', 'MJRNet',
    # UMIS-YOLO 模块（频域增强融合 + 残差特征融合）
    'DyT', 'FDFEF', 'RFF','FrequencyFocusedDownSampling2',
    # 跨模态融合模块
    'CIDAF', 'CGAFusion', 'DAF', 'DynamicAlignFusion', 'WDAF',
    'DyT', 'FDFEF', 'RFF','SymmetricFreqGuidedFusion','DecoupledFreqGuidedFusion','DecoupledFreqGuidedFusion_Pro_Safe',
    'DecoupledFreqGuidedFusion_BiFocus','DecoupledFreqGuidedFusion_FDFEF',
    'DecoupledFreqGuidedFusion_HFP','DecoupledFreqGuidedFusion_GCB','DecoupledFreqGuidedFusion_RD','DecoupledFreqGuidedFusion_IIA',
    'SymmetricFreqGuidedFusion_new','DySample','FrequencyFocusedDownSampling','DecoupledFreqGuidedFusion_HFBypass','LAGFusion','HeavyDFGF',
    'DFGF_DWconv_CA','DFGF_BiFocus','Deep_CFFM','SymmetricFreqGuidedFusion_attn','DecoupledFreqGuidedFusion_attn','DecoupledFreqGuidedFusion_trans',
    'GetIndex','ContextGuideFusionModuleV2','DecoupledFreqGuidedFusion_re','WTConv2dMaxPool','WTConv2d_imp','FocusWNC','SPPFCSPC'

)
