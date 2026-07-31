from __future__ import annotations
"""Fusion modules package aggregating fusion-related blocks.
This subpackage hosts fusion modules for multimodal/RGBT feature interaction,
including CFFormer-style FCM/FFN blocks, ICAFusion variants, CTF, and
DEYOLO's DEA/BiFocus family (DEA, DECA, DEPA, BiFocus, C2f_BiFocus).
"""


from .FCM_FFN import (
    FeatureFusion,
    FeatureInteraction,
    ChannelEmbed,
    CrossAttention,
    FCM,
    FCMFeatureFusion,
    ConvFFN_GLU,
)
from .CAM import CAM


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
from .icafusion import NiNfusion
from .ctf import CrossTransformerFusion, MultiHeadCrossAttention
from .deyolo import DEA, DECA, DEPA, BiFocus, C2f_BiFocus

from .RD import DConv, RepNCSPELAND
from .UniRGB_IR import (
    SpatialPriorModuleLite,
    ConvMixFusion,
    ScalarGate,
    ChannelGate,
    ncc,
)

from .mrod import GCB, MJRNet

from .dyt import DyT
from .fdfef import FDFEF
from .rff import RFF

from .cidaf import CIDAF
from .cgafusion import CGAFusion
from .daf import DAF, DynamicAlignFusion
from .wdaf import WDAF
from .mine import (SymmetricFreqGuidedFusion,DecoupledFreqGuidedFusion,DecoupledFreqGuidedFusion_Pro_Safe,
                   DecoupledFreqGuidedFusion_BiFocus,DecoupledFreqGuidedFusion_FDFEF,FrequencyFocusedDownSampling2,
                   DecoupledFreqGuidedFusion_HFP,DecoupledFreqGuidedFusion_GCB,
                   DecoupledFreqGuidedFusion_RD,DecoupledFreqGuidedFusion_IIA
                  ,SymmetricFreqGuidedFusion_new,DySample,FrequencyFocusedDownSampling,DecoupledFreqGuidedFusion_HFBypass,
                   LAGFusion,HeavyDFGF,DFGF_DWconv_CA,DFGF_BiFocus,Deep_CFFM,SymmetricFreqGuidedFusion_attn,
                   DecoupledFreqGuidedFusion_attn,DecoupledFreqGuidedFusion_trans,
                  GetIndex,ContextGuideFusionModuleV2,DecoupledFreqGuidedFusion_re,Ablation_Sym_Only_DPFR,Ablation_Only_DPFR,
                   Ablation_Sym_DPFR_PMDA,Ablation_DPFR_PMDA,
                   DecoupledFreqGuidedFusion_4Mode,
                   DecoupledFreqGuidedFusion_DPFRv2,
                   DecoupledFreqGuidedFusion_Step1,
                   DecoupledFreqGuidedFusion_Step2,
                   DecoupledFreqGuidedFusion_Step3,
                   DecoupledFreqGuidedFusion_Step4,
                   DecoupledFreqGuidedFusion_Step4Lite,
                   DecoupledFreqGuidedFusion_NoMask,
                   DecoupledFreqGuidedFusion_ExpA_RGBGuide,
                   DecoupledFreqGuidedFusion_ExpB_NoGate,
                   DecoupledFreqGuidedFusion_ExpC_RGBGuide_NoMask,
                   DecoupledFreqGuidedFusion_ExpD_SymNoMask,
                   DecoupledFreqGuidedFusion_F1_RGBGuide_Gate3x3,
                   DecoupledFreqGuidedFusion_F2_RGBGuide_FAFFM,
                   DecoupledFreqGuidedFusion_PMDA_SoftMask,
                   DecoupledFreqGuidedFusion_PMDA_Enhanced,
                   DecoupledFreqGuidedFusion_PMDA_DualMask,
                   DecoupledFreqGuidedFusion_PMDA_SoftMask_v2,
                   DecoupledFreqGuidedFusion_PMDA_Enhanced_v2,
                   DecoupledFreqGuidedFusion_PMDA_DualMask_v2,
                   DecoupledFreqGuidedFusion_PMDA_SoftEnhanced,
                   DecoupledFreqGuidedFusion_PMDA_DualEnhanced,
                   DecoupledFreqGuidedFusion_PMDA_Enhanced_g8,
                   DecoupledFreqGuidedFusion_PMDA_Enhanced_Gate3x3,
                   DecoupledFreqGuidedFusion_PMDA_Enhanced_ExpA,
                   DecoupledFreqGuidedFusion_PMDA_Enhanced_Refine,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_MDAAv2,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_Soft,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_MultiScale,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_LearnAttn,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_SymPMDA,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_OffsetReg,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_SymFreq,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_AlignLoss,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_CycleLoss,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_AlignV2,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_Contrastive,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_AlignBox,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveV2,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_DualContrast,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_SimCLR,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveLite,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveMoreNeg,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveAllLevels,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_CrossAttn,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_LearnFreq,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_Uncertainty,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_NeckFuse,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_DynamicGroup,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_DCNAlign,
                   DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveNeckFuse,
                   DecoupledFreqGuidedFusion_PMDA_Final_LearnTau,
                   DecoupledFreqGuidedFusion_PMDA_Final_DeepSE,
                   DecoupledFreqGuidedFusion_PMDA_Final_MultiLevel,
                   DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauDeepSE,
                   DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTau05,
                   DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauCos,
                   DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauDeepSECos,
                   DecoupledFreqGuidedFusion_PMDA_Final_LearnTau_IR,
                   DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauDeepSE_IR,
                   DecoupledFreqGuidedFusion_RTDETR_Enhanced,
                   DecoupledFreqGuidedFusion_RTDETR_ExpA,
                   DecoupledFreqGuidedFusion_RTDETR_NeckFuse,
                   DecoupledFreqGuidedFusion_RTDETR_LearnTau,
                   DecoupledFreqGuidedFusion_RTDETR_LearnTauDeepSE)
from .wtconv2d_imp import (WTConv2dMaxPool,WTConv2d_imp,FocusWNC,SPPFCSPC)
__all__ = (
    'FeatureFusion', 'FeatureInteraction', 'ChannelEmbed', 'CrossAttention', 'FCM', 'FCMFeatureFusion', 'CAM',

    'SequenceShuffleAttention', 'FeatureComplementaryMapping', 'TokenSelectiveAttention', 'SEFN', 'EDFFN',
    'FusionConvMSAA', 'IIA', 'HighFrequencyPerception', 'SpatialDependencyPerception', 'MSC', 'PST',
    'ConvFFN_GLU', 'NiNfusion', 'CrossTransformerFusion', 'MultiHeadCrossAttention',
    'DEA', 'DECA', 'DEPA', 'BiFocus', 'C2f_BiFocus',
    'DConv', 'RepNCSPELAND',
    'SpatialPriorModuleLite', 'ConvMixFusion', 'ScalarGate', 'ChannelGate', 'ncc',

    'GCB', 'MJRNet',

    'DyT', 'FDFEF', 'RFF','FrequencyFocusedDownSampling2',

    'CIDAF', 'CGAFusion', 'DAF', 'DynamicAlignFusion', 'WDAF',
    'DyT', 'FDFEF', 'RFF','SymmetricFreqGuidedFusion','DecoupledFreqGuidedFusion','DecoupledFreqGuidedFusion_Pro_Safe',
    'DecoupledFreqGuidedFusion_BiFocus','DecoupledFreqGuidedFusion_FDFEF',
    'DecoupledFreqGuidedFusion_HFP','DecoupledFreqGuidedFusion_GCB','DecoupledFreqGuidedFusion_RD','DecoupledFreqGuidedFusion_IIA',
    'SymmetricFreqGuidedFusion_new','DySample','FrequencyFocusedDownSampling','DecoupledFreqGuidedFusion_HFBypass','LAGFusion','HeavyDFGF',
    'DFGF_DWconv_CA','DFGF_BiFocus','Deep_CFFM','SymmetricFreqGuidedFusion_attn','DecoupledFreqGuidedFusion_attn',
    'DecoupledFreqGuidedFusion_trans','Ablation_Sym_Only_DPFR',
    'Ablation_Only_DPFR','Ablation_Sym_DPFR_PMDA','Ablation_DPFR_PMDA',
    'GetIndex','ContextGuideFusionModuleV2','DecoupledFreqGuidedFusion_re','DecoupledFreqGuidedFusion_4Mode','DecoupledFreqGuidedFusion_DPFRv2','DecoupledFreqGuidedFusion_Step1','DecoupledFreqGuidedFusion_Step2','DecoupledFreqGuidedFusion_Step3','DecoupledFreqGuidedFusion_Step4','DecoupledFreqGuidedFusion_Step4Lite','DecoupledFreqGuidedFusion_NoMask','DecoupledFreqGuidedFusion_ExpA_RGBGuide','DecoupledFreqGuidedFusion_ExpB_NoGate','DecoupledFreqGuidedFusion_ExpC_RGBGuide_NoMask','DecoupledFreqGuidedFusion_ExpD_SymNoMask','DecoupledFreqGuidedFusion_F1_RGBGuide_Gate3x3','DecoupledFreqGuidedFusion_F2_RGBGuide_FAFFM','DecoupledFreqGuidedFusion_PMDA_SoftMask','DecoupledFreqGuidedFusion_PMDA_Enhanced','DecoupledFreqGuidedFusion_PMDA_DualMask','DecoupledFreqGuidedFusion_PMDA_SoftMask_v2','DecoupledFreqGuidedFusion_PMDA_Enhanced_v2','DecoupledFreqGuidedFusion_PMDA_DualMask_v2','DecoupledFreqGuidedFusion_PMDA_SoftEnhanced','DecoupledFreqGuidedFusion_PMDA_DualEnhanced','DecoupledFreqGuidedFusion_PMDA_Enhanced_g8','DecoupledFreqGuidedFusion_PMDA_Enhanced_Gate3x3','DecoupledFreqGuidedFusion_PMDA_Enhanced_ExpA','DecoupledFreqGuidedFusion_PMDA_Enhanced_Refine','DecoupledFreqGuidedFusion_PMDA_ExpA_MDAAv2','DecoupledFreqGuidedFusion_PMDA_ExpA_Soft','DecoupledFreqGuidedFusion_PMDA_ExpA_MultiScale','DecoupledFreqGuidedFusion_PMDA_ExpA_LearnAttn','DecoupledFreqGuidedFusion_PMDA_ExpA_SymPMDA','DecoupledFreqGuidedFusion_PMDA_ExpA_OffsetReg','DecoupledFreqGuidedFusion_PMDA_ExpA_SymFreq','DecoupledFreqGuidedFusion_PMDA_ExpA_AlignLoss','DecoupledFreqGuidedFusion_PMDA_ExpA_CycleLoss','DecoupledFreqGuidedFusion_PMDA_ExpA_AlignV2','DecoupledFreqGuidedFusion_PMDA_ExpA_Contrastive','DecoupledFreqGuidedFusion_PMDA_ExpA_AlignBox','DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveV2','DecoupledFreqGuidedFusion_PMDA_ExpA_DualContrast','DecoupledFreqGuidedFusion_PMDA_ExpA_SimCLR','DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveLite','DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveMoreNeg','DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveAllLevels','DecoupledFreqGuidedFusion_PMDA_ExpA_CrossAttn','DecoupledFreqGuidedFusion_PMDA_ExpA_LearnFreq','DecoupledFreqGuidedFusion_PMDA_ExpA_Uncertainty','DecoupledFreqGuidedFusion_PMDA_ExpA_NeckFuse','DecoupledFreqGuidedFusion_PMDA_ExpA_DynamicGroup','DecoupledFreqGuidedFusion_PMDA_ExpA_DCNAlign','DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveNeckFuse','DecoupledFreqGuidedFusion_PMDA_Final_LearnTau','DecoupledFreqGuidedFusion_PMDA_Final_DeepSE','DecoupledFreqGuidedFusion_PMDA_Final_MultiLevel','DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauDeepSE','DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTau05','DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauCos','DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauDeepSECos','DecoupledFreqGuidedFusion_PMDA_Final_LearnTau_IR','DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauDeepSE_IR','DecoupledFreqGuidedFusion_RTDETR_Enhanced','DecoupledFreqGuidedFusion_RTDETR_ExpA','DecoupledFreqGuidedFusion_RTDETR_NeckFuse','DecoupledFreqGuidedFusion_RTDETR_LearnTau','DecoupledFreqGuidedFusion_RTDETR_LearnTauDeepSE','WTConv2dMaxPool','WTConv2d_imp','FocusWNC','SPPFCSPC'

)

