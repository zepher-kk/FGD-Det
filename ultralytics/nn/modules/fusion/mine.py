from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from collections import OrderedDict
from ultralytics.nn.modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ultralytics.nn.modules.block_new import get_activation, ConvNormLayer, WTConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, \
    Bottleneck
from ultralytics.nn.modules.fusion.attn import EMA, SimAM, SpatialGroupEnhance, BiLevelRoutingAttention, BiLevelRoutingAttention_nchw, TripletAttention,CoordAtt, BAMBlock, EfficientAttention, LSKBlock, SEAttention, CPCA, MPCA, deformable_LKA,EffectiveSEModule, LSKA, SegNext_Attention, DAttention, FocusedLinearAttention, MLCA, TransNeXt_AggregatedAttention,HiLo, LocalWindowAttention, ELA, CAA, EfficientAdditiveAttnetion, AFGCAttention, DualDomainSelectionMechanism,AttentionTSSA,CBAM

__all__ = ['DySample', 'SPDConv', 'MFFF', 'FrequencyFocusedDownSampling', 'SemanticAlignmenCalibration',
           'WaveletTransform','FrequencyFocusedDownSampling2',
           'ImprovedWaveletKernel', 'MFFF_W', 'ChannelShuffle', 'CBAM',
           'SymmetricFreqGuidedFusion', 'AsymmetricFreqGuidedFusion', 'AsymmetricFreqGuidedFusion_V2',
           'AsymmetricFreqGuidedFusion_V3',
           'SafeWaveletAlignFusion', 'FreqGuidedDyAlignFusion', 'AsymFreqDynamicAlignFusion', 'AsymWaveletStarFusion',
           'FreqDiffGuidedFusion', 'DualFreqAlignFusion',
           'EdgeConstrainedFreqFusion', 'FreqSpatialAttnFusion', 'DecoupledFreqGuidedFusion',
           'DecoupledSymmetricFreqFusion', 'SceneAwareDecoupledFusion',
           'CrossAttnFreqGuidedFusion', 'DetailPreservingFreqFusion','DecoupledFreqGuidedFusion_Pro_Safe','DecoupledFreqGuidedFusion_BiFocus',
            'DecoupledFreqGuidedFusion_FDFEF','DecoupledFreqGuidedFusion_HFP','DecoupledFreqGuidedFusion_GCB',
           'DecoupledFreqGuidedFusion_RD','DecoupledFreqGuidedFusion_IIA',
           'SymmetricFreqGuidedFusion_new','DecoupledFreqGuidedFusion_HFBypass','LAGFusion','HeavyDFGF','DFGF_DWconv_CA',
          'DFGF_BiFocus','Deep_CFFM','SymmetricFreqGuidedFusion_attn','DecoupledFreqGuidedFusion_attn','DecoupledFreqGuidedFusion_trans',
           'ContextGuideFusionModuleV2','DecoupledFreqGuidedFusion_re','Ablation_Sym_Only_DPFR','Ablation_Only_DPFR','Ablation_Sym_DPFR_PMDA',
          'Ablation_DPFR_PMDA',
          'PhaseDiffAlignMap', 'CrossModalAmpComplement', 'DecoupledFreqGuidedFusion_4Mode',
          'ForegroundAwareFFM', 'JointGate', 'AdaptiveFocus', 'DualModalEdgeMask', 'DecoupledFreqGuidedFusion_DPFRv2',
          'DecoupledFreqGuidedFusion_Step1',
          'DecoupledFreqGuidedFusion_Step2',
          'DecoupledFreqGuidedFusion_Step3',
          'DecoupledFreqGuidedFusion_Step4',
          'DecoupledFreqGuidedFusion_Step4Lite',
          'DecoupledFreqGuidedFusion_NoMask',
          'DecoupledFreqGuidedFusion_ExpA_RGBGuide', 'DecoupledFreqGuidedFusion_ExpB_NoGate',
          'DecoupledFreqGuidedFusion_ExpC_RGBGuide_NoMask', 'DecoupledFreqGuidedFusion_ExpD_SymNoMask',
          'DecoupledFreqGuidedFusion_F1_RGBGuide_Gate3x3', 'DecoupledFreqGuidedFusion_F2_RGBGuide_FAFFM',
          'DecoupledFreqGuidedFusion_PMDA_SoftMask', 'DecoupledFreqGuidedFusion_PMDA_Enhanced',
          'DecoupledFreqGuidedFusion_PMDA_DualMask',
          'DecoupledFreqGuidedFusion_PMDA_SoftMask_v2', 'DecoupledFreqGuidedFusion_PMDA_Enhanced_v2',
          'DecoupledFreqGuidedFusion_PMDA_DualMask_v2',
          'DecoupledFreqGuidedFusion_PMDA_SoftEnhanced', 'DecoupledFreqGuidedFusion_PMDA_DualEnhanced',
          'DecoupledFreqGuidedFusion_PMDA_Enhanced_g8',
          'DecoupledFreqGuidedFusion_PMDA_Enhanced_Gate3x3',
          'DecoupledFreqGuidedFusion_PMDA_Enhanced_ExpA',
          'DecoupledFreqGuidedFusion_PMDA_Enhanced_Refine',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_MDAAv2', 'DecoupledFreqGuidedFusion_PMDA_ExpA_Soft',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_MultiScale',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_LearnAttn',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_SymPMDA',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_OffsetReg',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_SymFreq',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_AlignLoss',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_CycleLoss',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_AlignV2',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_Contrastive',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_AlignBox',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveV2',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_DualContrast',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_SimCLR',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveLite',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveMoreNeg',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveAllLevels',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_CrossAttn',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_LearnFreq',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_Uncertainty',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_NeckFuse',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_DynamicGroup',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_DCNAlign',
          'DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveNeckFuse',
          'DecoupledFreqGuidedFusion_PMDA_Final_LearnTau',
          'DecoupledFreqGuidedFusion_PMDA_Final_DeepSE',
          'DecoupledFreqGuidedFusion_PMDA_Final_MultiLevel',
          'DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauDeepSE',
          'DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTau05',
          'DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauCos',
          'DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauDeepSECos',
          'DecoupledFreqGuidedFusion_RTDETR_Enhanced',
          'DecoupledFreqGuidedFusion_RTDETR_ExpA',
          'DecoupledFreqGuidedFusion_RTDETR_NeckFuse',
          'DecoupledFreqGuidedFusion_RTDETR_LearnTau',
          'DecoupledFreqGuidedFusion_RTDETR_LearnTauDeepSE']


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        x_reshaped = x.reshape(B * self.groups, -1, H, W)
        return F.grid_sample(x_reshaped, coords.type_as(x_reshaped), mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class SPDConv(nn.Module):

    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x


class FFM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):

        orig_dtype = x.dtype
        
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)


        x2_32 = x2.to(torch.float32)
        x2_fft = torch.fft.fft2(x2_32, norm='backward')


        x1_32 = x1.to(torch.float32)
        out = x1_32 * x2_fft


        out = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
        

        out = torch.abs(out).to(orig_dtype)

        return out * self.alpha + x * self.beta


class ImprovedFFTKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.SiLU()


        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)



        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ffm = FFM(dim)


        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        orig_dtype = x.dtype
        out = self.in_conv(x)
        
        x_att = self.fac_conv(self.fac_pool(out))
        

        out_32 = out.to(torch.float32)
        x_fft = torch.fft.fft2(out_32, norm='backward')
        
        x_att_32 = x_att.to(torch.float32)
        x_fft = x_att_32 * x_fft
        
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca).to(orig_dtype) 



        x_sca1 = self.conv1x1(x_fca)
        x_sca2 = self.conv3x3(x_fca)
        x_sca3 = self.conv5x5(x_fca)
        x_sca = x_sca1 + x_sca2 + x_sca3

        channel_weights = self.channel_attention(x_att)
        x_sca = x_sca * channel_weights

        x_sca = self.ffm(x_sca)

        out = x + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


class MFFF(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = ImprovedFFTKernel(int(dim * self.e))

    def forward(self, x):
        c1 = round(x.size(1) * self.e)
        c2 = x.size(1) - c1
        ok_branch, identity = torch.split(self.cv1(x), [c1, c2], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))


class ADown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class FrequencyFocusedDownSampling(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.ffm = FFM(self.c)


        self.conv_reduce = Conv(self.c * 2, self.c, 1, 1)


        self.conv_resize = Conv(self.c, self.c, 3, 2, 1)


    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)


        fgm_out = self.ffm(x2)
        fgm_out = self.conv_resize(fgm_out)
        pooled_out = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        pooled_out = self.cv2(pooled_out)


        x2 = torch.cat((fgm_out, pooled_out), 1)


        x2 = self.conv_reduce(x2)

        return torch.cat((x1, x2), 1)

class FrequencyFocusedDownSampling2(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.c_in = c1 // 2
        self.c_out = c2 // 2
        

        self.cv1 = Conv(self.c_in, self.c_out, 3, 2, 1)
        

        self.cv2 = Conv(self.c_in, self.c_out, 1, 1, 0)
        

        self.ffm = FFM(self.c_in)  


        self.conv_resize = Conv(self.c_in, self.c_out, 3, 2, 1)


        self.conv_reduce = Conv(self.c_out * 2, self.c_out, 1, 1)

    def forward(self, x):

        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        

        x1, x2 = x.chunk(2, 1)
        

        x1 = self.cv1(x1)


        fgm_out = self.ffm(x2)  
        fgm_out = self.conv_resize(fgm_out)
        
        pooled_out = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        pooled_out = self.cv2(pooled_out)


        x2 = torch.cat((fgm_out, pooled_out), 1)


        x2 = self.conv_reduce(x2)


        return torch.cat((x1, x2), 1)
    
class SemanticAlignmenCalibration(nn.Module):
    def __init__(self, inc):
        super(SemanticAlignmenCalibration, self).__init__()
        hidden_channels = inc[0]

        self.groups = 2
        self.spatial_conv = Conv(inc[0], hidden_channels, 3)
        self.semantic_conv = Conv(inc[1], hidden_channels, 3)


        self.frequency_enhancer = FFM(hidden_channels)

        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)


        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64),
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()
        self.offset_conv[1].weight.data.zero_()

    def init_weights(self):

        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        coarse_features, semantic_features = x
        batch_size, _, out_h, out_w = coarse_features.size()


        semantic_features = self.semantic_conv(semantic_features)
        semantic_features = F.interpolate(semantic_features, coarse_features.size()[2:], mode='bilinear',
                                          align_corners=True)


        enhanced_frequency = self.frequency_enhancer(semantic_features)


        gate = torch.sigmoid(self.gating_conv(semantic_features))
        fused_features = semantic_features * (1 - gate) + enhanced_frequency * gate


        coarse_features = self.spatial_conv(coarse_features)


        conv_results = self.offset_conv(torch.cat([coarse_features, fused_features], 1))


        fused_features = fused_features.reshape(batch_size * self.groups, -1, out_h, out_w)
        coarse_features = coarse_features.reshape(batch_size * self.groups, -1, out_h, out_w)


        offset_low = conv_results[:, 0:self.groups * 2, :, :].reshape(batch_size * self.groups, -1, out_h, out_w)
        offset_high = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(batch_size * self.groups, -1,
                                                                                     out_h, out_w)


        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(fused_features).to(fused_features.device)
        grid_w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        grid_h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        base_grid = torch.cat((grid_h.unsqueeze(2), grid_w.unsqueeze(2)), 2)
        base_grid = base_grid.repeat(batch_size * self.groups, 1, 1, 1).type_as(fused_features).to(
            fused_features.device)


        adjusted_grid_l = base_grid + offset_low.permute(0, 2, 3, 1) / normalization_factors
        adjusted_grid_h = base_grid + offset_high.permute(0, 2, 3, 1) / normalization_factors


        coarse_features = F.grid_sample(coarse_features, adjusted_grid_l.type_as(coarse_features), align_corners=True)
        fused_features = F.grid_sample(fused_features, adjusted_grid_h.type_as(fused_features), align_corners=True)


        coarse_features = coarse_features.reshape(batch_size, -1, out_h, out_w)
        fused_features = fused_features.reshape(batch_size, -1, out_h, out_w)


        attention_weights = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        final_features = fused_features * attention_weights[:, 0:1, :, :] + coarse_features * attention_weights[:, 1:2,
                                                                                              :, :]

        return final_features




import pywt


class WaveletTransform(nn.Module):

    '''功能：初始化小波基函数和滤波器系数
        参数说明：
            wavelet: 小波基名称（'haar','db4','bior2.2'等）
            mode: 边界处理模式（'zero','symmetric'等）
        关键技术点：
            使用pywt.Wavelet获取标准小波系数
            将滤波器转换为可训练参数nn.Parameter，支持梯度传播
            滤波器形状调整为(1,1,N)，适配卷积核维度'''

    def __init__(self, wavelet='haar', mode='zero'):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode


        coeffs = pywt.Wavelet(wavelet).filter_bank

        self.dec_lo = nn.Parameter(
            torch.tensor(coeffs[0], dtype=torch.float32).view(1, 1, 1, -1)
        )
        self.dec_hi = nn.Parameter(
            torch.tensor(coeffs[1], dtype=torch.float32).view(1, 1, 1, -1)
        )


        self.rec_lo = nn.Parameter(
            torch.tensor(coeffs[2], dtype=torch.float32).view(1, 1, 1, -1)
        )
        self.rec_hi = nn.Parameter(
            torch.tensor(coeffs[3], dtype=torch.float32).view(1, 1, 1, -1)
        )
        self.enhance_mode = None
        self.cbam = None
        self.shuffle = ChannelShuffle()

    def forward(self, x, inverse=False, enhance_mode=None):




        self.enhance_mode = enhance_mode
        if not inverse:
            return self.dwt2d(x)
        else:
            return self.idwt2d(x)

    def dwt2d(self, x, enhance_mode=None):

        B, C, H, W = x.shape
        x = x.view(-1, 1, H, W)


        lo_row = nn.functional.conv2d(
            x,
            self.dec_lo,
            padding=(0, (self.dec_lo.shape[-1] - 1) // 2),
            stride=(1, 2)
        )
        '''(1) self.dec_lo 的维度**
                    假设小波滤波器 self.dec_lo 的初始形状为 (1, 1, N)，其中：
                    N 是小波滤波器的长度（例如 Haar 小波的 N=2，DB4 小波的 N=8）
                    (1, 1, N)表示 (out_channels=1, in_channels=1, kernel_width=N)
                    在二维卷积中，conv2d 要求卷积核的维度为 (out_channels, in_channels, kernel_height, kernel_width)。
                    因此，代码中可能通过 unsqueeze 或 view 将其调整为 (1, 1, 1, N)，即：
                        kernel_height=1**：仅在宽度方向（水平）进行滤波
                        kernel_width=N：滤波器覆盖的宽度范围'''
        '''(2)padding=(0, pad)**
                参数格式：(padding_height, padding_width)
                计算方式：pad = (self.dec_lo.shape[-1] - 1) // 2  # 滤波器宽度为N时，pad = (N-1)//2
                物理意义：
                    左侧不填充：padding_height=0（垂直方向不填充）
                    右侧填充 pad：padding_width=pad（水平方向右侧填充 pad 列）
                目的：
                    保持输出宽度为输入宽度的一半（配合 stride=(1,2) 实现下采样）
                    避免因卷积核长度导致的边界信息丢失'''
        '''(3) stride=(1, 2)**
                参数格式：(stride_height, stride_width)
                物理意义：
                    垂直方向步长 1：保持高度不变，逐行扫描
                    水平方向步长 2：每移动一次滤波器，跳过 2 列，实现宽度下采样（输出宽度 ≈ 输入宽度/2）
                作用：
                    在水平方向进行下采样，将特征图宽度压缩为原来的一半
                    保留垂直方向的全分辨率，供后续列滤波使用'''

        lo_row = nn.functional.conv2d(x, self.dec_lo, padding=(0, (self.dec_lo.shape[-1] - 1) // 2), stride=(1, 2))
        hi_row = nn.functional.conv2d(x, self.dec_hi, padding=(0, (self.dec_hi.shape[-1] - 1) // 2), stride=(1, 2))


        LL = nn.functional.conv2d(lo_row, self.dec_lo.permute(0, 1, 3, 2),
                                  padding=((self.dec_lo.shape[-1] - 1) // 2, 0), stride=(2, 1))
        LH = nn.functional.conv2d(lo_row, self.dec_hi.permute(0, 1, 3, 2),
                                  padding=((self.dec_hi.shape[-1] - 1) // 2, 0), stride=(2, 1))
        HL = nn.functional.conv2d(hi_row, self.dec_lo.permute(0, 1, 3, 2),
                                  padding=((self.dec_lo.shape[-1] - 1) // 2, 0), stride=(2, 1))
        HH = nn.functional.conv2d(hi_row, self.dec_hi.permute(0, 1, 3, 2),
                                  padding=((self.dec_hi.shape[-1] - 1) // 2, 0), stride=(2, 1))
        '''torch.cat([LL, LH, HL, HH], dim=1) 
            通道合并：torch.cat 是沿指定维度拼接张量，不进行数值加和。
            具体操作：将四个子带张量 LL, LH, HL, HH 沿通道维度（dim=1）拼接。
                    输入每个子带的形状：(B, C, H//2, W//2)（假设原始输入为 (B, C, H, W)）
                    拼接后的形状：(B, 4*C, H//2, W//2)（通道数变为原来的4倍）'''

        '''二维离散小波变换（DWT）的四个子带：
        LL (Low-Low)：
            含义：行和列均经过低通滤波的低频近似分量。
            特征：保留图像的主体结构和平滑区域。
        LH (Low-High)：
        含义：行低通 + 列高通滤波的水平细节分量。
        特征：捕捉垂直方向的高频信息（如水平边缘）。
        HL (High-Low)：
            含义：行高通 + 列低通滤波的垂直细节分量。
            特征：捕捉水平方向的高频信息（如垂直边缘）。
            HH (High-High)：
            含义：行和列均经过高通滤波的对角线细节分量。
            特征：捕捉对角线方向的高频信息（如纹理和噪声）。'''
        '''view(B, C * 4, H // 2, W // 2) 的意义与原因
            目的：将拼接后的多通道数据整理为标准的四维张量格式。
            操作解析：
                输入形状：拼接后的张量形状为 (B, 4*C, H//2, W//2)。
                **view 的作用**：显式声明维度，确保数据排列符合后续网络层的输入要求。
            必要性：
                通道扩展：小波变换将原始通道数 C 分解为 4C，通过 view 明确通道维度的扩展。
                空间下采样：每个子带的长宽为原图的一半（H//2, W//2），符合下采样后的特征图尺寸。
                兼容性：使输出张量可直接输入到卷积层等模块，无需额外调整维度。'''

        LL = LL.view(B, C, H // 2, W // 2)
        LH = LH.view(B, C, H // 2, W // 2)
        HL = HL.view(B, C, H // 2, W // 2)
        HH = HH.view(B, C, H // 2, W // 2)

        return self._process_high_freq(LL, LH, HL, HH, B, C, H // 2, W // 2, enhance_mode)

    '''操作流程：
            列重构：用逆滤波器对垂直方向进行反卷积上采样
            行重构：对水平方向进行反卷积，合并高低频分量
            恢复尺寸：输出形状恢复为原始尺寸(B, C, H*2, W*2)
        关键技术点：
            使用conv_transpose2d实现插值上采样
            加法操作融合低高频分量
            保持滤波器方向与分解时一致'''

    def _process_high_freq(self, LL, LH, HL, HH, batch, channels, h, w, mode):


        highs = torch.cat([LH, HL, HH], dim=1)
        if self.enhance_mode == 'cbam' and self.cbam is None:
            self.cbam = CBAM(channels=3 * channels).to(LL.device)


        if mode == 'shuffle':
            highs = self.shuffle(highs)
        elif mode == 'cbam':
            highs = self.cbam(highs)


        LH, HL, HH = torch.chunk(highs, 3, dim=1)


        return torch.cat([
            LL.view(batch, channels, h, w),
            LH, HL, HH], dim=1).view(batch, channels * 4, h, w)

    def idwt2d(self, y):
        B, C_total, H, W = y.shape
        C = C_total // 4
        y = y.view(B, C, 4, H, W)


        LL = y[:, :, 0, :, :]
        LH = y[:, :, 1, :, :]
        HL = y[:, :, 2, :, :]
        HH = y[:, :, 3, :, :]


        pad_ver = (self.rec_lo.shape[-1] - 1) // 2
        lo_col = nn.functional.conv_transpose2d(
            LL,
            self.rec_lo.permute(0, 1, 3, 2).expand(C, 1, 6, 1),
            stride=(2, 1),
            padding=(pad_ver, 0),
            groups=C
        ) + nn.functional.conv_transpose2d(
            LH,
            self.rec_hi.permute(0, 1, 3, 2).expand(C, 1, 6, 1),
            stride=(2, 1),
            padding=(pad_ver, 0),
            groups=C
        )

        hi_col = nn.functional.conv_transpose2d(
            HL,
            self.rec_lo.permute(0, 1, 3, 2).expand(C, 1, 6, 1),
            stride=(2, 1),
            padding=(pad_ver, 0),
            groups=C
        ) + nn.functional.conv_transpose2d(
            HH,
            self.rec_hi.permute(0, 1, 3, 2).expand(C, 1, 6, 1),
            stride=(2, 1),
            padding=(pad_ver, 0),
            groups=C
        )


        pad_hor = (self.rec_lo.shape[-1] - 1) // 2
        x = nn.functional.conv_transpose2d(
            lo_col,
            self.rec_lo.expand(C, 1, 1, 6),
            stride=(1, 2),
            padding=(0, pad_hor),
            groups=C
        ) + nn.functional.conv_transpose2d(
            hi_col,
            self.rec_hi.expand(C, 1, 1, 6),
            stride=(1, 2),
            padding=(0, pad_hor),
            groups=C
        )

        return x


class ImprovedWaveletKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()


        self.wavelet = WaveletTransform(wavelet='bior2.2')

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)


        self.wave_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 4, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim * 4, 1),
            nn.Sigmoid()
        )


        self.conv_low = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_mid = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_high = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)


        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.act = nn.SiLU()

    def forward(self, x):

        out = self.in_conv(x)



        wave = self.wavelet(out)


        B, C, H, W = wave.shape
        wave_att = self.wave_att(wave)
        wave_processed = wave * wave_att


        restored = self.wavelet(wave_processed, inverse=True)


        low = self.conv_low(restored)
        mid = self.conv_mid(restored)
        high = self.conv_high(restored)


        channel_weights = self.channel_att(restored)
        fused = (low + mid + high) * channel_weights


        out = x + fused
        return self.act(self.out_conv(out))


class MFFF_W(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = nn.Conv2d(dim, dim, 1)
        self.cv2 = nn.Conv2d(dim, dim, 1)
        self.m = ImprovedWaveletKernel(int(dim * self.e))

    def forward(self, x):
        c1 = round(x.size(1) * self.e)
        c2 = x.size(1) - c1
        ok_branch, identity = torch.split(self.cv1(x), [c1, c2], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))


class ADown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = nn.Conv2d(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = nn.Conv2d(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)






class ChannelShuffle(nn.Module):


    def forward(self, x):
        batch, channels, h, w = x.size()
        groups = 3
        return x.view(batch, groups, -1, h, w).transpose(1, 2).reshape_as(x)



class AsymmetricFreqGuidedFusion(nn.Module):





    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(AsymmetricFreqGuidedFusion, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.frequency_enhancer = FFM(hidden_channels)


        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)



        out_channels = self.groups * 4 + self.groups * 2
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        self.offset_conv[1].weight.data.zero_()

    def forward(self, x):

        rgb_feat, ir_feat = x
        batch_size, _, out_h, out_w = rgb_feat.size()


        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(out_h, out_w), mode='bilinear', align_corners=True)


        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate


        rgb_feat = self.rgb_conv(rgb_feat)


        conv_results = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))


        ir_guided = ir_guided.reshape(batch_size * self.groups, -1, out_h, out_w)
        rgb_feat = rgb_feat.reshape(batch_size * self.groups, -1, out_h, out_w)



        offset_rgb = conv_results[:, 0:self.groups * 2, :, :].reshape(batch_size * self.groups, 2, out_h, out_w)

        offset_ir = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(batch_size * self.groups, 2, out_h,
                                                                                   out_w)


        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(ir_guided).to(ir_guided.device)
        grid_w = torch.linspace(-1.0, 1.0, out_w).view(1, -1).repeat(out_h, 1)
        grid_h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        base_grid = torch.cat((grid_w.unsqueeze(2), grid_h.unsqueeze(2)), dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(batch_size * self.groups, 1, 1, 1).type_as(ir_guided).to(
            ir_guided.device)


        adjusted_grid_rgb = base_grid + offset_rgb.permute(0, 2, 3, 1) / (normalization_factors * 0.5)
        adjusted_grid_ir = base_grid + offset_ir.permute(0, 2, 3, 1) / (normalization_factors * 0.5)


        rgb_aligned = F.grid_sample(rgb_feat, adjusted_grid_rgb.type_as(rgb_feat), align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided, adjusted_grid_ir.type_as(ir_guided), align_corners=True, padding_mode='border')


        rgb_aligned = rgb_aligned.reshape(batch_size, -1, out_h, out_w)
        ir_aligned = ir_aligned.reshape(batch_size, -1, out_h, out_w)



        attention_logits = conv_results[:, self.groups * 4:, :, :]

        attention_weights = 1 + torch.tanh(attention_logits)


        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups,
                                                                              dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups,
                                                                              dim=1)

        final_features = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return final_features


class SymmetricFreqGuidedFusion(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(SymmetricFreqGuidedFusion, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.frequency_enhancer = FFM(hidden_channels)


        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)


        out_channels = self.groups * 4 + self.groups * 2
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()
        self.out_channels = hidden_channels
    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()

    def forward(self, x):

        rgb_feat, ir_feat = x
        batch_size, _, out_h, out_w = rgb_feat.size()


        ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)


        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate


        conv_results = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))


        offset_rgb_raw = conv_results[:, 0:self.groups * 2, :, :]
        offset_ir_raw = conv_results[:, self.groups * 2:self.groups * 4, :, :]


        if self.tiny_mode:

            offset_rgb = torch.zeros_like(offset_rgb_raw)
            offset_ir = offset_ir_raw
            interp_method = 'nearest'
        else:

            offset_rgb = offset_rgb_raw
            offset_ir = offset_ir_raw
            interp_method = 'bilinear'



        ir_guided = ir_guided.reshape(batch_size * self.groups, -1, out_h, out_w)
        rgb_feat = rgb_feat.reshape(batch_size * self.groups, -1, out_h, out_w)

        offset_rgb = offset_rgb.reshape(batch_size * self.groups, 2, out_h, out_w)
        offset_ir = offset_ir.reshape(batch_size * self.groups, 2, out_h, out_w)


        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(ir_guided).to(ir_guided.device)
        grid_w = torch.linspace(-1.0, 1.0, out_w).view(1, -1).repeat(out_h, 1)
        grid_h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        base_grid = torch.cat((grid_w.unsqueeze(2), grid_h.unsqueeze(2)), dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(batch_size * self.groups, 1, 1, 1).type_as(ir_guided).to(ir_guided.device)

        adjusted_grid_rgb = base_grid + offset_rgb.permute(0, 2, 3, 1) / (normalization_factors * 0.5)
        adjusted_grid_ir = base_grid + offset_ir.permute(0, 2, 3, 1) / (normalization_factors * 0.5)


        rgb_aligned = F.grid_sample(rgb_feat, adjusted_grid_rgb.type_as(rgb_feat), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided, adjusted_grid_ir.type_as(ir_guided), mode=interp_method, align_corners=True, padding_mode='border')


        rgb_aligned = rgb_aligned.reshape(batch_size, -1, out_h, out_w)
        ir_aligned = ir_aligned.reshape(batch_size, -1, out_h, out_w)


        attention_logits = conv_results[:, self.groups * 4:, :, :]
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        final_features = ir_aligned * attn_ir + rgb_aligned * attn_rgb
        return final_features

class SymmetricFreqGuidedFusion_new(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(SymmetricFreqGuidedFusion_new, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)


        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )


        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),  
            nn.Sigmoid()
        )


        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        ir_feat = self.ir_conv(ir_feat)
        






        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)

        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        if self.tiny_mode:
            offset_rgb = torch.zeros_like(offsets[:, 0:self.groups * 2, :, :])
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'nearest'
        else:
            offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'bilinear'

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out

class DecoupledFreqGuidedFusion(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(DecoupledFreqGuidedFusion, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        

        self.tiny_mode = tiny_mode  

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)

        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))


        if self.tiny_mode:

            offset_rgb = torch.zeros_like(offsets[:, 0:self.groups * 2, :, :])
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'nearest'
        else:

            offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'bilinear'


        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)


        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out



class DecoupledFreqGuidedFusion_HFBypass(nn.Module):



    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(DecoupledFreqGuidedFusion_HFBypass, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),  
            nn.Sigmoid()
        )

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        

        if self.tiny_mode:
            self.residual_scale = nn.Parameter(torch.ones(1))

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)
        

        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)

        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))


        if self.tiny_mode:

            offset_rgb = torch.zeros_like(offsets[:, 0:self.groups * 2, :, :])
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'nearest'
        else:

            offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'bilinear'


        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)


        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)


        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb


        if self.tiny_mode:



            high_freq_residual = ir_freq * edge_mask * self.residual_scale
            fused_out = fused_out + high_freq_residual


        return fused_out

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


from .dyt import DyT
from .FCM_FFN import ConvFFN_GLU
from .mine import FFM

class DyT(nn.Module):

    def __init__(self, channels: int, alpha_init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(self.alpha * x)
        return out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv



class DecoupledFreqGuidedFusion_Pro_Safe(nn.Module):






    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(DecoupledFreqGuidedFusion_Pro_Safe, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels


        self.rgb_align = nn.Conv2d(c_rgb, hidden_channels, 1, bias=False) if c_rgb != hidden_channels else nn.Identity()
        self.ir_align = nn.Conv2d(c_ir, hidden_channels, 1, bias=False) if c_ir != hidden_channels else nn.Identity()

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )


        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            DyT(64), 
            nn.SiLU(inplace=True), 
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        
        self.spatial_refiner = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

        self.post_ffn = ConvFFN_GLU(in_channels=hidden_channels, out_channels=hidden_channels, expand=2)

        self.out_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        



        nn.init.normal_(self.offset_conv[1].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fusion_attn_conv[3].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        rgb_safe = self.rgb_align(rgb_feat)
        ir_safe = self.ir_align(ir_feat)

        ir_safe_resized = F.interpolate(ir_safe, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_safe_resized)
        gate = torch.sigmoid(self.gating_conv(ir_safe_resized))
        ir_guided = ir_safe_resized * (1 - gate) + ir_freq * gate

        edge_mask = self.edge_mask_gen(ir_freq) 



        offsets_raw = self.offset_conv(torch.cat([rgb_safe, ir_guided], dim=1))
        offsets = torch.tanh(offsets_raw) * 5.0
        
        offset_rgb = (offsets[:, 0:self.groups * 2, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)
        offset_ir = (offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        

        rgb_safe_shaped = rgb_safe.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_safe_shaped, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_safe_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        
        ir_guided_shaped = ir_guided.reshape(B * self.groups, -1, H, W)
        ir_aligned = F.grid_sample(ir_guided_shaped, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)


        ir_spatial_mask = self.spatial_refiner(ir_freq)
        rgb_refined = rgb_aligned * ir_spatial_mask + rgb_aligned 

        fusion_input = torch.cat([rgb_refined, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        
        attention_weights = 1 + torch.tanh(attention_logits)
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_refined * attn_rgb
        ffn_out = self.post_ffn(fused_out)


        return rgb_safe + self.out_proj(ffn_out)

class SymmetricFreqGuidedFusion_attn(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2, attn_type='default'):
        super(SymmetricFreqGuidedFusion_attn, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)

        out_channels = self.groups * 4 + self.groups * 2
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False)
        )



        if attn_type == 'default' or attn_type is None:
            self.plug_attn = nn.Identity()
        else:
            self.plug_attn = eval(attn_type)(hidden_channels)


        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()

    def forward(self, x):

        rgb_feat, ir_feat = x
        batch_size, _, out_h, out_w = rgb_feat.size()

        ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)

        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        conv_results = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        offset_rgb_raw = conv_results[:, 0:self.groups * 2, :, :]
        offset_ir_raw = conv_results[:, self.groups * 2:self.groups * 4, :, :]

        if self.tiny_mode:
            offset_rgb = torch.zeros_like(offset_rgb_raw)
            offset_ir = offset_ir_raw
            interp_method = 'nearest'
        else:
            offset_rgb = offset_rgb_raw
            offset_ir = offset_ir_raw
            interp_method = 'bilinear'

        ir_guided = ir_guided.reshape(batch_size * self.groups, -1, out_h, out_w)
        rgb_feat = rgb_feat.reshape(batch_size * self.groups, -1, out_h, out_w)

        offset_rgb = offset_rgb.reshape(batch_size * self.groups, 2, out_h, out_w)
        offset_ir = offset_ir.reshape(batch_size * self.groups, 2, out_h, out_w)

        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(ir_guided).to(ir_guided.device)
        grid_w = torch.linspace(-1.0, 1.0, out_w).view(1, -1).repeat(out_h, 1)
        grid_h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        base_grid = torch.cat((grid_w.unsqueeze(2), grid_h.unsqueeze(2)), dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(batch_size * self.groups, 1, 1, 1).type_as(ir_guided).to(ir_guided.device)

        adjusted_grid_rgb = base_grid + offset_rgb.permute(0, 2, 3, 1) / (normalization_factors * 0.5)
        adjusted_grid_ir = base_grid + offset_ir.permute(0, 2, 3, 1) / (normalization_factors * 0.5)

        rgb_aligned = F.grid_sample(rgb_feat, adjusted_grid_rgb.type_as(rgb_feat), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided, adjusted_grid_ir.type_as(ir_guided), mode=interp_method, align_corners=True, padding_mode='border')

        rgb_aligned = rgb_aligned.reshape(batch_size, -1, out_h, out_w)
        ir_aligned = ir_aligned.reshape(batch_size, -1, out_h, out_w)

        attention_logits = conv_results[:, self.groups * 4:, :, :]
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        final_features = ir_aligned * attn_ir + rgb_aligned * attn_rgb



        return self.plug_attn(final_features)


class DecoupledFreqGuidedFusion_attn(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2, attn_type='default'):
        super(DecoupledFreqGuidedFusion_attn, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),  
            nn.Sigmoid()
        )



        if attn_type == 'default' or attn_type is None:
            self.fusion_attn_conv = nn.Sequential(
                nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
            )
        else:

            if attn_type == 'CoordAtt':

                plug_module = eval(attn_type)(hidden_channels, hidden_channels)
            else:
                plug_module = eval(attn_type)(hidden_channels)
                
            self.fusion_attn_conv = nn.Sequential(

                nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),

                plug_module,

                nn.Conv2d(hidden_channels, self.groups * 2, kernel_size=3, padding=1, bias=False)
            )


        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        
        if isinstance(self.fusion_attn_conv[-1], nn.Conv2d):
            self.fusion_attn_conv[-1].weight.data.zero_()

    def forward(self, x):

        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)

        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        if self.tiny_mode:
            offset_rgb = torch.zeros_like(offsets[:, 0:self.groups * 2, :, :])
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'nearest'
        else:
            offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'bilinear'

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        

        attention_logits = self.fusion_attn_conv(fusion_input)
        
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out

class FocusH(nn.Module):
    def __init__(self, c1, c2, kernel=3, stride=1):
        super().__init__()
        self.c2 = c2
        self.conv1 = Conv(c1, c2, kernel, stride)
        self.conv2 = Conv(c1, c2, kernel, stride)

    def forward(self, x):
        b, _, h, w = x.shape
        result = torch.zeros(size=[b, self.c2, h, w], device=x.device, dtype=x.dtype)
        x1 = torch.zeros(size=[b, self.c2, h, w // 2], device=x.device, dtype=x.dtype)
        x2 = torch.zeros(size=[b, self.c2, h, w // 2], device=x.device, dtype=x.dtype)
        x1[..., ::2, :], x1[..., 1::2, :] = x[..., ::2, ::2], x[..., 1::2, 1::2]
        x2[..., ::2, :], x2[..., 1::2, :] = x[..., ::2, 1::2], x[..., 1::2, ::2]
        x1, x2 = self.conv1(x1), self.conv2(x2)
        result[..., ::2, ::2] = x1[..., ::2, :]
        result[..., 1::2, 1::2] = x1[..., 1::2, :]
        result[..., ::2, 1::2] = x2[..., ::2, :]
        result[..., 1::2, ::2] = x2[..., 1::2, :]
        return result

class FocusV(nn.Module):
    def __init__(self, c1, c2, kernel=3, stride=1):
        super().__init__()
        self.c2 = c2
        self.conv1 = Conv(c1, c2, kernel, stride)
        self.conv2 = Conv(c1, c2, kernel, stride)

    def forward(self, x):
        b, _, h, w = x.shape
        result = torch.zeros(size=[b, self.c2, h, w], device=x.device, dtype=x.dtype)
        x1 = torch.zeros(size=[b, self.c2, h // 2, w], device=x.device, dtype=x.dtype)
        x2 = torch.zeros(size=[b, self.c2, h // 2, w], device=x.device, dtype=x.dtype)
        x1[..., ::2], x1[..., 1::2] = x[..., ::2, ::2], x[..., 1::2, 1::2]
        x2[..., ::2], x2[..., 1::2] = x[..., 1::2, ::2], x[..., ::2, 1::2]
        x1, x2 = self.conv1(x1), self.conv2(x2)
        result[..., ::2, ::2] = x1[..., ::2]
        result[..., 1::2, 1::2] = x1[..., 1::2]
        result[..., 1::2, ::2] = x2[..., ::2]
        result[..., ::2, 1::2] = x2[..., 1::2]
        return result

class BiFocus(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.focus_h = FocusH(c1, c1, 3, 1)
        self.focus_v = FocusV(c1, c1, 3, 1)

        self.depth_conv = Conv(3 * c1, 3 * c1, 3, 1, 1, g=3 * c1)
        self.point_conv = Conv(3 * c1, c2, 1, 1, 0)

    def forward(self, x):
        out = torch.cat([x, self.focus_h(x), self.focus_v(x)], dim=1)
        return self.point_conv(self.depth_conv(out))


class DecoupledFreqGuidedFusion_BiFocus(nn.Module):
    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, self.groups * 4, kernel_size=3, padding=1, bias=False)
        )

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        

        self.bifocus = BiFocus(hidden_channels, hidden_channels)

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape


        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)

        rgb_feat = self.rgb_conv(rgb_feat)
        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        offset_rgb = (offsets[:, 0:self.groups * 2, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)
        offset_ir = (offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_feat_shaped = rgb_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_feat_shaped, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        ir_guided_shaped = ir_guided.reshape(B * self.groups, -1, H, W)
        ir_aligned = F.grid_sample(ir_guided_shaped, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)


        attention_logits = self.fusion_attn_conv(torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1))
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb


        return self.bifocus(fused_out)

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv



class DecoupledFreqGuidedFusion_FDFEF(nn.Module):
    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, self.groups * 4, kernel_size=3, padding=1, bias=False)
        )

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )


        self.alpha_rgb = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0.5)
        self.alpha_ir = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0.5)
        
        self.beta_rgb = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0.5)
        self.beta_ir = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0.5)


        self.out_conv = Conv(hidden_channels, hidden_channels, 3)

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape


        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)

        rgb_feat = self.rgb_conv(rgb_feat)
        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        offset_rgb = (offsets[:, 0:self.groups * 2, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)
        offset_ir = (offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0


        rgb_feat_shaped = rgb_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_feat_shaped, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        
        ir_guided_shaped = ir_guided.reshape(B * self.groups, -1, H, W)
        ir_aligned = F.grid_sample(ir_guided_shaped, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)





        orig_dtype = rgb_aligned.dtype
        rgb_aligned_f32 = rgb_aligned.to(torch.float32)
        ir_aligned_f32 = ir_aligned.to(torch.float32)

        f_rgb = torch.fft.rfft2(rgb_aligned_f32, norm='ortho')
        f_ir = torch.fft.rfft2(ir_aligned_f32, norm='ortho')

        amp_rgb, amp_ir = torch.abs(f_rgb), torch.abs(f_ir)
        phase_rgb, phase_ir = torch.angle(f_rgb), torch.angle(f_ir)


        alpha_rgb_f32 = self.alpha_rgb.to(torch.float32)
        alpha_ir_f32 = self.alpha_ir.to(torch.float32)
        beta_rgb_f32 = self.beta_rgb.to(torch.float32)
        beta_ir_f32 = self.beta_ir.to(torch.float32)

        amp_fused = alpha_rgb_f32 * amp_rgb + alpha_ir_f32 * amp_ir
        phase_fused = beta_rgb_f32 * phase_rgb + beta_ir_f32 * phase_ir

        f_recon = amp_fused * torch.exp(1j * phase_fused)

        fused_spatial_f32 = torch.fft.irfft2(f_recon, s=(H, W), norm='ortho')


        fused_spatial = fused_spatial_f32.to(orig_dtype)


        return self.out_conv(fused_spatial)

class HighFrequencyPerception(nn.Module):

    def __init__(self, c, ratio: tuple[float, float] = (0.25, 0.25), patch: tuple[int, int] = (8, 8), groups: int = 32) -> None:
        super().__init__()
        self.ratio = ratio
        self.ph, self.pw = int(patch[0]), int(patch[1])
        g = max(1, min(int(groups), c))
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.channel_conv1 = nn.Conv2d(c, c, kernel_size=1, groups=g)
        self.channel_conv2 = nn.Conv2d(c, c, kernel_size=1, groups=g)
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, c), num_channels=c),
        )

    def _mask_fft(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        



        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        

        xf = torch.fft.rfft2(x_f32, dim=(-2, -1))
        h0 = int(H * self.ratio[0])
        w0 = int((W // 2 + 1) * self.ratio[1])
        mask = torch.ones_like(xf, dtype=xf.dtype)
        mask[:, :, :h0, :w0] = 0
        xf = xf * mask
        xh = torch.fft.irfft2(xf, s=(H, W))
        

        return xh.to(orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf = self._mask_fft(x)
        spa = self.spatial_conv(hf) * x
        amax = torch.nn.functional.adaptive_max_pool2d(hf, output_size=(self.ph, self.pw))
        aavg = torch.nn.functional.adaptive_avg_pool2d(hf, output_size=(self.ph, self.pw))
        amax = torch.sum(torch.relu(amax), dim=(2, 3), keepdim=True)
        aavg = torch.sum(torch.relu(aavg), dim=(2, 3), keepdim=True)
        ch = self.channel_conv1(amax) + self.channel_conv1(aavg)
        ch = torch.sigmoid(self.channel_conv2(ch))
        cha = ch * x
        return self.out_conv(spa + cha)


class DecoupledFreqGuidedFusion_HFP(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.rgb_align = nn.Conv2d(c_rgb, hidden_channels, 1, bias=False) if c_rgb != hidden_channels else nn.Identity()
        self.ir_align = nn.Conv2d(c_ir, hidden_channels, 1, bias=False) if c_ir != hidden_channels else nn.Identity()

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, self.groups * 4, kernel_size=3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, 1, 7, padding=3), nn.Sigmoid())

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.SiLU(inplace=True), 
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )


        self.hfp_refiner = HighFrequencyPerception(hidden_channels)
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        rgb_safe = self.rgb_align(rgb_feat)
        ir_safe = self.ir_align(ir_feat)

        ir_safe_resized = F.interpolate(ir_safe, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_safe_resized)
        gate = torch.sigmoid(self.gating_conv(ir_safe_resized))
        ir_guided = ir_safe_resized * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq) 

        offsets_raw = self.offset_conv(torch.cat([rgb_safe, ir_guided], dim=1))
        offsets = torch.tanh(offsets_raw) * 5.0 
        
        offset_rgb = (offsets[:, 0:self.groups * 2, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)
        offset_ir = (offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        
        rgb_safe_shaped = rgb_safe.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_safe_shaped, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_safe_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        
        ir_guided_shaped = ir_guided.reshape(B * self.groups, -1, H, W)
        ir_aligned = F.grid_sample(ir_guided_shaped, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)

        attention_logits = self.fusion_attn_conv(torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1))
        attention_weights = 1 + torch.tanh(attention_logits)
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb
        

        refined_out = self.hfp_refiner(fused_out)


        return rgb_safe + self.gamma * refined_out

class GCB(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)
        self.conv_attn = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.LayerNorm([mid_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        attn = self.conv_attn(x).view(B, 1, -1)
        attn = F.softmax(attn, dim=-1)  
        x_flat = x.view(B, C, -1)  
        context = torch.bmm(x_flat, attn.transpose(1, 2)).view(B, C, 1, 1)  
        context = self.bottleneck(context)  

        return x + context 


class DecoupledFreqGuidedFusion_GCB(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.rgb_align = nn.Conv2d(c_rgb, hidden_channels, 1, bias=False) if c_rgb != hidden_channels else nn.Identity()
        self.ir_align = nn.Conv2d(c_ir, hidden_channels, 1, bias=False) if c_ir != hidden_channels else nn.Identity()


        self.gcb_rgb = GCB(hidden_channels)
        self.gcb_ir = GCB(hidden_channels)

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, self.groups * 4, kernel_size=3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, 1, 7, padding=3), nn.Sigmoid())

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.SiLU(inplace=True), 
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        rgb_safe = self.rgb_align(rgb_feat)
        ir_safe = self.ir_align(ir_feat)


        rgb_safe = self.gcb_rgb(rgb_safe)
        ir_safe = self.gcb_ir(ir_safe)

        ir_safe_resized = F.interpolate(ir_safe, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_safe_resized)
        gate = torch.sigmoid(self.gating_conv(ir_safe_resized))
        ir_guided = ir_safe_resized * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq) 


        offsets_raw = self.offset_conv(torch.cat([rgb_safe, ir_guided], dim=1))
        offsets = torch.tanh(offsets_raw) * 5.0 
        
        offset_rgb = (offsets[:, 0:self.groups * 2, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)
        offset_ir = (offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        
        rgb_safe_shaped = rgb_safe.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_safe_shaped, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_safe_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        
        ir_guided_shaped = ir_guided.reshape(B * self.groups, -1, H, W)
        ir_aligned = F.grid_sample(ir_guided_shaped, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)

        attention_logits = self.fusion_attn_conv(torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1))
        attention_weights = 1 + torch.tanh(attention_logits)
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb
        
        return rgb_safe + self.gamma * fused_out

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv





class DConv(nn.Module):

    def __init__(self, c1: int, alpha: float = 0.8, atoms: int = 512) -> None:
        super().__init__()
        self.alpha = float(alpha)

        self.CG = nn.Conv2d(c1, atoms, 1, bias=False)

        self.GIE = nn.Conv2d(atoms, atoms, 5, padding=2, groups=atoms, bias=False)

        self.D = nn.Conv2d(atoms, c1, 1, bias=False)

    @staticmethod
    def _pono(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + eps)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = self.CG(r)
        x = self.GIE(x)
        x = self._pono(x)
        x = self.D(x)

        return self.alpha * x + (1.0 - self.alpha) * r


class DecoupledFreqGuidedFusion_RD(nn.Module):
    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.rgb_align = nn.Conv2d(c_rgb, hidden_channels, 1, bias=False) if c_rgb != hidden_channels else nn.Identity()
        self.ir_align = nn.Conv2d(c_ir, hidden_channels, 1, bias=False) if c_ir != hidden_channels else nn.Identity()

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, self.groups * 4, kernel_size=3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, 1, 7, padding=3), nn.Sigmoid())

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            DyT(64), 
            nn.SiLU(inplace=True), 
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )


        self.dconv_refiner = DConv(hidden_channels, alpha=0.8, atoms=hidden_channels * 2)
        self.gamma = nn.Parameter(torch.zeros(1)) 

        self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.offset_conv[1].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fusion_attn_conv[3].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        rgb_safe = self.rgb_align(rgb_feat)
        ir_safe = self.ir_align(ir_feat)

        ir_safe_resized = F.interpolate(ir_safe, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_safe_resized)
        gate = torch.sigmoid(self.gating_conv(ir_safe_resized))
        ir_guided = ir_safe_resized * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq) 

        offsets_raw = self.offset_conv(torch.cat([rgb_safe, ir_guided], dim=1))
        offsets = torch.tanh(offsets_raw) * 5.0 
        
        offset_rgb = (offsets[:, 0:self.groups * 2, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)
        offset_ir = (offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        
        rgb_safe_shaped = rgb_safe.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_safe_shaped, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_safe_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        
        ir_guided_shaped = ir_guided.reshape(B * self.groups, -1, H, W)
        ir_aligned = F.grid_sample(ir_guided_shaped, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)

        attention_logits = self.fusion_attn_conv(torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1))
        attention_weights = 1 + torch.tanh(attention_logits)
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb
        

        refined_out = self.dconv_refiner(fused_out)


        return rgb_safe + self.gamma * refined_out

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class IIA(nn.Module):

    def __init__(self, channel: int, kernel_size: int = 7) -> None:
        super().__init__()
        p = kernel_size // 2

        self.conv_h = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, kernel_size), padding=(0, p), bias=False),
            nn.Sigmoid(),
        )

        self.conv_w = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(kernel_size, 1), padding=(p, 0), bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        avg = torch.mean(x, dim=1, keepdim=True)          
        maxv, _ = torch.max(x, dim=1, keepdim=True)       
        pooled = torch.cat([avg, maxv], dim=1)            
        

        attn_h = self.conv_h(pooled)                      
        attn_w = self.conv_w(pooled)                      
        return x + x * attn_h + x * attn_w


class DecoupledFreqGuidedFusion_IIA(nn.Module):
    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.rgb_align = nn.Conv2d(c_rgb, hidden_channels, 1, bias=False) if c_rgb != hidden_channels else nn.Identity()
        self.ir_align = nn.Conv2d(c_ir, hidden_channels, 1, bias=False) if c_ir != hidden_channels else nn.Identity()

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, self.groups * 4, kernel_size=3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, 1, 7, padding=3), nn.Sigmoid())

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            DyT(64), 
            nn.SiLU(inplace=True), 
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )


        self.iia_refiner = IIA(channel=hidden_channels, kernel_size=7)
        self.out_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1)) 

        self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.offset_conv[1].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fusion_attn_conv[3].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        rgb_safe = self.rgb_align(rgb_feat)
        ir_safe = self.ir_align(ir_feat)

        ir_safe_resized = F.interpolate(ir_safe, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_safe_resized)
        gate = torch.sigmoid(self.gating_conv(ir_safe_resized))
        ir_guided = ir_safe_resized * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq) 

        offsets_raw = self.offset_conv(torch.cat([rgb_safe, ir_guided], dim=1))
        offsets = torch.tanh(offsets_raw) * 5.0 
        
        offset_rgb = (offsets[:, 0:self.groups * 2, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)
        offset_ir = (offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask).reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        
        rgb_safe_shaped = rgb_safe.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_safe_shaped, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_safe_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        
        ir_guided_shaped = ir_guided.reshape(B * self.groups, -1, H, W)
        ir_aligned = F.grid_sample(ir_guided_shaped, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)

        attention_logits = self.fusion_attn_conv(torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1))
        attention_weights = 1 + torch.tanh(attention_logits)
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb
        

        refined_out = self.iia_refiner(fused_out)

        return rgb_safe + self.gamma * self.out_proj(refined_out)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k//2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class LAGFusion(nn.Module):





    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(LAGFusion, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)



        self.tap_squeeze = nn.Conv2d(hidden_channels, 1, 1)
        self.tap_excite = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.Sigmoid()
        )


        if not self.tiny_mode:
            out_offset_channels = self.groups * 4
            self.offset_conv = nn.Sequential(
                Conv(hidden_channels * 2, 64, 1),
                nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
            )


        self.fusion_attn = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        

        self.gamma_ir = nn.Parameter(torch.zeros(1))
        self.gamma_rgb = nn.Parameter(torch.zeros(1))

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        if not self.tiny_mode:
            self.offset_conv[1].weight.data.zero_()
        self.fusion_attn[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape


        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_conv(rgb_feat)




        ir_squeeze = self.tap_squeeze(ir_feat)
        local_max = F.max_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        local_avg = F.avg_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        thermal_anomaly_map = F.relu(local_max - local_avg)
        

        tap_gate = self.tap_excite(thermal_anomaly_map)


        if self.tiny_mode:



            

            ir_aligned = ir_feat * tap_gate 
            

            rgb_aligned = rgb_feat + (rgb_feat * tap_gate * self.gamma_rgb)
            
        else:


            offsets = self.offset_conv(torch.cat([rgb_feat, ir_feat], dim=1))
            
            offset_rgb = offsets[:, 0:self.groups * 2, :, :].reshape(B * self.groups, 2, H, W)
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :].reshape(B * self.groups, 2, H, W)

            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                            torch.arange(W, device=ir_feat.device), indexing='ij')
            base_grid = torch.stack((grid_x, grid_y), dim=0).float()
            base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
            normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

            grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
            grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

            rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
            ir_feat_g = ir_feat.reshape(B * self.groups, -1, H, W)

            rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1), mode='bilinear', align_corners=False)
            ir_aligned = F.grid_sample(ir_feat_g, grid_norm_ir.permute(0, 2, 3, 1), mode='bilinear', align_corners=False)

            rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
            ir_aligned = ir_aligned.reshape(B, -1, H, W)


        fusion_input = torch.cat([rgb_aligned, ir_aligned], dim=1)
        attention_logits = self.fusion_attn(fusion_input)
        

        attention_weights = torch.sigmoid(attention_logits) 

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb


        if self.tiny_mode:
            fused_out = fused_out + (ir_feat * tap_gate * self.gamma_ir)

        return fused_out
class HeavyWindowCrossAttention(nn.Module):




    def __init__(self, dim, num_heads=8, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5


        self.q_rgb = nn.Linear(dim, dim)
        self.kv_ir = nn.Linear(dim, dim * 2)
        
        self.q_ir = nn.Linear(dim, dim)
        self.kv_rgb = nn.Linear(dim, dim * 2)

        self.proj_rgb = nn.Linear(dim, dim)
        self.proj_ir = nn.Linear(dim, dim)

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, rgb, ir):
        B, C, H, W = rgb.shape
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        rgb = F.pad(rgb, (0, pad_r, 0, pad_b))
        ir = F.pad(ir, (0, pad_r, 0, pad_b))
        _, _, Hp, Wp = rgb.shape

        rgb = rgb.permute(0, 2, 3, 1) 
        ir = ir.permute(0, 2, 3, 1)

        rgb_win = self.window_partition(rgb, self.window_size) 
        ir_win = self.window_partition(ir, self.window_size)


        q_rgb = self.q_rgb(rgb_win).reshape(-1, self.window_size**2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_ir = self.kv_ir(ir_win).reshape(-1, self.window_size**2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_ir, v_ir = kv_ir[0], kv_ir[1]

        attn_rgb = (q_rgb @ k_ir.transpose(-2, -1)) * self.scale
        attn_rgb = attn_rgb.softmax(dim=-1)
        out_rgb = (attn_rgb @ v_ir).transpose(1, 2).reshape(-1, self.window_size**2, C)
        out_rgb = self.proj_rgb(out_rgb)


        q_ir = self.q_ir(ir_win).reshape(-1, self.window_size**2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_rgb = self.kv_rgb(rgb_win).reshape(-1, self.window_size**2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_rgb, v_rgb = kv_rgb[0], kv_rgb[1]

        attn_ir = (q_ir @ k_rgb.transpose(-2, -1)) * self.scale
        attn_ir = attn_ir.softmax(dim=-1)
        out_ir = (attn_ir @ v_rgb).transpose(1, 2).reshape(-1, self.window_size**2, C)
        out_ir = self.proj_ir(out_ir)

        rgb_fused = self.window_reverse(out_rgb, self.window_size, Hp, Wp)
        ir_fused = self.window_reverse(out_ir, self.window_size, Hp, Wp)

        rgb_fused = rgb_fused[:, :H, :W, :].permute(0, 3, 1, 2).contiguous()
        ir_fused = ir_fused[:, :H, :W, :].permute(0, 3, 1, 2).contiguous()

        return rgb_fused, ir_fused


class HeavyDFGF(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(HeavyDFGF, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.tap_squeeze = nn.Conv2d(hidden_channels, 1, 1)
        self.tap_excite = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.Sigmoid()
        )


        if not self.tiny_mode:
            out_offset_channels = self.groups * 4
            self.offset_conv = nn.Sequential(
                Conv(hidden_channels * 2, 64, 1),
                nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
            )


        self.heavy_cross_attn = HeavyWindowCrossAttention(hidden_channels, num_heads=8, window_size=7)
        

        self.out_conv = Conv(hidden_channels * 2, hidden_channels, 1)


        self.gamma_ir = nn.Parameter(torch.ones(1))
        self.gamma_rgb = nn.Parameter(torch.ones(1))

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape
        current_dtype = rgb_feat.dtype
        current_device = rgb_feat.device

        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_conv(rgb_feat)


        ir_squeeze = self.tap_squeeze(ir_feat)
        local_max = F.max_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        local_avg = F.avg_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        ir_freq_prior = F.relu(local_max - local_avg) 
        freq_gate = self.tap_excite(ir_freq_prior)


        if self.tiny_mode:

            ir_aligned = ir_feat * freq_gate 
            rgb_aligned = rgb_feat + (rgb_feat * freq_gate * self.gamma_rgb)
        else:

            offsets = self.offset_conv(torch.cat([rgb_feat, ir_feat], dim=1))
            offset_rgb = offsets[:, 0:self.groups * 2, :, :].reshape(B * self.groups, 2, H, W)
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :].reshape(B * self.groups, 2, H, W)

            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=current_device),
                                            torch.arange(W, device=current_device), indexing='ij')
            base_grid = torch.stack((grid_x, grid_y), dim=0).to(current_dtype)
            base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
            normalizer = torch.tensor([W - 1, H - 1], device=current_device, dtype=current_dtype).view(1, 2, 1, 1)

            grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
            grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

            rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
            ir_feat_g = ir_feat.reshape(B * self.groups, -1, H, W)

            rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).to(current_dtype), mode='bilinear', align_corners=False).reshape(B, -1, H, W)
            ir_aligned = F.grid_sample(ir_feat_g, grid_norm_ir.permute(0, 2, 3, 1).to(current_dtype), mode='bilinear', align_corners=False).reshape(B, -1, H, W)



        rgb_fused_attn, ir_fused_attn = self.heavy_cross_attn(rgb_aligned, ir_aligned)
        

        fused_out = self.out_conv(torch.cat([rgb_aligned + rgb_fused_attn, ir_aligned + ir_fused_attn], dim=1))


        if self.tiny_mode:

            fused_out = fused_out + (ir_feat * freq_gate * self.gamma_ir)

        return fused_out

class CoordAtt(nn.Module):




    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class DFGF_DWconv_CA(nn.Module):






    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(DFGF_DWconv_CA, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.tap_squeeze = nn.Conv2d(hidden_channels, 1, 1)
        self.tap_excite = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.Sigmoid()
        )


        if not self.tiny_mode:
            out_offset_channels = self.groups * 4

            self.offset_conv = nn.Sequential(
                Conv(hidden_channels * 2, 64, 1),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, out_offset_channels, kernel_size=1, bias=False)
            )


        self.fusion_attn = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        

        self.coord_att = CoordAtt(hidden_channels, hidden_channels)


        self.gamma_ir = nn.Parameter(torch.zeros(1))
        self.gamma_rgb = nn.Parameter(torch.zeros(1))

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        if not self.tiny_mode:

            self.offset_conv[-1].weight.data.zero_()
        self.fusion_attn[-1].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape
        
        current_dtype = rgb_feat.dtype
        current_device = rgb_feat.device

        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_conv(rgb_feat)


        ir_squeeze = self.tap_squeeze(ir_feat)
        local_max = F.max_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        local_avg = F.avg_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        thermal_anomaly_map = F.relu(local_max - local_avg) 
        tap_gate = self.tap_excite(thermal_anomaly_map)

        if self.tiny_mode:

            ir_aligned = ir_feat * tap_gate 
            rgb_aligned = rgb_feat + (rgb_feat * tap_gate * self.gamma_rgb)
        else:

            offsets = self.offset_conv(torch.cat([rgb_feat, ir_feat], dim=1))
            
            offset_rgb = offsets[:, 0:self.groups * 2, :, :].reshape(B * self.groups, 2, H, W)
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :].reshape(B * self.groups, 2, H, W)

            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=current_device),
                                            torch.arange(W, device=current_device), indexing='ij')
            base_grid = torch.stack((grid_x, grid_y), dim=0).to(current_dtype)
            base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
            normalizer = torch.tensor([W - 1, H - 1], device=current_device, dtype=current_dtype).view(1, 2, 1, 1)

            grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
            grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

            rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
            ir_feat_g = ir_feat.reshape(B * self.groups, -1, H, W)

            rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).to(current_dtype), mode='bilinear', align_corners=False)
            ir_aligned = F.grid_sample(ir_feat_g, grid_norm_ir.permute(0, 2, 3, 1).to(current_dtype), mode='bilinear', align_corners=False)

            rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
            ir_aligned = ir_aligned.reshape(B, -1, H, W)


        fusion_input = torch.cat([rgb_aligned, ir_aligned], dim=1)
        attention_weights = torch.sigmoid(self.fusion_attn(fusion_input)) 

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb


        fused_out = self.coord_att(fused_out)

        if self.tiny_mode:
            fused_out = fused_out + (ir_feat * tap_gate * self.gamma_ir)

        return fused_out

class DFGF_BiFocus(nn.Module):






    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(DFGF_BiFocus, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.tap_squeeze = nn.Conv2d(hidden_channels, 1, 1)
        self.tap_excite = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.Sigmoid()
        )


        if not self.tiny_mode:
            out_offset_channels = self.groups * 4
            self.offset_conv = nn.Sequential(
                Conv(hidden_channels * 2, 64, 1),
                nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
                nn.Conv2d(64, out_offset_channels, 1, bias=False)
            )


        self.comp_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels * 2, hidden_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, hidden_channels * 2, 1),
            nn.Softmax(dim=1)
        )
        

        self.refine_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )

        self.gamma_ir = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma_rgb = nn.Parameter(torch.ones(1) * 0.5)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if not self.tiny_mode:
            self.offset_conv[-1].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape
        dtype, device = rgb_feat.dtype, rgb_feat.device

        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_conv(rgb_feat)


        ir_saliency = self.tap_squeeze(ir_feat)
        ir_gate = self.tap_excite(F.max_pool2d(ir_saliency, 3, 1, 1) - F.avg_pool2d(ir_saliency, 3, 1, 1))
        

        ir_feat = ir_feat * (1 + ir_gate) 

        if self.tiny_mode:
            ir_aligned, rgb_aligned = ir_feat, rgb_feat
        else:

            offsets = self.offset_conv(torch.cat([rgb_feat, ir_feat], dim=1))
            offset_rgb = offsets[:, 0:self.groups * 2, :, :].reshape(B * self.groups, 2, H, W)
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :].reshape(B * self.groups, 2, H, W)

            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            base_grid = torch.stack((grid_x, grid_y), dim=0).to(dtype).unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
            norm = torch.tensor([W - 1, H - 1], device=device, dtype=dtype).view(1, 2, 1, 1)

            grid_rgb = (2.0 * (base_grid + offset_rgb) / norm - 1.0).permute(0, 2, 3, 1)
            grid_ir = (2.0 * (base_grid + offset_ir) / norm - 1.0).permute(0, 2, 3, 1)

            rgb_aligned = F.grid_sample(rgb_feat.reshape(B*self.groups, -1, H, W), grid_rgb, align_corners=False).reshape(B, -1, H, W)
            ir_aligned = F.grid_sample(ir_feat.reshape(B*self.groups, -1, H, W), grid_ir, align_corners=False).reshape(B, -1, H, W)



        combined = torch.cat([rgb_aligned, ir_aligned], dim=1)
        weights = self.comp_attn(combined)
        w_rgb, w_ir = torch.split(weights, self.hidden_channels, dim=1)
        
        fused = rgb_aligned * w_rgb + ir_aligned * w_ir
        

        fused = self.refine_conv(fused)
        
        if self.tiny_mode:
            fused = fused + (ir_feat * ir_gate * self.gamma_ir)

        return fused

class GlobalIlluminationEstimator(nn.Module):





    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4, bias=False),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.pool(x).view(b, c)
        illum_score = self.fc(v).view(b, 1, 1, 1) 
        return illum_score

class FFCM(nn.Module):




    def __init__(self, dim):
        super().__init__()

        self.dw3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dw5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)
        

        self.freq_weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x, guide=None):

        local_feat = self.dw3(x) + self.dw5(x)
        local_feat = self.act(self.bn(local_feat))
        


        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        if guide is not None:

            guide_fft = torch.fft.rfft2(guide, norm='ortho')
            amp_guide = torch.abs(guide_fft)
            x_fft = x_fft * amp_guide * self.freq_weight
        else:
            x_fft = x_fft * self.freq_weight


        global_feat = torch.fft.irfft2(x_fft, s=(x.size(2), x.size(3)), norm='ortho')
        

        return x + local_feat + global_feat

class IAF_FFCM_Fusion(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256):
        super().__init__()

        self.rgb_conv = nn.Sequential(nn.Conv2d(c_rgb, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.SiLU())
        self.ir_conv = nn.Sequential(nn.Conv2d(c_ir, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.SiLU())
        

        self.gle = GlobalIlluminationEstimator(hidden_channels)
        

        self.rgb_ffcm = FFCM(hidden_channels)
        self.ir_ffcm = FFCM(hidden_channels)
        

        self.w_rgb = nn.Parameter(torch.ones(1))
        self.w_ir = nn.Parameter(torch.ones(1))
        self.epsilon = 1e-4
        

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU()
        )

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, C, H, W = rgb_feat.shape
        
        rgb_feat = self.rgb_conv(rgb_feat)
        ir_feat = self.ir_conv(ir_feat)
        

        if ir_feat.shape[2:] != (H, W):
            ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=False)
            

        L = self.gle(rgb_feat)
        


        rgb_enhanced = self.rgb_ffcm(rgb_feat, guide=ir_feat)

        ir_enhanced = self.ir_ffcm(ir_feat, guide=None) 
        



        weight_rgb = F.relu(self.w_rgb) * L
        weight_ir = F.relu(self.w_ir) * (1.0 - L)
        
        fused = (weight_rgb * rgb_enhanced + weight_ir * ir_enhanced) / (weight_rgb + weight_ir + self.epsilon)
        
        return self.out_conv(fused)


class RGB_P3_Refiner(nn.Module):




    def __init__(self, in_channels, out_channels):
        super(RGB_P3_Refiner, self).__init__()
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        

        self.dilated_conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1, groups=out_channels, bias=False)
        self.dilated_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2, groups=out_channels, bias=False)
        self.dilated_conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4, groups=out_channels, bias=False)
        self.bn_dilated = nn.BatchNorm2d(out_channels)
        self.act_dilated = nn.GELU()


        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        d1 = self.dilated_conv1(x)
        d2 = self.dilated_conv2(x)
        d3 = self.dilated_conv3(x)
        x_expanded = self.act_dilated(self.bn_dilated(d1 + d2 + d3))
        gate = self.spatial_gate(x_expanded)
        return x_expanded * gate

class Deep_CFFM(nn.Module):





    def __init__(self, c_rgb, c_ir, hidden_channels=256, use_p3_refiner=False, groups=4):
        super(Deep_CFFM, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        



        if use_p3_refiner:

            self.rgb_proj = RGB_P3_Refiner(c_rgb, hidden_channels)
        else:

            self.rgb_proj = nn.Sequential(
                nn.Conv2d(c_rgb, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU() 
            )


        self.ir_proj = nn.Sequential(
            nn.Conv2d(c_ir, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels)
        )


        self.amp_mix_weight = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0.5)
        

        out_offset_channels = self.groups * 4
        self.offset_generator = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2, groups=hidden_channels, bias=False),
            nn.Conv2d(hidden_channels, 128, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(128, out_offset_channels, kernel_size=1, bias=False)
        )

        self.fusion_aggregator = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, C, H, W = rgb_feat.shape
        dtype, device = rgb_feat.dtype, rgb_feat.device


        if ir_feat.shape[2:] != (H, W):
            ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=False)

        rgb_feat = self.rgb_proj(rgb_feat)
        ir_feat = self.ir_proj(ir_feat)





        current_dtype = rgb_feat.dtype
        

        rgb_feat_f32 = rgb_feat.to(torch.float32)
        ir_feat_f32 = ir_feat.to(torch.float32)
        mix_weight_f32 = self.amp_mix_weight.to(torch.float32)


        fft_rgb = torch.fft.rfft2(rgb_feat_f32, norm='ortho')
        fft_ir = torch.fft.rfft2(ir_feat_f32, norm='ortho')

        amp_rgb, pha_rgb = torch.abs(fft_rgb), torch.angle(fft_rgb)
        amp_ir = torch.abs(fft_ir)


        amp_fused = amp_rgb * (1.0 - mix_weight_f32) + amp_ir * mix_weight_f32
        fft_fused = amp_fused * torch.exp(1j * pha_rgb)
        

        freq_guide_feat_f32 = torch.fft.irfft2(fft_fused, s=(H, W), norm='ortho')
        

        freq_guide_feat = freq_guide_feat_f32.to(current_dtype)



        offsets = self.offset_generator(freq_guide_feat)
        offset_rgb = offsets[:, 0:self.groups * 2, :, :].reshape(B * self.groups, 2, H, W)
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :].reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).to(dtype).unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        norm = torch.tensor([W - 1, H - 1], device=device, dtype=dtype).view(1, 2, 1, 1)

        grid_rgb = (2.0 * (base_grid + offset_rgb) / norm - 1.0).permute(0, 2, 3, 1)
        grid_ir = (2.0 * (base_grid + offset_ir) / norm - 1.0).permute(0, 2, 3, 1)

        rgb_aligned = F.grid_sample(rgb_feat.reshape(B*self.groups, -1, H, W), grid_rgb, align_corners=False).reshape(B, -1, H, W)
        ir_aligned = F.grid_sample(ir_feat.reshape(B*self.groups, -1, H, W), grid_ir, align_corners=False).reshape(B, -1, H, W)

        aligned_concat = torch.cat([rgb_aligned, ir_aligned], dim=1)
        attn_weights = self.fusion_aggregator(aligned_concat)
        
        fused_out = self.final_conv(aligned_concat * attn_weights)
        

        return fused_out + freq_guide_feat
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

class DecoupledFreqGuidedFusion_trans(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(DecoupledFreqGuidedFusion_trans, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),  
            nn.Sigmoid()
        )



        self.reduce_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        

        self.num_tokens = 4
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_channels, 
            nhead=4,
            dim_feedforward=hidden_channels * 2, 
            dropout=0.0, 
            activation='gelu',
            batch_first=True
        )
        

        self.to_logits = nn.Conv2d(hidden_channels, self.groups * 2, kernel_size=3, padding=1, bias=False)


        self.init_weights()


        self.gamma = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.to_logits.weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape


        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)


        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        if self.tiny_mode:

            offset_rgb = torch.zeros_like(offsets[:, 0:self.groups * 2, :, :])
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'nearest'
        else:

            offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'bilinear'

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)


        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        

        local_feat = self.reduce_conv(fusion_input) 
        


        tokens_2d = F.adaptive_avg_pool2d(local_feat, (self.num_tokens, self.num_tokens)) 
        tokens_1d = tokens_2d.flatten(2).transpose(1, 2)
        


        enhanced_tokens = self.transformer_encoder(tokens_1d)
        

        enhanced_tokens_2d = enhanced_tokens.transpose(1, 2).view(B, self.hidden_channels, self.num_tokens, self.num_tokens)
        global_feat = F.interpolate(enhanced_tokens_2d, size=(H, W), mode='bilinear', align_corners=False)
        
        hybrid_feat = local_feat + (1.0 - edge_mask) * global_feat
        

        attention_logits = self.to_logits(hybrid_feat)


        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out
    
class GetIndex(nn.Module):
    def __init__(self, *args):

        super().__init__()

        self.index = int(args[2] if len(args) >= 3 else args[1])

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return x[self.index]
        raise TypeError(
            f"GetIndex: 期望来自多输出主干的 list/tuple，实际收到 {type(x).__name__}。"
            f"请确认 YAML 里 from 指向了多输出主干层，且该主干已在 parse_model 中进入 save。"
        )
    
class SEnetV2(nn.Module):
    def __init__(self, in_channels, dim, reduction_ratio=4):





        super().__init__()
        self.in_channels = in_channels
        self.dim = dim*2
        mid_dim = self.dim // reduction_ratio



        self.h_avg_pool = nn.AdaptiveAvgPool2d((None, 1))

        self.w_avg_pool = nn.AdaptiveAvgPool2d((1, None))

        self.fc1 = nn.Sequential(
            nn.Linear(self.dim, mid_dim),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.dim, mid_dim),
            nn.ReLU(inplace=True)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.dim, mid_dim),
            nn.Sigmoid()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(self.dim, mid_dim),
            nn.Sigmoid()
        )

        self.fc_all = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid()
        )

    def forward(self, x):

        B, C, H, W = x.size()


        h_pool = self.h_avg_pool(x)
        w_pool = self.w_avg_pool(x)
        h_pool = h_pool.permute(0, 1, 3, 2)
        hw_pool = torch.cat([h_pool, w_pool], dim=3)
        hw_pool = hw_pool.squeeze(2)
        fc1 = self.fc1(hw_pool)
        fc2 = self.fc2(hw_pool)
        fc3 = self.fc3(hw_pool)
        fc4 = self.fc4(hw_pool)

        fc_all = torch.cat([fc1, fc2, fc3, fc4], dim=2)

        fc_all = self.fc_all(fc_all)

        assert H==W
        x1, x2 = fc_all[:, :, 0:H], fc_all[:, :, H:2*H]
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(3)
        return x * x1 * x2 + x


import torch
import torch.nn as nn

class DynamicGroupedCoordAtt(nn.Module):



    def __init__(self, in_channels, reduction_ratio=4, groups=4):
        super().__init__()
        self.groups = groups
        

        mid_channels = max(in_channels // reduction_ratio, groups)
        mid_channels = (mid_channels // groups) * groups


        self.h_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.w_pool = nn.AdaptiveAvgPool2d((1, None))



        self.conv_shared = nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True)


        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, groups=groups, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, groups=groups, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()


        x_h = self.h_pool(x).permute(0, 1, 3, 2)
        x_w = self.w_pool(x)
        x_cat = torch.cat([x_h, x_w], dim=3)


        out = self.act(self.bn(self.conv_shared(x_cat)))


        out_h, out_w = torch.split(out, [H, W], dim=3)
        out_h = out_h.permute(0, 1, 3, 2)


        weight_h = self.sigmoid(self.conv_h(out_h))
        weight_w = self.sigmoid(self.conv_w(out_w))


        return x * weight_h * weight_w + x

class ContextGuideFusionModuleV2(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = nn.Conv2d(inc[0], inc[1], 1)
            


        self.se = DynamicGroupedCoordAtt(in_channels=inc[1] * 2, reduction_ratio=4, groups=4)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        

        x_concat = torch.cat([x0, x1], dim=1)
        

        x_concat = self.se(x_concat)
        

        x0_weight, x1_weight = torch.split(x_concat, [x0.size(1), x1.size(1)], dim=1)

        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight

        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)
import torch
import torch.nn as nn
import torch.nn.functional as F





class Focus(nn.Module):




    def __init__(self):
        super().__init__()

    def forward(self, x):

        return torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)

class DecoupledFreqGuidedFusion_re(nn.Module):



    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(DecoupledFreqGuidedFusion_re, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        

        self.focus = Focus()

        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)

        self.ir_conv = Conv(c_ir, hidden_channels, 3)



        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)


        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1), 
            nn.Sigmoid()
        )

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):

        pass

    def forward(self, x):
        rgb_feat, ir_feat = x
        


        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        




        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate


        edge_mask = self.edge_mask_gen(ir_freq)
        

        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape


        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
        
        interp_method = 'bilinear'

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out

import torch
import torch.nn as nn
import torch.nn.functional as F

class Ablation_Only_DPFR(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(Ablation_Only_DPFR, self).__init__()
        self.hidden_channels = hidden_channels
        

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        

        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)


        self.fallback_fusion = Conv(hidden_channels * 2, hidden_channels, 1)
        
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        rgb_feat, ir_feat = x
        

        ir_feat = self.focus(ir_feat)              
        ir_feat = self.ir_focus_conv(ir_feat)      
        ir_feat = self.ir_conv(ir_feat)            
        
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        rgb_feat = self.rgb_conv(rgb_feat)


        fused_out = self.fallback_fusion(torch.cat([rgb_feat, ir_guided], dim=1))

        return fused_out

    import torch
import torch.nn as nn
import torch.nn.functional as F

class Ablation_DPFR_PMDA(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(Ablation_DPFR_PMDA, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        

        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)


        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1), 
            nn.Sigmoid()
        )


        self.fallback_fusion = Conv(hidden_channels * 2, hidden_channels, 1)

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        rgb_feat, ir_feat = x
        

        ir_feat = self.focus(ir_feat)              
        ir_feat = self.ir_focus_conv(ir_feat)      
        ir_feat = self.ir_conv(ir_feat)            
        
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        edge_mask = self.edge_mask_gen(ir_freq)
        
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape


        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
        
        interp_method = 'bilinear'

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)


        fused_out = self.fallback_fusion(torch.cat([rgb_aligned, ir_aligned], dim=1))

        return fused_out

    import torch
import torch.nn as nn
import torch.nn.functional as F

class Ablation_Sym_Only_DPFR(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(Ablation_Sym_Only_DPFR, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)


        self.fallback_fusion = Conv(hidden_channels * 2, hidden_channels, 1)

        self.out_channels = hidden_channels
        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        rgb_feat, ir_feat = x


        ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)


        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate


        fused_out = self.fallback_fusion(torch.cat([rgb_feat, ir_guided], dim=1))
        
        return fused_out

import torch
import torch.nn as nn
import torch.nn.functional as F

class Ablation_Sym_DPFR_PMDA(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(Ablation_Sym_DPFR_PMDA, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)


        out_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False)
        )


        self.fallback_fusion = Conv(hidden_channels * 2, hidden_channels, 1)

        self.out_channels = hidden_channels
        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        batch_size, _, out_h, out_w = rgb_feat.size()


        ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)


        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate


        conv_results = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        offset_rgb_raw = conv_results[:, 0:self.groups * 2, :, :]
        offset_ir_raw = conv_results[:, self.groups * 2:self.groups * 4, :, :]


        if self.tiny_mode:
            offset_rgb = torch.zeros_like(offset_rgb_raw)
            offset_ir = offset_ir_raw
            interp_method = 'nearest'
        else:
            offset_rgb = offset_rgb_raw
            offset_ir = offset_ir_raw
            interp_method = 'bilinear'


        ir_guided_g = ir_guided.reshape(batch_size * self.groups, -1, out_h, out_w)
        rgb_feat_g = rgb_feat.reshape(batch_size * self.groups, -1, out_h, out_w)

        offset_rgb = offset_rgb.reshape(batch_size * self.groups, 2, out_h, out_w)
        offset_ir = offset_ir.reshape(batch_size * self.groups, 2, out_h, out_w)


        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(ir_guided_g).to(ir_guided_g.device)
        grid_w = torch.linspace(-1.0, 1.0, out_w).view(1, -1).repeat(out_h, 1)
        grid_h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        base_grid = torch.cat((grid_w.unsqueeze(2), grid_h.unsqueeze(2)), dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(batch_size * self.groups, 1, 1, 1).type_as(ir_guided_g).to(ir_guided_g.device)

        adjusted_grid_rgb = base_grid + offset_rgb.permute(0, 2, 3, 1) / (normalization_factors * 0.5)
        adjusted_grid_ir = base_grid + offset_ir.permute(0, 2, 3, 1) / (normalization_factors * 0.5)


        rgb_aligned = F.grid_sample(rgb_feat_g, adjusted_grid_rgb.type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, adjusted_grid_ir.type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')

        rgb_aligned = rgb_aligned.reshape(batch_size, -1, out_h, out_w)
        ir_aligned = ir_aligned.reshape(batch_size, -1, out_h, out_w)


        fused_out = self.fallback_fusion(torch.cat([rgb_aligned, ir_aligned], dim=1))

        return fused_out












































































class PhaseDiffAlignMap(nn.Module):















    def __init__(self, dim):
        super().__init__()
        self.conv_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, fft_ir, fft_rgb):

        phase_ir = torch.angle(fft_ir)
        phase_rgb = torch.angle(fft_rgb)

        phase_diff = phase_ir - phase_rgb


        complex_diff = torch.exp(1j * phase_diff.to(torch.float32))
        spatial_diff = torch.abs(torch.fft.ifft2(
            complex_diff, dim=(-2, -1), norm='backward'
        ))


        target_dtype = self.conv_proj[0].weight.dtype
        return self.conv_proj(spatial_diff.to(target_dtype))


class CrossModalAmpComplement(nn.Module):










    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.ir_adapt = nn.Conv2d(dim, dim, 1)
        self.rgb_adapt = nn.Conv2d(dim, dim, 1)

    def forward(self, amp_ir, amp_rgb):


        target_dtype = self.ir_adapt.weight.dtype
        amp_ir_d = amp_ir.to(target_dtype)
        amp_rgb_d = amp_rgb.to(target_dtype)

        enhanced_amp_ir = amp_ir_d + torch.sigmoid(self.alpha) * self.ir_adapt(amp_rgb_d)
        enhanced_amp_rgb = amp_rgb_d + torch.sigmoid(self.beta) * self.rgb_adapt(amp_ir_d)
        return enhanced_amp_ir, enhanced_amp_rgb


class DecoupledFreqGuidedFusion_4Mode(nn.Module):














    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels


        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)



        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)




        self.amp_complement = CrossModalAmpComplement(hidden_channels)


        self.phase_align = PhaseDiffAlignMap(hidden_channels)


        self.frequency_enhancer_ir = FFM(hidden_channels)
        self.frequency_enhancer_rgb = FFM(hidden_channels)


        self.gating_conv_ir = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)



        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )




        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )


        self.init_weights()

    def init_weights(self):

        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    @staticmethod
    def amp_guided_phase_denoise(amp, phase, eps=0.1):













        amp_norm = amp / (amp.max(dim=-1, keepdim=True).values
                              .max(dim=-2, keepdim=True).values + 1e-8)
        phase_conf = torch.sigmoid((amp_norm - eps) * 10)
        return phase * phase_conf, phase_conf

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape



        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)


        rgb_feat = self.rgb_conv(rgb_feat)


        fft_ir = torch.fft.fft2(ir_feat.to(torch.float32), norm='backward')
        fft_rgb = torch.fft.fft2(rgb_feat.to(torch.float32), norm='backward')

        amp_ir, phase_ir = torch.abs(fft_ir), torch.angle(fft_ir)
        amp_rgb, phase_rgb = torch.abs(fft_rgb), torch.angle(fft_rgb)


        phase_ir_clean, _ = self.amp_guided_phase_denoise(amp_ir, phase_ir)
        phase_rgb_clean, _ = self.amp_guided_phase_denoise(amp_rgb, phase_rgb)


        fft_ir_clean = amp_ir * torch.exp(1j * phase_ir_clean)
        fft_rgb_clean = amp_rgb * torch.exp(1j * phase_rgb_clean)
        align_map = self.phase_align(fft_ir_clean, fft_rgb_clean)


        enhanced_amp_ir, enhanced_amp_rgb = self.amp_complement(amp_ir, amp_rgb)


        fft_ir_enhanced = enhanced_amp_ir * torch.exp(1j * phase_ir_clean)
        fft_rgb_enhanced = enhanced_amp_rgb * torch.exp(1j * phase_rgb_clean)

        ir_enhanced = torch.abs(torch.fft.ifft2(
            fft_ir_enhanced, dim=(-2, -1), norm='backward'
        )).to(rgb_feat.dtype)
        rgb_enhanced = torch.abs(torch.fft.ifft2(
            fft_rgb_enhanced, dim=(-2, -1), norm='backward'
        )).to(rgb_feat.dtype)


        ir_freq = self.frequency_enhancer_ir(ir_enhanced)
        rgb_freq = self.frequency_enhancer_rgb(rgb_enhanced)

        gate_ir = torch.sigmoid(self.gating_conv_ir(ir_enhanced))
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_enhanced))

        ir_guided = ir_enhanced * (1 - gate_ir) + ir_freq * gate_ir
        rgb_guided = rgb_enhanced * (1 - gate_rgb) + rgb_freq * gate_rgb


        offsets = self.offset_conv(torch.cat([rgb_guided, ir_guided], dim=1))





        align_map_scaled = align_map.to(offsets.dtype)
        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * (1.0 + align_map_scaled)
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * (1.0 + align_map_scaled)

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)


        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=ir_feat.device),
            torch.arange(W, device=ir_feat.device), indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(
            rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )
        ir_aligned = F.grid_sample(
            ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)




        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq, rgb_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out

























class ForegroundAwareFFM(nn.Module):











    def __init__(self, dim):
        super().__init__()

        self.saliency_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )

        self.ffm = FFM(dim)

    def forward(self, x):
        saliency = self.saliency_conv(x)
        x_fg = x * saliency
        return self.ffm(x_fg)


class JointGate(nn.Module):












    def __init__(self, dim):
        super().__init__()

        self.ch_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.sp_conv = nn.Sequential(
            nn.Conv2d(dim, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ch_gate = self.ch_conv(self.ch_pool(x))
        sp_gate = self.sp_conv(x)
        return ch_gate * sp_gate


class AdaptiveFocus(nn.Module):













    def __init__(self, c_in, c_out):
        super().__init__()
        self.focus = Focus()
        self.focus_conv = nn.Conv2d(c_in * 4, c_out, 1)
        self.stride_conv = nn.Conv2d(c_in, c_out, 3, 2, 1)


        self.mix_pool = nn.AdaptiveAvgPool2d(1)
        self.mix_fc = nn.Sequential(
            nn.Conv2d(c_in, c_in // 4, 1),
            nn.ReLU(),
            nn.Conv2d(c_in // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.focus_conv(self.focus(x))
        s = self.stride_conv(x)
        alpha = self.mix_fc(self.mix_pool(x))
        return alpha * f + (1 - alpha) * s


class DualModalEdgeMask(nn.Module):















    def __init__(self, dim):
        super().__init__()
        self.ir_edge_conv = nn.Sequential(
            nn.Conv2d(dim, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.rgb_edge_conv = nn.Sequential(
            nn.Conv2d(dim, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.trust_conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, ir_freq, rgb_feat):
        ir_edge = self.ir_edge_conv(ir_freq)
        rgb_edge = self.rgb_edge_conv(rgb_feat)
        w = self.trust_conv(torch.cat([ir_edge, rgb_edge], dim=1))
        return w * ir_edge + (1 - w) * rgb_edge


class DecoupledFreqGuidedFusion_DPFRv2(nn.Module):





















    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)




        self.adaptive_focus = AdaptiveFocus(c_ir, c_ir)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)




        self.frequency_enhancer = ForegroundAwareFFM(hidden_channels)

        self.gating = JointGate(hidden_channels)




        self.edge_mask_gen = DualModalEdgeMask(hidden_channels)



        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )



        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )


        self.init_weights()

    def init_weights(self):

        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x



        ir_feat = self.adaptive_focus(ir_feat)
        ir_feat = self.ir_conv(ir_feat)


        rgb_feat = self.rgb_conv(rgb_feat)


        ir_freq = self.frequency_enhancer(ir_feat)
        gate = self.gating(ir_feat)
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate


        edge_mask = self.edge_mask_gen(ir_freq, rgb_feat)


        B, _, H, W = rgb_feat.shape
        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))




        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * (1.0 + edge_mask)
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * (1.0 + edge_mask)

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)


        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=ir_feat.device),
            torch.arange(W, device=ir_feat.device), indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(
            rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )
        ir_aligned = F.grid_sample(
            ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)


        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out




















class DecoupledFreqGuidedFusion_Step2(nn.Module):









    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)


        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.frequency_enhancer = ForegroundAwareFFM(hidden_channels)

        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)


        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )


        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )


        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x


        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)


        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate


        edge_mask = self.edge_mask_gen(ir_freq)

        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape


        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)


        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=ir_feat.device),
            torch.arange(W, device=ir_feat.device), indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(
            rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )
        ir_aligned = F.grid_sample(
            ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)


        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out
































class DecoupledFreqGuidedFusion_Step3(nn.Module):








    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)


        self.adaptive_focus = AdaptiveFocus(c_ir, c_ir)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.frequency_enhancer = ForegroundAwareFFM(hidden_channels)

        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)


        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )


        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )


        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x


        ir_feat = self.adaptive_focus(ir_feat)
        ir_feat = self.ir_conv(ir_feat)


        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate


        edge_mask = self.edge_mask_gen(ir_freq)

        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape


        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)


        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=ir_feat.device),
            torch.arange(W, device=ir_feat.device), indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(
            rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )
        ir_aligned = F.grid_sample(
            ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)


        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out


























class DecoupledFreqGuidedFusion_Step4(nn.Module):










    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels


        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer_ir = ForegroundAwareFFM(hidden_channels)
        self.gating_conv_ir = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)



        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = ForegroundAwareFFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)



        self.edge_mask_gen = DualModalEdgeMask(hidden_channels)



        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )


        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x


        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer_ir(ir_feat)
        gate_ir = torch.sigmoid(self.gating_conv_ir(ir_feat))
        ir_guided = ir_feat * (1 - gate_ir) + ir_freq * gate_ir


        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb


        edge_mask = self.edge_mask_gen(ir_freq, rgb_feat)

        B, _, H, W = rgb_feat.shape


        offsets = self.offset_conv(torch.cat([rgb_guided, ir_guided], dim=1))


        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * (1.0 + edge_mask)
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * (1.0 + edge_mask)

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)


        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=ir_feat.device),
            torch.arange(W, device=ir_feat.device), indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(
            rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )
        ir_aligned = F.grid_sample(
            ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)


        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq, rgb_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out



class DecoupledFreqGuidedFusion_Step1(nn.Module):









    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)


        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.frequency_enhancer = FFM(hidden_channels)

        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)


        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )


        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )


        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):

        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x


        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)


        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate


        edge_mask = self.edge_mask_gen(ir_freq)

        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape


        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)


        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=ir_feat.device),
            torch.arange(W, device=ir_feat.device), indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(
            rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )
        ir_aligned = F.grid_sample(
            ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)


        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out

















class DecoupledFreqGuidedFusion_Step4Lite(nn.Module):








    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)


        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)


        self.edge_mask_gen = DualModalEdgeMask(hidden_channels)


        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )


        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x


        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate


        rgb_feat = self.rgb_conv(rgb_feat)


        edge_mask = self.edge_mask_gen(ir_freq, rgb_feat)

        B, _, H, W = rgb_feat.shape


        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * (1.0 + edge_mask)
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * (1.0 + edge_mask)

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)


        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=ir_feat.device),
            torch.arange(W, device=ir_feat.device), indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(
            rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )
        ir_aligned = F.grid_sample(
            ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)


        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out


















class DecoupledFreqGuidedFusion_NoMask(nn.Module):






    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)

        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x

        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape


        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        offset_rgb = offsets[:, 0:self.groups * 2, :, :]
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :]

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=ir_feat.device),
            torch.arange(W, device=ir_feat.device), indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(
            rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )
        ir_aligned = F.grid_sample(
            ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out



















class DecoupledFreqGuidedFusion_ExpA_RGBGuide(nn.Module):





    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels


        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)


        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)


        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )


        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )


        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x


        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)


        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb


        edge_mask = self.edge_mask_gen(rgb_freq)

        B, _, H, W = rgb_feat.shape


        offsets = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=ir_feat.device),
            torch.arange(W, device=ir_feat.device), indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_feat_g = ir_feat.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(
            rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )
        ir_aligned = F.grid_sample(
            ir_feat_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_feat_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out
















class DecoupledFreqGuidedFusion_ExpB_NoGate(nn.Module):






    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)

        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)

        self.frequency_enhancer = FFM(hidden_channels)


        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x

        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)

        ir_freq = self.frequency_enhancer(ir_feat)

        ir_guided = ir_freq

        edge_mask = self.edge_mask_gen(ir_freq)

        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape

        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=ir_feat.device),
            torch.arange(W, device=ir_feat.device), indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()

        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0

        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)

        rgb_aligned = F.grid_sample(
            rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )
        ir_aligned = F.grid_sample(
            ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g),
            mode='bilinear', align_corners=True, padding_mode='border'
        )

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(
            self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out








class DecoupledFreqGuidedFusion_ExpC_RGBGuide_NoMask(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, 3, padding=1, bias=False))
        self.init_weights()

    def init_weights(self):
        for l in self.children():
            if isinstance(l, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(l.weight)
                if l.bias is not None: nn.init.constant_(l.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        B, _, H, W = rgb_feat.shape
        offsets = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        offset_rgb = offsets[:, 0:self.groups * 2, :, :]
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :]
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W); offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W-1, H-1], device=ir_feat.device).view(1,2,1,1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_aligned = F.grid_sample(rgb_guided.reshape(B*self.groups,-1,H,W), grid_norm_rgb.permute(0,2,3,1).type_as(rgb_feat), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_feat.reshape(B*self.groups,-1,H,W), grid_norm_ir.permute(0,2,3,1).type_as(ir_feat), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W); ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attn_w = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attn_w[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attn_w[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb








class DecoupledFreqGuidedFusion_ExpD_SymNoMask(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer_ir = FFM(hidden_channels)
        self.gating_conv_ir = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, 3, padding=1, bias=False))
        self.init_weights()

    def init_weights(self):
        for l in self.children():
            if isinstance(l, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(l.weight)
                if l.bias is not None: nn.init.constant_(l.bias, 0)
        self.offset_conv[1].weight.data.zero_()
        self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer_ir(ir_feat)
        gate_ir = torch.sigmoid(self.gating_conv_ir(ir_feat))
        ir_guided = ir_feat * (1 - gate_ir) + ir_freq * gate_ir
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        B, _, H, W = rgb_feat.shape
        offsets = self.offset_conv(torch.cat([rgb_guided, ir_guided], dim=1))
        offset_rgb = offsets[:, 0:self.groups * 2, :, :]
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :]
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W); offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W-1, H-1], device=ir_feat.device).view(1,2,1,1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_aligned = F.grid_sample(rgb_guided.reshape(B*self.groups,-1,H,W), grid_norm_rgb.permute(0,2,3,1).type_as(rgb_feat), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided.reshape(B*self.groups,-1,H,W), grid_norm_ir.permute(0,2,3,1).type_as(ir_feat), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W); ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq, rgb_freq], dim=1)
        attn_w = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attn_w[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attn_w[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb










class DecoupledFreqGuidedFusion_F1_RGBGuide_Gate3x3(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, 1, 3, padding=1), nn.Sigmoid())
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, 3, padding=1, bias=False))
        self.init_weights()

    def init_weights(self):
        for l in self.children():
            if isinstance(l, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(l.weight)
                if l.bias is not None: nn.init.constant_(l.bias, 0)
        self.offset_conv[1].weight.data.zero_(); self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W); offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W-1, H-1], device=ir_feat.device).view(1,2,1,1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_aligned = F.grid_sample(rgb_guided.reshape(B*self.groups,-1,H,W), grid_norm_rgb.permute(0,2,3,1).type_as(rgb_feat), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_feat.reshape(B*self.groups,-1,H,W), grid_norm_ir.permute(0,2,3,1).type_as(ir_feat), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W); ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attn_w = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attn_w[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attn_w[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb










class DecoupledFreqGuidedFusion_F2_RGBGuide_FAFFM(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer_rgb = ForegroundAwareFFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, 1, 3, padding=1), nn.Sigmoid())
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, 3, padding=1, bias=False))
        self.init_weights()

    def init_weights(self):
        for l in self.children():
            if isinstance(l, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(l.weight)
                if l.bias is not None: nn.init.constant_(l.bias, 0)
        self.offset_conv[1].weight.data.zero_(); self.fusion_attn_conv[3].weight.data.zero_()

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
        offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W); offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W-1, H-1], device=ir_feat.device).view(1,2,1,1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_aligned = F.grid_sample(rgb_guided.reshape(B*self.groups,-1,H,W), grid_norm_rgb.permute(0,2,3,1).type_as(rgb_feat), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_feat.reshape(B*self.groups,-1,H,W), grid_norm_ir.permute(0,2,3,1).type_as(ir_feat), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W); ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attn_w = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attn_w[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attn_w[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb










class DecoupledFreqGuidedFusion_PMDA_SoftMask(nn.Module):













    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(DecoupledFreqGuidedFusion_PMDA_SoftMask, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        

        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        

        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1), 
            nn.Sigmoid()
        )
        


        self.lambda_mask = nn.Parameter(torch.tensor(0.3))
        

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        
        self.init_weights()
    
    def init_weights(self):
        pass
    
    def forward(self, x):
        rgb_feat, ir_feat = x
        

        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        
        edge_mask = self.edge_mask_gen(ir_freq)
        
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape
        

        offsets_raw = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        

        soft_mask = edge_mask + self.lambda_mask
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * soft_mask
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * soft_mask
        

        offset_rgb = torch.tanh(offset_rgb) * 5.0
        offset_ir = torch.tanh(offset_ir) * 5.0
        
        interp_method = 'bilinear'
        
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        
        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)
        
        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')
        
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)
        
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        
        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb
        
        return fused_out



class DecoupledFreqGuidedFusion_PMDA_Enhanced(nn.Module):













    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super(DecoupledFreqGuidedFusion_PMDA_Enhanced, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        

        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        



        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        



        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3),
            nn.Sigmoid()
        )
        

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        
        self.init_weights()
    
    def init_weights(self):

        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
    
    def forward(self, x):
        rgb_feat, ir_feat = x
        

        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        

        edge_mask = self.edge_mask_gen(ir_freq)
        
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape
        

        offsets_raw = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        

        edge_mask_expanded = edge_mask.repeat_interleave(2, dim=1)
        
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * edge_mask_expanded
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * edge_mask_expanded
        

        offset_rgb = torch.tanh(offset_rgb) * 5.0
        offset_ir = torch.tanh(offset_ir) * 5.0
        
        interp_method = 'bilinear'
        

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        
        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)
        
        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')
        
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)
        
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        
        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb
        
        return fused_out



class DecoupledFreqGuidedFusion_PMDA_DualMask(nn.Module):















    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(DecoupledFreqGuidedFusion_PMDA_DualMask, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        

        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        

        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        



        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 7, padding=3),
            nn.Sigmoid()
        )
        

        self.lambda_mask = nn.Parameter(torch.tensor(0.3))
        

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        
        self.init_weights()
    
    def init_weights(self):
        pass
    
    def forward(self, x):
        rgb_feat, ir_feat = x
        

        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape
        

        dual_input = torch.cat([rgb_feat, ir_freq], dim=1)
        edge_mask = self.edge_mask_gen(dual_input)
        

        offsets_raw = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        

        soft_mask = edge_mask + self.lambda_mask
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * soft_mask
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * soft_mask
        

        offset_rgb = torch.tanh(offset_rgb) * 5.0
        offset_ir = torch.tanh(offset_ir) * 5.0
        
        interp_method = 'bilinear'
        
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        
        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)
        
        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')
        
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)
        
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        
        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb
        
        return fused_out









class DecoupledFreqGuidedFusion_PMDA_SoftMask_v2(nn.Module):








    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(DecoupledFreqGuidedFusion_PMDA_SoftMask_v2, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.lambda_mask = nn.Parameter(torch.tensor(0.3))
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        soft_mask = edge_mask + self.lambda_mask
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * soft_mask
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * soft_mask
        interp_method = 'bilinear'
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_Enhanced_v2(nn.Module):









    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super(DecoupledFreqGuidedFusion_PMDA_Enhanced_v2, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3),
            nn.Sigmoid()
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        edge_mask_expanded = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * edge_mask_expanded
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * edge_mask_expanded
        interp_method = 'bilinear'
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_DualMask_v2(nn.Module):








    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(DecoupledFreqGuidedFusion_PMDA_DualMask_v2, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 7, padding=3),
            nn.Sigmoid()
        )

        self.lambda_mask = nn.Parameter(torch.tensor(0.3))
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape
        dual_input = torch.cat([rgb_feat, ir_freq], dim=1)
        edge_mask = self.edge_mask_gen(dual_input)
        offsets_raw = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        soft_mask = edge_mask + self.lambda_mask
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * soft_mask
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * soft_mask
        interp_method = 'bilinear'
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb









class DecoupledFreqGuidedFusion_PMDA_SoftEnhanced(nn.Module):










    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super(DecoupledFreqGuidedFusion_PMDA_SoftEnhanced, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3),
            nn.Sigmoid()
        )

        self.lambda_mask = nn.Parameter(torch.tensor(0.3))
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        soft_mask = edge_mask + self.lambda_mask
        edge_mask_expanded = soft_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * edge_mask_expanded
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * edge_mask_expanded
        interp_method = 'bilinear'
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_DualEnhanced(nn.Module):










    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super(DecoupledFreqGuidedFusion_PMDA_DualEnhanced, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups, 7, padding=3),
            nn.Sigmoid()
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape

        dual_input = torch.cat([rgb_feat, ir_freq], dim=1)
        edge_mask = self.edge_mask_gen(dual_input)
        offsets_raw = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        edge_mask_expanded = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * edge_mask_expanded
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * edge_mask_expanded
        interp_method = 'bilinear'
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_Enhanced_g8(nn.Module):










    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=8):
        super(DecoupledFreqGuidedFusion_PMDA_Enhanced_g8, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 192, 1),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3),
            nn.Sigmoid()
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        edge_mask_expanded = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * edge_mask_expanded
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * edge_mask_expanded
        interp_method = 'bilinear'
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb









class DecoupledFreqGuidedFusion_PMDA_Enhanced_Gate3x3(nn.Module):







    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)

        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3),
            nn.Sigmoid()
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        edge_mask_expanded = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * edge_mask_expanded
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * edge_mask_expanded
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_feat_g = rgb_feat.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_Enhanced_ExpA(nn.Module):











    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3),
            nn.Sigmoid()
        )
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x

        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)

        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb

        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape

        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        edge_mask_expanded = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * edge_mask_expanded
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * edge_mask_expanded
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_feat_g = ir_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_feat_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_feat_g), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_Enhanced_Refine(nn.Module):












    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        out_offset_channels = self.groups * 4


        self.offset_conv1 = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.offset_conv2 = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3),
            nn.Sigmoid()
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv1[4].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.offset_conv2[4].weight, mean=0.0, std=0.01)

    def _grid_align(self, feat, offsets, B, H, W):

        offsets = offsets.reshape(B * self.groups, 2, H, W)
        feat_g = feat.reshape(B * self.groups, -1, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=feat.device),
                                        torch.arange(W, device=feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=feat.device).view(1, 2, 1, 1).float()
        grid_norm = 2.0 * (base_grid + offsets) / normalizer - 1.0
        aligned = F.grid_sample(feat_g, grid_norm.permute(0, 2, 3, 1).type_as(feat_g),
                                mode='bilinear', align_corners=True, padding_mode='border')
        return aligned.reshape(B, -1, H, W)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape


        offsets1 = self.offset_conv1(torch.cat([rgb_feat, ir_guided], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb1 = offsets1[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir1 = offsets1[:, self.groups * 2:self.groups * 4, :, :] * mask_exp
        rgb_a1 = self._grid_align(rgb_feat, offset_rgb1, B, H, W)
        ir_a1 = self._grid_align(ir_guided, offset_ir1, B, H, W)


        offsets2 = self.offset_conv2(torch.cat([rgb_a1, ir_a1], dim=1))
        offset_rgb2 = offsets2[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir2 = offsets2[:, self.groups * 2:self.groups * 4, :, :] * mask_exp

        offset_rgb_total = offset_rgb1 + offset_rgb2
        offset_ir_total = offset_ir1 + offset_ir2
        rgb_aligned = self._grid_align(rgb_feat, offset_rgb_total, B, H, W)
        ir_aligned = self._grid_align(ir_guided, offset_ir_total, B, H, W)


        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb










class DecoupledFreqGuidedFusion_PMDA_ExpA_MDAAv2(nn.Module):







    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3),
            nn.Sigmoid()
        )
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_feat_g = ir_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_feat_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_feat_g), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_Soft(nn.Module):










    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3),
            nn.Sigmoid()
        )

        self.lambda_mask = nn.Parameter(torch.tensor(0.2))
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))

        soft_mask = edge_mask + self.lambda_mask
        mask_exp = soft_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_feat_g = ir_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_feat_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_feat_g), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_MultiScale(nn.Module):












    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3),
            nn.Sigmoid()
        )
        out_offset_channels = self.groups * 4


        self.offset_pre = Conv(hidden_channels * 2, 96, 1)

        self.offset_branch_a = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.offset_branch_b = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=5, padding=4, dilation=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        self.offset_fuse = nn.Conv2d(out_offset_channels * 2, out_offset_channels, 1, bias=False)

        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_fuse.weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape


        x_cat = self.offset_pre(torch.cat([rgb_guided, ir_feat], dim=1))
        off_a = self.offset_branch_a(x_cat)
        off_b = self.offset_branch_b(x_cat)
        offsets_raw = self.offset_fuse(torch.cat([off_a, off_b], dim=1))

        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_feat_g = ir_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_feat_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_feat_g), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb









class DecoupledFreqGuidedFusion_PMDA_ExpA_LearnAttn(nn.Module):












    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )



        self.attn_gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups, 7, padding=3),
            nn.Sigmoid()
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        B, _, H, W = rgb_feat.shape



        attn = self.attn_gate(torch.cat([rgb_guided, ir_feat], dim=1))

        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = attn.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_feat_g = ir_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_feat_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_feat_g), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_SymPMDA(nn.Module):













    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.ir_conv_in = Conv(c_ir, hidden_channels, 3)

        self.frequency_enhancer_ir = FFM(hidden_channels)
        self.gating_conv_ir = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)

        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        ir_feat = self.ir_conv_in(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)

        ir_freq = self.frequency_enhancer_ir(ir_feat)
        gate_ir = torch.sigmoid(self.gating_conv_ir(ir_feat))
        ir_guided = ir_feat * (1 - gate_ir) + ir_freq * gate_ir
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb


        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_guided], dim=1))
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :]
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :]

        offset_rgb = torch.tanh(offset_rgb * 0.1) * 15.0
        offset_ir = torch.tanh(offset_ir * 0.1) * 15.0
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_guided_g = ir_guided.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_OffsetReg(nn.Module):












    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3),
            nn.Sigmoid()
        )
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def _compute_offset_reg(self, offset_rgb, offset_ir):


        drgb_dy = (offset_rgb[:, :, 1:, :] - offset_rgb[:, :, :-1, :]).abs().mean()
        drgb_dx = (offset_rgb[:, :, :, 1:] - offset_rgb[:, :, :, :-1]).abs().mean()
        dir_dy  = (offset_ir[:, :, 1:, :]  - offset_ir[:, :, :-1, :]).abs().mean()
        dir_dx  = (offset_ir[:, :, :, 1:]  - offset_ir[:, :, :, :-1]).abs().mean()
        smooth_loss = (drgb_dy + drgb_dx + dir_dy + dir_dx) / 4.0

        mag_loss = (offset_rgb.pow(2).mean() + offset_ir.pow(2).mean()) / 2.0
        return smooth_loss, mag_loss

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat)
        ir_feat = self.ir_focus_conv(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * mask_exp


        smooth_loss, mag_loss = self._compute_offset_reg(offset_rgb, offset_ir)

        self.smooth_loss = smooth_loss.detach().item()
        self.mag_loss = mag_loss.detach().item()

        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float()
        base_grid = base_grid.unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir = 2.0 * (base_grid + offset_ir) / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_feat_g = ir_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_feat_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_feat_g), mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb









class DecoupledFreqGuidedFusion_PMDA_ExpA_SymFreq(nn.Module):











    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.ir_down = Conv(c_ir, c_ir, 3, 2)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer_ir = FFM(hidden_channels)
        self.gating_conv_ir = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)

        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        ir_feat = self.ir_down(ir_feat)
        ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer_ir(ir_feat)
        gate_ir = torch.sigmoid(self.gating_conv_ir(ir_feat))
        ir_guided = ir_feat * (1 - gate_ir) + ir_freq * gate_ir

        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb

        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_guided], dim=1))
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :]
        offset_ir  = offsets_raw[:, self.groups * 2:self.groups * 4, :, :]
        offset_rgb = torch.tanh(offset_rgb * 0.1) * 15.0
        offset_ir  = torch.tanh(offset_ir * 0.1) * 15.0
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir  = 2.0 * (base_grid + offset_ir)  / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_guided_g  = ir_guided.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned  = F.grid_sample(ir_guided_g,  grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g),  mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned  = ir_aligned.reshape(B, -1, H, W)
        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir  = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_AlignLoss(nn.Module):








    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3), nn.Sigmoid()
        )
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.align_loss = 0.0

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir  = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir  = 2.0 * (base_grid + offset_ir)  / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_feat_g    = ir_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned  = F.grid_sample(ir_feat_g,    grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_feat_g),    mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned  = ir_aligned.reshape(B, -1, H, W)


        cos_sim = F.cosine_similarity(rgb_aligned.flatten(1), ir_aligned.flatten(1), dim=1)
        self.align_loss = (1.0 - cos_sim.mean()).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir  = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_CycleLoss(nn.Module):









    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3), nn.Sigmoid()
        )
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.cycle_loss = 0.0

    def _warp(self, feat, offsets, B, H, W):
        offsets = offsets.reshape(B * self.groups, 2, H, W)
        feat_g = feat.reshape(B * self.groups, -1, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=feat.device),
                                        torch.arange(W, device=feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=feat.device).view(1, 2, 1, 1).float()
        grid_norm = 2.0 * (base_grid + offsets) / normalizer - 1.0
        warped = F.grid_sample(feat_g, grid_norm.permute(0, 2, 3, 1).type_as(feat_g),
                               mode='bilinear', align_corners=True, padding_mode='border')
        return warped.reshape(B, -1, H, W)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir  = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * mask_exp


        rgb_aligned = self._warp(rgb_guided, offset_rgb, B, H, W)
        ir_aligned  = self._warp(ir_feat,    offset_ir,  B, H, W)


        rgb_cycle = self._warp(rgb_aligned, -offset_rgb, B, H, W)
        ir_cycle  = self._warp(ir_aligned,  -offset_ir,  B, H, W)
        cycle_rgb = F.l1_loss(rgb_cycle, rgb_guided)
        cycle_ir  = F.l1_loss(ir_cycle,  ir_feat)
        self.cycle_loss = ((cycle_rgb + cycle_ir) / 2.0).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir  = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb









class DecoupledFreqGuidedFusion_PMDA_ExpA_AlignV2(nn.Module):








    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3), nn.Sigmoid()
        )
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.align_loss = 0.0

        self.lambda_l2 = 0.1

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir  = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir  = 2.0 * (base_grid + offset_ir)  / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_feat_g    = ir_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned  = F.grid_sample(ir_feat_g,    grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_feat_g),    mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned  = ir_aligned.reshape(B, -1, H, W)


        cos_sim = F.cosine_similarity(rgb_aligned.flatten(1), ir_aligned.flatten(1), dim=1)
        cos_loss = (1.0 - cos_sim.mean())
        l2_loss  = F.mse_loss(rgb_aligned, ir_aligned)
        self.align_loss = (cos_loss + self.lambda_l2 * l2_loss).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir  = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_Contrastive(nn.Module):









    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3), nn.Sigmoid()
        )
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0
        self.tau = 0.07

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir  = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir  = 2.0 * (base_grid + offset_ir)  / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_feat_g    = ir_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned  = F.grid_sample(ir_feat_g,    grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_feat_g),    mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned  = ir_aligned.reshape(B, -1, H, W)



        num_samples = min(H * W, 64)
        rand_idx = torch.randperm(H * W, device=rgb_aligned.device)[:num_samples]
        rgb_sampled = rgb_aligned.flatten(2)[:, :, rand_idx]
        ir_sampled  = ir_aligned.flatten(2)[:, :, rand_idx]

        rgb_norm = F.normalize(rgb_sampled, dim=1)
        ir_norm  = F.normalize(ir_sampled,  dim=1)

        logits = torch.bmm(rgb_norm.transpose(1, 2), ir_norm) / self.tau
        labels = torch.arange(num_samples, device=logits.device).unsqueeze(0).repeat(B, 1)
        loss_rgb2ir = F.cross_entropy(logits.reshape(B * num_samples, num_samples), labels.reshape(-1))
        loss_ir2rgb = F.cross_entropy(logits.transpose(1, 2).reshape(B * num_samples, num_samples), labels.reshape(-1))
        self.contrast_loss = ((loss_rgb2ir + loss_ir2rgb) / 2.0).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir  = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_AlignBox(nn.Module):








    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.focus = Focus()
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, self.groups, 7, padding=3), nn.Sigmoid()
        )
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.align_loss = 0.0

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        rgb_guided = rgb_feat * (1 - gate_rgb) + rgb_freq * gate_rgb
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir  = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device),
                                        torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        grid_norm_rgb = 2.0 * (base_grid + offset_rgb) / normalizer - 1.0
        grid_norm_ir  = 2.0 * (base_grid + offset_ir)  / normalizer - 1.0
        rgb_guided_g = rgb_guided.reshape(B * self.groups, -1, H, W)
        ir_feat_g    = ir_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_guided_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_guided_g), mode='bilinear', align_corners=True, padding_mode='border')
        ir_aligned  = F.grid_sample(ir_feat_g,    grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_feat_g),    mode='bilinear', align_corners=True, padding_mode='border')
        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned  = ir_aligned.reshape(B, -1, H, W)



        edge_weight = edge_mask.mean(dim=1, keepdim=True) + 0.5
        cos_sim_map = F.cosine_similarity(rgb_aligned, ir_aligned, dim=1)
        weighted_cos = (edge_weight.squeeze(1) * cos_sim_map).sum() / edge_weight.sum()
        self.align_loss = (1.0 - weighted_cos).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        attention_weights = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        attn_ir  = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * attn_ir + rgb_aligned * attn_rgb









class DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveV2(nn.Module):







    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, self.groups, 7, padding=3), nn.Sigmoid())
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, 3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, 3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

        self.log_tau = nn.Parameter(torch.tensor(-2.66))
        self.contrast_loss = 0.0

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2, :, :] * mask_exp
        offset_ir  = offsets_raw[:, self.groups * 2:self.groups * 4, :, :] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        normalizer = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (base_grid + o.reshape(B * self.groups, 2, H, W)) / normalizer - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)


        tau = torch.exp(self.log_tau) + 0.01
        num_samples = min(H * W, 64)
        rand_idx = torch.randperm(H * W, device=rgb_aligned.device)[:num_samples]
        rgb_n = F.normalize(rgb_aligned.flatten(2)[:, :, rand_idx], dim=1)
        ir_n  = F.normalize(ir_aligned.flatten(2)[:, :, rand_idx], dim=1)
        logits = torch.bmm(rgb_n.transpose(1, 2), ir_n) / tau
        labels = torch.arange(num_samples, device=logits.device).unsqueeze(0).repeat(B, 1)


        N = num_samples
        mask_pos = torch.eye(N, device=logits.device).bool().unsqueeze(0)
        neg_logits = logits.masked_fill(mask_pos, float('-inf'))
        k = max(N // 2, 1)
        topk_vals, _ = neg_logits.topk(k, dim=-1)
        threshold = topk_vals[:, :, -1:].detach()
        hard_mask = (neg_logits >= threshold)
        final_mask = mask_pos | hard_mask
        logits_masked = logits.masked_fill(~final_mask, float('-inf'))

        loss_r2i = F.cross_entropy(logits_masked.reshape(B * N, N), labels.reshape(-1))
        loss_i2r = F.cross_entropy(logits_masked.transpose(1, 2).reshape(B * N, N), labels.reshape(-1))
        self.contrast_loss = ((loss_r2i + loss_i2r) / 2.0).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * a_ir + rgb_aligned * a_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_DualContrast(nn.Module):








    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, self.groups, 7, padding=3), nn.Sigmoid())
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, 3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, 3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0
        self.tau = 0.07

    def _info_nce(self, feat_a, feat_b):
        B, C, H, W = feat_a.shape
        N = min(H * W, 64)
        idx = torch.randperm(H * W, device=feat_a.device)[:N]
        a = F.normalize(feat_a.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(feat_b.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        l1 = F.cross_entropy(logits.reshape(B * N, N), labels.reshape(-1))
        l2 = F.cross_entropy(logits.transpose(1, 2).reshape(B * N, N), labels.reshape(-1))
        return (l1 + l2) / 2.0

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = (offsets_raw[:, 0:self.groups * 2] * mask_exp).reshape(B * self.groups, 2, H, W)
        offset_ir  = (offsets_raw[:, self.groups * 2:self.groups * 4] * mask_exp).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)


        loss_pre  = self._info_nce(rgb_guided, ir_feat)
        loss_post = self._info_nce(rgb_aligned, ir_aligned)
        self.contrast_loss = ((loss_pre + loss_post) / 2.0).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * a_ir + rgb_aligned * a_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_SimCLR(nn.Module):








    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, self.groups, 7, padding=3), nn.Sigmoid())
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_offset_channels, 3, padding=1, bias=False)
        )

        self.proj_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, 3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0
        self.tau = 0.07

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = (offsets_raw[:, 0:self.groups * 2] * mask_exp).reshape(B * self.groups, 2, H, W)
        offset_ir  = (offsets_raw[:, self.groups * 2:self.groups * 4] * mask_exp).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)


        rgb_proj = self.proj_head(rgb_aligned); ir_proj = self.proj_head(ir_aligned)
        N = min(H * W, 64)
        idx = torch.randperm(H * W, device=rgb_proj.device)[:N]
        a = F.normalize(rgb_proj.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(ir_proj.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        l1 = F.cross_entropy(logits.reshape(B * N, N), labels.reshape(-1))
        l2 = F.cross_entropy(logits.transpose(1, 2).reshape(B * N, N), labels.reshape(-1))
        self.contrast_loss = ((l1 + l2) / 2.0).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * a_ir + rgb_aligned * a_rgb









class DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveLite(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, self.groups * 4, 3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, 3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0; self.tau = 0.07

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))

        offset_rgb = offsets_raw[:, 0:self.groups * 2].reshape(B * self.groups, 2, H, W)
        offset_ir  = offsets_raw[:, self.groups * 2:self.groups * 4].reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)

        N = min(H * W, 64); idx = torch.randperm(H * W, device=rgb_aligned.device)[:N]
        a = F.normalize(rgb_aligned.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(ir_aligned.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        l1 = F.cross_entropy(logits.reshape(B * N, N), labels.reshape(-1))
        l2 = F.cross_entropy(logits.transpose(1, 2).reshape(B * N, N), labels.reshape(-1))
        self.contrast_loss = ((l1 + l2) / 2.0).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * a_ir + rgb_aligned * a_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveMoreNeg(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, self.groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, self.groups * 4, 3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, 3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0
        self.tau = 0.1
        self.num_samples = 256

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2] * mask_exp
        offset_ir  = offsets_raw[:, self.groups * 2:self.groups * 4] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)


        N = min(H * W, self.num_samples)
        idx = torch.randperm(H * W, device=rgb_aligned.device)[:N]
        a = F.normalize(rgb_aligned.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(ir_aligned.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        l1 = F.cross_entropy(logits.reshape(B * N, N), labels.reshape(-1))
        l2 = F.cross_entropy(logits.transpose(1, 2).reshape(B * N, N), labels.reshape(-1))
        self.contrast_loss = ((l1 + l2) / 2.0).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * a_ir + rgb_aligned * a_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveAllLevels(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, self.groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, self.groups * 4, 3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.groups * 2, 3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0; self.tau = 0.07

    def _info_nce(self, fa, fb):
        B, _, H, W = fa.shape
        N = min(H * W, 64); idx = torch.randperm(H * W, device=fa.device)[:N]
        a = F.normalize(fa.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(fb.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        return (F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) +
                F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets_raw[:, 0:self.groups * 2] * mask_exp
        offset_ir  = offsets_raw[:, self.groups * 2:self.groups * 4] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)


        loss_pre  = self._info_nce(rgb_guided, ir_feat)
        loss_post = self._info_nce(rgb_aligned, ir_aligned)
        self.contrast_loss = ((loss_pre + loss_post) / 2.0).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * a_ir + rgb_aligned * a_rgb









class CrossModalOffsetAttn(nn.Module):

    def __init__(self, dim, groups, head_dim=64, pool_size=4):
        super().__init__()
        self.groups = groups; self.head_dim = head_dim
        self.num_heads = dim // head_dim; self.pool_size = pool_size
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim * 2, groups * 4, 1)
        self.scale = head_dim ** -0.5

    def forward(self, rgb, ir):
        B, C, H, W = rgb.shape

        ps = self.pool_size
        rgb_p = F.avg_pool2d(rgb, ps, ps)
        ir_p  = F.avg_pool2d(ir,  ps, ps)
        Hp, Wp = rgb_p.shape[2], rgb_p.shape[3]
        q = self.q_proj(ir_p).reshape(B, self.num_heads, self.head_dim, Hp*Wp).permute(0,1,3,2)
        k = self.k_proj(rgb_p).reshape(B, self.num_heads, self.head_dim, Hp*Wp)
        v = self.v_proj(rgb_p).reshape(B, self.num_heads, self.head_dim, Hp*Wp).permute(0,1,3,2)
        attn = torch.softmax((q @ k) * self.scale, dim=-1)
        out_p = (attn @ v).permute(0,1,3,2).reshape(B, C, Hp, Wp)
        out = F.interpolate(out_p, size=(H, W), mode='bilinear', align_corners=True)
        return self.out_proj(torch.cat([out, ir], dim=1))


class LearnableFrequencyBasis(nn.Module):

    def __init__(self, dim, num_bases=16):
        super().__init__()
        self.dim = dim; self.num_bases = num_bases

        self.h_basis = nn.Parameter(torch.randn(dim, 1, num_bases, 31) * 0.02)
        self.v_basis = nn.Parameter(torch.randn(dim, 1, num_bases, 31) * 0.02)
        self.h_weights = nn.Parameter(torch.ones(dim, num_bases, 1, 1) / num_bases)
        self.v_weights = nn.Parameter(torch.ones(dim, num_bases, 1, 1) / num_bases)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        w_h = F.softmax(self.h_weights, dim=1)
        w_v = F.softmax(self.v_weights, dim=1)

        x_h = F.conv2d(x, self.h_basis.reshape(C*self.num_bases, 1, 1, 31),
                       padding=(0, 15), groups=C)
        x_h = x_h.reshape(B, C, self.num_bases, H, W)
        x_h = (x_h * w_h.view(1, C, self.num_bases, 1, 1)).sum(dim=2)

        x_v = F.conv2d(x, self.v_basis.reshape(C*self.num_bases, 1, 31, 1),
                       padding=(15, 0), groups=C)
        x_v = x_v.reshape(B, C, self.num_bases, H, W)
        x_v = (x_v * w_v.view(1, C, self.num_bases, 1, 1)).sum(dim=2)
        return (x_h + x_v) * self.alpha + x * self.beta


class DecoupledFreqGuidedFusion_PMDA_ExpA_CrossAttn(nn.Module):





    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.cross_attn = CrossModalOffsetAttn(hidden_channels, groups)
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, groups * 2, 3, padding=1, bias=False)
        )
        self.contrast_loss = 0.0; self.tau = 0.07

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.cross_attn(rgb_guided, ir_feat)
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = (offsets_raw[:, 0:self.groups * 2] * mask_exp).reshape(B * self.groups, 2, H, W)
        offset_ir  = (offsets_raw[:, self.groups * 2:self.groups * 4] * mask_exp).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)
        N = min(H * W, 64); idx = torch.randperm(H * W, device=rgb_aligned.device)[:N]
        a = F.normalize(rgb_aligned.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(ir_aligned.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        l1 = F.cross_entropy(logits.reshape(B * N, N), labels.reshape(-1))
        l2 = F.cross_entropy(logits.transpose(1, 2).reshape(B * N, N), labels.reshape(-1))
        self.contrast_loss = ((l1 + l2) / 2.0).detach().item()
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * a_ir + rgb_aligned * a_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_LearnFreq(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = LearnableFrequencyBasis(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, groups * 4, 3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, groups * 2, 3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0; self.tau = 0.07

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = (offsets_raw[:, 0:self.groups * 2] * mask_exp).reshape(B * self.groups, 2, H, W)
        offset_ir  = (offsets_raw[:, self.groups * 2:self.groups * 4] * mask_exp).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)
        N = min(H * W, 64); idx = torch.randperm(H * W, device=rgb_aligned.device)[:N]
        a = F.normalize(rgb_aligned.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(ir_aligned.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        l1 = F.cross_entropy(logits.reshape(B * N, N), labels.reshape(-1))
        l2 = F.cross_entropy(logits.transpose(1, 2).reshape(B * N, N), labels.reshape(-1))
        self.contrast_loss = ((l1 + l2) / 2.0).detach().item()
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * a_ir + rgb_aligned * a_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_Uncertainty(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, groups * 4 + 2, 3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, groups * 2, 3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0; self.tau = 0.07

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = (offsets_raw[:, 0:self.groups * 2] * mask_exp).reshape(B * self.groups, 2, H, W)
        offset_ir  = (offsets_raw[:, self.groups * 2:self.groups * 4] * mask_exp).reshape(B * self.groups, 2, H, W)

        logvar_rgb = offsets_raw[:, self.groups * 4:self.groups * 4 + 1].mean(dim=(2,3), keepdim=True)
        logvar_ir  = offsets_raw[:, self.groups * 4 + 1:self.groups * 4 + 2].mean(dim=(2,3), keepdim=True)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)
        N = min(H * W, 64); idx = torch.randperm(H * W, device=rgb_aligned.device)[:N]
        a = F.normalize(rgb_aligned.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(ir_aligned.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        l1 = F.cross_entropy(logits.reshape(B * N, N), labels.reshape(-1))
        l2 = F.cross_entropy(logits.transpose(1, 2).reshape(B * N, N), labels.reshape(-1))

        uncert_weight = torch.exp(-0.5 * (logvar_rgb.squeeze() + logvar_ir.squeeze()))
        self.contrast_loss = ((l1 + l2) / 2.0 * uncert_weight.mean()).detach().item()
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * a_ir + rgb_aligned * a_rgb









class DecoupledFreqGuidedFusion_PMDA_ExpA_NeckFuse(nn.Module):




    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, groups * 2, 3, padding=1, bias=False)
        )

        self.se_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels // 4, 1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden_channels // 4, hidden_channels, 1), nn.Sigmoid()
        )

        self.refine = Conv(hidden_channels, hidden_channels, 3)
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0; self.tau = 0.07

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = (offsets_raw[:, 0:self.groups * 2] * mask_exp).reshape(B * self.groups, 2, H, W)
        offset_ir  = (offsets_raw[:, self.groups * 2:self.groups * 4] * mask_exp).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)

        N = min(H * W, 64); idx = torch.randperm(H * W, device=rgb_aligned.device)[:N]
        a = F.normalize(rgb_aligned.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(ir_aligned.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) +
                               F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        fused = ir_aligned * a_ir + rgb_aligned * a_rgb

        se_weight = self.se_fc(fused)
        fused = fused * se_weight
        fused = fused + self.refine(fused)
        return fused



class DecoupledFreqGuidedFusion_PMDA_ExpA_DynamicGroup(nn.Module):





    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, groups * 2, 3, padding=1, bias=False)
        )

        self.group_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden_channels // 2, groups, 1)
        )
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0; self.tau = 0.07

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = (offsets_raw[:, 0:self.groups * 2] * mask_exp).reshape(B * self.groups, 2, H, W)
        offset_ir  = (offsets_raw[:, self.groups * 2:self.groups * 4] * mask_exp).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)

        N = min(H * W, 64); idx = torch.randperm(H * W, device=rgb_aligned.device)[:N]
        a = F.normalize(rgb_aligned.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(ir_aligned.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) +
                               F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))

        group_logits = self.group_predictor(rgb_freq).squeeze(-1).squeeze(-1)
        group_weights = F.softmax(group_logits, dim=1)
        a_ir  = (aw[:, 0:self.groups] * group_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        a_rgb = (aw[:, self.groups:] * group_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        a_ir  = a_ir.repeat(1, self.hidden_channels, 1, 1)
        a_rgb = a_rgb.repeat(1, self.hidden_channels, 1, 1)
        return ir_aligned * a_ir + rgb_aligned * a_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_DCNAlign(nn.Module):








    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)

        out_offset_channels = self.groups * 4
        out_mod_channels = self.groups * 2
        total_out = out_offset_channels + out_mod_channels
        self.offset_mod_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, total_out, 3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, groups * 2, 3, padding=1, bias=False)
        )
        nn.init.normal_(self.offset_mod_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0; self.tau = 0.07

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        B, _, H, W = rgb_feat.shape

        raw = self.offset_mod_conv(torch.cat([rgb_guided, ir_feat], dim=1))

        offset_part = raw[:, :self.groups * 4]
        mod_part    = raw[:, self.groups * 4:]
        modulation_rgb = torch.sigmoid(mod_part[:, 0:self.groups])
        modulation_ir  = torch.sigmoid(mod_part[:, self.groups:])


        offset_rgb = offset_part[:, 0:self.groups * 2] * modulation_rgb.repeat_interleave(2, dim=1)
        offset_ir  = offset_part[:, self.groups * 2:self.groups * 4] * modulation_ir.repeat_interleave(2, dim=1)
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)

        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)

        N = min(H * W, 64); idx = torch.randperm(H * W, device=rgb_aligned.device)[:N]
        a = F.normalize(rgb_aligned.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(ir_aligned.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) +
                               F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()

        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return ir_aligned * a_ir + rgb_aligned * a_rgb



class DecoupledFreqGuidedFusion_PMDA_ExpA_ContrastiveNeckFuse(nn.Module):






    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False)
        )
        self.fusion_attn_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, groups * 2, 3, padding=1, bias=False)
        )
        self.se_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels // 4, 1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden_channels // 4, hidden_channels, 1), nn.Sigmoid()
        )
        self.refine = Conv(hidden_channels, hidden_channels, 3)
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0; self.tau = 0.07

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        offsets_raw = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = (offsets_raw[:, 0:self.groups * 2] * mask_exp).reshape(B * self.groups, 2, H, W)
        offset_ir  = (offsets_raw[:, self.groups * 2:self.groups * 4] * mask_exp).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W),
            (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f),
            mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        rgb_aligned = G(offset_rgb, rgb_guided); ir_aligned = G(offset_ir, ir_feat)
        N = min(H * W, 64); idx = torch.randperm(H * W, device=rgb_aligned.device)[:N]
        a = F.normalize(rgb_aligned.flatten(2)[:, :, idx], dim=1)
        b = F.normalize(ir_aligned.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) +
                               F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()
        fusion_input = torch.cat([rgb_aligned, ir_aligned, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fusion_input))
        a_ir = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        a_rgb = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        fused = ir_aligned * a_ir + rgb_aligned * a_rgb
        fused = fused * self.se_fc(fused) + self.refine(fused)
        return fused









class DecoupledFreqGuidedFusion_PMDA_Final_LearnTau(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, groups * 2, 3, padding=1, bias=False))
        self.se_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels // 4, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels // 4, hidden_channels, 1), nn.Sigmoid())
        self.refine = Conv(hidden_channels, hidden_channels, 3)
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.log_tau = nn.Parameter(torch.tensor(-2.66))
        self.contrast_loss = 0.0

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        o = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        m = edge_mask.repeat_interleave(2, dim=1)
        or_ = (o[:, 0:self.groups * 2] * m).reshape(B * self.groups, 2, H, W)
        oi_ = (o[:, self.groups * 2:self.groups * 4] * m).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        ra = G(or_, rgb_guided); ia = G(oi_, ir_feat)
        tau = torch.exp(self.log_tau) + 0.01
        N = min(H * W, 64); idx = torch.randperm(H * W, device=ra.device)[:N]
        a = F.normalize(ra.flatten(2)[:, :, idx], dim=1); b = F.normalize(ia.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) + F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()
        fi = torch.cat([ra, ia, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fi))
        fused = ia * aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1) + ra * aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return fused * self.se_fc(fused) + self.refine(fused)



class DecoupledFreqGuidedFusion_PMDA_Final_DeepSE(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, groups * 2, 3, padding=1, bias=False))

        self.se_deep = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels // 8, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 8, hidden_channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, hidden_channels, 1), nn.Sigmoid()
        )

        self.refine1 = Conv(hidden_channels, hidden_channels, 3)
        self.refine2 = Conv(hidden_channels, hidden_channels, 3)
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0; self.tau = 0.07

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        o = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        m = edge_mask.repeat_interleave(2, dim=1)
        or_ = (o[:, 0:self.groups * 2] * m).reshape(B * self.groups, 2, H, W)
        oi_ = (o[:, self.groups * 2:self.groups * 4] * m).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        ra = G(or_, rgb_guided); ia = G(oi_, ir_feat)
        N = min(H * W, 64); idx = torch.randperm(H * W, device=ra.device)[:N]
        a = F.normalize(ra.flatten(2)[:, :, idx], dim=1); b = F.normalize(ia.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) + F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()
        fi = torch.cat([ra, ia, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fi))
        fused = ia * aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1) + ra * aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        fused = fused * self.se_deep(fused) + self.refine1(fused) + self.refine2(fused)
        return fused



class DecoupledFreqGuidedFusion_PMDA_Final_MultiLevel(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, groups * 2, 3, padding=1, bias=False))
        self.se_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels // 4, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels // 4, hidden_channels, 1), nn.Sigmoid())
        self.refine = Conv(hidden_channels, hidden_channels, 3)
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.contrast_loss = 0.0; self.tau = 0.07

    def _nce(self, fa, fb):
        B, _, H, W = fa.shape
        N = min(H * W, 64); idx = torch.randperm(H * W, device=fa.device)[:N]
        a = F.normalize(fa.flatten(2)[:, :, idx], dim=1); b = F.normalize(fb.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / self.tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        return (F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) + F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        o = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1))
        m = edge_mask.repeat_interleave(2, dim=1)
        or_ = (o[:, 0:self.groups * 2] * m).reshape(B * self.groups, 2, H, W)
        oi_ = (o[:, self.groups * 2:self.groups * 4] * m).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        ra = G(or_, rgb_guided); ia = G(oi_, ir_feat)

        loss_pre  = self._nce(rgb_guided, ir_feat)
        loss_post = self._nce(ra, ia)

        loss_group = 0
        for g in range(self.groups):
            c_start = g * (self.hidden_channels // self.groups)
            c_end = c_start + (self.hidden_channels // self.groups)
            loss_group += self._nce(ra[:, c_start:c_end], ia[:, c_start:c_end])
        loss_group /= self.groups
        self.contrast_loss = ((loss_pre + loss_post + loss_group) / 3.0).detach().item()

        fi = torch.cat([ra, ia, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fi))
        fused = ia * aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1) + ra * aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return fused * self.se_fc(fused) + self.refine(fused)








class DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauDeepSE(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, groups * 2, 3, padding=1, bias=False))
        self.se_deep = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels // 8, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels // 8, hidden_channels // 4, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels // 4, hidden_channels, 1), nn.Sigmoid())
        self.refine1 = Conv(hidden_channels, hidden_channels, 3)
        self.refine2 = Conv(hidden_channels, hidden_channels, 3)
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.log_tau = nn.Parameter(torch.tensor(-2.66))
        self.contrast_loss = 0.0

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        o = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1)); m = edge_mask.repeat_interleave(2, dim=1)
        or_ = (o[:, 0:self.groups * 2] * m).reshape(B * self.groups, 2, H, W)
        oi_ = (o[:, self.groups * 2:self.groups * 4] * m).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        ra = G(or_, rgb_guided); ia = G(oi_, ir_feat)
        tau = torch.exp(self.log_tau) + 0.01
        N = min(H * W, 64); idx = torch.randperm(H * W, device=ra.device)[:N]
        a = F.normalize(ra.flatten(2)[:, :, idx], dim=1); b = F.normalize(ia.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) + F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()
        fi = torch.cat([ra, ia, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fi))
        fused = ia * aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1) + ra * aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return fused * self.se_deep(fused) + self.refine1(fused) + self.refine2(fused)



class DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTau05(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, groups * 2, 3, padding=1, bias=False))
        self.se_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels // 4, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels // 4, hidden_channels, 1), nn.Sigmoid())
        self.refine = Conv(hidden_channels, hidden_channels, 3)
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.log_tau = nn.Parameter(torch.tensor(-3.0))
        self.contrast_loss = 0.0

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        o = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1)); m = edge_mask.repeat_interleave(2, dim=1)
        or_ = (o[:, 0:self.groups * 2] * m).reshape(B * self.groups, 2, H, W)
        oi_ = (o[:, self.groups * 2:self.groups * 4] * m).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        ra = G(or_, rgb_guided); ia = G(oi_, ir_feat)
        tau = torch.exp(self.log_tau) + 0.01
        N = min(H * W, 64); idx = torch.randperm(H * W, device=ra.device)[:N]
        a = F.normalize(ra.flatten(2)[:, :, idx], dim=1); b = F.normalize(ia.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) + F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()
        fi = torch.cat([ra, ia, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fi))
        fused = ia * aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1) + ra * aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return fused * self.se_fc(fused) + self.refine(fused)



class DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauCos(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, groups * 2, 3, padding=1, bias=False))
        self.se_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels // 4, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels // 4, hidden_channels, 1), nn.Sigmoid())
        self.refine = Conv(hidden_channels, hidden_channels, 3)
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.log_tau = nn.Parameter(torch.tensor(-2.66))
        self.contrast_loss = 0.0

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        o = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1)); m = edge_mask.repeat_interleave(2, dim=1)
        or_ = (o[:, 0:self.groups * 2] * m).reshape(B * self.groups, 2, H, W)
        oi_ = (o[:, self.groups * 2:self.groups * 4] * m).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        ra = G(or_, rgb_guided); ia = G(oi_, ir_feat)
        tau = torch.exp(self.log_tau) + 0.01
        N = min(H * W, 64); idx = torch.randperm(H * W, device=ra.device)[:N]
        a = F.normalize(ra.flatten(2)[:, :, idx], dim=1); b = F.normalize(ia.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        nce_loss = (F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) + F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0

        cos_sim = F.cosine_similarity(ra.flatten(1), ia.flatten(1), dim=1).mean()
        self.contrast_loss = (nce_loss + (1.0 - cos_sim) * 0.5).detach().item()
        fi = torch.cat([ra, ia, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fi))
        fused = ia * aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1) + ra * aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return fused * self.se_fc(fused) + self.refine(fused)



class DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauDeepSECos(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3); self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, groups * 2, 3, padding=1, bias=False))
        self.se_deep = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels // 8, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels // 8, hidden_channels // 4, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels // 4, hidden_channels, 1), nn.Sigmoid())
        self.refine1 = Conv(hidden_channels, hidden_channels, 3); self.refine2 = Conv(hidden_channels, hidden_channels, 3)
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)
        self.log_tau = nn.Parameter(torch.tensor(-2.66))
        self.contrast_loss = 0.0

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat)
        rgb_guided = rgb_feat * (1 - torch.sigmoid(self.gating_conv_rgb(rgb_feat))) + rgb_freq * torch.sigmoid(self.gating_conv_rgb(rgb_feat))
        edge_mask = self.edge_mask_gen(rgb_freq)
        B, _, H, W = rgb_feat.shape
        o = self.offset_conv(torch.cat([rgb_guided, ir_feat], dim=1)); m = edge_mask.repeat_interleave(2, dim=1)
        or_ = (o[:, 0:self.groups * 2] * m).reshape(B * self.groups, 2, H, W)
        oi_ = (o[:, self.groups * 2:self.groups * 4] * m).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        ra = G(or_, rgb_guided); ia = G(oi_, ir_feat)
        tau = torch.exp(self.log_tau) + 0.01
        N = min(H * W, 64); idx = torch.randperm(H * W, device=ra.device)[:N]
        a = F.normalize(ra.flatten(2)[:, :, idx], dim=1); b = F.normalize(ia.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        nce = (F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) + F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0
        cos = (1.0 - F.cosine_similarity(ra.flatten(1), ia.flatten(1), dim=1).mean())
        self.contrast_loss = (nce + cos * 0.3).detach().item()
        fi = torch.cat([ra, ia, rgb_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fi))
        fused = ia * aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1) + ra * aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return fused * self.se_deep(fused) + self.refine1(fused) + self.refine2(fused)









class DecoupledFreqGuidedFusion_RTDETR_Enhanced(nn.Module):






    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels; self.tiny_mode = tiny_mode
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.reduce_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True))
        self.num_tokens = 4
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=4, dim_feedforward=hidden_channels*2, dropout=0.0, activation='gelu', batch_first=True)
        self.to_logits = nn.Conv2d(hidden_channels, groups * 2, 3, padding=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        device = rgb_feat.device; dtype = rgb_feat.dtype
        B, _, H, W = rgb_feat.shape
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)
        rgb_feat = self.rgb_conv(rgb_feat)
        ir_feat_conv = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat_conv)
        gate = torch.sigmoid(self.gating_conv(ir_feat_conv))
        ir_guided = ir_feat_conv * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)
        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets[:, 0:self.groups * 2] * mask_exp
        offset_ir  = offsets[:, self.groups * 2:self.groups * 4] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).to(dtype).unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=device, dtype=dtype).view(1, 2, 1, 1)
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1), align_corners=False).reshape(B, -1, H, W)
        ra = G(offset_rgb, rgb_feat); ia = G(offset_ir, ir_guided)

        x_in = self.reduce_conv(torch.cat([ra, ia, ir_freq], dim=1))
        B2, C2, H2, W2 = x_in.shape; N = self.num_tokens
        tokens = F.adaptive_avg_pool2d(x_in, (N, N)).flatten(2).transpose(1, 2)
        tokens = self.transformer_encoder(tokens)
        tokens = tokens.transpose(1, 2).reshape(B2, C2, N, N)
        tokens = F.interpolate(tokens, size=(H2, W2), mode='bilinear', align_corners=True)
        attn = self.to_logits(tokens)
        aw = 1 + torch.tanh(attn)
        ai = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        ar = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        fused = ra * ar + ia * ai

        residual = self.gamma * tokens
        conv = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1).to(device) if not hasattr(self, '_final') else self._final

        return fused + residual



class DecoupledFreqGuidedFusion_RTDETR_ExpA(nn.Module):






    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels; self.tiny_mode = tiny_mode
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.reduce_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True))
        self.num_tokens = 4
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=4, dim_feedforward=hidden_channels*2, dropout=0.0, activation='gelu', batch_first=True)
        self.to_logits = nn.Conv2d(hidden_channels, groups * 2, 3, padding=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        device = rgb_feat.device; dtype = rgb_feat.dtype
        B, _, H, W = rgb_feat.shape
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)

        rgb_feat_conv = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat_conv)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat_conv))
        rgb_guided = rgb_feat_conv * (1 - gate_rgb) + rgb_freq * gate_rgb
        ir_feat_conv = self.ir_conv(ir_feat)
        edge_mask = self.edge_mask_gen(rgb_freq)
        offsets = self.offset_conv(torch.cat([rgb_guided, ir_feat_conv], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets[:, 0:self.groups * 2] * mask_exp
        offset_ir  = offsets[:, self.groups * 2:self.groups * 4] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).to(dtype).unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=device, dtype=dtype).view(1, 2, 1, 1)
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1), align_corners=False).reshape(B, -1, H, W)
        ra = G(offset_rgb, rgb_guided); ia = G(offset_ir, ir_feat_conv)
        x_in = self.reduce_conv(torch.cat([ra, ia, rgb_freq], dim=1))
        B2, C2, H2, W2 = x_in.shape; N = self.num_tokens
        tokens = F.adaptive_avg_pool2d(x_in, (N, N)).flatten(2).transpose(1, 2)
        tokens = self.transformer_encoder(tokens)
        tokens = tokens.transpose(1, 2).reshape(B2, C2, N, N)
        tokens = F.interpolate(tokens, size=(H2, W2), mode='bilinear', align_corners=True)
        attn = self.to_logits(tokens)
        aw = 1 + torch.tanh(attn)
        ai = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        ar = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        fused = ra * ar + ia * ai
        return fused + self.gamma * tokens



class DecoupledFreqGuidedFusion_RTDETR_NeckFuse(nn.Module):





    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels; self.tiny_mode = tiny_mode
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.reduce_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True))
        self.num_tokens = 4
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=4, dim_feedforward=hidden_channels*2, dropout=0.0, activation='gelu', batch_first=True)
        self.to_logits = nn.Conv2d(hidden_channels, groups * 2, 3, padding=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        self.se_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels//4, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels//4, hidden_channels, 1), nn.Sigmoid())
        self.refine = Conv(hidden_channels, hidden_channels, 3)
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        device = rgb_feat.device; dtype = rgb_feat.dtype
        B, _, H, W = rgb_feat.shape
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)
        rgb_feat = self.rgb_conv(rgb_feat)
        ir_feat_conv = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat_conv)
        gate = torch.sigmoid(self.gating_conv(ir_feat_conv))
        ir_guided = ir_feat_conv * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)
        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = offsets[:, 0:self.groups * 2] * mask_exp
        offset_ir  = offsets[:, self.groups * 2:self.groups * 4] * mask_exp
        offset_rgb = offset_rgb.reshape(B * self.groups, 2, H, W)
        offset_ir  = offset_ir.reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).to(dtype).unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=device, dtype=dtype).view(1, 2, 1, 1)
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1), align_corners=False).reshape(B, -1, H, W)
        ra = G(offset_rgb, rgb_feat); ia = G(offset_ir, ir_guided)
        x_in = self.reduce_conv(torch.cat([ra, ia, ir_freq], dim=1))
        B2, C2, H2, W2 = x_in.shape; N = self.num_tokens
        tokens = F.adaptive_avg_pool2d(x_in, (N, N)).flatten(2).transpose(1, 2)
        tokens = self.transformer_encoder(tokens)
        tokens = tokens.transpose(1, 2).reshape(B2, C2, N, N)
        tokens = F.interpolate(tokens, size=(H2, W2), mode='bilinear', align_corners=True)
        attn = self.to_logits(tokens)
        aw = 1 + torch.tanh(attn)
        ai = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        ar = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        fused = ra * ar + ia * ai + self.gamma * tokens

        fused = fused * self.se_fc(fused) + self.refine(fused)
        return fused










class DecoupledFreqGuidedFusion_RTDETR_LearnTau(nn.Module):






    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels; self.tiny_mode = tiny_mode

        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())

        self.reduce_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True))
        self.num_tokens = 4
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=4, dim_feedforward=hidden_channels*2, dropout=0.0, activation='gelu', batch_first=True)
        self.to_logits = nn.Conv2d(hidden_channels, groups * 2, 3, padding=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        self.se_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels//4, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels//4, hidden_channels, 1), nn.Sigmoid())
        self.refine = Conv(hidden_channels, hidden_channels, 3)

        self.log_tau = nn.Parameter(torch.tensor(-2.66))
        self.contrast_loss = 0.0
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        device = rgb_feat.device; dtype = rgb_feat.dtype
        B, _, H, W = rgb_feat.shape
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)

        rgb_feat_conv = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat_conv)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat_conv))
        rgb_guided = rgb_feat_conv * (1 - gate_rgb) + rgb_freq * gate_rgb
        ir_feat_conv = self.ir_conv(ir_feat)
        edge_mask = self.edge_mask_gen(rgb_freq)

        offsets = self.offset_conv(torch.cat([rgb_guided, ir_feat_conv], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = (offsets[:, 0:self.groups * 2] * mask_exp).reshape(B * self.groups, 2, H, W)
        offset_ir  = (offsets[:, self.groups * 2:self.groups * 4] * mask_exp).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).to(dtype).unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=device, dtype=dtype).view(1, 2, 1, 1)
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1), align_corners=False).reshape(B, -1, H, W)
        ra = G(offset_rgb, rgb_guided); ia = G(offset_ir, ir_feat_conv)

        tau = torch.exp(self.log_tau) + 0.01
        N = min(H * W, 64); idx = torch.randperm(H * W, device=device)[:N]
        a = F.normalize(ra.flatten(2)[:, :, idx], dim=1); b = F.normalize(ia.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / tau
        labels = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) + F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()

        x_in = self.reduce_conv(torch.cat([ra, ia, rgb_freq], dim=1))
        B2, C2, H2, W2 = x_in.shape; tN = self.num_tokens
        tokens = F.adaptive_avg_pool2d(x_in, (tN, tN)).flatten(2).transpose(1, 2)
        tokens = self.transformer_encoder(tokens)
        tokens = tokens.transpose(1, 2).reshape(B2, C2, tN, tN)
        tokens = F.interpolate(tokens, size=(H2, W2), mode='bilinear', align_corners=True)
        attn = self.to_logits(tokens)
        aw = 1 + torch.tanh(attn)
        ai = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        ar = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        fused = ra * ar + ia * ai + self.gamma * tokens

        fused = fused * self.se_fc(fused) + self.refine(fused)
        return fused



class DecoupledFreqGuidedFusion_RTDETR_LearnTauDeepSE(nn.Module):





    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels; self.tiny_mode = tiny_mode
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer_rgb = FFM(hidden_channels)
        self.gating_conv_rgb = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False)
        )
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.reduce_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True))
        self.num_tokens = 4
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=4, dim_feedforward=hidden_channels*2, dropout=0.0, activation='gelu', batch_first=True)
        self.to_logits = nn.Conv2d(hidden_channels, groups * 2, 3, padding=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        self.se_deep = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels//8, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels//8, hidden_channels//4, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels//4, hidden_channels, 1), nn.Sigmoid())
        self.refine1 = Conv(hidden_channels, hidden_channels, 3)
        self.refine2 = Conv(hidden_channels, hidden_channels, 3)
        self.log_tau = nn.Parameter(torch.tensor(-2.66))
        self.contrast_loss = 0.0
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        device = rgb_feat.device; dtype = rgb_feat.dtype
        B, _, H, W = rgb_feat.shape
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)
        rgb_feat_conv = self.rgb_conv(rgb_feat)
        rgb_freq = self.frequency_enhancer_rgb(rgb_feat_conv)
        gate_rgb = torch.sigmoid(self.gating_conv_rgb(rgb_feat_conv))
        rgb_guided = rgb_feat_conv * (1 - gate_rgb) + rgb_freq * gate_rgb
        ir_feat_conv = self.ir_conv(ir_feat)
        edge_mask = self.edge_mask_gen(rgb_freq)
        offsets = self.offset_conv(torch.cat([rgb_guided, ir_feat_conv], dim=1))
        mask_exp = edge_mask.repeat_interleave(2, dim=1)
        offset_rgb = (offsets[:, 0:self.groups * 2] * mask_exp).reshape(B * self.groups, 2, H, W)
        offset_ir  = (offsets[:, self.groups * 2:self.groups * 4] * mask_exp).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).to(dtype).unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=device, dtype=dtype).view(1, 2, 1, 1)
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1), align_corners=False).reshape(B, -1, H, W)
        ra = G(offset_rgb, rgb_guided); ia = G(offset_ir, ir_feat_conv)
        tau = torch.exp(self.log_tau) + 0.01
        N = min(H * W, 64); idx = torch.randperm(H * W, device=device)[:N]
        a = F.normalize(ra.flatten(2)[:, :, idx], dim=1); b = F.normalize(ia.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / tau
        labels = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) + F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()
        x_in = self.reduce_conv(torch.cat([ra, ia, rgb_freq], dim=1))
        B2, C2, H2, W2 = x_in.shape; tN = self.num_tokens
        tokens = F.adaptive_avg_pool2d(x_in, (tN, tN)).flatten(2).transpose(1, 2)
        tokens = self.transformer_encoder(tokens)
        tokens = tokens.transpose(1, 2).reshape(B2, C2, tN, tN)
        tokens = F.interpolate(tokens, size=(H2, W2), mode='bilinear', align_corners=True)
        attn = self.to_logits(tokens)
        aw = 1 + torch.tanh(attn)
        ai = aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        ar = aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        fused = ra * ar + ia * ai + self.gamma * tokens

        fused = fused * self.se_deep(fused) + self.refine1(fused) + self.refine2(fused)
        return fused








class DecoupledFreqGuidedFusion_PMDA_Final_LearnTau_IR(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, groups * 2, 3, padding=1, bias=False))
        self.se_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels//4, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels//4, hidden_channels, 1), nn.Sigmoid())
        self.refine = Conv(hidden_channels, hidden_channels, 3)
        self.log_tau = nn.Parameter(torch.tensor(-2.66))
        self.contrast_loss = 0.0
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape
        o = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1)); m = edge_mask.repeat_interleave(2, dim=1)
        or_ = (o[:, 0:self.groups * 2] * m).reshape(B * self.groups, 2, H, W)
        oi_ = (o[:, self.groups * 2:self.groups * 4] * m).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        ra = G(or_, rgb_feat); ia = G(oi_, ir_guided)
        tau = torch.exp(self.log_tau) + 0.01
        N = min(H * W, 64); idx = torch.randperm(H * W, device=ir_feat.device)[:N]
        a = F.normalize(ra.flatten(2)[:, :, idx], dim=1); b = F.normalize(ia.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) + F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()
        fi = torch.cat([ra, ia, ir_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fi))
        fused = ia * aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1) + ra * aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return fused * self.se_fc(fused) + self.refine(fused)



class DecoupledFreqGuidedFusion_PMDA_Ultimate_LearnTauDeepSE_IR(nn.Module):

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=4):
        super().__init__()
        self.groups = groups; self.hidden_channels = hidden_channels
        self.focus = Focus(); self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.edge_mask_gen = nn.Sequential(nn.Conv2d(hidden_channels, groups, 7, padding=3), nn.Sigmoid())
        self.offset_conv = nn.Sequential(Conv(hidden_channels * 2, 128, 1), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, groups * 4, 3, padding=1, bias=False))
        self.fusion_attn_conv = nn.Sequential(nn.Conv2d(hidden_channels * 3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, groups * 2, 3, padding=1, bias=False))
        self.se_deep = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_channels, hidden_channels//8, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels//8, hidden_channels//4, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels//4, hidden_channels, 1), nn.Sigmoid())
        self.refine1 = Conv(hidden_channels, hidden_channels, 3); self.refine2 = Conv(hidden_channels, hidden_channels, 3)
        self.log_tau = nn.Parameter(torch.tensor(-2.66))
        self.contrast_loss = 0.0
        nn.init.normal_(self.offset_conv[4].weight, mean=0.0, std=0.01)

    def forward(self, x):
        rgb_feat, ir_feat = x
        ir_feat = self.focus(ir_feat); ir_feat = self.ir_focus_conv(ir_feat); ir_feat = self.ir_conv(ir_feat)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape
        o = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1)); m = edge_mask.repeat_interleave(2, dim=1)
        or_ = (o[:, 0:self.groups * 2] * m).reshape(B * self.groups, 2, H, W)
        oi_ = (o[:, self.groups * 2:self.groups * 4] * m).reshape(B * self.groups, 2, H, W)
        gy, gx = torch.meshgrid(torch.arange(H, device=ir_feat.device), torch.arange(W, device=ir_feat.device), indexing='ij')
        bg = torch.stack((gx, gy), dim=0).float().unsqueeze(0).repeat(B * self.groups, 1, 1, 1)
        nz = torch.tensor([W - 1, H - 1], device=ir_feat.device).view(1, 2, 1, 1).float()
        G = lambda o, f: F.grid_sample(f.reshape(B * self.groups, -1, H, W), (2.0 * (bg + o) / nz - 1.0).permute(0, 2, 3, 1).type_as(f), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        ra = G(or_, rgb_feat); ia = G(oi_, ir_guided)
        tau = torch.exp(self.log_tau) + 0.01
        N = min(H * W, 64); idx = torch.randperm(H * W, device=ir_feat.device)[:N]
        a = F.normalize(ra.flatten(2)[:, :, idx], dim=1); b = F.normalize(ia.flatten(2)[:, :, idx], dim=1)
        logits = torch.bmm(a.transpose(1, 2), b) / tau
        labels = torch.arange(N, device=logits.device).unsqueeze(0).repeat(B, 1)
        self.contrast_loss = ((F.cross_entropy(logits.reshape(B*N,N), labels.reshape(-1)) + F.cross_entropy(logits.transpose(1,2).reshape(B*N,N), labels.reshape(-1))) / 2.0).detach().item()
        fi = torch.cat([ra, ia, ir_freq], dim=1)
        aw = 1 + torch.tanh(self.fusion_attn_conv(fi))
        fused = ia * aw[:, 0:self.groups].repeat_interleave(self.hidden_channels // self.groups, dim=1) + ra * aw[:, self.groups:].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        return fused * self.se_deep(fused) + self.refine1(fused) + self.refine2(fused)

