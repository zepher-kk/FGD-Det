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
           'ContextGuideFusionModuleV2','DecoupledFreqGuidedFusion_re']


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
    # Changing the dimension of the Tensor
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
        # 1. 记录原始输入精度 (如 float16)
        orig_dtype = x.dtype
        
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        # 2. 强制转为 float32 以规避 ComplexHalf 算子缺失问题
        x2_32 = x2.to(torch.float32)
        x2_fft = torch.fft.fft2(x2_32, norm='backward')

        # 3. 确保 x1 也是 float32 参与计算
        x1_32 = x1.to(torch.float32)
        out = x1_32 * x2_fft

        # 4. 执行逆变换
        out = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
        
        # 5. 取绝对值后转回原始精度
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

        # 改进后的 SCA 部分
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ffm = FFM(dim)

        # 通道注意力
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
        
        # --- 频域计算精度保护开始 ---
        out_32 = out.to(torch.float32)
        x_fft = torch.fft.fft2(out_32, norm='backward')
        
        x_att_32 = x_att.to(torch.float32)
        x_fft = x_att_32 * x_fft
        
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca).to(orig_dtype) 
        # --- 频域计算精度保护结束 ---

        # 后续 SCA 分支计算
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


class ADown(nn.Module):  # Downsample x2分支
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


class FrequencyFocusedDownSampling(nn.Module):  # Downsample x2分支 with parallel FGM
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.ffm = FFM(self.c)  # FGM 模块处理 x2 分支

        # 1x1 卷积用于在拼接后减少通道数
        self.conv_reduce = Conv(self.c * 2, self.c, 1, 1)

        # 新增的卷积层用于调整 fgm_out 的空间尺寸
        self.conv_resize = Conv(self.c, self.c, 3, 2, 1)

    # 经过池化后分成两个分支，一个分支经过 cv1 处理，另一个分支经过 fgm + maxpool cv2 处理，然后将两个分支拼接在一起，最后使用 1x1 卷积将通道数减少到预期的值。公式写一个表达一下，x1,x2用文字描述一下是什么，cv1,cv2也是呀
    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)

        # 并联处理 x2 分支
        fgm_out = self.ffm(x2)  # FGM 处理的输出
        fgm_out = self.conv_resize(fgm_out)  # 调整 fgm_out 的空间尺寸
        pooled_out = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        pooled_out = self.cv2(pooled_out)

        # 将 FGM 输出和 MaxPool2d + Conv 输出拼接
        x2 = torch.cat((fgm_out, pooled_out), 1)

        # 使用 1x1 卷积将通道数减少到预期的值
        x2 = self.conv_reduce(x2)

        return torch.cat((x1, x2), 1)

class FrequencyFocusedDownSampling2(nn.Module):  # Downsample x2分支 with parallel FGM
    def __init__(self, c1, c2):
        super().__init__()
        # 🌟 核心修复：分离输入通道和输出通道的逻辑
        self.c_in = c1 // 2   # 劈裂后的实际输入通道
        self.c_out = c2 // 2  # 我们期望每个分支最终输出的通道
        
        # cv1 负责将 x1 从 c_in 映射到 c_out，并完成下采样
        self.cv1 = Conv(self.c_in, self.c_out, 3, 2, 1)
        
        # cv2 负责池化后的 x2 通道映射
        self.cv2 = Conv(self.c_in, self.c_out, 1, 1, 0)
        
        # 🌟 FFM 必须接收它真实的输入通道 c_in！
        self.ffm = FFM(self.c_in)  

        # 🌟 调整 fgm_out 空间尺寸的同时，也必须把通道从 c_in 映射到 c_out
        self.conv_resize = Conv(self.c_in, self.c_out, 3, 2, 1)

        # 1x1 卷积用于在 x2 内部拼接后减少通道数
        self.conv_reduce = Conv(self.c_out * 2, self.c_out, 1, 1)

    def forward(self, x):
        # 这里的 avg_pool2d 会导致 40x40 变成 39x39，是正常的特征重叠过渡
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        
        # 劈裂后，x1 和 x2 各有 c_in 个通道
        x1, x2 = x.chunk(2, 1)
        
        # x1 分支处理 (输出尺寸变回正常的下采样尺寸，如 20x20)
        x1 = self.cv1(x1)

        # 并联处理 x2 分支
        fgm_out = self.ffm(x2)  
        fgm_out = self.conv_resize(fgm_out)  # 尺寸变 20x20，通道变 c_out
        
        pooled_out = torch.nn.functional.max_pool2d(x2, 3, 2, 1) # 尺寸变 20x20
        pooled_out = self.cv2(pooled_out) # 通道变 c_out

        # 将 FGM 输出和 MaxPool2d + Conv 输出拼接
        x2 = torch.cat((fgm_out, pooled_out), 1)

        # 使用 1x1 卷积将通道数减少到预期的 c_out
        x2 = self.conv_reduce(x2)

        # 最终合并 x1 (c_out) 和 x2 (c_out) 得到总通道数 c2
        return torch.cat((x1, x2), 1)
    
class SemanticAlignmenCalibration(nn.Module):  #
    def __init__(self, inc):
        super(SemanticAlignmenCalibration, self).__init__()
        hidden_channels = inc[0]

        self.groups = 2
        self.spatial_conv = Conv(inc[0], hidden_channels, 3)  # 用于处理高分辨率的空间特征
        self.semantic_conv = Conv(inc[1], hidden_channels, 3)  # 用于处理低分辨率的语义特征

        # FGM模块：用于在频域中增强特征
        self.frequency_enhancer = FFM(hidden_channels)
        # 门控卷积：结合空间和频域特征
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)

        # 用于生成偏移量的卷积序列
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64),  # 处理拼接后的特征
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)  # 生成偏移量
        )

        self.init_weights()
        self.offset_conv[1].weight.data.zero_()  # 初始化最后一层卷积的权重为零

    def init_weights(self):
        # 初始化卷积层的权重
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        coarse_features, semantic_features = x
        batch_size, _, out_h, out_w = coarse_features.size()

        # 处理低分辨率的语义特征 (1/8 下采样)
        semantic_features = self.semantic_conv(semantic_features)
        semantic_features = F.interpolate(semantic_features, coarse_features.size()[2:], mode='bilinear',
                                          align_corners=True)

        # 频域增强特征
        enhanced_frequency = self.frequency_enhancer(semantic_features)

        # 门控机制融合频域和空间域的特征
        gate = torch.sigmoid(self.gating_conv(semantic_features))
        fused_features = semantic_features * (1 - gate) + enhanced_frequency * gate

        # 处理高分辨率的空间特征 (1/8 下采样)
        coarse_features = self.spatial_conv(coarse_features)

        # 拼接处理后的空间特征和融合后的特征
        conv_results = self.offset_conv(torch.cat([coarse_features, fused_features], 1))

        # 调整特征维度以适应分组
        fused_features = fused_features.reshape(batch_size * self.groups, -1, out_h, out_w)
        coarse_features = coarse_features.reshape(batch_size * self.groups, -1, out_h, out_w)

        # 获取偏移量
        offset_low = conv_results[:, 0:self.groups * 2, :, :].reshape(batch_size * self.groups, -1, out_h, out_w)
        offset_high = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(batch_size * self.groups, -1,
                                                                                     out_h, out_w)

        # 生成归一化网格用于偏移校正
        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(fused_features).to(fused_features.device)
        grid_w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        grid_h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        base_grid = torch.cat((grid_h.unsqueeze(2), grid_w.unsqueeze(2)), 2)
        base_grid = base_grid.repeat(batch_size * self.groups, 1, 1, 1).type_as(fused_features).to(
            fused_features.device)

        # 使用生成的偏移量对网格进行调整
        adjusted_grid_l = base_grid + offset_low.permute(0, 2, 3, 1) / normalization_factors
        adjusted_grid_h = base_grid + offset_high.permute(0, 2, 3, 1) / normalization_factors

        # 进行特征采样
        coarse_features = F.grid_sample(coarse_features, adjusted_grid_l.type_as(coarse_features), align_corners=True)
        fused_features = F.grid_sample(fused_features, adjusted_grid_h.type_as(fused_features), align_corners=True)

        # 调整维度回到原始形状
        coarse_features = coarse_features.reshape(batch_size, -1, out_h, out_w)
        fused_features = fused_features.reshape(batch_size, -1, out_h, out_w)

        # 融合增强后的特征
        attention_weights = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        final_features = fused_features * attention_weights[:, 0:1, :, :] + coarse_features * attention_weights[:, 1:2,
                                                                                              :, :]

        return final_features


#########################################################wavlet#########################################################

import pywt


class WaveletTransform(nn.Module):
    """可微分小波变换模块（支持自动求导）"""
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

        # 从pywt库加载小波滤波器系数
        coeffs = pywt.Wavelet(wavelet).filter_bank
        # 分解滤波器调整为四维 (out_c, in_c, kernel_h, kernel_w)
        self.dec_lo = nn.Parameter(
            torch.tensor(coeffs[0], dtype=torch.float32).view(1, 1, 1, -1)
        )
        self.dec_hi = nn.Parameter(
            torch.tensor(coeffs[1], dtype=torch.float32).view(1, 1, 1, -1)
        )

        # 重构滤波器同理
        self.rec_lo = nn.Parameter(
            torch.tensor(coeffs[2], dtype=torch.float32).view(1, 1, 1, -1)
        )
        self.rec_hi = nn.Parameter(
            torch.tensor(coeffs[3], dtype=torch.float32).view(1, 1, 1, -1)
        )
        self.enhance_mode = None  # 新增模式控制
        self.cbam = None
        self.shuffle = ChannelShuffle()

    def forward(self, x, inverse=False, enhance_mode=None):
        """新增enhance_mode参数控制增强方式
        Args:
            enhance_mode: None/'shuffle'/'cbam'
        """
        self.enhance_mode = enhance_mode  # 设置当前增强模式
        if not inverse:
            return self.dwt2d(x)
        else:
            return self.idwt2d(x)

    def dwt2d(self, x, enhance_mode=None):
        """二维离散小波变换"""
        B, C, H, W = x.shape
        x = x.view(-1, 1, H, W)  # 合并批次和通道维度

        # 行滤波（水平方向卷积）
        lo_row = nn.functional.conv2d(
            x,
            self.dec_lo,
            padding=(0, (self.dec_lo.shape[-1] - 1) // 2),
            stride=(1, 2)  # 此时合法，卷积核为四维
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
        # 行滤波
        lo_row = nn.functional.conv2d(x, self.dec_lo, padding=(0, (self.dec_lo.shape[-1] - 1) // 2), stride=(1, 2))
        hi_row = nn.functional.conv2d(x, self.dec_hi, padding=(0, (self.dec_hi.shape[-1] - 1) // 2), stride=(1, 2))

        # 列滤波
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
        # 调整维度为 [B, C, H//2, W//2]
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
        """高频子带增强处理"""
        # 合并三个高频子带 [B, C*3, H, W]
        highs = torch.cat([LH, HL, HH], dim=1)
        if self.enhance_mode == 'cbam' and self.cbam is None:
            self.cbam = CBAM(channels=3 * channels).to(LL.device)

        # 应用增强模块
        if mode == 'shuffle':
            highs = self.shuffle(highs)
        elif mode == 'cbam':
            highs = self.cbam(highs)

        # 重新分割处理后的子带
        LH, HL, HH = torch.chunk(highs, 3, dim=1)

        # 保持LL不变并与处理后的高频拼接
        return torch.cat([
            LL.view(batch, channels, h, w),
            LH, HL, HH], dim=1).view(batch, channels * 4, h, w)

    def idwt2d(self, y):
        B, C_total, H, W = y.shape  # 输入通道数 C_total = 原通道数 * 4
        C = C_total // 4  # 恢复原始通道数
        y = y.view(B, C, 4, H, W)  # 形状调整为 (B, C, 4, H, W)

        # 分离四个子带
        LL = y[:, :, 0, :, :]  # (B, C, H, W)
        LH = y[:, :, 1, :, :]
        HL = y[:, :, 2, :, :]
        HH = y[:, :, 3, :, :]

        # 列方向反卷积（垂直上采样）
        pad_ver = (self.rec_lo.shape[-1] - 1) // 2
        lo_col = nn.functional.conv_transpose2d(
            LL,
            self.rec_lo.permute(0, 1, 3, 2).expand(C, 1, 6, 1),  # 扩展为 (C, 1, 6, 1)
            stride=(2, 1),
            padding=(pad_ver, 0),
            groups=C  # 分组卷积，独立处理每个通道
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

        # 行方向反卷积（水平上采样）
        pad_hor = (self.rec_lo.shape[-1] - 1) // 2
        x = nn.functional.conv_transpose2d(
            lo_col,
            self.rec_lo.expand(C, 1, 1, 6),  # 扩展为 (C, 1, 1, 6)
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

        return x  # 输出形状 (B, C, H*2, W*2)


class ImprovedWaveletKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        # 初始化小波变换模块
        self.wavelet = WaveletTransform(wavelet='bior2.2')

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)

        # 小波域处理模块
        self.wave_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 4, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim * 4, 1),
            nn.Sigmoid()
        )

        # 多尺度卷积
        self.conv_low = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_mid = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_high = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.act = nn.SiLU()

    def forward(self, x):
        # 输入预处理
        out = self.in_conv(x)

        # ----------------- 小波变换处理 -----------------
        # 分解到小波域
        wave = self.wavelet(out)

        # 小波域注意力
        B, C, H, W = wave.shape
        wave_att = self.wave_att(wave)
        wave_processed = wave * wave_att

        # 逆变换恢复空间域
        restored = self.wavelet(wave_processed, inverse=True)

        # ----------------- 多尺度特征融合 -----------------
        low = self.conv_low(restored)
        mid = self.conv_mid(restored)
        high = self.conv_high(restored)

        # 通道注意力加权
        channel_weights = self.channel_att(restored)
        fused = (low + mid + high) * channel_weights

        # 残差连接
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


############################################process_high_freq########################



class ChannelShuffle(nn.Module):
    """自适应通道数的混洗模块"""

    def forward(self, x):
        batch, channels, h, w = x.size()
        groups = 3  # 固定按3个子带分组
        return x.view(batch, groups, -1, h, w).transpose(1, 2).reshape_as(x)


####################################################################################################
class AsymmetricFreqGuidedFusion(nn.Module):
    """
    非对称跨尺度多模态融合模块
    使用低分辨率红外特征的频域先验，引导高分辨率可见光特征进行空间形变对齐与融合。
    """

    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(AsymmetricFreqGuidedFusion, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        # 独立特征空间映射
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)  # 处理高分辨率可见光
        self.ir_conv = Conv(c_ir, hidden_channels, 3)  # 处理低分辨率红外

        # 频域增强模块 (基于 FFM)
        self.frequency_enhancer = FFM(hidden_channels)

        # 频域-空域门控卷积
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)

        # 偏移量与注意力生成器 (输入为拼接后的 rgb + guided_ir)
        # 输出通道数: groups * 2(RGB的x,y偏移) + groups * 2(IR的x,y偏移) + groups * 2(RGB和IR的注意力权重)
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
        # 将最后一层偏移量初始化为0，确保初始状态为恒等映射
        self.offset_conv[1].weight.data.zero_()

    def forward(self, x):
        # 期望输入为 list 或 tuple: [rgb_features(高分), ir_features(低分)]
        rgb_feat, ir_feat = x
        batch_size, _, out_h, out_w = rgb_feat.size()

        # 1. 红外特征处理与上采样对齐
        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(out_h, out_w), mode='bilinear', align_corners=True)

        # 2. 提取红外稀疏频域先验并做门控融合
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate  # 带有强频域信息的红外先验

        # 3. 可见光特征映射
        rgb_feat = self.rgb_conv(rgb_feat)

        # 4. 生成形变偏移量与注意力权重 (利用红外先验引导)
        conv_results = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        # 整理特征维度以适应内部 Channel 分组 (groups)
        ir_guided = ir_guided.reshape(batch_size * self.groups, -1, out_h, out_w)
        rgb_feat = rgb_feat.reshape(batch_size * self.groups, -1, out_h, out_w)

        # 提取偏移量
        # RGB 偏移量
        offset_rgb = conv_results[:, 0:self.groups * 2, :, :].reshape(batch_size * self.groups, 2, out_h, out_w)
        # IR 偏移量
        offset_ir = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(batch_size * self.groups, 2, out_h,
                                                                                   out_w)

        # 5. 构建基础坐标网格
        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(ir_guided).to(ir_guided.device)
        grid_w = torch.linspace(-1.0, 1.0, out_w).view(1, -1).repeat(out_h, 1)
        grid_h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        base_grid = torch.cat((grid_w.unsqueeze(2), grid_h.unsqueeze(2)), dim=2)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).repeat(batch_size * self.groups, 1, 1, 1).type_as(ir_guided).to(
            ir_guided.device)

        # 施加偏移量 (通过除以宽高归一化到 [-1, 1])
        adjusted_grid_rgb = base_grid + offset_rgb.permute(0, 2, 3, 1) / (normalization_factors * 0.5)
        adjusted_grid_ir = base_grid + offset_ir.permute(0, 2, 3, 1) / (normalization_factors * 0.5)

        # 6. 利用形变网格对齐特征 (Grid Sample)
        rgb_aligned = F.grid_sample(rgb_feat, adjusted_grid_rgb.type_as(rgb_feat), align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided, adjusted_grid_ir.type_as(ir_guided), align_corners=True, padding_mode='border')

        # 恢复维度
        rgb_aligned = rgb_aligned.reshape(batch_size, -1, out_h, out_w)
        ir_aligned = ir_aligned.reshape(batch_size, -1, out_h, out_w)

        # 7. 基于注意力的自适应特征融合
        # 提取剩余的通道作为注意力权重
        attention_logits = conv_results[:, self.groups * 4:, :, :]
        # 1 + tanh 保证权重在 [0, 2] 之间，1代表完全保留
        attention_weights = 1 + torch.tanh(attention_logits)

        # 通道拆分，按比例叠加
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups,
                                                                              dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups,
                                                                              dim=1)

        final_features = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return final_features


class SymmetricFreqGuidedFusion(nn.Module):
    """
    对称同尺度多模态融合模块 (支持自适应尺度切换)
    """
    # 【新增参数】：tiny_mode，默认 False
    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(SymmetricFreqGuidedFusion, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  # 存为类属性

        # 独立特征空间映射
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)

        # 频域增强模块 (基于 FFM)
        self.frequency_enhancer = FFM(hidden_channels)

        # 频域-空域门控卷积
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)

        # 偏移量与注意力生成器
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
        # 期望输入为 list: [rgb_features, ir_features] (二者 H,W 必须相同)
        rgb_feat, ir_feat = x
        batch_size, _, out_h, out_w = rgb_feat.size()

        # 1. 特征映射
        ir_feat = self.ir_conv(ir_feat)
        rgb_feat = self.rgb_conv(rgb_feat)

        # 2. 提取红外稀疏频域先验并做门控融合
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        # 3. 生成形变偏移量与注意力权重 (利用红外频域先验引导对齐)
        conv_results = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        # 提取原始偏移量
        offset_rgb_raw = conv_results[:, 0:self.groups * 2, :, :]
        offset_ir_raw = conv_results[:, self.groups * 2:self.groups * 4, :, :]

        # ================= ★ 优雅的自适应开关逻辑 ★ =================
        if self.tiny_mode:
            # 救命模式：RGB锚点绝不动，IR去贴合，全局用 nearest 保护单像素边缘
            offset_rgb = torch.zeros_like(offset_rgb_raw)
            offset_ir = offset_ir_raw
            interp_method = 'nearest'
        else:
            # SOTA模式：正常双向形变，bilinear 带来平滑过渡
            offset_rgb = offset_rgb_raw
            offset_ir = offset_ir_raw
            interp_method = 'bilinear'
        # ============================================================

        # 整理特征维度以适应内部 Channel 分组
        ir_guided = ir_guided.reshape(batch_size * self.groups, -1, out_h, out_w)
        rgb_feat = rgb_feat.reshape(batch_size * self.groups, -1, out_h, out_w)

        offset_rgb = offset_rgb.reshape(batch_size * self.groups, 2, out_h, out_w)
        offset_ir = offset_ir.reshape(batch_size * self.groups, 2, out_h, out_w)

        # 4. 构建基础坐标网格并计算形变
        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(ir_guided).to(ir_guided.device)
        grid_w = torch.linspace(-1.0, 1.0, out_w).view(1, -1).repeat(out_h, 1)
        grid_h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        base_grid = torch.cat((grid_w.unsqueeze(2), grid_h.unsqueeze(2)), dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(batch_size * self.groups, 1, 1, 1).type_as(ir_guided).to(ir_guided.device)

        adjusted_grid_rgb = base_grid + offset_rgb.permute(0, 2, 3, 1) / (normalization_factors * 0.5)
        adjusted_grid_ir = base_grid + offset_ir.permute(0, 2, 3, 1) / (normalization_factors * 0.5)

        # 5. 形变采样 (Grid Sample) - 传入动态计算的 interp_method
        rgb_aligned = F.grid_sample(rgb_feat, adjusted_grid_rgb.type_as(rgb_feat), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided, adjusted_grid_ir.type_as(ir_guided), mode=interp_method, align_corners=True, padding_mode='border')

        # 恢复维度
        rgb_aligned = rgb_aligned.reshape(batch_size, -1, out_h, out_w)
        ir_aligned = ir_aligned.reshape(batch_size, -1, out_h, out_w)

        # 6. 注意力加权融合
        attention_logits = conv_results[:, self.groups * 4:, :, :]
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        final_features = ir_aligned * attn_ir + rgb_aligned * attn_rgb
        return final_features

class SymmetricFreqGuidedFusion_new(nn.Module):
    """
    真正血统纯正的：对称同尺度多模态融合模块
    (算法逻辑与 Decoupled 完全一致，仅去除了跨尺度上采样)
    """
    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(SymmetricFreqGuidedFusion_new, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  

        # 独立特征映射
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        # 形变偏移量生成
        out_offset_channels = self.groups * 4
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64, 1),
            nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
        )

        # 边缘掩码生成 (保留 Decoupled 中的先进设计)
        self.edge_mask_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),  
            nn.Sigmoid()
        )

        # 后置注意力生成 (保留 Decoupled 中的先进设计)
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
        
        # ================= ★ 唯一的区别在此 ★ =================
        # 因为是对称同尺度融合，输入的 IR 和 RGB 尺寸已经一样大，
        # 所以我们直接注释/删除了 Decoupled 里面的 F.interpolate 这一行！
        # ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)
        # =======================================================

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
###############################################################################################################################################################
class DecoupledFreqGuidedFusion(nn.Module):
    """
    终极方案: 解耦频域引导融合 (支持自适应尺度切换)
    """
    # 【新增参数】：tiny_mode，默认 False（常规模式）
    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(DecoupledFreqGuidedFusion, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        
        # 将开关存为类属性
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
            nn.Conv2d(hidden_channels, 1, 3, padding=1),  # 为了安全，这里统一用 3x3 替代 7x7，大小目标均受益
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

        # ================= ★ 优雅的自适应开关逻辑 ★ =================
        if self.tiny_mode:
            # VEDAI 专属：锁死可见光形变，采用 nearest 保留像素级锐度
            offset_rgb = torch.zeros_like(offsets[:, 0:self.groups * 2, :, :])
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'nearest'
        else:
            # M3FD 专属：正常的双向形变，采用 bilinear 获得平滑边缘
            offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'bilinear'
        # ============================================================

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

        # 这里的 mode 直接传入刚刚动态判断的 interp_method
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

    ###################################################################################################################################
# 假设 Conv 和 FFM 等基础组件已在外部定义
class DecoupledFreqGuidedFusion_HFBypass(nn.Module):
    """
    终极方案: 解耦频域引导融合 (支持自适应尺度切换 + 高频空间保真直连)
    """
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
        
        # 【新增】：如果需要严格控制注入的高频残差强度，可以加一个轻量级的缩放因子
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
        
        # ir_freq 包含了最纯粹的红外高频信息（热点、边缘）
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)

        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        # ================= ★ 优雅的自适应开关逻辑 ★ =================
        if self.tiny_mode:
            # 锁死可见光形变，采用 nearest 保留像素级锐度
            offset_rgb = torch.zeros_like(offsets[:, 0:self.groups * 2, :, :])
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'nearest'
        else:
            # 正常的双向形变，采用 bilinear 获得平滑边缘
            offset_rgb = offsets[:, 0:self.groups * 2, :, :] * edge_mask
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'bilinear'
        # ============================================================

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

        # 空间形变重采样：这里对于小目标来说是“有损”的
        rgb_aligned = F.grid_sample(rgb_feat_g, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_g), mode=interp_method, align_corners=True, padding_mode='border')
        ir_aligned = F.grid_sample(ir_guided_g, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_g), mode=interp_method, align_corners=True, padding_mode='border')

        rgb_aligned = rgb_aligned.reshape(B, -1, H, W)
        ir_aligned = ir_aligned.reshape(B, -1, H, W)

        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        # 基础融合特征
        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        # ================= ★ SCI 核心创新：高频残差空间直连 ★ =================
        if self.tiny_mode:
            # 绕过 grid_sample 的破坏，将纯粹的红外高频特征 (ir_freq) 
            # 通过 edge_mask (只关注高频热点区域) 过滤后，强制注入到最终特征中。
            # 可学习参数 residual_scale 让网络自己决定注入的力度。
            high_freq_residual = ir_freq * edge_mask * self.residual_scale
            fused_out = fused_out + high_freq_residual
        # =====================================================================

        return fused_out

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

# 请确保从你的对应文件中导入 DyT, FFM 和 ConvFFN_GLU
from .dyt import DyT
from .FCM_FFN import ConvFFN_GLU
from .mine import FFM

class DyT(nn.Module):
    """动态 Tanh (DyT) 归一化"""
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

# 确保导入了 FFM, DyT 和 ConvFFN_GLU

class DecoupledFreqGuidedFusion_Pro_Safe(nn.Module):
    """
    终极防休克修复版：
    1. 破解零初始化陷阱，打通反向传播大动脉。
    2. 引入形变防越界锁，防止 F.grid_sample 丢失梯度。
    3. 严格保留了用户的 F.grid_sample 和 type_as 写法。
    """
    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(DecoupledFreqGuidedFusion_Pro_Safe, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        # 1. 安全通道映射 
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

        # 【修复1】：将 ReLU 换成 SiLU，防止大面积死神经元切断梯度
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
        
        # 【修复2：打破零初始化死局】
        # 放弃绝对归 0，改用标准差 0.01 的微小噪声。
        # 这样第一轮反向传播时，梯度乘数不再是 0，而是 0.01，大动脉彻底打通！
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

        # 【修复3：形变防越界锁】
        # 计算 offset 后，必须用 Tanh 压住它！锁死最大形变在 ±5 个像素。
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
        
        # 【严格保留：用户的 reshape、type_as 和 grid_sample 逻辑】
        rgb_safe_shaped = rgb_safe.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_safe_shaped, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_safe_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        
        ir_guided_shaped = ir_guided.reshape(B * self.groups, -1, H, W)
        ir_aligned = F.grid_sample(ir_guided_shaped, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)

        # 3. 语义融合
        ir_spatial_mask = self.spatial_refiner(ir_freq)
        rgb_refined = rgb_aligned * ir_spatial_mask + rgb_aligned 

        fusion_input = torch.cat([rgb_refined, ir_aligned, ir_freq], dim=1)
        attention_logits = self.fusion_attn_conv(fusion_input)
        
        attention_weights = 1 + torch.tanh(attention_logits)
        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_refined * attn_rgb
        ffn_out = self.post_ffn(fused_out)

        # 残差回加，彻底保证梯度不断
        return rgb_safe + self.out_proj(ffn_out)
################################################################################################################################################
class SymmetricFreqGuidedFusion_attn(nn.Module):
    """
    对称同尺度多模态融合模块 (支持自适应尺度切换 + YAML即插即用注意力)
    """
    # 【改动点】：末尾新增 attn_type='default'，完美兼容 YAML 解析
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

        # ================= ★ SFGF 即插即用扩展区 ★ =================
        # 由于原版逻辑中注意力与 offset 耦合，我们在特征输出的最后一环追加注意力标定
        if attn_type == 'default' or attn_type is None:
            self.plug_attn = nn.Identity()  # 默认不加任何额外操作，保持原汁原味
        else:
            self.plug_attn = eval(attn_type)(hidden_channels) # 动态实例化传入的顶会模块
        # ==========================================================

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.offset_conv[1].weight.data.zero_()

    def forward(self, x):
        # 你的前向传播逻辑【一字未改】
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

        # ================= ★ SFGF 终极输出修改 ★ =================
        # 通过额外附加的注意力机制，进行全局维度的重标定，不影响前面任何逻辑！
        return self.plug_attn(final_features)


class DecoupledFreqGuidedFusion_attn(nn.Module):
    """
    终极方案: 解耦频域引导融合 (支持自适应尺度切换 + YAML即插即用注意力)
    """
    # 【改动点】：末尾新增 attn_type='default'，完美兼容 YAML 解析
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

        # ================= ★ DFGF 融合注意力内联改造区 ★ =================
        # 将原有简单的 Conv->BN->ReLU->Conv 替换为包含即插即用模块的序列
        if attn_type == 'default' or attn_type is None:
            self.fusion_attn_conv = nn.Sequential(
                nn.Conv2d(hidden_channels * 3, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, self.groups * 2, kernel_size=3, padding=1, bias=False)
            )
        else:
            # 🛡️ 针对需要传多个参数的特殊模块做兼容
            if attn_type == 'CoordAtt':
                # CoordAtt 需要同时传入 inp 和 oup
                plug_module = eval(attn_type)(hidden_channels, hidden_channels)
            else:
                plug_module = eval(attn_type)(hidden_channels)
                
            self.fusion_attn_conv = nn.Sequential(
                # 先降维
                nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                # 动态嵌入刚刚安全实例化好的顶会注意力机制
                plug_module,
                # 恢复到所需输出维度
                nn.Conv2d(hidden_channels, self.groups * 2, kernel_size=3, padding=1, bias=False)
            )
        # ==============================================================

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
        # 你的前向传播逻辑【一字未改】
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
        
        # 重点：此处调用的是被我们在 __init__ 中重新封装好的 nn.Sequential
        attention_logits = self.fusion_attn_conv(fusion_input)
        
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out
################################################################################################################################################
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
        # 深度可分离卷积，控制参数量
        self.depth_conv = Conv(3 * c1, 3 * c1, 3, 1, 1, g=3 * c1)
        self.point_conv = Conv(3 * c1, c2, 1, 1, 0)

    def forward(self, x):
        out = torch.cat([x, self.focus_h(x), self.focus_v(x)], dim=1)
        return self.point_conv(self.depth_conv(out))

# ================= 主模块 =================
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
        
        # 引入 BiFocus 进行边缘重聚焦
        self.bifocus = BiFocus(hidden_channels, hidden_channels)

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        # 1. 对齐阶段 (与你原版一致)
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

        # 2. 语义融合
        attention_logits = self.fusion_attn_conv(torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1))
        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        # 3. BiFocus 锐化边缘
        return self.bifocus(fused_out)
#########################################################################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

# 确保 FFM 已经在 mine.py 中定义

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

        # 引入频域幅值与相位的可学习融合权重
        self.alpha_rgb = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0.5)
        self.alpha_ir = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0.5)
        
        self.beta_rgb = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0.5)
        self.beta_ir = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0.5)

        # 输出稳定层
        self.out_conv = Conv(hidden_channels, hidden_channels, 3)

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, _, H, W = rgb_feat.shape

        # 1. 几何对齐 (完全保留你的原版逻辑)
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

        # 【严格保留】你的 reshape、type_as 和 grid_sample 逻辑
        rgb_feat_shaped = rgb_feat.reshape(B * self.groups, -1, H, W)
        rgb_aligned = F.grid_sample(rgb_feat_shaped, grid_norm_rgb.permute(0, 2, 3, 1).type_as(rgb_feat_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)
        
        ir_guided_shaped = ir_guided.reshape(B * self.groups, -1, H, W)
        ir_aligned = F.grid_sample(ir_guided_shaped, grid_norm_ir.permute(0, 2, 3, 1).type_as(ir_guided_shaped), mode='bilinear', align_corners=True, padding_mode='border').reshape(B, -1, H, W)

# ==========================================================
        # 2. 纯频域解耦融合 (修复 FP16 非2幂次方报错)
        # ==========================================================
        # 【关键修复】: 强制转换为 float32 进行 FFT 计算，绕过 cuFFT 的底层限制，并增加数值稳定性
        orig_dtype = rgb_aligned.dtype
        rgb_aligned_f32 = rgb_aligned.to(torch.float32)
        ir_aligned_f32 = ir_aligned.to(torch.float32)

        f_rgb = torch.fft.rfft2(rgb_aligned_f32, norm='ortho')
        f_ir = torch.fft.rfft2(ir_aligned_f32, norm='ortho')

        amp_rgb, amp_ir = torch.abs(f_rgb), torch.abs(f_ir)
        phase_rgb, phase_ir = torch.angle(f_rgb), torch.angle(f_ir)

        # 确保可学习权重也对齐到 float32 参与计算
        alpha_rgb_f32 = self.alpha_rgb.to(torch.float32)
        alpha_ir_f32 = self.alpha_ir.to(torch.float32)
        beta_rgb_f32 = self.beta_rgb.to(torch.float32)
        beta_ir_f32 = self.beta_ir.to(torch.float32)

        amp_fused = alpha_rgb_f32 * amp_rgb + alpha_ir_f32 * amp_ir
        phase_fused = beta_rgb_f32 * phase_rgb + beta_ir_f32 * phase_ir

        f_recon = amp_fused * torch.exp(1j * phase_fused)

        fused_spatial_f32 = torch.fft.irfft2(f_recon, s=(H, W), norm='ortho')

        # 【关键修复】: 将频域重建后的特征安全地转换回原本的类型 (如 float16)，保证后续网络不报错
        fused_spatial = fused_spatial_f32.to(orig_dtype)

        # 3. 最终通过一层卷积稳定输出
        return self.out_conv(fused_spatial)
    ###########################################################################################################################################################
class HighFrequencyPerception(nn.Module):
    """提取自 hfp.py 的高频感知增强模块"""
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
        
        # ==========================================
        # 【关键修复】: 强制升格为 float32 避开 cuFFT 报错
        # ==========================================
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        
        # 在 float32 精度下进行傅里叶变换
        xf = torch.fft.rfft2(x_f32, dim=(-2, -1))
        h0 = int(H * self.ratio[0])
        w0 = int((W // 2 + 1) * self.ratio[1])
        mask = torch.ones_like(xf, dtype=xf.dtype)
        mask[:, :, :h0, :w0] = 0
        xf = xf * mask
        xh = torch.fft.irfft2(xf, s=(H, W))
        
        # 算完之后，安全退回到原来的精度 (比如 float16)
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
    """频域解耦融合 + 高频感知细节保真"""
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

        # 引入 HFP 模块用于融合后的高频锐化
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
        
        # 使用 HFP 增强融合后的高频物理边缘
        refined_out = self.hfp_refiner(fused_out)

        # 残差回传，保护底层预训练权重
        return rgb_safe + self.gamma * refined_out
################################################################################################################################################
class GCB(nn.Module):
    """提取自 mrod.py 的全局上下文块，极其轻量"""
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
        # 将全局光照/环境信息叠加回原图
        return x + context 


class DecoupledFreqGuidedFusion_GCB(nn.Module):
    """全局上下文驱动的解耦频域融合"""
    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super().__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels

        self.rgb_align = nn.Conv2d(c_rgb, hidden_channels, 1, bias=False) if c_rgb != hidden_channels else nn.Identity()
        self.ir_align = nn.Conv2d(c_ir, hidden_channels, 1, bias=False) if c_ir != hidden_channels else nn.Identity()

        # 在特征输入初期引入 GCB 感知全局环境
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

        # 赋予双模态特征全局光照感知能力
        rgb_safe = self.gcb_rgb(rgb_safe)
        ir_safe = self.gcb_ir(ir_safe)

        ir_safe_resized = F.interpolate(ir_safe, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_safe_resized)
        gate = torch.sigmoid(self.gating_conv(ir_safe_resized))
        ir_guided = ir_safe_resized * (1 - gate) + ir_freq * gate
        edge_mask = self.edge_mask_gen(ir_freq) 

        # 此时计算出的 offset 具备了对强光/黑夜场景的抗干扰能力
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
###########################################################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
# 确保你的 FFM 和 DyT 正常导入
# from .mine import FFM 
# from .dyt import DyT

# ================= 依赖模块：DConv 字典注入 (源自 RD.py) =================
class DConv(nn.Module):
    """基于字典检索的特征增强机制"""
    def __init__(self, c1: int, alpha: float = 0.8, atoms: int = 512) -> None:
        super().__init__()
        self.alpha = float(alpha)
        # 映射到字典空间
        self.CG = nn.Conv2d(c1, atoms, 1, bias=False)
        # 空间信息交互
        self.GIE = nn.Conv2d(atoms, atoms, 5, padding=2, groups=atoms, bias=False)
        # 解码回原维度
        self.D = nn.Conv2d(atoms, c1, 1, bias=False)

    @staticmethod
    def _pono(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Position-wise Normalization: 消除全局光照偏置"""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + eps)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = self.CG(r)
        x = self.GIE(x)
        x = self._pono(x)
        x = self.D(x)
        # alpha 动态加权残差融合
        return self.alpha * x + (1.0 - self.alpha) * r

# ================= 主模块 =================
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

        # 引入 RD.py 的 DConv 模块作为后置语义重构，atoms 设为通道数的两倍保证容量
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
        
        # 通过 DConv 字典映射消除光照偏置
        refined_out = self.dconv_refiner(fused_out)

        # 同样保留 ReZero 残差回加，打通全局梯度
        return rgb_safe + self.gamma * refined_out
    #####################################################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

# ================= 依赖模块：IIA 正交注意力 (源自 iia.py) =================
class IIA(nn.Module):
    """强化方向性结构一致性模块"""
    def __init__(self, channel: int, kernel_size: int = 7) -> None:
        super().__init__()
        p = kernel_size // 2
        # 沿 W 方向（水平）增强
        self.conv_h = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, kernel_size), padding=(0, p), bias=False),
            nn.Sigmoid(),
        )
        # 沿 H 方向（垂直）增强
        self.conv_w = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(kernel_size, 1), padding=(p, 0), bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道维做均值和最大值池化
        avg = torch.mean(x, dim=1, keepdim=True)          
        maxv, _ = torch.max(x, dim=1, keepdim=True)       
        pooled = torch.cat([avg, maxv], dim=1)            
        
        # 分别生成十字方向的注意力并加回原图
        attn_h = self.conv_h(pooled)                      
        attn_w = self.conv_w(pooled)                      
        return x + x * attn_h + x * attn_w

# ================= 主模块 =================
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

        # 引入 IIA 模块修复空间拉扯形变，kernel=7 保证十字感受野够大
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
        
        # 使用 IIA 进行十字结构校准
        refined_out = self.iia_refiner(fused_out)

        return rgb_safe + self.gamma * self.out_proj(refined_out)
#################################################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    # 标准的 Conv+BN+Act，假设你外部有定义，这里做个简单占位确保代码完整性
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k//2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class LAGFusion(nn.Module):
    """
    顶会级/SCI级方案: Local-Anomaly Guided Fusion (局部异常引导的尺度解耦融合)
    - 针对大目标: 采用 Deformable Offset 进行语义空间对齐
    - 针对小目标: 采用 Thermal Anomaly Prior 进行无网格能量注入 (Grid-Free Injection)
    """
    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(LAGFusion, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  

        # 1. 模态独立特征嵌入
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)

        # 2. 小目标专属: 局部热异常提取 (Thermal Anomaly Prior)
        # 通过 1x1 卷积降维提取单通道的热力突变图
        self.tap_squeeze = nn.Conv2d(hidden_channels, 1, 1)
        self.tap_excite = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 3. 大目标专属: 空间形变对齐 (Deformable Alignment)
        if not self.tiny_mode:
            out_offset_channels = self.groups * 4
            self.offset_conv = nn.Sequential(
                Conv(hidden_channels * 2, 64, 1),
                nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
            )

        # 4. 协同融合自注意力 (Synergistic Self-Attention)
        self.fusion_attn = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        
        # 残差缩放因子，让网络自适应学习注入强度
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

        # 统一步长与通道
        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_conv(rgb_feat)

        # ================= ★ SCI 核心一：Thermal Anomaly Prior (TAP) ★ =================
        # 物理意义：小目标在红外中表现为局部高亮。MaxPool 提取突变点，AvgPool 提取局部背景。
        # 两者相减，完美过滤掉大面积背景，只保留极小的“热点异常” (Infrared Small Target Assumption)
        ir_squeeze = self.tap_squeeze(ir_feat)
        local_max = F.max_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        local_avg = F.avg_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        thermal_anomaly_map = F.relu(local_max - local_avg) # 只保留正向突变（热点）
        
        # 生成跨模态能量注入掩码
        tap_gate = self.tap_excite(thermal_anomaly_map)
        # =================================================================================

        if self.tiny_mode:
            # ================= ★ SCI 核心二：无网格能量注入 (小目标专属) ★ =================
            # 【彻底放弃 grid_sample 形变】，避免任何插值带来的低通滤波破坏。
            # 策略：直接利用红外热异常掩码 (tap_gate)，在可见光特征图的对应位置“点亮”特征。
            
            # 红外自身特征提纯 (抑制背景噪声)
            ir_aligned = ir_feat * tap_gate 
            
            # 可见光特征补偿 (利用红外热点线索增强可见光的微小暗弱目标)
            rgb_aligned = rgb_feat + (rgb_feat * tap_gate * self.gamma_rgb)
            
        else:
            # ================= ★ SCI 核心三：语义空间对齐 (大目标专属) ★ =================
            # 保留传统的形变对齐，用于解决大视差下大目标的边缘重合问题
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

        # 4. 最终自适应通道/空间融合
        fusion_input = torch.cat([rgb_aligned, ir_aligned], dim=1)
        attention_logits = self.fusion_attn(fusion_input)
        
        # 使用 sigmoid 替代 tanh，获得更稳定的 0-1 门控权重
        attention_weights = torch.sigmoid(attention_logits) 

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        # 若为小目标模式，强制保留一份未被注意力抹除的绝对热点残差
        if self.tiny_mode:
            fused_out = fused_out + (ir_feat * tap_gate * self.gamma_ir)

        return fused_out
class HeavyWindowCrossAttention(nn.Module):
    """
    替换掉你原来轻量级 Conv 的重型特征交互引擎
    在 7x7 的局部窗口内，计算 RGB 和 IR 每一个像素的密集相关性
    """
    def __init__(self, dim, num_heads=8, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        # 巨大的参数量来源：Q, K, V 密集投影
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

        # RGB query -> IR key/value
        q_rgb = self.q_rgb(rgb_win).reshape(-1, self.window_size**2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_ir = self.kv_ir(ir_win).reshape(-1, self.window_size**2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_ir, v_ir = kv_ir[0], kv_ir[1]

        attn_rgb = (q_rgb @ k_ir.transpose(-2, -1)) * self.scale
        attn_rgb = attn_rgb.softmax(dim=-1)
        out_rgb = (attn_rgb @ v_ir).transpose(1, 2).reshape(-1, self.window_size**2, C)
        out_rgb = self.proj_rgb(out_rgb)

        # IR query -> RGB key/value
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
# ==========================================================

class HeavyDFGF(nn.Module):
    """
    终极版: 重型解耦频域引导融合 (Heavy Decoupled Frequency-Guided Fusion)
    完美继承 DFGF 的 SCI 创新点，同时拉满计算量专攻无人机极小目标。
    """
    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(HeavyDFGF, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  

        # 1. 模态特征嵌入
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)

        # 2. 【继承】：热异常频域先验提取 (TAP)
        self.tap_squeeze = nn.Conv2d(hidden_channels, 1, 1)
        self.tap_excite = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 3. 【继承】：大目标专属的空间形变对齐
        if not self.tiny_mode:
            out_offset_channels = self.groups * 4
            self.offset_conv = nn.Sequential(
                Conv(hidden_channels * 2, 64, 1),
                nn.Conv2d(64, out_offset_channels, kernel_size=3, padding=1, bias=False)
            )

        # 4. 【算力升级】：重型窗口交叉注意力 (替换原来的 1x1 卷积门控)
        self.heavy_cross_attn = HeavyWindowCrossAttention(hidden_channels, num_heads=8, window_size=7)
        
        # 降维输出
        self.out_conv = Conv(hidden_channels * 2, hidden_channels, 1)

        # 5. 【继承】：高频残差直连自适应因子
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

        # ================= ★ SCI核心一：频域热异常提取 ★ =================
        ir_squeeze = self.tap_squeeze(ir_feat)
        local_max = F.max_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        local_avg = F.avg_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        ir_freq_prior = F.relu(local_max - local_avg) 
        freq_gate = self.tap_excite(ir_freq_prior)

        # ================= ★ SCI核心二：尺度感知对齐 ★ =================
        if self.tiny_mode:
            # 放弃网格形变，进行无损纯净特征引导
            ir_aligned = ir_feat * freq_gate 
            rgb_aligned = rgb_feat + (rgb_feat * freq_gate * self.gamma_rgb)
        else:
            # 常规形变对齐 (包含 AMP 类型对齐修复)
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

        # ================= ★ SCI核心三：重型交叉注意力特征榨取 ★ =================
        # 以前这里只是简单 concat + conv，现在是极其耗算力的像素级互感知
        rgb_fused_attn, ir_fused_attn = self.heavy_cross_attn(rgb_aligned, ir_aligned)
        
        # 残差连接并融合
        fused_out = self.out_conv(torch.cat([rgb_aligned + rgb_fused_attn, ir_aligned + ir_fused_attn], dim=1))

        # ================= ★ SCI核心四：高频残差直接注入 ★ =================
        if self.tiny_mode:
            # 最纯粹的高频能量，跨越所有的形变和注意力网络，直接打入输出底层！
            fused_out = fused_out + (ir_feat * freq_gate * self.gamma_ir)

        return fused_out
    #########################################################################################################################################################################
class CoordAtt(nn.Module):
    """
    轻量级位置敏感注意力 (Coordinate Attention)
    同时捕捉跨通道信息以及方向感知、位置敏感的空间信息，极大提升抗背景干扰能力
    """
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
    """
    DFGF 提点升级版 (Context-Aware & Coordinate-Refined)
    1. 引入 3x3 深度可分离卷积扩大视差偏移量 (Offset) 的感受野
    2. 引入 CoordAtt 彻底过滤对齐后的跨模态噪声
    保持极低计算量，专为提点设计。
    """
    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(DFGF_DWconv_CA, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  

        # 1. 模态独立特征嵌入
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)

        # 2. 频域热异常提取 (TAP)
        self.tap_squeeze = nn.Conv2d(hidden_channels, 1, 1)
        self.tap_excite = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 3. 大目标专属: 上下文感知的空间形变对齐 (Context-Aware Deformable Alignment)
        if not self.tiny_mode:
            out_offset_channels = self.groups * 4
            # 【升级点 1】：加入 groups=64 的 DW-Conv，零成本扩大感受野，更精准地捕捉双目相机物理视差
            self.offset_conv = nn.Sequential(
                Conv(hidden_channels * 2, 64, 1),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False), # DW 卷积
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, out_offset_channels, kernel_size=1, bias=False)
            )

        # 4. 协同融合自注意力 (Synergistic Self-Attention) - 恢复为轻量级
        self.fusion_attn = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, self.groups * 2, kernel_size=3, padding=1, bias=False)
        )
        
        # 【升级点 2】：引入坐标注意力机制，用于最后提纯融合后的特征
        self.coord_att = CoordAtt(hidden_channels, hidden_channels)

        # 残差缩放因子
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
            # 初始化 offset 的最后一层为 0，保证初始阶段不发生突变形变
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

        # === 提取频域热异常 ===
        ir_squeeze = self.tap_squeeze(ir_feat)
        local_max = F.max_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        local_avg = F.avg_pool2d(ir_squeeze, kernel_size=3, stride=1, padding=1)
        thermal_anomaly_map = F.relu(local_max - local_avg) 
        tap_gate = self.tap_excite(thermal_anomaly_map)

        if self.tiny_mode:
            # === 小目标：纯净高频能量直连 ===
            ir_aligned = ir_feat * tap_gate 
            rgb_aligned = rgb_feat + (rgb_feat * tap_gate * self.gamma_rgb)
        else:
            # === 常规目标：形变对齐 ===
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

        # === 自适应注意力融合 ===
        fusion_input = torch.cat([rgb_aligned, ir_aligned], dim=1)
        attention_weights = torch.sigmoid(self.fusion_attn(fusion_input)) 

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        # 【升级点 3】：最终特征提纯，消除背景噪声，锁死目标空间位置
        fused_out = self.coord_att(fused_out)

        if self.tiny_mode:
            fused_out = fused_out + (ir_feat * tap_gate * self.gamma_ir)

        return fused_out
###################################################################################################################
class DFGF_BiFocus(nn.Module):
    """
    DFGF 提点终极版: Bi-Focus (双向聚焦融合)
    1. 动态热异常对比度拉伸 (Saliency Boost)
    2. 基于显著性的模态竞争机制 (Competitive Fusion)
    3. 修复型深度卷积 (Spatial Refine)
    """
    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(DFGF_BiFocus, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode

        # 1. 模态特征投影
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        self.ir_conv = Conv(c_ir, hidden_channels, 3)

        # 2. 增强型 TAP
        self.tap_squeeze = nn.Conv2d(hidden_channels, 1, 1)
        self.tap_excite = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 3. 偏移量预测 (加入 DW-Conv 增加上下文)
        if not self.tiny_mode:
            out_offset_channels = self.groups * 4
            self.offset_conv = nn.Sequential(
                Conv(hidden_channels * 2, 64, 1),
                nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
                nn.Conv2d(64, out_offset_channels, 1, bias=False)
            )

        # 4. 模态竞争注意力机制
        self.comp_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels * 2, hidden_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, hidden_channels * 2, 1),
            nn.Softmax(dim=1) # 在通道/模态间进行竞争
        )
        
        # 5. 空域精炼层
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

        # === 显著性提取与对比度拉伸 ===
        ir_saliency = self.tap_squeeze(ir_feat)
        ir_gate = self.tap_excite(F.max_pool2d(ir_saliency, 3, 1, 1) - F.avg_pool2d(ir_saliency, 3, 1, 1))
        
        # 显著性引导的 IR 增强 (Saliency Boost)
        ir_feat = ir_feat * (1 + ir_gate) 

        if self.tiny_mode:
            ir_aligned, rgb_aligned = ir_feat, rgb_feat
        else:
            # === 上下文感知形变对齐 ===
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

        # === 模态竞争融合 (Competitive Fusion) ===
        # 使用 Softmax 引导的权重，让模型在“纹理”和“热量”之间二选一或权衡
        combined = torch.cat([rgb_aligned, ir_aligned], dim=1)
        weights = self.comp_attn(combined)
        w_rgb, w_ir = torch.split(weights, self.hidden_channels, dim=1)
        
        fused = rgb_aligned * w_rgb + ir_aligned * w_ir
        
        # === 空间修复与残差注入 ===
        fused = self.refine_conv(fused)
        
        if self.tiny_mode:
            fused = fused + (ir_feat * ir_gate * self.gamma_ir)

        return fused
    #########################################################################################################################
class GlobalIlluminationEstimator(nn.Module):
    """
    创新点 3：全局光照估计器 (GLE)
    基于可见光图像的全局空间特征，输出连续的光照置信度 L ∈ [0, 1]
    趋近 1 代表光照充足（白天），趋近 0 代表低照度（黑夜）
    """
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4, bias=False),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 1, bias=False),
            nn.Sigmoid() # 将得分压缩到 0~1 之间
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.pool(x).view(b, c)
        illum_score = self.fc(v).view(b, 1, 1, 1) 
        return illum_score

class FFCM(nn.Module):
    """
    创新点 2：傅里叶卷积混合器 (Fourier Convolution Mixer)
    包含：3x3, 5x5 局部多尺度空间卷积 + 快速傅里叶变换 (FFT) 频域调制
    """
    def __init__(self, dim):
        super().__init__()
        # 局部多尺度空间特征 (捕获小目标边缘)
        self.dw3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dw5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)
        
        # 频域调制可学习参数
        self.freq_weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x, guide=None):
        # 1. Local Spatial (空域局部纹理)
        local_feat = self.dw3(x) + self.dw5(x)
        local_feat = self.act(self.bn(local_feat))
        
        # 2. Global Spectral (频域全局调制)
        # 将输入特征转换到频域 -> [B, C, H, W/2+1]
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        if guide is not None:
            # 如果存在引导模态（如 IR），利用 IR 的振幅谱 (Amplitude) 指导 RGB 的频谱过滤
            guide_fft = torch.fft.rfft2(guide, norm='ortho')
            amp_guide = torch.abs(guide_fft)
            x_fft = x_fft * amp_guide * self.freq_weight
        else:
            x_fft = x_fft * self.freq_weight

        # 逆快速傅里叶变换 (iFFT) 恢复到空域
        global_feat = torch.fft.irfft2(x_fft, s=(x.size(2), x.size(3)), norm='ortho')
        
        # 3. 空域 + 频域特征耦合
        return x + local_feat + global_feat

class IAF_FFCM_Fusion(nn.Module):
    """
    终极提点模块：光照感知 + 傅里叶频域混合 + BiFPN 动态加权融合
    专治复杂照度变化 (M3FD) 与 极小目标丢失 (DroneVehicle)
    """
    def __init__(self, c_rgb, c_ir, hidden_channels=256):
        super().__init__()
        # 嵌入层
        self.rgb_conv = nn.Sequential(nn.Conv2d(c_rgb, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.SiLU())
        self.ir_conv = nn.Sequential(nn.Conv2d(c_ir, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.SiLU())
        
        # GLE 模块
        self.gle = GlobalIlluminationEstimator(hidden_channels)
        
        # 双模态 FFCM 频域特征提取
        self.rgb_ffcm = FFCM(hidden_channels)
        self.ir_ffcm = FFCM(hidden_channels)
        
        # 创新点 1：基于 BiFPN 思想的动态可学习权重
        self.w_rgb = nn.Parameter(torch.ones(1))
        self.w_ir = nn.Parameter(torch.ones(1))
        self.epsilon = 1e-4 # 防止分母为 0
        
        # 融合输出层
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
        
        # 处理非对称跨尺度的分辨率匹配
        if ir_feat.shape[2:] != (H, W):
            ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=False)
            
        # === Step 1: 环境光照感知 (GLE) ===
        L = self.gle(rgb_feat) # L 趋近1为白天，0为黑夜
        
        # === Step 2: 频域与多尺度空域增强 (FFCM) ===
        # 利用红外的振幅谱 (IR guide) 强行“清洗”RGB在夜间的杂波
        rgb_enhanced = self.rgb_ffcm(rgb_feat, guide=ir_feat)
        # 红外不受光照影响，依靠自身增强
        ir_enhanced = self.ir_ffcm(ir_feat, guide=None) 
        
        # === Step 3: 光照引导的 BiFPN 加权 (IAF) ===
        # 公式: w_i * L_i / (sum(w) + eps)
        # 使用 ReLU 保证权重非负
        weight_rgb = F.relu(self.w_rgb) * L
        weight_ir = F.relu(self.w_ir) * (1.0 - L)
        
        fused = (weight_rgb * rgb_enhanced + weight_ir * ir_enhanced) / (weight_rgb + weight_ir + self.epsilon)
        
        return self.out_conv(fused)
##############################################################################################################################################

class RGB_P3_Refiner(nn.Module):
    """
    P3层 RGB 侧专属增强模块 (Receptive Field Expansion & Spatial Denoising)
    用于解决跨尺度融合时的感受野不匹配，及高分辨率下的背景噪声干扰。
    """
    def __init__(self, in_channels, out_channels):
        super(RGB_P3_Refiner, self).__init__()
        
        # 1. 降维
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # 2. 多尺度膨胀卷积 (抹平感受野鸿沟)
        self.dilated_conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1, groups=out_channels, bias=False)
        self.dilated_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2, groups=out_channels, bias=False)
        self.dilated_conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4, groups=out_channels, bias=False)
        self.bn_dilated = nn.BatchNorm2d(out_channels)
        self.act_dilated = nn.GELU()

        # 3. 空间提纯 (过滤背景噪声)
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
    """
    深度跨模态傅里叶融合模块 (Deep Cross-modal Fourier Fusion Module)
    支持通过 `use_p3_refiner` 开关，为高分辨率底层 (P3) 启用专属的 RGB 特征强化。
    """
    # 增加 use_p3_refiner 开关，默认为 False
    def __init__(self, c_rgb, c_ir, hidden_channels=256, use_p3_refiner=False, groups=4):
        super(Deep_CFFM, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        
        # ==========================================
        # 优雅的开关逻辑：根据配置决定 RGB 的投影方式
        # ==========================================
        if use_p3_refiner:
            # P3 层：开启大感受野与空间降噪
            self.rgb_proj = RGB_P3_Refiner(c_rgb, hidden_channels)
        else:
            # P4/P5 层：保持轻量化，避免过度提取
            self.rgb_proj = nn.Sequential(
                nn.Conv2d(c_rgb, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU() 
            )

        # IR 保持极简 1x1，保护热点
        self.ir_proj = nn.Sequential(
            nn.Conv2d(c_ir, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels)
        )

        # 频域调制参数
        self.amp_mix_weight = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0.5)
        
        # 空间对齐与聚合 (DCNv3 逻辑平替版)
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
            nn.Conv2d(hidden_channels, hidden_channels * 2, 1), # <--- 改这里，输出 2 倍 hidden_channels
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)

    def forward(self, x):
        rgb_feat, ir_feat = x
        B, C, H, W = rgb_feat.shape
        dtype, device = rgb_feat.dtype, rgb_feat.device

        # IR 分辨率对齐
        if ir_feat.shape[2:] != (H, W):
            ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=False)

        rgb_feat = self.rgb_proj(rgb_feat)
        ir_feat = self.ir_proj(ir_feat)

        # ==========================================
        # 安全 FFT 模块：局部提升至 FP32，规避 cuFFT 的 2 的幂次方限制与半精度溢出
        # ==========================================
        # 1. 记录当前的数据类型 (通常是 float16)
        current_dtype = rgb_feat.dtype
        
        # 2. 临时升维到 float32 进行高精度傅里叶变换
        rgb_feat_f32 = rgb_feat.to(torch.float32)
        ir_feat_f32 = ir_feat.to(torch.float32)
        mix_weight_f32 = self.amp_mix_weight.to(torch.float32)

        # 3. 在 FP32 下执行 FFT (支持任意尺寸如 20x20)
        fft_rgb = torch.fft.rfft2(rgb_feat_f32, norm='ortho')
        fft_ir = torch.fft.rfft2(ir_feat_f32, norm='ortho')

        amp_rgb, pha_rgb = torch.abs(fft_rgb), torch.angle(fft_rgb)
        amp_ir = torch.abs(fft_ir) # IR 只需要振幅

        # 4. 动态混合振幅 (保持 FP32)
        amp_fused = amp_rgb * (1.0 - mix_weight_f32) + amp_ir * mix_weight_f32
        fft_fused = amp_fused * torch.exp(1j * pha_rgb)
        
        # 5. iFFT 变换回空域
        freq_guide_feat_f32 = torch.fft.irfft2(fft_fused, s=(H, W), norm='ortho')
        
        # 6. 安全转回原来的数据类型 (float16)，无缝衔接后续网络
        freq_guide_feat = freq_guide_feat_f32.to(current_dtype)
        # ==========================================

        # 动态形变对齐
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
        
        # 残差注入，保全频域高频信号
        return fused_out + freq_guide_feat
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

class DecoupledFreqGuidedFusion_trans(nn.Module):
    """
    终极 Transformer 进化版: 解耦频域引导融合 (Trans-DFGF)
    结合 CNN 的像素级对齐与 Transformer 的全局语义聚合
    """
    def __init__(self, c_rgb, c_ir, hidden_channels=256, tiny_mode=False, groups=2):
        super(DecoupledFreqGuidedFusion_trans, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        self.tiny_mode = tiny_mode  

        # ================= 前端 CNN 特征映射与对齐机制 (保持你的创新不动) =================
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

        # ================= ★ 新增：区域感知 Transformer 融合模块 ★ =================
        # 1. 降维卷积 (将拼接的 RGB + IR + Freq 压回 hidden_channels)
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 轻量级标准 Transformer Encoder (处理区域 Tokens)
        self.num_tokens = 4  # 划分 4x4 = 16 个区域 Token，完美规避显存爆炸
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_channels, 
            nhead=4,  # 4 个注意力头
            dim_feedforward=hidden_channels * 2, 
            dropout=0.0, 
            activation='gelu',
            batch_first=True
        )
        
        # 3. 输出注意力逻辑值
        self.to_logits = nn.Conv2d(hidden_channels, self.groups * 2, kernel_size=3, padding=1, bias=False)
        # ==========================================================================

        self.init_weights()
        # 初始化一个针对所有通道的极小标量 (LayerScale 机制)
        # 初始设为 0，意味着一开始网络完全依靠 CNN，随着训练慢慢引入 Transformer
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

        # ================= 1. 解耦频域与门控 (你的原版逻辑) =================
        ir_feat = self.ir_conv(ir_feat)
        ir_feat = F.interpolate(ir_feat, size=(H, W), mode='bilinear', align_corners=True)
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        edge_mask = self.edge_mask_gen(ir_freq)
        rgb_feat = self.rgb_conv(rgb_feat)

        # ================= 2. 空间对齐与保护机制 (你的原版逻辑) =================
        offsets = self.offset_conv(torch.cat([rgb_feat, ir_guided], dim=1))

        if self.tiny_mode:
            # 锁定可见光形变，采用 nearest 保留像素级锐度
            offset_rgb = torch.zeros_like(offsets[:, 0:self.groups * 2, :, :])
            offset_ir = offsets[:, self.groups * 2:self.groups * 4, :, :] * edge_mask
            interp_method = 'nearest'
        else:
            # 正常的双向形变，采用 bilinear 获得平滑边缘
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

        # ================= 3. ★ Transformer 增强融合 ★ =================
        fusion_input = torch.cat([rgb_aligned, ir_aligned, ir_freq], dim=1)
        
        # 3.1 CNN 降维提取局部细节
        local_feat = self.reduce_conv(fusion_input) 
        
        # 3.2 区域 Token 化 (Adaptive Pooling 到 4x4)
        # 将庞大的 HxW 压缩为 16 个包含宏观语义的 Token
        tokens_2d = F.adaptive_avg_pool2d(local_feat, (self.num_tokens, self.num_tokens)) 
        tokens_1d = tokens_2d.flatten(2).transpose(1, 2) # 形状: (B, 16, C)
        
        # 3.3 Transformer 全局交互
        # 让 16 个区域产生跨模态的全局注意力关联
        enhanced_tokens = self.transformer_encoder(tokens_1d) # 形状: (B, 16, C)
        
        # 3.4 恢复 2D 形状并平滑广播 (Broadcast) 到原分辨率
        enhanced_tokens_2d = enhanced_tokens.transpose(1, 2).view(B, self.hidden_channels, self.num_tokens, self.num_tokens)
        global_feat = F.interpolate(enhanced_tokens_2d, size=(H, W), mode='bilinear', align_corners=False)
        
        hybrid_feat = local_feat + (1.0 - edge_mask) * global_feat
        
        # 3.6 输出权重分配逻辑
        attention_logits = self.to_logits(hybrid_feat)
        # ===============================================================

        attention_weights = 1 + torch.tanh(attention_logits)

        attn_ir = attention_weights[:, 0:self.groups, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)
        attn_rgb = attention_weights[:, self.groups:, :, :].repeat_interleave(self.hidden_channels // self.groups, dim=1)

        fused_out = ir_aligned * attn_ir + rgb_aligned * attn_rgb

        return fused_out
    
class GetIndex(nn.Module):
    def __init__(self, *args):
        """args: (c1, c2, index)  —— c1/c2 由 parse_model 注入，index 是第 3 个参数"""
        super().__init__()
        # 兼容老调用：若只传了 2 个参数，第二个就是 index；否则第三个
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
        """
        Args:
            in_channels: 输入特征图的通道数C
            reduction_ratio: 全连接层的压缩比
        """
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim*2
        mid_dim = self.dim // reduction_ratio


        # 高度方向的平均池化分支
        self.h_avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # 保持W维度
        # 宽度方向的平均池化分支
        self.w_avg_pool = nn.AdaptiveAvgPool2d((1, None))  # 保持H维度

        self.fc1 = nn.Sequential(
            nn.Linear(self.dim, mid_dim),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.dim, mid_dim),
            nn.ReLU(inplace=True)
        )
        # 双路特征融合
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
        """ 输入: [B, C, H, W] """
        B, C, H, W = x.size()

        # 高度方向处理路径
        h_pool = self.h_avg_pool(x)  # [B, C, H, 1]
        w_pool = self.w_avg_pool(x)  # [B, C, 1, W]
        h_pool = h_pool.permute(0, 1, 3, 2)  # [B, C, 1, H]
        hw_pool = torch.cat([h_pool, w_pool], dim=3)  # [B, C, 1, H+W]
        hw_pool = hw_pool.squeeze(2) # [B, C, H+W]
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
    #uavdetr-r18-wtdownsamplem4.yaml取消sev2的残差

import torch
import torch.nn as nn

class DynamicGroupedCoordAtt(nn.Module):
    """
    自适应分组双向坐标注意力机制 (支持 4组/8组 动态配置)
    """
    def __init__(self, in_channels, reduction_ratio=4, groups=4):
        super().__init__()
        self.groups = groups
        
        # 自动计算中间通道数，并强制确保其能被 groups 完美整除
        mid_channels = max(in_channels // reduction_ratio, groups)
        mid_channels = (mid_channels // groups) * groups

        # 空间双向池化 (保持不变，提取 H 和 W 方向的感知特征)
        self.h_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.w_pool = nn.AdaptiveAvgPool2d((1, None))

        # 🌟 核心优化 1：使用 1x1 Grouped Conv 完美替代并行的 4 个 FC
        # 彻底解耦分辨率，完美支持 YOLO 的多尺度动态尺寸
        self.conv_shared = nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True) # 统一使用 SiLU，比混用 ReLU/Sigmoid 更利于梯度流动

        # 🌟 核心优化 2：分组还原通道，并生成两个方向的注意力权重
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, groups=groups, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, groups=groups, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        # 提取 H 和 W 方向的特征并拼接 (由于要沿宽度拼接，我们确保它是 B x C x 1 x (H+W))
        x_h = self.h_pool(x).permute(0, 1, 3, 2)  # [B, C, 1, H]
        x_w = self.w_pool(x)                      # [B, C, 1, W]
        x_cat = torch.cat([x_h, x_w], dim=3)      # [B, C, 1, H+W]

        # 跨组信息交互与压缩 (单次并行计算，速度远超此前的切分运算)
        out = self.act(self.bn(self.conv_shared(x_cat)))

        # 将 H 和 W 特征重新拆解开
        out_h, out_w = torch.split(out, [H, W], dim=3)
        out_h = out_h.permute(0, 1, 3, 2)         # [B, mid_channels, H, 1]

        # 生成两个方向的自适应注意力权重
        weight_h = self.sigmoid(self.conv_h(out_h))
        weight_w = self.sigmoid(self.conv_w(out_w))

        # 空间双向加权 + 残差连接
        return x * weight_h * weight_w + x

class ContextGuideFusionModuleV2(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = nn.Conv2d(inc[0], inc[1], 1)
            
        # 🌟 现在只需要传入自适应的通道数 (inc[1] * 2) 即可！
        # 无需再传入高度和宽度。你可以随时在这里把 groups 改为 4 或 8。
        self.se = DynamicGroupedCoordAtt(in_channels=inc[1] * 2, reduction_ratio=4, groups=4)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        
        # 拼接跨模态特征
        x_concat = torch.cat([x0, x1], dim=1)  # [B, 2C, H, W]
        
        # 通过自适应分组注意力网络
        x_concat = self.se(x_concat)
        
        # 将注意力权重重新拆分给 RGB 和 IR
        x0_weight, x1_weight = torch.split(x_concat, [x0.size(1), x1.size(1)], dim=1)

        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight

        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)
import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 Conv 和 FFM 等基础模块你已经导入了
# class Conv(...): ...
# class FFM(...): ...

class Focus(nn.Module):
    """
    无损空域-通道投影模块 (Space-to-Depth)
    将空间分辨率减半，通道数翻4倍，实现 100% 无损的高频信息重组
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([
            x[..., ::2, ::2],      # top-left
            x[..., 1::2, ::2],     # bottom-left
            x[..., ::2, 1::2],     # top-right
            x[..., 1::2, 1::2]     # bottom-right
        ], dim=1)

class DecoupledFreqGuidedFusion_re(nn.Module):
    """
    终极方案: 解耦频域引导融合 (高至低非对称引导版 + Focus无损下采样)
    """
    def __init__(self, c_rgb, c_ir, hidden_channels=256, groups=2):
        super(DecoupledFreqGuidedFusion_re, self).__init__()
        self.groups = groups
        self.hidden_channels = hidden_channels
        
        # RGB 分支 (正常卷积)
        self.rgb_conv = Conv(c_rgb, hidden_channels, 3)
        
        # ================= IR 分支 (你的核心创新) =================
        self.focus = Focus()
        # Focus 会让通道数变成 4 倍 (c_ir * 4)，用 1x1 卷积将其降维回 c_ir 或其他维度
        self.ir_focus_conv = Conv(c_ir * 4, c_ir, 1)
        # 随后接原来的 3x3 卷积，提炼特征并对齐至 hidden_channels
        self.ir_conv = Conv(c_ir, hidden_channels, 3)
        # =========================================================

        # 频域增强与门控解耦
        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)

        # 偏移量预测 (由于前面已经完成了分辨率对齐，这里按正常逻辑走)
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
        # 初始化逻辑保持不变...
        pass

    def forward(self, x):
        rgb_feat, ir_feat = x
        
        # ====== 你的神来之笔：无损下采样与高频提纯 ======
        # 假设此时 ir_feat 是 P2 (大图), rgb_feat 是 P3 (小图)
        ir_feat = self.focus(ir_feat)              # [B, C, 2H, 2W] -> [B, 4C, H, W]
        ir_feat = self.ir_focus_conv(ir_feat)      # 1x1 降维
        ir_feat = self.ir_conv(ir_feat)            # 3x3 提炼 -> [B, hidden_channels, H, W]
        
        # 因为在前面 Focus 已经把 IR 降到了和 RGB 一样的 (H, W)，所以彻底删除了 F.interpolate！
        # ================================================

        # 提取高频先验并解耦
        ir_freq = self.frequency_enhancer(ir_feat)
        gate = torch.sigmoid(self.gating_conv(ir_feat))
        ir_guided = ir_feat * (1 - gate) + ir_freq * gate

        # 生成物理掩膜
        edge_mask = self.edge_mask_gen(ir_freq)
        
        # RGB 正常卷积提取
        rgb_feat = self.rgb_conv(rgb_feat)
        B, _, H, W = rgb_feat.shape

        # ====== 后续的偏移量计算和 Grid Sample 保持你原来的神仙写法 ======
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