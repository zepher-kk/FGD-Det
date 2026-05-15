# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import dill as pickle
import math
from ultralytics.nn.modules.conv import Conv
import numpy as np

import pywt
import pywt.data

__all__ = ['WTConv2dMaxPool','WTConv2d_imp','FocusWNC','SPPFCSPC']

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """创建小波分解和重构滤波器
        Args:
            wave: 小波基名称，如'db1'
            in_size: 输入通道数（分解滤波器组数量）
            out_size: 输出通道数（重构滤波器组数量）
            type: 张量数据类型
        Returns:
            dec_filters: 分解滤波器组 [4, in_size, 1, kernel_size, kernel_size]
            rec_filters: 重构滤波器组 [4, out_size, 1, kernel_size, kernel_size]
        """
    # 获取指定小波基的滤波器系数
    w = pywt.Wavelet(wave)

    # 分解滤波器（高通和低通）
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type) # 高通滤波器系数反转
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type) # 低通滤波器系数反转

    # 构造二维可分离滤波器组：LL, LH, HL, HH
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0) # [4, kernel_size, kernel_size]

    # 按输入通道数扩展维度 [4, in_size, 1, kernel_size, kernel_size]
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # 重构滤波器（系数需要额外翻转）
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])

    # 构造重构滤波器组
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    # 按输出通道数扩展维度 [4, out_size, 1, kernel_size, kernel_size]
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    """执行二维小波分解
        Args:
            x: 输入张量 [B, C, H, W]
            filters: 分解滤波器组 [4*C, 1, kernel_h, kernel_w]
        Returns:
            分解后的4个子带 [B, C, 4, H//2, W//2]
        """
    b, c, h, w = x.shape
    # 计算填充尺寸（保持下采样后尺寸为整数）
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters.to(device=x.device, dtype=x.dtype),
                 stride=2, groups=c, padding=pad)

    # 重排维度：[B, C*4, H//2, W//2] -> [B, C, 4, H//2, W//2]
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    """执行二维小波重构
        Args:
            x: 输入子带 [B, C, 4, H, W]
            filters: 重构滤波器组 [4*C, 1, kernel_h, kernel_w]
        Returns:
            重构后的图像 [B, C, H*2, W*2]
        """
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # 合并子带维度：[B, C, 4, H, W] -> [B, C*4, H, W]
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters.to(device=x.device, dtype=x.dtype),
                           stride=2, groups=c, padding=pad)
    return x


# Define the WaveletTransform class
class WaveletTransform(Function):
    """自定义小波变换函数（支持自动求导）"""
    @staticmethod
    def forward(ctx, input, filters):
        # 前向传播：执行小波分解，保存滤波器供反向使用
        ctx.filters = filters
        with torch.no_grad():# 分解过程不计算梯度
            x = wavelet_transform(input, filters)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：用重构滤波器计算梯度
        grad = inverse_wavelet_transform(grad_output, ctx.filters)
        return grad, None # 滤波器不需要梯度


# Define the InverseWaveletTransform class
class InverseWaveletTransform(Function):
    """自定义小波逆变换函数（支持自动求导）"""
    @staticmethod
    def forward(ctx, input, filters):
        ctx.filters = filters
        with torch.no_grad():
            x = inverse_wavelet_transform(input, filters)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad = wavelet_transform(grad_output, ctx.filters)
        return grad, None

# Initialize the WaveletTransform
def wavelet_transform_init(filters):
    """创建小波变换函数闭包"""
    def apply(input):
        return WaveletTransform.apply(input, filters)
    return apply

# Initialize the InverseWaveletTransform
def inverse_wavelet_transform_init(filters):
    """创建小波逆变换函数闭包"""
    def apply(input):
        return InverseWaveletTransform.apply(input, filters)
    return apply

class WTConv2d_imp(nn.Module):
    """集成小波变换的卷积层
        Args:
            in_channels: 输入/输出通道数（必须相同）
            kernel_size: 基础卷积核尺寸
            wt_levels: 小波分解层数
            wt_type: 小波基类型
        """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d_imp, self).__init__()

        #assert in_channels == out_channels # 当前实现要求输入输出通道相同

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wt_levels = wt_levels # 小波分解层数
        self.stride = stride
        self.dilation = 1

        # 初始化小波滤波器（固定参数，不参与训练）
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 绑定小波变换函数
        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        if in_channels != out_channels:
            self.channel_align = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.channel_align = nn.Identity()

        # 基础卷积层（处理低频分量）
        # 深度可分离卷积
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        # 小波域卷积层（处理高频分量）
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        # 高频分量缩放系数（初始值较小以稳定训练）
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )
        # 下采样处理（如果需要）
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in,
                                                   self.stride_filter.to(device=x_in.device, dtype=x_in.dtype),  # 加 dtype
                                                   bias=None, 
                                                   stride=self.stride, 
                                                   groups=in_channels,
                                                  )
        else:
            self.do_stride = None

    def forward(self, x):
        # 多级小波分解与处理 --------------------------------
        x_ll_in_levels = [] # 存储各层低频分量
        x_h_in_levels = []  # 存储各层高频分量
        shapes_in_levels = []   # 记录原始尺寸

        curr_x_ll = x   # 当前处理的低频分量

        for i in range(self.wt_levels):
            # 保存当前尺寸（用于后续裁剪）
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # 处理奇数尺寸（补零）
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 小波分解得到4个子带 [B, C, 4, H//2, W//2]
            curr_x = self.wt_function(curr_x_ll)
            # 分离低频（LL）和高频（LH,HL,HH）
            curr_x_ll = curr_x[:,:,0,:,:]   # 下一层的输入

            # 处理高频分量
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            # 保存处理后的分量
            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        # 逆向重构过程 ------------------------------------
        next_x_ll = 0   # 初始化为零（残差连接）
        for i in range(self.wt_levels-1, -1, -1):
            # 取出当前层数据
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            # 合并低频残差
            curr_x_ll = curr_x_ll + next_x_ll
            # 拼接所有子带
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            # 小波逆变换
            next_x_ll = self.iwt_function(curr_x)
            # 裁剪到原始尺寸（处理补零情况）
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # 合并基础卷积与高频残差 --------------------------
        x_tag = next_x_ll   # 最终重构结果
        assert len(x_ll_in_levels) == 0 # 确保所有层级已处理

        # 基础卷积路径
        x = self.base_scale(self.base_conv(x))
        # 残差连接
        x = x + x_tag

        # 下采样（如果需要）
        if self.do_stride is not None:
            x = self.do_stride(x)
        x = self.channel_align(x)
        return x

class _ScaleModule(nn.Module):
    """可学习的缩放层（逐通道缩放）"""
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None    # 可扩展为添加偏置
    
    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv2dMaxPool(nn.Module):
    def __init__(self, c1, c2, e=0.25, kernel_size=5, stride=2, use_avgpool=False,pool_kernel=3):
        """
                Args:
                    in_channels: 输入通道数
                    out_channels: 输出通道数
                    kernel_size: WTConv2d的卷积核大小
                    stride: 下采样步长
                    use_avgpool: 是否启用前置AvgPool
                    pool_kernel: AvgPool的核大小（仅在use_avgpool=True时生效）
                """
        super().__init__()
        assert 0 <= e <= 1, "e must be in [0,1]"
        # 计算普通卷积分支通道数 (必须为整数)
        base_conv_ch = int(round(c2 * e))
        wt_conv_ch = c2 - base_conv_ch  # 剩余通道给WT分支

        self.use_avgpool = use_avgpool
        if self.use_avgpool:
            self.avgpool = nn.AvgPool2d(
                kernel_size=pool_kernel,
                stride=1,  # 仅平滑，不下采样
                padding=pool_kernel // 2
            )
        else:
            self.avgpool = nn.Identity()

        # --------------------- 普通卷积分支 ---------------------
        self.base_conv = nn.Sequential(
            nn.Conv2d(c1, base_conv_ch*2, kernel_size=3,stride=stride, padding=1),  # 下采样
            nn.Conv2d(base_conv_ch*2, base_conv_ch, kernel_size=1)
        )
        # WTConv2d 分支
        self.wt_conv = WTConv2d_imp(c1, wt_conv_ch, kernel_size=kernel_size, stride=stride  # WTConv2d内部下采样
        )

        # MaxPool 分支（保持通道数一致）
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_kernel, stride=stride, padding=pool_kernel // 2),
            nn.Conv2d(c1, wt_conv_ch, kernel_size=1)  # 通道对齐
        )

        # 合并方式（可选 Concat 或 Add）
        self.merge_mode = "add"  # 或 "concat"
        self.outconv = nn.Sequential(nn.Conv2d(c2, c2, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(c2)
                                    )
                                                    

    def forward(self, x):

        x = self.avgpool(x)

        x_base = self.base_conv(x)

        x_wt = self.wt_conv(x)
        x_mp = self.maxpool(x)

        if self.merge_mode == "add":
            xw =  x_wt + x_mp  # 通道数需相同
        elif self.merge_mode == "concat":
            xw =  torch.cat([x_wt, x_mp], dim=1)  # 通道数翻倍

        downresult = torch.cat([x_base, xw], dim=1)
        # 通道拼接
        return downresult



#################################################################################################################################
class FocusWNC(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = nn.Conv2d(c1 * 4, c2, k, s)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        x = self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        return x
        # return self.conv(self.contract(x))

class SPPFCSPC(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, self.m(x3)), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
