from __future__ import annotations
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10786275
    论文题目：CFFormer: A Cross-Fusion Transformer Framework for the Semantic Segmentation of Multisource Remote Sensing Images （TGRS 2025）
    中文题目：CFFormer：一种用于多源遥感图像语义分割的交叉融合Transformer框架（IEEE TGRS 2025）
    多源特征融合模块（Feature Fusion Module，FFM）：
        实际意义：①跨模态信息的全局交互不足：多模态遥感图像（如光学与 SAR/DSM）的互补信息需要通过全局建模。传统方法（简单相加或拼接）仅能实现局部或浅层的特征交互，无法捕捉不同模态间的长距离依赖关系。
                ②特征冗余与噪声干扰问题：多模态数据可能存在特征冗余（如重复的背景信息）或因传感器差异的噪声，直接融合会导致模型性能下降。
        实现方式：多头交叉注意力机制 + 特征增强与融合
"""


class CrossAttention(nn.Module):







































    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, qk_scale=None):







        super(CrossAttention, self).__init__()


        assert dim % num_heads == 0, f"dim {dim} 必须能被头数 {num_heads} 整除"

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.sr_ratio = sr_ratio


        if sr_ratio > 1:

            self.sr1 = nn.Conv2d(
                dim, dim,
                kernel_size=sr_ratio + 1,
                stride=sr_ratio,
                padding=sr_ratio // 2,
                groups=dim
            )
            self.norm1 = nn.LayerNorm(dim)


            self.sr2 = nn.Conv2d(
                dim, dim,
                kernel_size=sr_ratio + 1,
                stride=sr_ratio,
                padding=sr_ratio // 2,
                groups=dim
            )
            self.norm2 = nn.LayerNorm(dim)


    def forward(self, x1, x2, H, W):






        B, N, C = x1.shape



        q1 = self.q1(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = self.q2(x2).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()


        if self.sr_ratio > 1:

            x_1 = x1.permute(0, 2, 1).reshape(B, C, H, W)
            x_1 = self.sr1(x_1)
            x_1 = x_1.reshape(B, C, -1).permute(0, 2, 1)
            x_1 = self.norm1(x_1)


            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)


            x_2 = x2.permute(0, 2, 1).reshape(B, C, H, W)
            x_2 = self.sr2(x_2)
            x_2 = x_2.reshape(B, C, -1).permute(0, 2, 1)
            x_2 = self.norm2(x_2)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:

            kv1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)


        k1, v1 = kv1[0], kv1[1]
        k2, v2 = kv2[0], kv2[1]


        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)



        main_out = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        aux_out = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)

        return main_out, aux_out



class FeatureInteraction(nn.Module):









































    def __init__(self, dim, reduction=1, num_heads=None, sr_ratio=None, norm_layer=nn.LayerNorm):







        super().__init__()


        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)


        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)


        self.cross_attn = CrossAttention(
            dim // reduction,
            num_heads=num_heads,
            sr_ratio=sr_ratio
        )


        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)


        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2, H, W):


        y1, z1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, z2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)


        c1, c2 = self.cross_attn(z1, z2, H, W)


        y1 = torch.cat((y1, c1), dim=-1)
        y2 = torch.cat((y2, c2), dim=-1)


        main_out = self.norm1(x1 + self.end_proj1(y1))
        aux_out = self.norm2(x2 + self.end_proj2(y2))

        return main_out, aux_out


class ChannelEmbed(nn.Module):







































    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):






        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels


        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),

            nn.Conv2d(
                out_channels // reduction,
                out_channels // reduction,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=out_channels // reduction
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):





        B, N, _C = x.shape


        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()


        residual = self.residual(x)


        x = self.channel_embed(x)


        out = self.norm(residual + x)

        return out


class FeatureFusion(nn.Module):







































    def __init__(self, dim, reduction=1, sr_ratio=1, num_heads=None, norm_layer=nn.BatchNorm2d):







        super().__init__()


        self.cross = FeatureInteraction(
            dim=dim,
            reduction=reduction,
            num_heads=num_heads,
            sr_ratio=sr_ratio
        )


        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2,
            out_channels=dim,
            reduction=reduction,
            norm_layer=norm_layer
        )


        self.apply(self._init_weights)


    @classmethod
    def _init_weights(cls, m):



        if isinstance(m, nn.Linear):

            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):

            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2=None):

        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1
        """
        :param x1: 输入特征1（形状：[B, C, H, W]）
        :param x2: 输入特征2（形状同x1）
        """
        B, C, H, W = x1.shape




        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)


        x1, x2 = self.cross(x1, x2, H, W)


        fuse = torch.cat((x1, x2), dim=-1)


        fuse = self.channel_emb(fuse, H, W)

        return fuse

if __name__ == "__main__":
    x1 = torch.randn(1, 32, 50, 50)
    x2 = torch.randn(1, 32, 50, 50)
    fusion_module = FeatureFusion(dim=32,  reduction=1,  sr_ratio=4, num_heads=8)
    output = fusion_module(x1, x2)
    print(f"输入张量1形状: {x1.shape}")
    print(f"输入张量2形状: {x2.shape}")
    print(f"输出张量形状: {output.shape}")





class ChannelWeights(nn.Module):








































    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, std, max), dim=1)
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights


class SpatialWeights(nn.Module):









































    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x)
        spatial_weights = spatial_weights.reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights


class FCM(nn.Module):











































    def __init__(self, dim, reduction=1, eps=1e-8):
        super(FCM, self).__init__()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2=None):

        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        spatial_weights = self.spatial_weights(x1, x2)
        x1_1 = x1 + fuse_weights[0] * spatial_weights[1] * x2
        x2_1 = x2 + fuse_weights[0] * spatial_weights[0] * x1
        channel_weights = self.channel_weights(x1_1, x2_1)
        main_out = x1_1 + fuse_weights[1] * channel_weights[1] * x2_1
        aux_out = x2_1 + fuse_weights[1] * channel_weights[0] * x1_1
        return main_out, aux_out


if __name__ == "__main__":

    x1 = torch.randn(1, 32, 50, 50)
    x2 = torch.randn(1, 32, 50, 50)
    fcm = FCM(dim=32)
    main_out, aux_out = fcm(x1, x2)
    print(f"FCM 输出张量1形状: {main_out.shape}")
    print(f"FCM 输出张量2形状: {aux_out.shape}")











class FCMFeatureFusion(nn.Module):












    def __init__(
        self,
        dim: int | None = None,
        reduction: int = 1,
        sr_ratio: int = 1,
        num_heads: int | None = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        detach_fcm: bool = False,
        return_pair: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.sr_ratio = sr_ratio
        self.num_heads = num_heads
        self.norm_layer = norm_layer
        self.detach_fcm = detach_fcm
        self.return_pair = return_pair


        self.fcm: FCM | None = None
        self.ffm: FeatureFusion | None = None

    def _build_if_needed(self, c: int) -> None:
        if self.fcm is None or self.ffm is None:
            dim = c if self.dim is None else self.dim
            if dim != c:
                raise AssertionError(f"FCMFeatureFusion: dim={self.dim} 与输入通道 {c} 不一致")

            self.fcm = FCM(dim=dim, reduction=self.reduction)
            self.ffm = FeatureFusion(
                dim=dim,
                reduction=self.reduction,
                sr_ratio=self.sr_ratio,
                num_heads=self.num_heads,
                norm_layer=self.norm_layer,
            )

    def forward(self, x1, x2=None):

        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1

        if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
            raise TypeError("FCMFeatureFusion 需要两路输入张量")
        if x1.shape != x2.shape:
            raise ValueError(f"两路输入形状需一致，got {x1.shape} vs {x2.shape}")

        _, c, _, _ = x1.shape
        self._build_if_needed(c)


        main, aux = self.fcm(x1, x2)
        if self.detach_fcm:
            main, aux = main.detach(), aux.detach()


        fused = self.ffm(main, aux)
        return (fused, (main, aux)) if self.return_pair else fused









class ConvFFN_GLU(nn.Module):













    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand: int = 4,
        dwk: int = 3,
        act: str = 'swiglu',
        norm_layer: nn.Module = nn.BatchNorm2d,
        share_dw: bool = False,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        assert dwk in (3, 5, 7), f"dwk {dwk} 不支持，推荐 3/5/7"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = max(out_channels * expand, 1)
        self.act = act.lower()
        self.share_dw = share_dw


        self.proj_u = nn.Conv2d(in_channels, self.hidden, kernel_size=1, bias=True)
        self.proj_g = nn.Conv2d(in_channels, self.hidden, kernel_size=1, bias=True)
        self.dw = nn.Conv2d(
            self.hidden, self.hidden, kernel_size=dwk, stride=1, padding=dwk // 2, groups=self.hidden, bias=True
        )
        self.proj_out = nn.Conv2d(self.hidden, out_channels, kernel_size=1, bias=True)


        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = norm_layer(out_channels)
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0.0 else nn.Identity()

        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _gate(self, g: torch.Tensor) -> torch.Tensor:
        if self.act == 'swiglu':
            return F.silu(g)
        if self.act == 'geglu':
            return F.gelu(g)
        if self.act == 'sigmoid':
            return torch.sigmoid(g)

        return F.silu(g)

    def forward(self, x, H: int | None = None, W: int | None = None):





        if x.dim() == 3:
            assert H is not None and W is not None, "当输入为 [B, N, 2C] 时需提供 H 与 W"
            B, N, C = x.shape
            x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        elif x.dim() == 4:
            B, C, H, W = x.shape
        else:
            raise ValueError(f"不支持的输入维度: {x.shape}")

        u = self.proj_u(x)
        g = self.proj_g(x)

        u = self.dw(u)
        if self.share_dw:
            g = self.dw(g)

        y = self._gate(g) * u
        y = self.proj_out(y)

        out = self.residual(x)
        out = self.norm(out + self.drop_path(y))
        return out


class DropPath(nn.Module):


    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor

