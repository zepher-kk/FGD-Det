# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""FLOPs calculation rules for YOLOMM modules.

This module provides a registry of per-module-type FLOPs calculation rules.
All rules use arithmetic FLOPs (multiply + add each count as 1), ensuring
consistency across all complexity reporting.
"""

from __future__ import annotations

import math

import torch.nn as nn


# ------------------------------------------------------------------
# Primitive helpers
# ------------------------------------------------------------------

def conv2d_flops(conv, in_shape, out_shape) -> float:
    """Calculate FLOPs for a Conv2d operation.

    Uses arithmetic FLOPs: 2 * output_h * output_w * out_channels * (in_channels / groups) * kernel_h * kernel_w

    Args:
        conv: The Conv2d module (or wrapper .conv attribute).
        in_shape: Input TensorShapeSpec.
        out_shape: Output TensorShapeSpec.

    Returns:
        FLOPs as float.
    """
    kernel_h, kernel_w = conv.kernel_size
    groups = max(int(conv.groups), 1)
    return float(
        2
        * out_shape.height
        * out_shape.width
        * conv.out_channels
        * (conv.in_channels // groups)
        * kernel_h
        * kernel_w
    )


def linear_flops(linear, batch_tokens: int) -> float:
    """Calculate FLOPs for a Linear operation.

    Args:
        linear: The Linear module.
        batch_tokens: Number of tokens in the batch (e.g., seq_len * batch_size).

    Returns:
        FLOPs as float.
    """
    return float(2 * batch_tokens * linear.in_features * linear.out_features)


def _conv_output_shape(in_shape, conv) -> tuple[int, int]:
    """Return output spatial dims for a conv-like module."""
    if isinstance(conv.kernel_size, tuple):
        kh, kw = conv.kernel_size
    else:
        kh = kw = conv.kernel_size

    if isinstance(conv.stride, tuple):
        sh, sw = conv.stride
    else:
        sh = sw = conv.stride

    if isinstance(conv.padding, tuple):
        ph, pw = conv.padding
    else:
        ph = pw = conv.padding

    if isinstance(conv.dilation, tuple):
        dh, dw = conv.dilation
    else:
        dh = dw = conv.dilation

    h_out = (in_shape.height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (in_shape.width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return int(h_out), int(w_out)


# ------------------------------------------------------------------
# Recursive traversal helper for complex head structures
# ------------------------------------------------------------------

def _compute_sequential_flops(module, in_shape) -> float:
    """递归遍历复合模块结构，累计所有 Conv2d 的 FLOPs.

    支持的模块类型：
    - Conv 包装层 (直接读取 .conv)
    - 原生 nn.Conv2d
    - nn.Sequential / nn.ModuleList (递归展开)
    - 嵌套的 nn.Sequential

    Args:
        module: 要遍历的模块
        in_shape: 输入 TensorShapeSpec

    Returns:
        累计的 FLOPs, (out_shape,)

    Examples:
        >>> # cv2[i] = nn.Sequential(Conv, Conv, nn.Conv2d)
        >>> flops, _ = _compute_sequential_flops(cv2[i], src_shape)
    """
    total = 0.0
    current_shape = in_shape

    # 处理 nn.Sequential 或 nn.ModuleList 或 tuple/list
    if isinstance(module, (tuple, list, nn.Sequential, nn.ModuleList)):
        for sub_module in module:
            sub_flops, current_shape = _compute_sequential_flops(sub_module, current_shape)
            total += sub_flops
        return total, current_shape

    # 处理 Conv 包装层 (项目中的标准 Conv)
    if hasattr(module, 'conv') and hasattr(module.conv, 'out_channels'):
        conv = module.conv
        h_out, w_out = _conv_output_shape(in_shape, conv)
        out_shape = type(in_shape)(
            channels=conv.out_channels,
            height=h_out,
            width=w_out,
        )
        return conv2d_flops(conv, in_shape, out_shape), out_shape

    # 处理原生 nn.Conv2d
    if isinstance(module, nn.Conv2d):
        h_out, w_out = _conv_output_shape(in_shape, module)
        out_shape = type(in_shape)(
            channels=module.out_channels,
            height=h_out,
            width=w_out,
        )
        return conv2d_flops(module, in_shape, out_shape), out_shape

    # 处理 DWConv (Depthwise Conv)
    if type(module).__name__ == 'DWConv':
        if hasattr(module, 'conv'):
            dconv = module.conv  # DWConv 内部的 depthwise conv
            h_out, w_out = _conv_output_shape(in_shape, dconv)
            out_shape = type(in_shape)(
                channels=dconv.out_channels,
                height=h_out,
                width=w_out,
            )
            return conv2d_flops(dconv, in_shape, out_shape), out_shape
        elif hasattr(module, 'dconv'):
            dconv = module.dconv
            h_out, w_out = _conv_output_shape(in_shape, dconv)
            out_shape = type(in_shape)(
                channels=dconv.out_channels,
                height=h_out,
                width=w_out,
            )
            return conv2d_flops(dconv, in_shape, out_shape), out_shape

    # 其他模块类型（如 Activation, BN 等）不产生算力，形状不变
    return 0.0, current_shape


# ------------------------------------------------------------------
# Rule registry
# ------------------------------------------------------------------

RULES = {}


def register(name):
    """Decorator to register a complexity rule for a module type."""
    def deco(fn):
        RULES[name] = fn
        return fn
    return deco


# ------------------------------------------------------------------
# Route-only rules (zero FLOPs)
# ------------------------------------------------------------------

@register("Concat")
@register("Upsample")
@register("Index")
def route_only_rule(node, input_shapes, output_shapes):
    """Route-only layers contribute no arithmetic FLOPs."""
    return 0.0


# ------------------------------------------------------------------
# Primitive module rules
# ------------------------------------------------------------------

@register("Conv")
def conv_rule(node, input_shapes, output_shapes):
    """Standard Conv wrapper (as used in YOLOMM)."""
    return conv2d_flops(node.module.conv, input_shapes[0], output_shapes[0])


@register("GhostConv")
def ghost_conv_rule(node, input_shapes, output_shapes):
    """GhostConv: primary conv + cheap conv."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]

    # Primary conv
    primary_cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    flops = conv2d_flops(module.cv1.conv, src, primary_cv1_out)

    # Cheap conv (operates on reduced channels)
    cheap_in = type(src)(channels=module.ghost_channels, height=out.height, width=out.width)
    cheap_out = type(src)(channels=module.ghost_channels, height=out.height, width=out.width)
    flops += conv2d_flops(module.cv2.conv, cheap_in, cheap_out)

    # Concat is zero FLOPs
    return flops


# ------------------------------------------------------------------
# C2f / C3k2 / C2PSA family (bottleneck-based blocks)
# ------------------------------------------------------------------

@register("C2f")
@register("C3k2")
def c2f_rule(node, input_shapes, output_shapes):
    """C2f/C3k2: cv1 -> bottlenecks -> cv2."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    hidden = module.c
    total = 0.0

    # cv1: expansion
    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)

    # Bottlenecks (each has cv1 + cv2)
    block_shape = type(src)(channels=hidden, height=out.height, width=out.width)
    for block in module.m:
        total += conv2d_flops(block.cv1.conv, block_shape, block_shape)
        total += conv2d_flops(block.cv2.conv, block_shape, block_shape)

    # cv2: combines (2 + n) * hidden channels
    cv2_in_ch = (2 + len(module.m)) * hidden
    cv2_in = type(src)(channels=cv2_in_ch, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)

    return total


@register("C2PSA")
def c2psa_rule(node, input_shapes, output_shapes):
    """C2PSA: cv1 -> split -> PSABlock*n -> concat -> cv2.

    Forward: a, b = self.cv1(x).split((self.c, self.c), dim=1)
             b = self.m(b)  # PSABlock sequence
             return self.cv2(torch.cat((a, b), 1))

    Note: cv2 input is ALWAYS 2*hidden (a + b concatenated), independent of n.
    This differs from C2f/C3k2 where cv2 input is (2 + n) * hidden.
    """
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    hidden = module.c
    total = 0.0

    # cv1: 输入 -> 2*hidden
    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)

    # PSABlock 序列 (每个: attn + ffn)
    block_shape = type(src)(channels=hidden, height=out.height, width=out.width)
    for block in module.m:
        # block.attn: Attention 模块 (qkv/proj/pe conv + matmul)
        total += _attention_flops(block.attn, block_shape)
        # block.ffn: nn.Sequential(Conv(c, c*2, 1), Conv(c*2, c, 1))
        ffn = block.ffn
        # ffn[0]: hidden -> 2*hidden
        ffn0_out = type(src)(channels=hidden * 2, height=out.height, width=out.width)
        total += conv2d_flops(ffn[0].conv, block_shape, ffn0_out)
        # ffn[1]: 2*hidden -> hidden
        total += conv2d_flops(ffn[1].conv, ffn0_out, block_shape)

    # cv2: 恒定输入 2*hidden (a 和 b 串联，与 n 无关)
    cv2_in = type(src)(channels=2 * hidden, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)

    return total


def _attention_flops(attn, in_shape):
    """Attention 模块 FLOPs 计算.

    Attention 结构 (ultralytics/nn/modules/block.py:1361):
    - qkv: Conv(c, c * 3, 1)
    - proj: Conv(c * 3 // 2, c, 1)
    - pe: Conv(c, c, 7, 1, 3, g=c)

    矩阵乘 (block.py:1379):
    - QK^T: seq_len * seq_len * head_dim * num_heads
    - AV: seq_len * head_dim * num_heads * seq_len

    Args:
        attn: Attention 模块实例
        in_shape: 输入 TensorShapeSpec

    Returns:
        FLOPs as float
    """
    total = 0.0
    h, w = in_shape.height, in_shape.width
    seq_len = h * w

    # qkv、proj、pe 三个卷积
    for attr in ['qkv', 'proj', 'pe']:
        conv = getattr(attn, attr, None)
        if conv and hasattr(conv, 'conv'):
            out_shape = type(in_shape)(channels=conv.conv.out_channels, height=h, width=w)
            total += conv2d_flops(conv.conv, in_shape, out_shape)

    # QK^T 和 AV 矩阵乘
    num_heads = getattr(attn, 'num_heads', 8)
    head_dim = getattr(attn, 'head_dim', in_shape.channels // num_heads)
    # QK^T + AV: 2 * seq_len^2 * head_dim * num_heads
    total += 2 * seq_len * seq_len * head_dim * num_heads

    return total


@register("C3")
def c3_rule(node, input_shapes, output_shapes):
    """C3 block: cv1 -> bottlenecks (3x) -> cv2."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # cv1
    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)

    # Bottlenecks
    hidden = module.cv1.conv.out_channels // 2
    block_shape = type(src)(channels=hidden, height=out.height, width=out.width)
    for block in module.m:
        total += conv2d_flops(block.cv1.conv, block_shape, block_shape)
        total += conv2d_flops(block.cv2.conv, block_shape, block_shape)

    # cv2
    cv2_in_ch = module.cv1.conv.out_channels
    cv2_in = type(src)(channels=cv2_in_ch, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)

    return total


# ------------------------------------------------------------------
# SPPF / SPP / SPPELAN family (pooling pyramids)
# ------------------------------------------------------------------

@register("SPPF")
def sppf_rule(node, input_shapes, output_shapes):
    """SPPF: cv1 -> maxpool cascade (k=5, c=3) -> cv2."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # cv1
    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)

    # MaxPool operations are zero FLOPs (just indexing)
    # cv2 concatenates the pooled features (4x)
    cv2_in_ch = module.cv1.conv.out_channels * 4
    cv2_in = type(src)(channels=cv2_in_ch, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)

    return total


@register("SPP")
def spp_rule(node, input_shapes, output_shapes):
    """SPP: parallel pooling at different scales."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # MaxPool branches are zero FLOPs
    # cv1
    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)

    # cv2 concatenates pooled features
    cv2_in_ch = module.cv1.conv.out_channels * 4  # 1x1 + 5x5 + 9x9 + 13x13
    cv2_in = type(src)(channels=cv2_in_ch, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)

    return total


@register("SPPELAN")
def sppelan_rule(node, input_shapes, output_shapes):
    """SPPELAN: enhanced SPP with ELAN structure."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # MaxPool branches are zero FLOPs
    # cv1-cv4 process different pool branches
    hidden = module.c
    base_shape = type(src)(channels=hidden, height=out.height, width=out.width)

    for cv_name in ['cv1', 'cv2', 'cv3', 'cv4']:
        cv = getattr(module, cv_name, None)
        if cv and hasattr(cv, 'conv'):
            total += conv2d_flops(cv.conv, base_shape, base_shape)

    # cv5 combines all branches
    cv5 = getattr(module, 'cv5', None)
    if cv5 and hasattr(cv5, 'conv'):
        cv5_in_ch = hidden * 4  # 4 branches
        cv5_in = type(src)(channels=cv5_in_ch, height=out.height, width=out.width)
        total += conv2d_flops(cv5.conv, cv5_in, out)

    return total


# ------------------------------------------------------------------
# Downsampling modules
# ------------------------------------------------------------------

@register("ADown")
def adown_rule(node, input_shapes, output_shapes):
    """ADown: parallel downsampling with different kernel sizes."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    cv1 = getattr(module, 'cv1', None)
    cv2 = getattr(module, 'cv2', None)
    if cv1 and hasattr(cv1, 'conv'):
        cv1_out = type(src)(channels=cv1.conv.out_channels, height=out.height, width=out.width)
        total += conv2d_flops(cv1.conv, src, cv1_out)
    if cv2 and hasattr(cv2, 'conv'):
        cv2_out = type(src)(channels=cv2.conv.out_channels, height=out.height, width=out.width)
        total += conv2d_flops(cv2.conv, src, cv2_out)

    return total


@register("AConv")
def aconv_rule(node, input_shapes, output_shapes):
    """AConv: downsampling with averaging."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    cv1 = getattr(module, 'cv1', None)
    if cv1 and hasattr(cv1, 'conv'):
        total += conv2d_flops(cv1.conv, src, out)

    return total


@register("SCDown")
def scdown_rule(node, input_shapes, output_shapes):
    """SCDown: channel-wise spatial downsampling."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    cv1 = getattr(module, 'cv1', None)
    cv2 = getattr(module, 'cv2', None)
    if cv1 and hasattr(cv1, 'conv'):
        total += conv2d_flops(cv1.conv, src, out)
    if cv2 and hasattr(cv2, 'conv'):
        total += conv2d_flops(cv2.conv, src, out)

    return total


# ------------------------------------------------------------------
# Fusion modules (multimodal feature fusion)
# ------------------------------------------------------------------

@register("FeatureFusion")
def feature_fusion_rule(node, input_shapes, output_shapes):
    """FeatureFusion: channel embedding -> fusion."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # channel_emb
    channel_emb = getattr(module, 'channel_emb', None)
    if channel_emb and hasattr(channel_emb, 'conv'):
        total += conv2d_flops(channel_emb.conv, src, out)

    # Additional fusion convs if present
    for attr in ['ffm', 'fusion_conv']:
        conv = getattr(module, attr, None)
        if conv and hasattr(conv, 'conv'):
            total += conv2d_flops(conv.conv, src, out)

    return total


@register("FCMFeatureFusion")
def fcm_feature_fusion_rule(node, input_shapes, output_shapes):
    """FCMFeatureFusion: FCM-based fusion."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    ffm = getattr(module, 'ffm', None)
    if ffm:
        # Use FeatureFusion rule on the internal ffm module
        total += feature_fusion_rule(
            type('Node', (), {'module': ffm, 'type_name': 'FeatureFusion'})(),
            input_shapes,
            output_shapes
        )

    # Direct channel projection
    if hasattr(module, 'dim'):
        dim = module.dim
        if dim and dim != src.channels:
            # Account for channel projection
            dummy_conv = type('Conv', (), {
                'in_channels': src.channels,
                'out_channels': dim,
                'kernel_size': (1, 1),
                'groups': 1
            })()
            total += conv2d_flops(dummy_conv, src, out)

    return total


@register("MCFGatedFusion")
def mcfgated_fusion_rule(node, input_shapes, output_shapes):
    """MCFGatedFusion: gated fusion with gate and post modules."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # Gate module
    gate = getattr(module, 'gate', None)
    if gate and hasattr(gate, 'conv'):
        gate_out = type(src)(channels=gate.conv.out_channels, height=out.height, width=out.width)
        total += conv2d_flops(gate.conv, src, gate_out)

    # Post module
    post = getattr(module, 'post', None)
    if post and hasattr(post, 'conv'):
        total += conv2d_flops(post.conv, src, out)

    # Additional convs if present
    for attr in ['cv1', 'cv2', 'cv3']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)

    return total


@register("CrossTransformerFusion")
def cross_transformer_fusion_rule(node, input_shapes, output_shapes):
    """CrossTransformerFusion: transformer-based cross-modal fusion."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    model_dim = getattr(module, 'model_dim', 0)
    if model_dim:
        # Projection convs
        for attr in ['proj_rgb', 'proj_x', 'proj_out']:
            proj = getattr(module, attr, None)
            if proj and hasattr(proj, 'conv'):
                total += conv2d_flops(proj.conv, src, out)

        # Attention: for simplicity, we use a simplified estimate
        # Full transformer FLOPs would require token counting
        seq_len = src.height * src.width
        if hasattr(module, 'num_heads') and hasattr(module, 'num_layers'):
            # Rough estimate: 2 * seq_len^2 * model_dim per layer * num_layers
            attn_flops = 2 * seq_len * seq_len * model_dim * module.num_layers
            total += attn_flops

    return total


@register("IIA")
def iia_rule(node, input_shapes, output_shapes):
    """IIA (Information Integration Augmentation)."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # IIA typically has reduction and expansion convs
    for attr in ['cv1', 'cv2', 'cv3', 'reduction', 'expansion']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)
            src = cv_out  # Chain for next conv

    return total


@register("CTF")
def ctf_rule(node, input_shapes, output_shapes):
    """CTF (Cross-Transformer Fusion)."""
    # Similar to CrossTransformerFusion
    return cross_transformer_fusion_rule(node, input_shapes, output_shapes)


@register("SEFN")
def sefn_rule(node, input_shapes, output_shapes):
    """SEFN (Semantic Enhancement Fusion Network)."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # SEFN typically has enhancement convs
    for attr in ['cv1', 'cv2', 'enhance', 'fusion']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)
            if attr != 'fusion':  # Don't chain the final fusion
                src = cv_out

    return total


@register("SequenceShuffleAttention")
def ssa_rule(node, input_shapes, output_shapes):
    """SequenceShuffleAttention: 分组1x1卷积门控."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # 主要计算量: 分组 1x1 Conv2d (lazy built, 从 gating 中获取)
    gating = getattr(module, 'gating', None)
    if gating is not None:
        for layer in gating:
            if isinstance(layer, nn.Conv2d):
                total += conv2d_flops(layer, src, out)

    return total


@register("RFF")
def rff_rule(node, input_shapes, output_shapes):
    """RFF (Residual Feature Fusion)."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    for attr in ['cv1', 'cv2', 'cv3', 'fusion']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)

    return total


@register("MSIA")
def msia_rule(node, input_shapes, output_shapes):
    """MSIA (Multi-Scale Information Aggregation)."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # MSIA typically has multiple scale-specific convs
    for attr in ['cv1', 'cv2', 'cv3', 'fusion']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)

    return total


@register("SOEP")
def soep_rule(node, input_shapes, output_shapes):
    """SOEP (Second-Order Enhancement Pyramid)."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # SOEP typically has enhancement pyramid convs
    for attr in ['cv1', 'cv2', 'cv3', 'cv4', 'fusion']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)

    return total


@register("MROD")
def mrod_rule(node, input_shapes, output_shapes):
    """MROD (Multi-modal Reasoning Object Detection)."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # MROD has reasoning-specific convs
    for attr in ['cv1', 'cv2', 'reasoning', 'fusion']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)

    return total


# ------------------------------------------------------------------
# Detection heads
# ------------------------------------------------------------------

@register("Detect")
def detect_rule(node, input_shapes, output_shapes):
    """YOLO Detect head: 递归遍历 cv2/cv3 的真实子结构.

    Detect 头结构:
    - cv2[i]: nn.Sequential(Conv, Conv, nn.Conv2d) for bbox regression
    - cv3[i]: nn.Sequential(Sequential(DWConv, Conv), Sequential(DWConv, Conv), nn.Conv2d) for classification

    使用 _compute_sequential_flops() 递归展开所有内部算子.
    """
    module = node.module
    total = 0.0

    cv2 = getattr(module, 'cv2', None)
    cv3 = getattr(module, 'cv3', None)

    # Each input scale gets separate bbox and class convs
    for i, src in enumerate(input_shapes):
        # cv2: bbox regression 分支
        if cv2 and hasattr(cv2, '__iter__') and len(cv2) > i:
            flops, _ = _compute_sequential_flops(cv2[i], src)
            total += flops

        # cv3: class prediction 分支
        if cv3 and hasattr(cv3, '__iter__') and len(cv3) > i:
            flops, _ = _compute_sequential_flops(cv3[i], src)
            total += flops

    return total


@register("Segment")
def segment_rule(node, input_shapes, output_shapes):
    """Segmentation head (similar to Detect but with mask output)."""
    module = node.module
    total = 0.0

    # cv2: bbox
    cv2 = getattr(module, 'cv2', None)
    if cv2 and hasattr(cv2, '__iter__'):
        for i, src in enumerate(input_shapes):
            if i < len(cv2):
                cv2_i = cv2[i]
                if hasattr(cv2_i, 'conv'):
                    cv2_out = type(src)(channels=cv2_i.conv.out_channels, height=src.height, width=src.width)
                    total += conv2d_flops(cv2_i.conv, src, cv2_out)

    # cv3: class
    cv3 = getattr(module, 'cv3', None)
    if cv3 and hasattr(cv3, '__iter__'):
        for i, src in enumerate(input_shapes):
            if i < len(cv3):
                cv3_i = cv3[i]
                if hasattr(cv3_i, 'conv'):
                    cv3_out = type(src)(channels=cv3_i.conv.out_channels, height=src.height, width=src.width)
                    total += conv2d_flops(cv3_i.conv, src, cv3_out)

    # mask_head
    mask_head = getattr(module, 'mask_head', None)
    if mask_head:
        # Simplified estimate for mask head
        for src in input_shapes:
            mask_out = type(src)(channels=32, height=src.height, width=src.width)  # Typical mask dim
            total += conv2d_flops(
                type('Conv', (), {'in_channels': src.channels, 'out_channels': 32, 'kernel_size': (1, 1), 'groups': 1})(),
                src, mask_out
            )

    return total


@register("Pose")
def pose_rule(node, input_shapes, output_shapes):
    """Pose estimation head."""
    module = node.module
    total = 0.0

    # Similar to Detect but outputs keypoints
    for src in input_shapes:
        # kpt_conv
        kpt_conv = getattr(module, 'kpt_conv', None)
        if kpt_conv and hasattr(kpt_conv, '__iter__'):
            for kc in kpt_conv:
                if hasattr(kc, 'conv'):
                    kpt_out = type(src)(channels=kc.conv.out_channels, height=src.height, width=src.width)
                    total += conv2d_flops(kc.conv, src, kpt_out)

    return total


@register("OBB")
def obb_rule(node, input_shapes, output_shapes):
    """OBB (Oriented Bounding Box) head."""
    # Similar to Detect but with angle prediction
    return detect_rule(node, input_shapes, output_shapes)


@register("Classification")
def classification_rule(node, input_shapes, output_shapes):
    """Classification head."""
    module = node.module
    src = input_shapes[0] if input_shapes else None
    if not src:
        return 0.0

    total = 0.0

    # Global average pooling (zero FLOPs)
    # Linear classifier
    fc = getattr(module, 'fc', None) or getattr(module, 'linear', None)
    if fc and isinstance(fc, type(lambda: None).__class__):  # Check if Linear-like
        batch_tokens = src.height * src.width  # Flattened spatial dims
        total += linear_flops(fc, batch_tokens)

    return total


# ------------------------------------------------------------------
# C2fAttn and A2C2f (attention variants)
# ------------------------------------------------------------------

@register("C2fAttn")
def c2f_attn_rule(node, input_shapes, output_shapes):
    """C2f with attention."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    hidden = module.c
    total = 0.0

    # cv1
    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)

    # Bottlenecks
    block_shape = type(src)(channels=hidden, height=out.height, width=out.width)
    for block in module.m:
        total += conv2d_flops(block.cv1.conv, block_shape, block_shape)
        total += conv2d_flops(block.cv2.conv, block_shape, block_shape)

    # cv2
    cv2_in_ch = (2 + len(module.m)) * hidden
    cv2_in = type(src)(channels=cv2_in_ch, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)

    # Attention module (if present)
    attn = getattr(module, 'attn', None)
    if attn:
        # Simplified: treat as additional conv operations
        # Full attention FLOPs would be seq_len^2 dependent
        pass

    return total


@register("A2C2f")
def a2c2f_rule(node, input_shapes, output_shapes):
    """A2-C2f variant."""
    return c2f_rule(node, input_shapes, output_shapes)


@register("BottleneckCSP")
def bottleneck_csp_rule(node, input_shapes, output_shapes):
    """BottleneckCSP (older YOLOv4-style block)."""
    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    # cv1
    cv1 = getattr(module, 'cv1', None)
    if cv1 and hasattr(cv1, 'conv'):
        cv1_out = type(src)(channels=cv1.conv.out_channels, height=out.height, width=out.width)
        total += conv2d_flops(cv1.conv, src, cv1_out)

    # Bottlenecks
    for block in module.m:
        if hasattr(block, 'cv1') and hasattr(block.cv1, 'conv'):
            total += conv2d_flops(block.cv1.conv, src, src)
        if hasattr(block, 'cv2') and hasattr(block.cv2, 'conv'):
            total += conv2d_flops(block.cv2.conv, src, src)

    # cv4 (final projection)
    cv4 = getattr(module, 'cv4', None)
    if cv4 and hasattr(cv4, 'conv'):
        total += conv2d_flops(cv4.conv, src, out)

    return total


# ------------------------------------------------------------------
# Multi-output special handling (FCM, MultiHeadCrossAttention)
# ------------------------------------------------------------------

@register("FCM")
def fcm_rule(node, input_shapes, output_shapes):
    """FCM (Fusion Calibration Module) - multi-output."""
    module = node.module
    src = input_shapes[0]
    total = 0.0

    # FCM typically has spatial and channel weighting paths
    spatial_weights = getattr(module, 'spatial_weights', None)
    if spatial_weights:
        # Spatial attention path
        for attr in ['conv1', 'conv2']:
            cv = getattr(spatial_weights, attr, None)
            if cv and hasattr(cv, 'conv'):
                cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
                total += conv2d_flops(cv.conv, src, cv_out)

    channel_weights = getattr(module, 'channel_weights', None)
    if channel_weights:
        # Channel attention path
        for attr in ['fc1', 'fc2']:
            fc = getattr(channel_weights, attr, None)
            if fc:
                # Global pooling reduces to (batch, channels, 1, 1)
                batch_tokens = src.height * src.width
                total += linear_flops(fc, batch_tokens)

    return total


@register("MultiHeadCrossAttention")
def multi_head_cross_attention_rule(node, input_shapes, output_shapes):
    """MultiHeadCrossAttention - multi-output."""
    module = node.module
    src = input_shapes[0]
    total = 0.0

    # Query/Key/Value projections
    for attr in ['query_vis', 'key_vis', 'value_vis']:
        proj = getattr(module, attr, None)
        if proj and isinstance(proj, type(lambda: None).__class__):
            batch_tokens = src.height * src.width
            total += linear_flops(proj, batch_tokens)

    # Attention computation: seq_len^2 * num_heads * head_dim
    seq_len = src.height * src.width
    num_heads = getattr(module, 'num_heads', 8)
    head_dim = getattr(module, 'head_dim', 64)
    total += 2 * seq_len * seq_len * num_heads * head_dim  # QK^T and softmax*V

    # Output projection
    fc_out = getattr(module, 'fc_out_vis', None)
    if fc_out and isinstance(fc_out, type(lambda: None).__class__):
        total += linear_flops(fc_out, seq_len)

    return total
