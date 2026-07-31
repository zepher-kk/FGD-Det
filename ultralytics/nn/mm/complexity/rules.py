from __future__ import annotations
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""FLOPs calculation rules for YOLOMM modules.

This module provides a registry of per-module-type FLOPs calculation rules.
All rules use arithmetic FLOPs (multiply + add each count as 1), ensuring
consistency across all complexity reporting.
"""

import math

import torch.nn as nn





def conv2d_flops(conv, in_shape, out_shape) -> float:












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









    return float(2 * batch_tokens * linear.in_features * linear.out_features)

def _conv_output_shape(in_shape, conv) -> tuple[int, int]:

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





def _compute_sequential_flops(module, in_shape) -> float:



















    total = 0.0
    current_shape = in_shape


    if isinstance(module, (tuple, list, nn.Sequential, nn.ModuleList)):
        for sub_module in module:
            sub_flops, current_shape = _compute_sequential_flops(sub_module, current_shape)
            total += sub_flops
        return total, current_shape


    if hasattr(module, 'conv') and hasattr(module.conv, 'out_channels'):
        conv = module.conv
        h_out, w_out = _conv_output_shape(in_shape, conv)
        out_shape = type(in_shape)(
            channels=conv.out_channels,
            height=h_out,
            width=w_out,
        )
        return conv2d_flops(conv, in_shape, out_shape), out_shape


    if isinstance(module, nn.Conv2d):
        h_out, w_out = _conv_output_shape(in_shape, module)
        out_shape = type(in_shape)(
            channels=module.out_channels,
            height=h_out,
            width=w_out,
        )
        return conv2d_flops(module, in_shape, out_shape), out_shape


    if type(module).__name__ == 'DWConv':
        if hasattr(module, 'conv'):
            dconv = module.conv
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


    return 0.0, current_shape





RULES = {}

def register(name):

    def deco(fn):
        RULES[name] = fn
        return fn
    return deco





@register("Concat")
@register("Upsample")
@register("Index")
def route_only_rule(node, input_shapes, output_shapes):

    return 0.0





@register("Conv")
def conv_rule(node, input_shapes, output_shapes):

    return conv2d_flops(node.module.conv, input_shapes[0], output_shapes[0])

@register("GhostConv")
def ghost_conv_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]


    primary_cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    flops = conv2d_flops(module.cv1.conv, src, primary_cv1_out)


    cheap_in = type(src)(channels=module.ghost_channels, height=out.height, width=out.width)
    cheap_out = type(src)(channels=module.ghost_channels, height=out.height, width=out.width)
    flops += conv2d_flops(module.cv2.conv, cheap_in, cheap_out)


    return flops





@register("C2f")
@register("C3k2")
def c2f_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    hidden = module.c
    total = 0.0


    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)


    block_shape = type(src)(channels=hidden, height=out.height, width=out.width)
    for block in module.m:
        total += conv2d_flops(block.cv1.conv, block_shape, block_shape)
        total += conv2d_flops(block.cv2.conv, block_shape, block_shape)


    cv2_in_ch = (2 + len(module.m)) * hidden
    cv2_in = type(src)(channels=cv2_in_ch, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)

    return total

@register("C2PSA")
def c2psa_rule(node, input_shapes, output_shapes):









    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    hidden = module.c
    total = 0.0


    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)


    block_shape = type(src)(channels=hidden, height=out.height, width=out.width)
    for block in module.m:

        total += _attention_flops(block.attn, block_shape)

        ffn = block.ffn

        ffn0_out = type(src)(channels=hidden * 2, height=out.height, width=out.width)
        total += conv2d_flops(ffn[0].conv, block_shape, ffn0_out)

        total += conv2d_flops(ffn[1].conv, ffn0_out, block_shape)


    cv2_in = type(src)(channels=2 * hidden, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)

    return total

def _attention_flops(attn, in_shape):


















    total = 0.0
    h, w = in_shape.height, in_shape.width
    seq_len = h * w


    for attr in ['qkv', 'proj', 'pe']:
        conv = getattr(attn, attr, None)
        if conv and hasattr(conv, 'conv'):
            out_shape = type(in_shape)(channels=conv.conv.out_channels, height=h, width=w)
            total += conv2d_flops(conv.conv, in_shape, out_shape)


    num_heads = getattr(attn, 'num_heads', 8)
    head_dim = getattr(attn, 'head_dim', in_shape.channels // num_heads)

    total += 2 * seq_len * seq_len * head_dim * num_heads

    return total

@register("C3")
def c3_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0


    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)


    hidden = module.cv1.conv.out_channels // 2
    block_shape = type(src)(channels=hidden, height=out.height, width=out.width)
    for block in module.m:
        total += conv2d_flops(block.cv1.conv, block_shape, block_shape)
        total += conv2d_flops(block.cv2.conv, block_shape, block_shape)


    cv2_in_ch = module.cv1.conv.out_channels
    cv2_in = type(src)(channels=cv2_in_ch, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)

    return total





@register("SPPF")
def sppf_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0


    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)



    cv2_in_ch = module.cv1.conv.out_channels * 4
    cv2_in = type(src)(channels=cv2_in_ch, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)

    return total

@register("SPP")
def spp_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0



    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)


    cv2_in_ch = module.cv1.conv.out_channels * 4
    cv2_in = type(src)(channels=cv2_in_ch, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)

    return total

@register("SPPELAN")
def sppelan_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0



    hidden = module.c
    base_shape = type(src)(channels=hidden, height=out.height, width=out.width)

    for cv_name in ['cv1', 'cv2', 'cv3', 'cv4']:
        cv = getattr(module, cv_name, None)
        if cv and hasattr(cv, 'conv'):
            total += conv2d_flops(cv.conv, base_shape, base_shape)


    cv5 = getattr(module, 'cv5', None)
    if cv5 and hasattr(cv5, 'conv'):
        cv5_in_ch = hidden * 4
        cv5_in = type(src)(channels=cv5_in_ch, height=out.height, width=out.width)
        total += conv2d_flops(cv5.conv, cv5_in, out)

    return total





@register("ADown")
def adown_rule(node, input_shapes, output_shapes):

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





@register("FeatureFusion")
def feature_fusion_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0


    channel_emb = getattr(module, 'channel_emb', None)
    if channel_emb and hasattr(channel_emb, 'conv'):
        total += conv2d_flops(channel_emb.conv, src, out)


    for attr in ['ffm', 'fusion_conv']:
        conv = getattr(module, attr, None)
        if conv and hasattr(conv, 'conv'):
            total += conv2d_flops(conv.conv, src, out)

    return total

@register("FCMFeatureFusion")
def fcm_feature_fusion_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    ffm = getattr(module, 'ffm', None)
    if ffm:

        total += feature_fusion_rule(
            type('Node', (), {'module': ffm, 'type_name': 'FeatureFusion'})(),
            input_shapes,
            output_shapes
        )


    if hasattr(module, 'dim'):
        dim = module.dim
        if dim and dim != src.channels:

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

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0


    gate = getattr(module, 'gate', None)
    if gate and hasattr(gate, 'conv'):
        gate_out = type(src)(channels=gate.conv.out_channels, height=out.height, width=out.width)
        total += conv2d_flops(gate.conv, src, gate_out)


    post = getattr(module, 'post', None)
    if post and hasattr(post, 'conv'):
        total += conv2d_flops(post.conv, src, out)


    for attr in ['cv1', 'cv2', 'cv3']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)

    return total

@register("CrossTransformerFusion")
def cross_transformer_fusion_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0

    model_dim = getattr(module, 'model_dim', 0)
    if model_dim:

        for attr in ['proj_rgb', 'proj_x', 'proj_out']:
            proj = getattr(module, attr, None)
            if proj and hasattr(proj, 'conv'):
                total += conv2d_flops(proj.conv, src, out)



        seq_len = src.height * src.width
        if hasattr(module, 'num_heads') and hasattr(module, 'num_layers'):

            attn_flops = 2 * seq_len * seq_len * model_dim * module.num_layers
            total += attn_flops

    return total

@register("IIA")
def iia_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0


    for attr in ['cv1', 'cv2', 'cv3', 'reduction', 'expansion']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)
            src = cv_out

    return total

@register("CTF")
def ctf_rule(node, input_shapes, output_shapes):


    return cross_transformer_fusion_rule(node, input_shapes, output_shapes)

@register("SEFN")
def sefn_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0


    for attr in ['cv1', 'cv2', 'enhance', 'fusion']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)
            if attr != 'fusion':
                src = cv_out

    return total

@register("SequenceShuffleAttention")
def ssa_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0


    gating = getattr(module, 'gating', None)
    if gating is not None:
        for layer in gating:
            if isinstance(layer, nn.Conv2d):
                total += conv2d_flops(layer, src, out)

    return total

@register("RFF")
def rff_rule(node, input_shapes, output_shapes):

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

@register("SOEP")
def soep_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0


    for attr in ['cv1', 'cv2', 'cv3', 'cv4', 'fusion']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)

    return total

@register("MROD")
def mrod_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0


    for attr in ['cv1', 'cv2', 'reasoning', 'fusion']:
        cv = getattr(module, attr, None)
        if cv and hasattr(cv, 'conv'):
            cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
            total += conv2d_flops(cv.conv, src, cv_out)

    return total





@register("Detect")
def detect_rule(node, input_shapes, output_shapes):








    module = node.module
    total = 0.0

    cv2 = getattr(module, 'cv2', None)
    cv3 = getattr(module, 'cv3', None)


    for i, src in enumerate(input_shapes):

        if cv2 and hasattr(cv2, '__iter__') and len(cv2) > i:
            flops, _ = _compute_sequential_flops(cv2[i], src)
            total += flops


        if cv3 and hasattr(cv3, '__iter__') and len(cv3) > i:
            flops, _ = _compute_sequential_flops(cv3[i], src)
            total += flops

    return total

@register("Segment")
def segment_rule(node, input_shapes, output_shapes):

    module = node.module
    total = 0.0


    cv2 = getattr(module, 'cv2', None)
    if cv2 and hasattr(cv2, '__iter__'):
        for i, src in enumerate(input_shapes):
            if i < len(cv2):
                cv2_i = cv2[i]
                if hasattr(cv2_i, 'conv'):
                    cv2_out = type(src)(channels=cv2_i.conv.out_channels, height=src.height, width=src.width)
                    total += conv2d_flops(cv2_i.conv, src, cv2_out)


    cv3 = getattr(module, 'cv3', None)
    if cv3 and hasattr(cv3, '__iter__'):
        for i, src in enumerate(input_shapes):
            if i < len(cv3):
                cv3_i = cv3[i]
                if hasattr(cv3_i, 'conv'):
                    cv3_out = type(src)(channels=cv3_i.conv.out_channels, height=src.height, width=src.width)
                    total += conv2d_flops(cv3_i.conv, src, cv3_out)


    mask_head = getattr(module, 'mask_head', None)
    if mask_head:

        for src in input_shapes:
            mask_out = type(src)(channels=32, height=src.height, width=src.width)
            total += conv2d_flops(
                type('Conv', (), {'in_channels': src.channels, 'out_channels': 32, 'kernel_size': (1, 1), 'groups': 1})(),
                src, mask_out
            )

    return total

@register("Pose")
def pose_rule(node, input_shapes, output_shapes):

    module = node.module
    total = 0.0


    for src in input_shapes:

        kpt_conv = getattr(module, 'kpt_conv', None)
        if kpt_conv and hasattr(kpt_conv, '__iter__'):
            for kc in kpt_conv:
                if hasattr(kc, 'conv'):
                    kpt_out = type(src)(channels=kc.conv.out_channels, height=src.height, width=src.width)
                    total += conv2d_flops(kc.conv, src, kpt_out)

    return total

@register("OBB")
def obb_rule(node, input_shapes, output_shapes):


    return detect_rule(node, input_shapes, output_shapes)

@register("Classification")
def classification_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0] if input_shapes else None
    if not src:
        return 0.0

    total = 0.0



    fc = getattr(module, 'fc', None) or getattr(module, 'linear', None)
    if fc and isinstance(fc, type(lambda: None).__class__):
        batch_tokens = src.height * src.width
        total += linear_flops(fc, batch_tokens)

    return total





@register("C2fAttn")
def c2f_attn_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    hidden = module.c
    total = 0.0


    cv1_out = type(src)(channels=module.cv1.conv.out_channels, height=out.height, width=out.width)
    total += conv2d_flops(module.cv1.conv, src, cv1_out)


    block_shape = type(src)(channels=hidden, height=out.height, width=out.width)
    for block in module.m:
        total += conv2d_flops(block.cv1.conv, block_shape, block_shape)
        total += conv2d_flops(block.cv2.conv, block_shape, block_shape)


    cv2_in_ch = (2 + len(module.m)) * hidden
    cv2_in = type(src)(channels=cv2_in_ch, height=out.height, width=out.width)
    total += conv2d_flops(module.cv2.conv, cv2_in, out)


    attn = getattr(module, 'attn', None)
    if attn:


        pass

    return total

@register("A2C2f")
def a2c2f_rule(node, input_shapes, output_shapes):

    return c2f_rule(node, input_shapes, output_shapes)

@register("BottleneckCSP")
def bottleneck_csp_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    out = output_shapes[0]
    total = 0.0


    cv1 = getattr(module, 'cv1', None)
    if cv1 and hasattr(cv1, 'conv'):
        cv1_out = type(src)(channels=cv1.conv.out_channels, height=out.height, width=out.width)
        total += conv2d_flops(cv1.conv, src, cv1_out)


    for block in module.m:
        if hasattr(block, 'cv1') and hasattr(block.cv1, 'conv'):
            total += conv2d_flops(block.cv1.conv, src, src)
        if hasattr(block, 'cv2') and hasattr(block.cv2, 'conv'):
            total += conv2d_flops(block.cv2.conv, src, src)


    cv4 = getattr(module, 'cv4', None)
    if cv4 and hasattr(cv4, 'conv'):
        total += conv2d_flops(cv4.conv, src, out)

    return total





@register("FCM")
def fcm_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    total = 0.0


    spatial_weights = getattr(module, 'spatial_weights', None)
    if spatial_weights:

        for attr in ['conv1', 'conv2']:
            cv = getattr(spatial_weights, attr, None)
            if cv and hasattr(cv, 'conv'):
                cv_out = type(src)(channels=cv.conv.out_channels, height=out.height, width=out.width)
                total += conv2d_flops(cv.conv, src, cv_out)

    channel_weights = getattr(module, 'channel_weights', None)
    if channel_weights:

        for attr in ['fc1', 'fc2']:
            fc = getattr(channel_weights, attr, None)
            if fc:

                batch_tokens = src.height * src.width
                total += linear_flops(fc, batch_tokens)

    return total

@register("MultiHeadCrossAttention")
def multi_head_cross_attention_rule(node, input_shapes, output_shapes):

    module = node.module
    src = input_shapes[0]
    total = 0.0


    for attr in ['query_vis', 'key_vis', 'value_vis']:
        proj = getattr(module, attr, None)
        if proj and isinstance(proj, type(lambda: None).__class__):
            batch_tokens = src.height * src.width
            total += linear_flops(proj, batch_tokens)


    seq_len = src.height * src.width
    num_heads = getattr(module, 'num_heads', 8)
    head_dim = getattr(module, 'head_dim', 64)
    total += 2 * seq_len * seq_len * num_heads * head_dim


    fc_out = getattr(module, 'fc_out_vis', None)
    if fc_out and isinstance(fc_out, type(lambda: None).__class__):
        total += linear_flops(fc_out, seq_len)

    return total

