from __future__ import annotations
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Per-module pruning operations for structured channel pruning.

Each function directly modifies module weights in-place.
"""

import torch
import torch.nn as nn

def _ssa_group_input_indices(channel_count: int, groups: int, device: torch.device) -> torch.Tensor:






    if groups <= 0:
        raise ValueError(f"SSA groups must be positive, got {groups}")
    if channel_count % groups != 0:
        raise ValueError(f"SSA channels {channel_count} must be divisible by groups {groups}")

    channels_per_group = channel_count // groups
    order = torch.arange(channel_count, device=device).reshape(channels_per_group, groups).permute(1, 0)
    return order.contiguous()





def prune_conv_out(module, keep_idx: torch.Tensor):






    conv = module.conv
    conv.weight = nn.Parameter(conv.weight.data[keep_idx])
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[keep_idx])
    conv.out_channels = len(keep_idx)

    if hasattr(module, "bn") and isinstance(module.bn, nn.BatchNorm2d):
        bn = module.bn
        bn.weight = nn.Parameter(bn.weight.data[keep_idx])
        bn.bias = nn.Parameter(bn.bias.data[keep_idx])
        bn.running_mean = bn.running_mean[keep_idx]
        bn.running_var = bn.running_var[keep_idx]
        bn.num_features = len(keep_idx)

def prune_conv_in(module, keep_idx: torch.Tensor):






    conv = module.conv
    if conv.groups == 1:
        conv.weight = nn.Parameter(conv.weight.data[:, keep_idx])
    else:

        conv.weight = nn.Parameter(conv.weight.data[keep_idx])
        conv.groups = len(keep_idx)
        conv.out_channels = len(keep_idx)

        if hasattr(module, "bn"):
            bn = module.bn
            bn.weight = nn.Parameter(bn.weight.data[keep_idx])
            bn.bias = nn.Parameter(bn.bias.data[keep_idx])
            bn.running_mean = bn.running_mean[keep_idx]
            bn.running_var = bn.running_var[keep_idx]
            bn.num_features = len(keep_idx)
    conv.in_channels = len(keep_idx)

def prune_raw_conv2d_in(conv2d: nn.Conv2d, keep_idx: torch.Tensor):

    conv2d.weight = nn.Parameter(conv2d.weight.data[:, keep_idx])
    conv2d.in_channels = len(keep_idx)

def prune_raw_conv2d_out(conv2d: nn.Conv2d, keep_idx: torch.Tensor):

    conv2d.weight = nn.Parameter(conv2d.weight.data[keep_idx])
    if conv2d.bias is not None:
        conv2d.bias = nn.Parameter(conv2d.bias.data[keep_idx])
    conv2d.out_channels = len(keep_idx)

def prune_batchnorm2d_out(bn: nn.BatchNorm2d, keep_idx: torch.Tensor):

    bn.weight = nn.Parameter(bn.weight.data[keep_idx])
    bn.bias = nn.Parameter(bn.bias.data[keep_idx])
    bn.running_mean = bn.running_mean[keep_idx]
    bn.running_var = bn.running_var[keep_idx]
    bn.num_features = len(keep_idx)





def prune_bottleneck_hidden(bn_module, keep_idx: torch.Tensor):








    prune_conv_out(bn_module.cv1, keep_idx)
    prune_conv_in(bn_module.cv2, keep_idx)

def prune_bottleneck_inout(bn_module, keep_idx: torch.Tensor):







    prune_conv_in(bn_module.cv1, keep_idx)

    prune_conv_out(bn_module.cv2, keep_idx)





def prune_c2f_out(module, keep_idx: torch.Tensor):

    prune_conv_out(module.cv2, keep_idx)

def prune_c2f_in(module, keep_idx: torch.Tensor):

    prune_conv_in(module.cv1, keep_idx)

def prune_c2f_internal(module, keep_hidden: torch.Tensor):
















    n = len(module.m)
    new_c = len(keep_hidden)


    keep_cv1_out = torch.cat([keep_hidden, keep_hidden + module.c])
    prune_conv_out(module.cv1, keep_cv1_out)


    for bottleneck in module.m:
        prune_bottleneck_inout(bottleneck, keep_hidden)

        c_ = bottleneck.cv1.conv.out_channels
        if c_ > 0:
            keep_bn_hidden = _compute_sub_keep(bottleneck.cv1, c_)
            prune_bottleneck_hidden(bottleneck, keep_bn_hidden)



    chunks = [keep_hidden + i * module.c for i in range(2 + n)]
    keep_cv2_in = torch.cat(chunks)
    prune_conv_in(module.cv2, keep_cv2_in)


    module.c = new_c

def _compute_sub_keep(conv_module, n_channels: int) -> torch.Tensor:

    device = conv_module.conv.weight.device
    return torch.arange(n_channels, device=device)





def prune_c3k_internal(module, keep_hidden: torch.Tensor):













    prune_conv_out(module.cv1, keep_hidden)

    prune_conv_out(module.cv2, keep_hidden)


    for bottleneck in module.m:
        prune_bottleneck_inout(bottleneck, keep_hidden)
        c_ = bottleneck.cv1.conv.out_channels
        if c_ > 0:
            keep_bn_hidden = torch.arange(c_, device=keep_hidden.device)
            prune_bottleneck_hidden(bottleneck, keep_bn_hidden)


    new_c = len(keep_hidden)
    keep_cv3_in = torch.arange(2 * new_c, device=keep_hidden.device)
    prune_conv_in(module.cv3, keep_cv3_in)





def prune_sppf_out(module, keep_idx: torch.Tensor):

    prune_conv_out(module.cv2, keep_idx)

def prune_sppf_in(module, keep_idx: torch.Tensor):

    prune_conv_in(module.cv1, keep_idx)

def prune_sppf_internal(module, keep_hidden: torch.Tensor):











    n = module.n
    prune_conv_out(module.cv1, keep_hidden)

    keep_cv2_in = torch.cat([keep_hidden + i * len(keep_hidden) for i in range(n + 1)])
    prune_conv_in(module.cv2, keep_cv2_in)





def prune_c2psa_out(module, keep_idx: torch.Tensor):





    prune_conv_out(module.cv2, keep_idx)

def prune_c2psa_in(module, keep_idx: torch.Tensor):

    prune_conv_in(module.cv1, keep_idx)

def prune_c2psa_internal(module, keep_hidden: torch.Tensor):











    old_c = module.c
    new_c = len(keep_hidden)


    keep_cv1_out = torch.cat([keep_hidden, keep_hidden + old_c])
    prune_conv_out(module.cv1, keep_cv1_out)


    for psa_block in module.m:
        _prune_psa_block(psa_block, keep_hidden)


    keep_cv2_in = torch.cat([keep_hidden, keep_hidden + old_c])
    prune_conv_in(module.cv2, keep_cv2_in)

    module.c = new_c

def _prune_psa_block(psa_block, keep_idx: torch.Tensor):










    new_c = len(keep_idx)


    attn = psa_block.attn
    if hasattr(attn, "qkv"):


        prune_conv_in(attn.qkv, keep_idx)



    if hasattr(attn, "proj"):

        prune_conv_in(attn.proj, keep_idx)
        prune_conv_out(attn.proj, keep_idx)
    if hasattr(attn, "pe"):

        if attn.pe is not None:
            prune_conv_in(attn.pe, keep_idx)
            prune_conv_out(attn.pe, keep_idx)


    ffn = psa_block.ffn
    if isinstance(ffn, nn.Sequential) and len(ffn) >= 2:

        prune_conv_in(ffn[0], keep_idx)

        prune_conv_out(ffn[1], keep_idx)





def prune_c3_out(module, keep_idx: torch.Tensor):













    prune_conv_out(module.cv3, keep_idx)

def prune_c3_in(module, keep_idx: torch.Tensor):








    prune_conv_in(module.cv1, keep_idx)
    prune_conv_in(module.cv2, keep_idx)

def prune_c3_internal(module, keep_hidden: torch.Tensor):


















    new_c = len(keep_hidden)


    prune_conv_out(module.cv1, keep_hidden)
    prune_conv_out(module.cv2, keep_hidden)


    for bottleneck in module.m:
        prune_bottleneck_inout(bottleneck, keep_hidden)

        c_ = bottleneck.cv1.conv.out_channels
        if c_ > 0:
            keep_bn_hidden = torch.arange(c_, device=keep_hidden.device)
            prune_bottleneck_hidden(bottleneck, keep_bn_hidden)


    keep_cv3_in = torch.arange(2 * new_c, device=keep_hidden.device)
    prune_conv_in(module.cv3, keep_cv3_in)





def prune_bottleneck_csp_out(module, keep_idx: torch.Tensor):
















    prune_conv_out(module.cv4, keep_idx)

def prune_bottleneck_csp_in(module, keep_idx: torch.Tensor):






    prune_conv_in(module.cv1, keep_idx)
    prune_conv_in(module.cv2, keep_idx)

def prune_bottleneck_csp_internal(module, keep_hidden: torch.Tensor):









    new_c = len(keep_hidden)


    prune_conv_out(module.cv1, keep_hidden)
    prune_conv_out(module.cv2, keep_hidden)


    for bottleneck in module.m:
        prune_bottleneck_inout(bottleneck, keep_hidden)
        c_ = bottleneck.cv1.conv.out_channels
        if c_ > 0:
            keep_bn_hidden = torch.arange(c_, device=keep_hidden.device)
            prune_bottleneck_hidden(bottleneck, keep_bn_hidden)




    keep_cv4_in = torch.cat([keep_hidden, keep_hidden + new_c])
    prune_conv_in(module.cv4, keep_cv4_in)





def prune_adown_out(module, keep_idx: torch.Tensor):













    old_out = module.cv1.conv.out_channels
    new_out = len(keep_idx)
    half = new_out




    prune_conv_out(module.cv1, torch.arange(half, device=keep_idx.device))


    prune_conv_out(module.cv2, torch.arange(half, device=keep_idx.device))


    module.c = half

def prune_adown_in(module, keep_idx: torch.Tensor):









    prune_conv_in(module.cv1, keep_idx)
    prune_conv_in(module.cv2, keep_idx)

def prune_adown_internal(module, keep_hidden: torch.Tensor):





    pass





def prune_sppelan_out(module, keep_idx: torch.Tensor):












    prune_conv_out(module.cv5, keep_idx)

def prune_sppelan_in(module, keep_idx: torch.Tensor):






    prune_conv_in(module.cv1, keep_idx)

def prune_sppelan_internal(module, keep_hidden: torch.Tensor):
















    new_c = len(keep_hidden)


    prune_conv_out(module.cv1, keep_hidden)





    keep_cv5_in = torch.cat([keep_hidden + i * new_c for i in range(4)])
    prune_conv_in(module.cv5, keep_cv5_in)





def prune_spp_out(module, keep_idx: torch.Tensor):












    prune_conv_out(module.cv2, keep_idx)

def prune_spp_in(module, keep_idx: torch.Tensor):






    prune_conv_in(module.cv1, keep_idx)

def prune_spp_internal(module, keep_hidden: torch.Tensor):
















    new_c = len(keep_hidden)


    prune_conv_out(module.cv1, keep_hidden)





    keep_cv2_in = torch.cat([keep_hidden + i * new_c for i in range(4)])
    prune_conv_in(module.cv2, keep_cv2_in)





def prune_ghostconv_out(module, keep_idx: torch.Tensor):


















    half = len(keep_idx) // 2
    new_c = half


    prune_conv_out(module.cv1, torch.arange(new_c, device=keep_idx.device))


    prune_conv_out(module.cv2, torch.arange(new_c, device=keep_idx.device))

def prune_ghostconv_in(module, keep_idx: torch.Tensor):








    prune_conv_in(module.cv1, keep_idx)





def prune_c2fattn_out(module, keep_idx: torch.Tensor):













    prune_c2f_out(module, keep_idx)

def prune_c2fattn_in(module, keep_idx: torch.Tensor):






    prune_c2f_in(module, keep_idx)





def prune_a2c2f_out(module, keep_idx: torch.Tensor):













    prune_conv_out(module.cv2, keep_idx)

def prune_a2c2f_in(module, keep_idx: torch.Tensor):






    prune_conv_in(module.cv1, keep_idx)





def prune_scdown_out(module, keep_idx: torch.Tensor):











    prune_conv_out(module.cv2, keep_idx)

def prune_scdown_in(module, keep_idx: torch.Tensor):






    prune_conv_in(module.cv1, keep_idx)





def prune_detect_in(detect, scale_idx: int, keep_idx: torch.Tensor):








    cv2_seq = detect.cv2[scale_idx]
    prune_conv_in(cv2_seq[0], keep_idx)



    cv3_seq = detect.cv3[scale_idx]
    first = cv3_seq[0]
    if isinstance(first, nn.Sequential):


        _prune_dwconv_module(first[0], keep_idx)

        prune_conv_in(first[1], keep_idx)
    else:

        prune_conv_in(first, keep_idx)

def _prune_dwconv_module(dwconv_module, keep_idx: torch.Tensor):

    conv = dwconv_module.conv
    conv.weight = nn.Parameter(conv.weight.data[keep_idx])
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[keep_idx])
    conv.in_channels = len(keep_idx)
    conv.out_channels = len(keep_idx)
    conv.groups = len(keep_idx)

    if hasattr(dwconv_module, "bn"):
        bn = dwconv_module.bn
        bn.weight = nn.Parameter(bn.weight.data[keep_idx])
        bn.bias = nn.Parameter(bn.bias.data[keep_idx])
        bn.running_mean = bn.running_mean[keep_idx]
        bn.running_var = bn.running_var[keep_idx]
        bn.num_features = len(keep_idx)





def prune_featurefusion_out(module, keep_idx: torch.Tensor):








    channel_emb = getattr(module, "channel_emb", None)
    if channel_emb is None:
        raise ValueError(f"FeatureFusion: channel_emb not found on {type(module).__name__}")

    channel_embed_seq = getattr(channel_emb, "channel_embed", None)
    if channel_embed_seq is None or len(channel_embed_seq) < 4:
        raise ValueError(f"FeatureFusion: channel_embed seq not found or too short")

    prune_raw_conv2d_out(channel_emb.residual, keep_idx)
    prune_raw_conv2d_out(channel_embed_seq[3], keep_idx)
    if len(channel_embed_seq) >= 5 and isinstance(channel_embed_seq[4], nn.BatchNorm2d):
        prune_batchnorm2d_out(channel_embed_seq[4], keep_idx)
    if isinstance(channel_emb.norm, nn.BatchNorm2d):
        prune_batchnorm2d_out(channel_emb.norm, keep_idx)
    channel_emb.out_channels = len(keep_idx)

def prune_fcmfeaturefusion_out(module, keep_idx: torch.Tensor):





    ffm = getattr(module, "ffm", None)
    if ffm is None:
        raise ValueError(f"FCMFeatureFusion: ffm (FeatureFusion) not found")
    prune_featurefusion_out(ffm, keep_idx)

def prune_mcfgatedfusion_out(module, keep_idx: torch.Tensor):









    post = getattr(module, "post", None)
    if post is not None:

        prune_conv_out(post, keep_idx)
    elif hasattr(module, "gate"):

        gate = module.gate
        gate.out_channels = len(keep_idx)
        gate.weight = nn.Parameter(gate.weight.data[keep_idx])
        if gate.bias is not None:
            gate.bias = nn.Parameter(gate.bias.data[keep_idx])
        if hasattr(module, "bn") and hasattr(module.bn, "weight"):
            bn = module.bn
            bn.num_features = len(keep_idx)
            bn.weight = nn.Parameter(bn.weight.data[keep_idx])
            if bn.bias is not None:
                bn.bias = nn.Parameter(bn.bias.data[keep_idx])
            bn.running_mean = bn.running_mean[keep_idx]
            bn.running_var = bn.running_var[keep_idx]

def prune_featurefusion_in(module, keep_idx: torch.Tensor):




    channel_emb = getattr(module, "channel_emb", None)
    if channel_emb is None:
        raise ValueError(f"FeatureFusion: channel_emb not found")

    prune_raw_conv2d_in(channel_emb.channel_embed[0], keep_idx)
    prune_raw_conv2d_in(channel_emb.residual, keep_idx)

def prune_mcfgatedfusion_in(module, keep_idx: torch.Tensor):







    gate = getattr(module, "gate", None)
    if gate is not None:
        gate.in_channels = len(keep_idx)
        gate.weight = nn.Parameter(gate.weight.data[:, keep_idx, :, :])





def prune_ssa_in(module, keep_idx: torch.Tensor):











    gating = getattr(module, "gating", None)
    if gating is None:
        return

    for layer in gating:
        if isinstance(layer, nn.Conv2d):
            c = layer.in_channels
            g = layer.groups
            group_map = _ssa_group_input_indices(c, g, keep_idx.device)
            kept_total = int(keep_idx.numel())
            if kept_total == 0:
                raise ValueError("SSA prune received empty keep_idx")
            if kept_total % g != 0:
                raise ValueError(
                    f"SSA prune requires keep_idx count divisible by groups: keep={kept_total}, groups={g}"
                )

            new_weight_parts = []
            bias_keep_parts = []
            kept_per_group = None
            for gi in range(g):
                original_group = group_map[gi]
                local_keep = torch.nonzero(torch.isin(original_group, keep_idx), as_tuple=False).flatten()
                if kept_per_group is None:
                    kept_per_group = int(local_keep.numel())
                    if kept_per_group <= 0:
                        raise ValueError(
                            f"SSA prune produced empty local group keep at group {gi}; keep={kept_total}, groups={g}"
                        )
                elif int(local_keep.numel()) != kept_per_group:
                    raise ValueError(
                        "SSA prune requires equal kept channels per shuffled group, "
                        f"but group {gi} has {int(local_keep.numel())} vs expected {kept_per_group}"
                    )

                start = gi * (c // g)
                end = start + (c // g)
                group_weight = layer.weight.data[start:end]
                new_weight_parts.append(group_weight[local_keep][:, local_keep])
                bias_keep_parts.append(local_keep + start)

            layer.weight = nn.Parameter(torch.cat(new_weight_parts, dim=0))
            if layer.bias is not None:
                layer.bias = nn.Parameter(layer.bias.data[torch.cat(bias_keep_parts)])

            new_c = len(keep_idx)
            if new_c % g != 0:
                raise ValueError(f"SSA pruned channels {new_c} must remain divisible by groups {g}")
            layer.in_channels = new_c
            layer.out_channels = new_c


    module._c = len(keep_idx)

