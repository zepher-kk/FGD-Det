from __future__ import annotations
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Multi-input consumer parameter adapters for structured channel pruning.

This module provides:
1. `rebuild_consumer_if_supported()` - rebuilds lazy-initialized fusion modules
   after input channel counts change.
2. Fixed-dimension consumer adapters that perform parameter-level pruning
   without rebuilding new modules (FeatureFusion, FCM, MCFGatedFusion,
   CrossTransformerFusion, etc.).

All functions modify modules in-place.
"""

import torch
import torch.nn as nn

from ultralytics.nn.mm.pruning.ops import (
    prune_batchnorm2d_out,
    prune_conv_in,
    prune_conv_out,
    prune_raw_conv2d_in,
    prune_raw_conv2d_out,
)





def _rebuild_equal_width_single_arg(module, ch: list[int]) -> None:




    c = ch[0]
    module._build_if_needed(c)

REBUILD_DISPATCH: dict[str, callable] = {

    "CAM": lambda module, ch: module._build_if_needed(ch[0], ch[1]),
    "RFF": lambda module, ch: module._build(ch[0], ch[1]),

    "SEFN": _rebuild_equal_width_single_arg,
    "FusionConvMSAA": _rebuild_equal_width_single_arg,
    "SpatialDependencyPerception": _rebuild_equal_width_single_arg,


}

def rebuild_consumer_if_supported(node_type: str, module, input_channels: list[int]) -> bool:












    rebuild = REBUILD_DISPATCH.get(node_type)
    if rebuild is None:
        return False
    rebuild(module, input_channels)
    return True





def _take_prefix_keep(width: int, new_width: int, device: torch.device) -> torch.Tensor:
    return torch.arange(min(width, new_width), device=device)

def _prune_linear_in(linear: nn.Linear, keep_idx: torch.Tensor) -> None:
    linear.weight = nn.Parameter(linear.weight.data[:, keep_idx])
    linear.in_features = len(keep_idx)

def _prune_linear_out(linear: nn.Linear, keep_idx: torch.Tensor) -> None:
    linear.weight = nn.Parameter(linear.weight.data[keep_idx])
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias.data[keep_idx])
    linear.out_features = len(keep_idx)

def _prune_layernorm(layernorm: nn.LayerNorm, keep_idx: torch.Tensor) -> None:
    if layernorm.elementwise_affine:
        layernorm.weight = nn.Parameter(layernorm.weight.data[keep_idx])
        layernorm.bias = nn.Parameter(layernorm.bias.data[keep_idx])
    layernorm.normalized_shape = (len(keep_idx),)

def prune_feature_interaction_inputs(module, left_keep: torch.Tensor, right_keep: torch.Tensor) -> None:

















    new_dim = len(left_keep)
    hidden_old = module.channel_proj1.out_features // 2
    reduction = max(module.channel_proj1.in_features // max(hidden_old, 1), 1)
    hidden_keep = _take_prefix_keep(hidden_old, max(new_dim // reduction, 1), left_keep.device)
    proj_keep = torch.cat([hidden_keep, hidden_keep + hidden_old])

    _prune_linear_in(module.channel_proj1, left_keep)
    _prune_linear_out(module.channel_proj1, proj_keep)
    _prune_linear_in(module.channel_proj2, right_keep)
    _prune_linear_out(module.channel_proj2, proj_keep)

    cross_attn = module.cross_attn
    hidden_kv_keep = torch.cat([hidden_keep, hidden_keep + hidden_old])
    for attr in ("q1", "q2"):
        linear = getattr(cross_attn, attr)
        _prune_linear_in(linear, hidden_keep)
        _prune_linear_out(linear, hidden_keep)
    for attr in ("kv1", "kv2"):
        linear = getattr(cross_attn, attr)
        _prune_linear_in(linear, hidden_keep)
        _prune_linear_out(linear, hidden_kv_keep)

    if getattr(cross_attn, "sr_ratio", 1) > 1:
        for attr in ("sr1", "sr2"):
            conv = getattr(cross_attn, attr)
            prune_raw_conv2d_in(conv, hidden_keep)
            prune_raw_conv2d_out(conv, hidden_keep)
            conv.groups = len(hidden_keep)
        for attr in ("norm1", "norm2"):
            _prune_layernorm(getattr(cross_attn, attr), hidden_keep)

    _prune_linear_in(module.end_proj1, proj_keep)
    _prune_linear_out(module.end_proj1, left_keep)
    _prune_linear_in(module.end_proj2, proj_keep)
    _prune_linear_out(module.end_proj2, right_keep)
    _prune_layernorm(module.norm1, left_keep)
    _prune_layernorm(module.norm2, right_keep)

def prune_channel_embed_inputs(module, total_in: int, left_width: int) -> None:










    del total_in, left_width

def prune_channel_embed_output(module, keep_idx: torch.Tensor) -> None:






    new_out = len(keep_idx)
    module.out_channels = new_out

def prune_featurefusion_consumer(
    module,
    left_keep: torch.Tensor,
    right_keep: torch.Tensor,
    out_keep: torch.Tensor | None = None,
) -> None:












    prune_feature_interaction_inputs(module.cross, left_keep, right_keep)
    orig_dim = module.channel_emb.out_channels
    final_keep = out_keep if out_keep is not None else torch.arange(len(left_keep), device=left_keep.device)
    hidden_old = module.channel_emb.channel_embed[0].out_channels
    reduction = max(orig_dim // max(hidden_old, 1), 1)
    hidden_keep = _take_prefix_keep(hidden_old, max(len(final_keep) // reduction, 1), left_keep.device)
    input_keep = torch.cat([left_keep, right_keep + orig_dim])

    prune_raw_conv2d_in(module.channel_emb.residual, input_keep)
    prune_raw_conv2d_out(module.channel_emb.residual, final_keep)

    prune_raw_conv2d_in(module.channel_emb.channel_embed[0], input_keep)
    prune_raw_conv2d_out(module.channel_emb.channel_embed[0], hidden_keep)

    dw = module.channel_emb.channel_embed[1]
    prune_raw_conv2d_in(dw, hidden_keep)
    prune_raw_conv2d_out(dw, hidden_keep)
    dw.groups = len(hidden_keep)

    prune_raw_conv2d_in(module.channel_emb.channel_embed[3], hidden_keep)
    prune_raw_conv2d_out(module.channel_emb.channel_embed[3], final_keep)

    if isinstance(module.channel_emb.channel_embed[4], nn.BatchNorm2d):
        prune_batchnorm2d_out(module.channel_emb.channel_embed[4], final_keep)
    if isinstance(module.channel_emb.norm, nn.BatchNorm2d):
        prune_batchnorm2d_out(module.channel_emb.norm, final_keep)
    module.channel_emb.out_channels = len(final_keep)

def prune_featurefusion_output(module, keep_idx: torch.Tensor) -> None:






    prune_channel_embed_output(module.channel_emb, keep_idx)





def prune_fcm_consumer(module, left_keep: torch.Tensor, right_keep: torch.Tensor) -> None:















    new_dim = len(left_keep)
    orig_dim = module.spatial_weights.dim

    spatial_conv1 = module.spatial_weights.mlp[0]
    spatial_conv2 = module.spatial_weights.mlp[2]
    spatial_hidden_old = spatial_conv1.out_channels
    spatial_reduction = max(orig_dim // max(spatial_hidden_old, 1), 1)
    spatial_hidden_keep = _take_prefix_keep(
        spatial_hidden_old, max(new_dim // spatial_reduction, 1), left_keep.device
    )
    fused_keep = torch.cat([left_keep, right_keep + orig_dim])

    prune_raw_conv2d_in(spatial_conv1, fused_keep)
    prune_raw_conv2d_out(spatial_conv1, spatial_hidden_keep)
    prune_raw_conv2d_in(spatial_conv2, spatial_hidden_keep)
    module.spatial_weights.dim = new_dim

    fc1 = module.channel_weights.mlp[0]
    fc2 = module.channel_weights.mlp[2]
    hidden_old = fc1.out_features
    reduction = max((orig_dim * 6) // max(hidden_old, 1), 1)
    hidden_keep = _take_prefix_keep(hidden_old, max((new_dim * 6) // reduction, 1), left_keep.device)
    stats_keep = torch.cat([
        left_keep,
        right_keep + orig_dim,
        left_keep + orig_dim * 2,
        right_keep + orig_dim * 3,
        left_keep + orig_dim * 4,
        right_keep + orig_dim * 5,
    ])
    out_keep = torch.cat([left_keep, right_keep + orig_dim])

    _prune_linear_in(fc1, stats_keep)
    _prune_linear_out(fc1, hidden_keep)
    _prune_linear_in(fc2, hidden_keep)
    _prune_linear_out(fc2, out_keep)
    module.channel_weights.dim = new_dim

def prune_fcm_output(module, keep_idx: torch.Tensor) -> None:









    pass





def prune_mcfgatedfusion_consumer(
    module,
    main_keep: torch.Tensor,
    aux_keep: torch.Tensor,
    out_keep: torch.Tensor,
) -> None:














    prune_raw_conv2d_in(module.gate, aux_keep)

    if module.mode == "concat" and module.post is not None:
        gate_width = module.gate.out_channels
        old_main_width = module.post.conv.in_channels - gate_width
        post_keep = torch.cat([main_keep, torch.arange(gate_width, device=main_keep.device) + old_main_width])
        prune_conv_in(module.post, post_keep)
        prune_conv_out(module.post, out_keep)
    else:
        final_keep = out_keep if out_keep is not None else main_keep
        prune_raw_conv2d_out(module.gate, final_keep)
        if isinstance(module.bn, nn.BatchNorm2d):
            prune_batchnorm2d_out(module.bn, final_keep)

def prune_mcfgatedfusion_output(module, keep_idx: torch.Tensor) -> None:






    mode = module.mode

    if mode == "concat":
        prune_conv_out(module.post, keep_idx)
    else:
        prune_raw_conv2d_out(module.gate, keep_idx)
        if isinstance(module.bn, nn.BatchNorm2d):
            prune_batchnorm2d_out(module.bn, keep_idx)





def prune_cross_transformer_fusion_consumer(
    module,
    left_keep: torch.Tensor,
    right_keep: torch.Tensor,
    out_keep: torch.Tensor | None = None,
) -> None:





























    new_dim = len(left_keep)

    device = module.encoder.embedding.weight.device
    dtype = module.encoder.embedding.weight.dtype
    hidden_dim = module.hidden_dim


    module.encoder.embedding = nn.Linear(new_dim, new_dim).to(device=device, dtype=dtype)


    module.encoder.positional_encoding = type(module.encoder.positional_encoding)(
        new_dim, module.encoder.positional_encoding.dropout.p
    ).to(device=device, dtype=dtype)


    num_layers = len(module.encoder.layers)
    num_heads = module.encoder.layers[0].cross_attention.num_heads
    dropout = module.encoder.layers[0].ff.dropout.p

    new_hidden = new_dim * 2

    module.encoder.layers = nn.ModuleList()
    for _ in range(num_layers):
        from ultralytics.nn.modules.fusion.ctf import (
            FeedForward,
            MultiHeadCrossAttention,
            TransformerEncoderLayer,
        )

        cross_attn = MultiHeadCrossAttention(new_dim, num_heads)
        norm1 = nn.LayerNorm(new_dim)
        ff = FeedForward(new_dim, new_hidden, dropout)
        norm2 = nn.LayerNorm(new_dim)
        layer = TransformerEncoderLayer(new_dim, num_heads, new_hidden, dropout)
        layer.cross_attention = cross_attn
        layer.norm1 = norm1
        layer.ff = ff
        layer.norm2 = norm2
        module.encoder.layers.append(layer)


    module.model_dim = new_dim
    module.hidden_dim = new_hidden

def prune_cross_transformer_fusion_output(module, keep_idx: torch.Tensor) -> None:












    new_total = len(keep_idx)
    if new_total % 2 != 0:
        raise ValueError(f"CrossTransformerFusion output channels must be even, got {new_total}")
    new_dim = new_total // 2

    device = module.encoder.embedding.weight.device
    dtype = module.encoder.embedding.weight.dtype
    num_layers = len(module.encoder.layers)
    num_heads = module.encoder.layers[0].cross_attention.num_heads
    dropout = module.encoder.layers[0].ff.dropout.p
    new_hidden = new_dim * 2


    module.encoder.embedding = nn.Linear(new_dim, new_dim).to(device=device, dtype=dtype)


    module.encoder.positional_encoding = type(module.encoder.positional_encoding)(
        new_dim, dropout
    ).to(device=device, dtype=dtype)


    from ultralytics.nn.modules.fusion.ctf import (
        FeedForward,
        MultiHeadCrossAttention,
        TransformerEncoderLayer,
    )

    module.encoder.layers = nn.ModuleList()
    for _ in range(num_layers):
        cross_attn = MultiHeadCrossAttention(new_dim, num_heads)
        norm1 = nn.LayerNorm(new_dim)
        ff = FeedForward(new_dim, new_hidden, dropout)
        norm2 = nn.LayerNorm(new_dim)
        layer = TransformerEncoderLayer(new_dim, num_heads, new_hidden, dropout)
        layer.cross_attention = cross_attn
        layer.norm1 = norm1
        layer.ff = ff
        layer.norm2 = norm2
        module.encoder.layers.append(layer)

    module.model_dim = new_dim
    module.hidden_dim = new_hidden





def prune_fcmfeaturefusion_consumer(
    module,
    left_keep: torch.Tensor,
    right_keep: torch.Tensor,
    out_keep: torch.Tensor | None = None,
) -> None:










    prune_fcm_consumer(module.fcm, left_keep, right_keep)
    prune_featurefusion_consumer(module.ffm, left_keep, right_keep, out_keep)

def prune_fcmfeaturefusion_output(module, keep_idx: torch.Tensor) -> None:

    prune_featurefusion_output(module.ffm, keep_idx)





CONSUMER_ADAPTERS: dict[str, callable] = {
    "FeatureFusion": prune_featurefusion_consumer,
    "FCM": prune_fcm_consumer,
    "FCMFeatureFusion": prune_fcmfeaturefusion_consumer,
    "MCFGatedFusion": prune_mcfgatedfusion_consumer,
    "CrossTransformerFusion": prune_cross_transformer_fusion_consumer,
}

def adapt_consumer(
    node_type: str,
    module,
    left_keep: torch.Tensor,
    right_keep: torch.Tensor,
    out_keep: torch.Tensor | None = None,
) -> bool:













    adapter = CONSUMER_ADAPTERS.get(node_type)
    if adapter is None:
        return False
    adapter(module, left_keep, right_keep, out_keep)
    return True

