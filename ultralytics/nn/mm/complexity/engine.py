from __future__ import annotations
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Graph-driven complexity engine for YOLOMM multimodal models.

This module provides a unified complexity analysis that:
1. Starts from explicit input semantics (not inferred from parameters)
2. Propagates tensor shapes through the prune graph using SHAPE_RULES
3. Applies per-module-type FLOPs rules
4. Returns stage-breakdown results

Phase 1: pruning pipeline
Phase 2: finetrain integration
Phase 3: YOLOMM mainline default-structure integration
Other family/logger integration is intentionally left for later phases.
"""

import math

from ultralytics.nn.mm.pruning.graph import PruneGraph, build_prune_graph
from ultralytics.utils import LOGGER

from .schema import (
    ComplexityInputSpec,
    ComplexityReport,
    NodeComplexity,
    RouteMode,
    StageKind,
    TensorShapeSpec,
)
from .rules import RULES




DEBUG = False

def _debug_log(msg: str):

    if DEBUG:
        LOGGER.info(f"[ComplexityDebug] {msg}")

def _ensure_same_spatial(node, input_shapes, context: str) -> tuple[int, int]:













    if not input_shapes:
        raise RuntimeError(f"{context} node {node.idx} ({node.type_name}) has no input shapes to validate")

    first_h, first_w = input_shapes[0].height, input_shapes[0].width
    mismatches = []
    for i, shape in enumerate(input_shapes[1:], start=1):
        if shape.height != first_h or shape.width != first_w:
            mismatches.append((i, shape.channels, shape.height, shape.width))

    if mismatches:
        upstream_info = [f"layer_{edge.node_idx}[slot_{edge.output_slot}]" for edge in node.input_edges]
        raise RuntimeError(
            f"{context} node {node.idx} ({node.type_name}) received spatially inconsistent inputs. "
            f"input0=(c={input_shapes[0].channels}, h={first_h}, w={first_w}), "
            f"mismatches={mismatches}, upstream={upstream_info}"
        )

    return first_h, first_w




"""
SHAPE_RULES 注册表设计原则：
1. 每个模块类型有明确的输出形状规则
2. 支持会改变空间尺寸的模块（Conv 下采样、Upsample 上采样）
3. Concat 必须校验所有输入空间尺寸一致
4. 未注册的节点 fail-fast，不使用默认值
"""

SHAPE_RULES = {}

def register_shape(name):

    def deco(fn):
        SHAPE_RULES[name] = fn
        return fn
    return deco





def _conv_output_shape(in_shape, conv_module):












    h_in, w_in = in_shape.height, in_shape.width


    kernel_size = conv_module.kernel_size
    stride = conv_module.stride
    padding = conv_module.padding
    dilation = conv_module.dilation


    if isinstance(kernel_size, (tuple, list)):
        kh, kw = kernel_size
    else:
        kh = kw = kernel_size

    if isinstance(stride, (tuple, list)):
        sh, sw = stride
    else:
        sh = sw = stride

    if isinstance(padding, (tuple, list)):
        ph, pw = padding
    else:
        ph = pw = padding

    if isinstance(dilation, (tuple, list)):
        dh, dw = dilation
    else:
        dh = dw = dilation


    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1


    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"Conv output spatial dims invalid: h_out={h_out}, w_out={w_out} "
            f"from input=({h_in}, {w_in}), kernel=({kh}, {kw}), stride=({sh}, {sw}), "
            f"padding=({ph}, {pw}), dilation=({dh}, {dw})"
        )

    return TensorShapeSpec(
        channels=conv_module.out_channels,
        height=int(h_out),
        width=int(w_out),
    )





@register_shape("Conv")
def shape_conv(node, input_shapes):

    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    conv = getattr(node.module, 'conv', None)
    if conv is None:
        raise RuntimeError(f"Conv node {node.idx} has no .conv attribute")

    return (_conv_output_shape(src, conv),)

@register_shape("GhostConv")
def shape_ghost_conv(node, input_shapes):

    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]

    cv1 = getattr(node.module, 'cv1', None)
    if cv1 is None:
        raise RuntimeError(f"GhostConv node {node.idx} has no .cv1 attribute")

    conv = getattr(cv1, 'conv', None)
    if conv is None:
        raise RuntimeError(f"GhostConv node {node.idx}.cv1 has no .conv attribute")

    return (_conv_output_shape(src, conv),)

@register_shape("SCDown")
def shape_scdown(node, input_shapes):

    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    module = node.module


    cv1 = getattr(module, 'cv1', None)
    if cv1 and hasattr(cv1, 'conv'):
        return (_conv_output_shape(src, cv1.conv),)

    cv2 = getattr(module, 'cv2', None)
    if cv2 and hasattr(cv2, 'conv'):
        return (_conv_output_shape(src, cv2.conv),)

    raise RuntimeError(f"SCDown node {node.idx} has no valid conv branch")

@register_shape("AConv")
def shape_aconv(node, input_shapes):

    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    cv1 = getattr(node.module, 'cv1', None)
    if cv1 is None:
        raise RuntimeError(f"AConv node {node.idx} has no .cv1 attribute")

    conv = getattr(cv1, 'conv', None)
    if conv is None:
        raise RuntimeError(f"AConv node {node.idx}.cv1 has no .conv attribute")

    return (_conv_output_shape(src, conv),)

@register_shape("ADown")
def shape_adown(node, input_shapes):

    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    module = node.module


    for cv_name in ['cv1', 'cv2']:
        cv = getattr(module, cv_name, None)
        if cv and hasattr(cv, 'conv'):
            return (_conv_output_shape(src, cv.conv),)

    raise RuntimeError(f"ADown node {node.idx} has no valid conv branch")





def _shape_bottleneck_block(node, input_shapes):

    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]

    return (
        TensorShapeSpec(
            channels=node.primary_out_channels,
            height=src.height,
            width=src.width,
        ),
    )

register_shape("C2f")(_shape_bottleneck_block)
register_shape("C3k2")(_shape_bottleneck_block)
register_shape("C2PSA")(_shape_bottleneck_block)
register_shape("C2fAttn")(_shape_bottleneck_block)
register_shape("A2C2f")(_shape_bottleneck_block)
register_shape("C3")(_shape_bottleneck_block)
register_shape("BottleneckCSP")(_shape_bottleneck_block)





def _shape_spp_variant(node, input_shapes):

    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    return (
        TensorShapeSpec(
            channels=node.primary_out_channels,
            height=src.height,
            width=src.width,
        ),
    )

register_shape("SPPF")(_shape_spp_variant)
register_shape("SPP")(_shape_spp_variant)
register_shape("SPPELAN")(_shape_spp_variant)





def _shape_fusion(node, input_shapes):

    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    h, w = _ensure_same_spatial(node, input_shapes, context="Fusion")
    return (
        TensorShapeSpec(
            channels=node.primary_out_channels,
            height=h,
            width=w,
        ),
    )

register_shape("FeatureFusion")(_shape_fusion)
register_shape("FCMFeatureFusion")(_shape_fusion)
register_shape("MCFGatedFusion")(_shape_fusion)
register_shape("CrossTransformerFusion")(_shape_fusion)
register_shape("IIA")(_shape_fusion)
register_shape("CTF")(_shape_fusion)
register_shape("SEFN")(_shape_fusion)
register_shape("RFF")(_shape_fusion)
register_shape("MSIA")(_shape_fusion)
register_shape("SOEP")(_shape_fusion)
register_shape("MROD")(_shape_fusion)
register_shape("SequenceShuffleAttention")(_shape_fusion)





@register_shape("Concat")
def shape_concat(node, input_shapes):

    if not input_shapes:
        raise RuntimeError(f"Concat node {node.idx} has no inputs")

    first_h, first_w = _ensure_same_spatial(node, input_shapes, context="Concat")


    channels = sum(shape.channels for shape in input_shapes)
    return (
        TensorShapeSpec(
            channels=channels,
            height=first_h,
            width=first_w,
        ),
    )





@register_shape("Upsample")
def shape_upsample(node, input_shapes):

    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    scale = int(getattr(node.module, "scale_factor", 2) or 2)

    h_out = src.height * scale
    w_out = src.width * scale


    if h_out > 10000 or w_out > 10000:
        _debug_log(
            f"Warning: Upsample node {node.idx} producing very large spatial size: "
            f"({src.height}, {src.width}) -> ({h_out}, {w_out})"
        )

    return (
        TensorShapeSpec(
            channels=src.channels,
            height=h_out,
            width=w_out,
        ),
    )





@register_shape("Index")
def shape_index(node, input_shapes):

    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    return (input_shapes[0],)





@register_shape("FCM")
def shape_fcm(node, input_shapes):

    if not input_shapes:
        return tuple(
            TensorShapeSpec(channels=c, height=1, width=1)
            for c in node.out_channels
        )

    src = input_shapes[0]
    return tuple(
        TensorShapeSpec(channels=c, height=src.height, width=src.width)
        for c in node.out_channels
    )

@register_shape("MultiHeadCrossAttention")
def shape_mhca(node, input_shapes):

    if not input_shapes:
        return tuple(
            TensorShapeSpec(channels=c, height=1, width=1)
            for c in node.out_channels
        )

    src = input_shapes[0]
    return tuple(
        TensorShapeSpec(channels=c, height=src.height, width=src.width)
        for c in node.out_channels
    )





def _shape_head(node, input_shapes):

    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)


    return tuple(
        TensorShapeSpec(channels=shape.channels, height=shape.height, width=shape.width)
        for shape in input_shapes
    )

register_shape("Detect")(_shape_head)
register_shape("Segment")(_shape_head)
register_shape("Pose")(_shape_head)
register_shape("OBB")(_shape_head)
register_shape("Classification")(_shape_head)





def compute_multimodal_complexity_report(
    model,
    imgsz: int = 640,
    route_mode: RouteMode = "dual",
    modality: str | None = None,
) -> ComplexityReport:

    graph = build_prune_graph(model)
    input_spec = build_complexity_input_spec(
        model=model,
        graph=graph,
        imgsz=imgsz,
        route_mode=route_mode,
        modality=modality,
    )
    return _run_complexity_engine(model=model, graph=graph, input_spec=input_spec)

def compute_default_multimodal_complexity_report(model, imgsz: int = 640) -> ComplexityReport:





    return compute_multimodal_complexity_report(
        model=model,
        imgsz=imgsz,
        route_mode="dual",
        modality=None,
    )

def compute_pruning_complexity_report(model, imgsz: int = 640) -> ComplexityReport:

    return compute_default_multimodal_complexity_report(model=model, imgsz=imgsz)





def build_complexity_input_spec(
    model,
    graph,
    imgsz: int,
    route_mode: RouteMode = "dual",
    modality: str | None = None,
) -> ComplexityInputSpec:

















    _ = modality


    router = getattr(model, "multimodal_router", None) or getattr(model, "mm_router", None)
    if router is not None and hasattr(router, "INPUT_SOURCES"):
        rgb_channels = int(router.INPUT_SOURCES.get("RGB", 3))
        x_channels = int(router.INPUT_SOURCES.get("X", 3))
        return ComplexityInputSpec(
            imgsz=(imgsz, imgsz),
            route_mode=route_mode,
            rgb_channels=rgb_channels,
            x_channels=x_channels,
        )


    x_entries = [
        node for node in graph.nodes
        if getattr(node.module, "_mm_input_source", None) == "X"
    ]
    if x_entries:
        x_channels = int(getattr(x_entries[0].module, "in_channels", 3) or 3)
        return ComplexityInputSpec(
            imgsz=(imgsz, imgsz),
            route_mode=route_mode,
            rgb_channels=3,
            x_channels=x_channels,
        )


    return ComplexityInputSpec(
        imgsz=(imgsz, imgsz),
        route_mode="rgb" if route_mode != "dual" else "dual",
        rgb_channels=3,
        x_channels=0,
    )





def _seed_entry_shapes(graph, input_spec):













    h, w = input_spec.imgsz
    seeded = {}

    for node in graph.nodes:
        if not node.is_entry:
            continue

        source = getattr(node.module, "_mm_input_source", None)

        if source == "RGB":
            seeded[node.idx] = (
                TensorShapeSpec(channels=input_spec.rgb_channels, height=h, width=w),
            )
        elif source == "X":
            seeded[node.idx] = (
                TensorShapeSpec(channels=input_spec.x_channels, height=h, width=w),
            )
        elif source == "Dual":
            seeded[node.idx] = (
                TensorShapeSpec(
                    channels=input_spec.rgb_channels + input_spec.x_channels,
                    height=h,
                    width=w,
                ),
            )
        else:

            seeded[node.idx] = (
                TensorShapeSpec(channels=input_spec.rgb_channels, height=h, width=w),
            )

    return seeded

def _infer_node_output_shapes(node, input_shapes):












    if not input_shapes:

        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)


    shape_rule = SHAPE_RULES.get(node.type_name)

    if shape_rule is None:

        raise RuntimeError(
            f"No shape rule registered for node type '{node.type_name}' at layer {node.idx}. "
            f"Register a shape rule in ultralytics/nn/mm/complexity/engine.py SHAPE_RULES."
        )

    return shape_rule(node, input_shapes)

def _propagate_shapes(graph, input_spec):











    node_outputs = {}
    entry_inputs = _seed_entry_shapes(graph, input_spec)

    for node in graph.nodes:

        if node.idx in entry_inputs:
            input_shapes = entry_inputs[node.idx]
        else:

            shapes = []
            for edge in node.input_edges:
                if edge.node_idx in node_outputs:
                    output_slot_shapes = node_outputs[edge.node_idx]
                    if edge.output_slot < len(output_slot_shapes):
                        shapes.append(output_slot_shapes[edge.output_slot])
            if not shapes:
                upstream_info = [f"layer_{edge.node_idx}[slot_{edge.output_slot}]" for edge in node.input_edges]
                raise RuntimeError(
                    f"Failed to resolve input shapes for node {node.idx} ({node.type_name}). "
                    f"upstream={upstream_info}"
                )
            input_shapes = tuple(shapes)


        output_shapes = _infer_node_output_shapes(node, input_shapes)
        node_outputs[node.idx] = output_shapes


        if DEBUG:
            _debug_log(
                f"Node {node.idx} ({node.type_name}): "
                f"input={[(s.channels, s.height, s.width) for s in input_shapes]} -> "
                f"output={[(s.channels, s.height, s.width) for s in output_shapes]}"
            )

    return entry_inputs, node_outputs





def _classify_stage(node) -> StageKind:








    if node.is_route_only:
        return "route_only"
    if node.is_head:
        return "head"
    if node.branch_kind == "rgb":
        return "rgb_branch"
    if node.branch_kind == "x":
        return "x_branch"
    return "fusion"





def _compute_node_flops(node, input_shapes, output_shapes) -> float:













    rule = RULES.get(node.type_name)

    if rule is None:

        if not node.is_route_only and not node.is_head:
            raise RuntimeError(
                f"No complexity rule registered for node type '{node.type_name}' at layer {node.idx}. "
                f"Register a rule in ultralytics/nn/mm/complexity/rules.py or mark as route-only/head."
            )
        return 0.0

    return float(rule(node, input_shapes, output_shapes))





def _run_complexity_engine(model, graph, input_spec) -> ComplexityReport:










    entry_inputs, node_outputs = _propagate_shapes(graph, input_spec)

    nodes = []
    for node in graph.nodes:

        if node.idx in entry_inputs:
            input_shapes = entry_inputs[node.idx]
        else:
            shapes = []
            for edge in node.input_edges:
                if edge.node_idx in node_outputs:
                    output_slot_shapes = node_outputs[edge.node_idx]
                    if edge.output_slot < len(output_slot_shapes):
                        shapes.append(output_slot_shapes[edge.output_slot])
            input_shapes = tuple(shapes) if shapes else ()


        output_shapes = node_outputs.get(node.idx, ())


        stage = _classify_stage(node)


        flops = _compute_node_flops(node, input_shapes, output_shapes)


        if DEBUG:
            _debug_log(
                f"Node {node.idx} ({node.type_name}, stage={stage}): "
                f"FLOPs={flops / 1e6:.2f}M"
            )

        nodes.append(
            NodeComplexity(
                node_idx=node.idx,
                type_name=node.type_name,
                stage=stage,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                flops=flops,
            )
        )

    return ComplexityReport(input_spec=input_spec, nodes=nodes)

