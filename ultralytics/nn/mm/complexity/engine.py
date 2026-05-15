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

from __future__ import annotations

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


# ------------------------------------------------------------------
# Debug mode (set to True for detailed shape/flops logging)
# ------------------------------------------------------------------
DEBUG = False


def _debug_log(msg: str):
    """Print debug message if DEBUG mode is enabled."""
    if DEBUG:
        LOGGER.info(f"[ComplexityDebug] {msg}")


def _ensure_same_spatial(node, input_shapes, context: str) -> tuple[int, int]:
    """Ensure all input tensors share the same spatial size.

    Args:
        node: PruneNode being processed.
        input_shapes: Input TensorShapeSpec tuple.
        context: Short label to make failures easier to localize.

    Returns:
        The common `(height, width)`.

    Raises:
        RuntimeError: If any input differs in spatial size.
    """
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


# ------------------------------------------------------------------
# Shape inference registry (SHAPE_RULES)
# ------------------------------------------------------------------
"""
SHAPE_RULES 注册表设计原则：
1. 每个模块类型有明确的输出形状规则
2. 支持会改变空间尺寸的模块（Conv 下采样、Upsample 上采样）
3. Concat 必须校验所有输入空间尺寸一致
4. 未注册的节点 fail-fast，不使用默认值
"""

SHAPE_RULES = {}


def register_shape(name):
    """Decorator to register a shape inference rule for a module type."""
    def deco(fn):
        SHAPE_RULES[name] = fn
        return fn
    return deco


# ------------------------------------------------------------------
# Helper: 卷积输出尺寸计算
# ------------------------------------------------------------------

def _conv_output_shape(in_shape, conv_module):
    """计算卷积输出的标准公式.

    H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    W_out = floor((W_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)

    Args:
        in_shape: 输入 TensorShapeSpec
        conv_module: nn.Conv2d 实例

    Returns:
        TensorShapeSpec for output
    """
    h_in, w_in = in_shape.height, in_shape.width

    # 读取卷积参数
    kernel_size = conv_module.kernel_size
    stride = conv_module.stride
    padding = conv_module.padding
    dilation = conv_module.dilation

    # 处理 tuple 参数
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

    # 标准卷积输出公式
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    # 防御性检查：输出尺寸不能 <= 0
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


# ------------------------------------------------------------------
# Shape rules: Conv 系列（含下采样）
# ------------------------------------------------------------------

@register_shape("Conv")
def shape_conv(node, input_shapes):
    """Conv 包装层：读取内部 .conv 的 stride 计算输出尺寸."""
    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    conv = getattr(node.module, 'conv', None)
    if conv is None:
        raise RuntimeError(f"Conv node {node.idx} has no .conv attribute")

    return (_conv_output_shape(src, conv),)


@register_shape("GhostConv")
def shape_ghost_conv(node, input_shapes):
    """GhostConv：主分支决定输出尺寸."""
    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    # GhostConv 的主分支是 cv1.conv
    cv1 = getattr(node.module, 'cv1', None)
    if cv1 is None:
        raise RuntimeError(f"GhostConv node {node.idx} has no .cv1 attribute")

    conv = getattr(cv1, 'conv', None)
    if conv is None:
        raise RuntimeError(f"GhostConv node {node.idx}.cv1 has no .conv attribute")

    return (_conv_output_shape(src, conv),)


@register_shape("SCDown")
def shape_scdown(node, input_shapes):
    """SCDown：通道式空间下采样，使用 cv1/cv2 中较大的 stride."""
    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    module = node.module

    # SCDown 有两个并行分支，取主分支的 stride
    cv1 = getattr(module, 'cv1', None)
    if cv1 and hasattr(cv1, 'conv'):
        return (_conv_output_shape(src, cv1.conv),)

    cv2 = getattr(module, 'cv2', None)
    if cv2 and hasattr(cv2, 'conv'):
        return (_conv_output_shape(src, cv2.conv),)

    raise RuntimeError(f"SCDown node {node.idx} has no valid conv branch")


@register_shape("AConv")
def shape_aconv(node, input_shapes):
    """AConv：带平均的下采样卷积."""
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
    """ADown：并行下采样，两个分支 stride 相同."""
    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    module = node.module

    # 任一分支都可以
    for cv_name in ['cv1', 'cv2']:
        cv = getattr(module, cv_name, None)
        if cv and hasattr(cv, 'conv'):
            return (_conv_output_shape(src, cv.conv),)

    raise RuntimeError(f"ADown node {node.idx} has no valid conv branch")


# ------------------------------------------------------------------
# Shape rules: C2f/C3k2/C2PSA/C3/BottleneckCSP（空间不变）
# ------------------------------------------------------------------

def _shape_bottleneck_block(node, input_shapes):
    """C2f/C3k2/C2PSA/C3 等瓶颈块：空间尺寸不变."""
    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    # 这些块的 cv1/cv2 都是 1x1 卷积，空间尺寸不变
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


# ------------------------------------------------------------------
# Shape rules: SPPF/SPP/SPPELAN（空间不变）
# ------------------------------------------------------------------

def _shape_spp_variant(node, input_shapes):
    """SPPF/SPP/SPPELAN：池化不改变空间尺寸."""
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


# ------------------------------------------------------------------
# Shape rules: 融合模块（空间不变）
# ------------------------------------------------------------------

def _shape_fusion(node, input_shapes):
    """融合模块：空间尺寸不变."""
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


# ------------------------------------------------------------------
# Shape rules: Concat（强校验）
# ------------------------------------------------------------------

@register_shape("Concat")
def shape_concat(node, input_shapes):
    """Concat：必须校验所有输入空间尺寸一致."""
    if not input_shapes:
        raise RuntimeError(f"Concat node {node.idx} has no inputs")

    first_h, first_w = _ensure_same_spatial(node, input_shapes, context="Concat")

    # 通道相加
    channels = sum(shape.channels for shape in input_shapes)
    return (
        TensorShapeSpec(
            channels=channels,
            height=first_h,
            width=first_w,
        ),
    )


# ------------------------------------------------------------------
# Shape rules: Upsample
# ------------------------------------------------------------------

@register_shape("Upsample")
def shape_upsample(node, input_shapes):
    """Upsample：scale_factor 放大空间尺寸."""
    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    src = input_shapes[0]
    scale = int(getattr(node.module, "scale_factor", 2) or 2)

    h_out = src.height * scale
    w_out = src.width * scale

    # 防御性检查
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


# ------------------------------------------------------------------
# Shape rules: Index
# ------------------------------------------------------------------

@register_shape("Index")
def shape_index(node, input_shapes):
    """Index：直接传递选中槽位的形状."""
    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    return (input_shapes[0],)


# ------------------------------------------------------------------
# Shape rules: 多输出模块
# ------------------------------------------------------------------

@register_shape("FCM")
def shape_fcm(node, input_shapes):
    """FCM：多输出，空间尺寸不变."""
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
    """MultiHeadCrossAttention：多输出，空间尺寸不变."""
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


# ------------------------------------------------------------------
# Shape rules: 检测头（保留输入形状用于多尺度）
# ------------------------------------------------------------------

def _shape_head(node, input_shapes):
    """检测头：保留各输入尺度形状."""
    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    # 检测头保留所有输入形状（用于多尺度检测）
    return tuple(
        TensorShapeSpec(channels=shape.channels, height=shape.height, width=shape.width)
        for shape in input_shapes
    )


register_shape("Detect")(_shape_head)
register_shape("Segment")(_shape_head)
register_shape("Pose")(_shape_head)
register_shape("OBB")(_shape_head)
register_shape("Classification")(_shape_head)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def compute_multimodal_complexity_report(
    model,
    imgsz: int = 640,
    route_mode: RouteMode = "dual",
    modality: str | None = None,
) -> ComplexityReport:
    """Compute graph-driven complexity report with explicit multimodal input semantics."""
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
    """Compute the default-structure complexity report for YOLOMM mainline flows.

    This intentionally ignores runtime modality switches and reports the model's
    structural complexity truth source using the shared graph-driven engine.
    """
    return compute_multimodal_complexity_report(
        model=model,
        imgsz=imgsz,
        route_mode="dual",
        modality=None,
    )


def compute_pruning_complexity_report(model, imgsz: int = 640) -> ComplexityReport:
    """Compute pruning-stage complexity report using the shared multimodal engine."""
    return compute_default_multimodal_complexity_report(model=model, imgsz=imgsz)


# ------------------------------------------------------------------
# Input spec construction
# ------------------------------------------------------------------

def build_complexity_input_spec(
    model,
    graph,
    imgsz: int,
    route_mode: RouteMode = "dual",
    modality: str | None = None,
) -> ComplexityInputSpec:
    """Build explicit input spec from router or graph metadata.

    Priority order:
    1. Read from multimodal_router.INPUT_SOURCES (most reliable)
    2. Fall back to graph entry node analysis
    3. Default to single-modal RGB if no multimodal info

    Args:
        model: The model being analyzed.
        graph: PruneGraph from build_prune_graph().
        imgsz: Image size.
        route_mode: Explicit route mode requested by caller.
        modality: Optional runtime modality tag for future callers.

    Returns:
        ComplexityInputSpec with explicit route_mode and channel counts.
    """
    _ = modality  # Reserved for future routing-sensitive input-spec branching.

    # Try to read from the real router
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

    # Fall back to graph entry analysis
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

    # No X modality detected
    return ComplexityInputSpec(
        imgsz=(imgsz, imgsz),
        route_mode="rgb" if route_mode != "dual" else "dual",
        rgb_channels=3,
        x_channels=0,
    )


# ------------------------------------------------------------------
# Shape propagation
# ------------------------------------------------------------------

def _seed_entry_shapes(graph, input_spec):
    """Seed input shapes for all entry nodes.

    Entry nodes are those that receive external image input, not output from
    other layers. This explicitly sets their shapes based on input_spec rather
    than inferring from upstream.

    Args:
        graph: PruneGraph instance.
        input_spec: ComplexityInputSpec with true external semantics.

    Returns:
        dict mapping node_idx -> (TensorShapeSpec,) for entry nodes.
    """
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
            # Default to RGB for unspecified entries
            seeded[node.idx] = (
                TensorShapeSpec(channels=input_spec.rgb_channels, height=h, width=w),
            )

    return seeded


def _infer_node_output_shapes(node, input_shapes):
    """Infer output tensor shapes for a node using SHAPE_RULES.

    Args:
        node: PruneNode instance.
        input_shapes: Tuple of TensorShapeSpec for each input.

    Returns:
        Tuple of TensorShapeSpec (single for most modules, multiple for FCM/etc).

    Raises:
        RuntimeError: If no shape rule is registered for this node type.
    """
    if not input_shapes:
        # Entry nodes already seeded, this shouldn't happen for non-entry
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    # 查找形状规则
    shape_rule = SHAPE_RULES.get(node.type_name)

    if shape_rule is None:
        # Fail-fast：未注册的节点类型必须显式处理
        raise RuntimeError(
            f"No shape rule registered for node type '{node.type_name}' at layer {node.idx}. "
            f"Register a shape rule in ultralytics/nn/mm/complexity/engine.py SHAPE_RULES."
        )

    return shape_rule(node, input_shapes)


def _propagate_shapes(graph, input_spec):
    """Propagate tensor shapes through the entire graph.

    Args:
        graph: PruneGraph instance.
        input_spec: ComplexityInputSpec.

    Returns:
        Tuple of (entry_inputs, node_outputs) where:
        - entry_inputs: dict node_idx -> input_shapes for entry nodes
        - node_outputs: dict node_idx -> output_shapes for all nodes
    """
    node_outputs = {}
    entry_inputs = _seed_entry_shapes(graph, input_spec)

    for node in graph.nodes:
        # Get input shapes
        if node.idx in entry_inputs:
            input_shapes = entry_inputs[node.idx]
        else:
            # Collect from upstream edges
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

        # Infer output shapes
        output_shapes = _infer_node_output_shapes(node, input_shapes)
        node_outputs[node.idx] = output_shapes

        # Debug logging
        if DEBUG:
            _debug_log(
                f"Node {node.idx} ({node.type_name}): "
                f"input={[(s.channels, s.height, s.width) for s in input_shapes]} -> "
                f"output={[(s.channels, s.height, s.width) for s in output_shapes]}"
            )

    return entry_inputs, node_outputs


# ------------------------------------------------------------------
# Stage classification
# ------------------------------------------------------------------

def _classify_stage(node) -> StageKind:
    """Classify a node into its multimodal stage.

    Args:
        node: PruneNode instance.

    Returns:
        One of: 'rgb_branch', 'x_branch', 'fusion', 'head', 'route_only'.
    """
    if node.is_route_only:
        return "route_only"
    if node.is_head:
        return "head"
    if node.branch_kind == "rgb":
        return "rgb_branch"
    if node.branch_kind == "x":
        return "x_branch"
    return "fusion"


# ------------------------------------------------------------------
# FLOPs computation
# ------------------------------------------------------------------

def _compute_node_flops(node, input_shapes, output_shapes) -> float:
    """Compute FLOPs for a single node using registered rules.

    Args:
        node: PruneNode instance.
        input_shapes: Tuple of input TensorShapeSpec.
        output_shapes: Tuple of output TensorShapeSpec.

    Returns:
        FLOPs as float.

    Raises:
        RuntimeError: If no rule is registered for a computational node.
    """
    rule = RULES.get(node.type_name)

    if rule is None:
        # Fail-fast for unregistered computational nodes
        if not node.is_route_only and not node.is_head:
            raise RuntimeError(
                f"No complexity rule registered for node type '{node.type_name}' at layer {node.idx}. "
                f"Register a rule in ultralytics/nn/mm/complexity/rules.py or mark as route-only/head."
            )
        return 0.0

    return float(rule(node, input_shapes, output_shapes))


# ------------------------------------------------------------------
# Main engine
# ------------------------------------------------------------------

def _run_complexity_engine(model, graph, input_spec) -> ComplexityReport:
    """Run the full complexity analysis pipeline.

    Args:
        model: The model (used for module access).
        graph: PruneGraph instance.
        input_spec: ComplexityInputSpec.

    Returns:
        ComplexityReport with per-node results and aggregates.
    """
    entry_inputs, node_outputs = _propagate_shapes(graph, input_spec)

    nodes = []
    for node in graph.nodes:
        # Get input shapes
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

        # Get output shapes
        output_shapes = node_outputs.get(node.idx, ())

        # Classify stage
        stage = _classify_stage(node)

        # Compute FLOPs
        flops = _compute_node_flops(node, input_shapes, output_shapes)

        # Debug logging
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
