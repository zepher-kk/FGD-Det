from __future__ import annotations
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Pruning graph data structures for YOLOMM multimodal models.

This module provides a DAG-like representation of the model topology that
captures the multimodal branch structure (RGB/X/Dual/Fusion) and multi-output
module semantics (FCM, MultiHeadCrossAttention) needed for structured pruning.
"""

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn





BranchKind = Literal["rgb", "x", "dual", "fusion", "head", "single", "unknown"]



_MULTI_OUTPUT_TYPES = frozenset({"FCM", "MultiHeadCrossAttention"})


_ROUTE_ONLY_TYPES = frozenset({"Concat", "Upsample", "Index"})


_HEAD_TYPES = frozenset({"Detect", "Segment", "Pose", "OBB", "Classification"})





@dataclass(frozen=True)
class EdgeRef:








    node_idx: int
    output_slot: int = 0

@dataclass
class PruneNode:





















    idx: int
    module: nn.Module
    type_name: str
    input_edges: tuple[EdgeRef, ...]
    in_channels: int
    out_channels: tuple[int, ...]
    branch_kind: BranchKind
    is_entry: bool
    is_head: bool
    is_multi_input: bool
    is_route_only: bool = False

    @property
    def primary_out_channels(self) -> int:

        return self.out_channels[0]

@dataclass
class PruneGraph:





    nodes: list[PruneNode]

    def node(self, idx: int) -> PruneNode:

        return self.nodes[idx]

    def prunable_nodes(self) -> list[PruneNode]:




        return [
            n for n in self.nodes
            if not n.is_route_only and n.type_name not in {"Concat", "Upsample", "Detect", "Index", "Segment", "Pose", "OBB", "Classification"}
        ]





def infer_branch_kind(layer: nn.Module, parents: list[PruneNode]) -> BranchKind:















    source = getattr(layer, "_mm_input_source", None)
    if source == "RGB":
        return "rgb"
    if source == "X":
        return "x"
    if source == "Dual":
        return "dual"

    parent_kinds = {p.branch_kind for p in parents}
    if not parent_kinds:
        return "unknown"

    if len(parent_kinds) > 1:

        return "fusion"

    return next(iter(parent_kinds))





def _get_layer_type_name(layer: nn.Module) -> str:

    return type(layer).__name__

def _get_in_channels(layer: nn.Module, ltype: str) -> int:

    if ltype == "SequenceShuffleAttention":
        gating = getattr(layer, "gating", None)
        if gating is not None:
            for sub_layer in gating:
                if isinstance(sub_layer, nn.Conv2d):
                    return sub_layer.in_channels
        return getattr(layer, "_c", 0) or 0
    return getattr(layer, "in_channels", 0) or 0

def _get_out_channels_tuple(layer: nn.Module, ltype: str) -> tuple[int, ...]:







    if ltype in _MULTI_OUTPUT_TYPES:
        oc = getattr(layer, "out_channels", None)
        if isinstance(oc, (tuple, list)):
            return tuple(oc)
        if ltype == "FCM":
            dim = getattr(getattr(layer, "spatial_weights", None), "dim", 0) or getattr(
                getattr(layer, "channel_weights", None), "dim", 0
            )
            return (dim, dim)
        if ltype == "MultiHeadCrossAttention":
            dim = getattr(getattr(layer, "query_vis", None), "out_features", 0) or getattr(
                getattr(layer, "fc_out_vis", None), "out_features", 0
            )
            return (dim, dim)
        raise ValueError(f"Unsupported multi-output producer '{ltype}' without explicit slot metadata")


    c = 0
    if ltype == "Conv":
        c = layer.conv.out_channels
    elif ltype in ("C3k2", "C2f", "C2PSA", "SPPF", "SPP", "C3", "GhostConv", "C2fAttn", "A2C2f", "SCDown"):
        c = getattr(getattr(layer, "cv2", None), "conv", None)
        c = c.out_channels if c is not None else 0
    elif ltype == "BottleneckCSP":
        c = getattr(getattr(layer, "cv4", None), "conv", None)
        c = c.out_channels if c is not None else 0
    elif ltype == "ADown":
        cv1 = getattr(getattr(layer, "cv1", None), "conv", None)
        cv2 = getattr(getattr(layer, "cv2", None), "conv", None)
        c = (cv1.out_channels if cv1 else 0) + (cv2.out_channels if cv2 else 0)
    elif ltype == "SPPELAN":
        c = getattr(getattr(layer, "cv5", None), "conv", None)
        c = c.out_channels if c is not None else 0
    elif ltype == "AConv":
        cv = getattr(getattr(layer, "cv1", None), "conv", None)
        c = cv.out_channels if cv else 0
    elif ltype == "FeatureFusion":
        channel_emb = getattr(layer, "channel_emb", None)
        c = getattr(channel_emb, "out_channels", 0) if channel_emb is not None else 0
    elif ltype == "FCMFeatureFusion":
        ffm = getattr(layer, "ffm", None)
        if ffm is not None:
            return _get_out_channels_tuple(ffm, "FeatureFusion")
        c = getattr(layer, "dim", 0) or 0
    elif ltype == "MCFGatedFusion":
        post = getattr(layer, "post", None)
        if post is not None and hasattr(post, "conv"):
            c = post.conv.out_channels
        else:
            gate = getattr(layer, "gate", None)
            c = gate.out_channels if gate is not None else 0
    elif ltype == "CrossTransformerFusion":
        model_dim = getattr(layer, "model_dim", 0)
        c = model_dim * 2 if model_dim else 0
    elif ltype == "SequenceShuffleAttention":

        gating = getattr(layer, "gating", None)
        if gating is not None:
            for l in gating:
                if isinstance(l, nn.Conv2d):
                    c = l.out_channels
                    break
        else:
            c = getattr(layer, "_c", 0) or 0
    elif ltype in ("Concat", "Upsample"):
        c = 0
    elif ltype == "Detect":
        c = 0
    elif isinstance(layer, nn.Upsample):
        c = 0
    elif hasattr(layer, "in_channels"):

        c = getattr(layer, "in_channels", 0) or 0

    return (c,)

def _is_multi_output(layer: nn.Module, ltype: str) -> bool:

    return ltype in _MULTI_OUTPUT_TYPES

def _is_route_only(ltype: str) -> bool:

    return ltype in _ROUTE_ONLY_TYPES

def _is_head(ltype: str) -> bool:

    return ltype in _HEAD_TYPES





def build_prune_graph(model: nn.Module) -> PruneGraph:














    layers_module = getattr(model, "model", model)
    layers = list(layers_module)


    nodes: list[PruneNode] = []
    entry_indices: set[int] = set()

    for i, layer in enumerate(layers):
        ltype = _get_layer_type_name(layer)
        in_ch = _get_in_channels(layer, ltype)
        out_ch = _get_out_channels_tuple(layer, ltype)

        is_head = _is_head(ltype)
        is_route_only = _is_route_only(ltype)
        is_entry = False


        mm_source = getattr(layer, "_mm_input_source", None)
        if mm_source in ("RGB", "X", "Dual"):
            is_entry = True
            entry_indices.add(i)


        if i == 0:
            is_entry = True
            entry_indices.add(i)

        node = PruneNode(
            idx=i,
            module=layer,
            type_name=ltype,
            input_edges=(),
            in_channels=in_ch,
            out_channels=out_ch,
            branch_kind="unknown",
            is_entry=is_entry,
            is_head=is_head,
            is_multi_input=False,
            is_route_only=is_route_only,
        )
        nodes.append(node)


    multi_output_producers: dict[int, int] = {}

    for i, node in enumerate(nodes):
        layer = node.module
        ltype = node.type_name
        f = getattr(layer, "f", None)




        if node.is_entry and getattr(layer, "_mm_new_input_start", False):
            from_list = []

        elif f is None:
            from_list: list = []
        elif isinstance(f, (list, tuple)):
            from_list = list(f)
        else:
            from_list = [f]


        edges: list[EdgeRef] = []
        for rel in from_list:
            abs_idx = rel if rel >= 0 else i + rel
            if 0 <= abs_idx < len(nodes):
                producer_node = nodes[abs_idx]
                slot = 0
                if ltype == "Index":
                    slot = int(getattr(layer, "index", 0))
                    if slot < 0 or slot >= len(producer_node.out_channels):
                        raise ValueError(
                            f"Index(layer={i}) selects output_slot={slot}, "
                            f"but producer layer {abs_idx} only has {len(producer_node.out_channels)} slot(s)"
                        )
                edges.append(EdgeRef(node_idx=abs_idx, output_slot=slot))

        is_multi_input = len(edges) > 1


        parents = [nodes[e.node_idx] for e in edges if e.node_idx < i]
        kind = infer_branch_kind(layer, parents)


        node.input_edges = tuple(edges)
        node.is_multi_input = is_multi_input
        node.branch_kind = kind


        if _is_multi_output(ltype, ltype):
            multi_output_producers[i] = len(node.out_channels)


        if ltype == "Concat":
            total = 0
            for e in edges:
                prod = nodes[e.node_idx]
                total += prod.out_channels[e.output_slot]
            node.out_channels = (total,)


        elif ltype == "Upsample":
            if edges:
                prod = nodes[edges[0].node_idx]
                node.out_channels = (prod.out_channels[edges[0].output_slot],)
        elif ltype == "Index":
            if edges:
                prod = nodes[edges[0].node_idx]
                slot = edges[0].output_slot
                node.out_channels = (prod.out_channels[slot],)

    return PruneGraph(nodes=nodes)

