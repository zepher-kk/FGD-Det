from __future__ import annotations
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Complexity analysis data structures for YOLOMM pruning stage.

This module defines the input specification, tensor shape, node complexity,
and report objects used by the graph-driven complexity engine. The design
ensures that complexity calculations start from explicit input semantics
rather than relying on implicit dummy input inference.
"""

from dataclasses import dataclass, field
from typing import Literal

RouteMode = Literal["dual", "rgb", "x"]
StageKind = Literal["rgb_branch", "x_branch", "fusion", "head", "route_only"]

@dataclass(frozen=True)
class ComplexityInputSpec:












    imgsz: tuple[int, int]
    route_mode: RouteMode
    rgb_channels: int
    x_channels: int

@dataclass(frozen=True)
class TensorShapeSpec:










    channels: int
    height: int
    width: int

@dataclass
class NodeComplexity:










    node_idx: int
    type_name: str
    stage: StageKind
    input_shapes: tuple[TensorShapeSpec, ...]
    output_shapes: tuple[TensorShapeSpec, ...]
    flops: float

@dataclass
class ComplexityReport:






    input_spec: ComplexityInputSpec
    nodes: list[NodeComplexity] = field(default_factory=list)

    @property
    def total_flops(self) -> float:

        return sum(node.flops for node in self.nodes)

    def stage_flops(self) -> dict[StageKind, float]:

        totals: dict[StageKind, float] = {}
        for node in self.nodes:
            totals[node.stage] = totals.get(node.stage, 0.0) + node.flops
        return totals

