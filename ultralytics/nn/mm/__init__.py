from __future__ import annotations




"""
Ultralytics Multimodal Router Module

This module provides a comprehensive multimodal routing system for RGB+X architectures,
supporting YOLO and RTDETR with zero-copy tensor routing and configuration-driven data flow.

Core Components:
- MultiModalRouter: Universal RGB+X multimodal data router
- MultiModalConfigParser: YAML configuration parsing for multimodal architectures
- Utility functions: System status, model validation, and configuration helpers

Supported Modalities:
- RGB: 3-channel visible light images
- X: 3-channel unified other modality (depth/thermal/lidar/etc.)
- Dual: 6-channel RGB+X concatenated input

Features:
- Zero-copy tensor view routing
- Configuration-driven data flow
- Thread-safe caching mechanisms
- X modality new input start redirection
- Universal framework for RGB+X multimodal detection
"""


from .router import MultiModalRouter


from .parser import MultiModalConfigParser


from .utils import (
    validate_mm_config_format,
    mm_system_status,
    check_mm_model_attributes,
    get_mm_system_info
)
from .generators import DepthGen, DEMGen, EdgeGen


from .source_matcher import MultiModalSourceMatcher


from .distill import (
    DistillConfig,
    TeacherSpec,
    FeatureMappingSpec,
    MappingSpec,
    OutputTeacherSpec,
    load_distill_config,
)


__version__ = "v1.0"
PROJECT_VERSION = "v0.1212"


__all__ = [

    "MultiModalRouter",
    "MultiModalConfigParser",


    "validate_mm_config_format",
    "mm_system_status",
    "check_mm_model_attributes",
    "get_mm_system_info",

    "DepthGen",
    "DEMGen",
    "EdgeGen",


    "MultiModalSourceMatcher",


    "DistillConfig",
    "TeacherSpec",
    "FeatureMappingSpec",
    "MappingSpec",
    "OutputTeacherSpec",
    "load_distill_config",


    "__version__",
    "PROJECT_VERSION",
]


__author__ = "YOLOMM Team"
__description__ = "Universal RGB+X Multimodal Routing System"
__supported_modalities__ = ["RGB", "X", "Dual"]
__supported_architectures__ = ["YOLO", "RTDETR"]

