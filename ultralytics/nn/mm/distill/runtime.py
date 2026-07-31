from __future__ import annotations



"""
Teacher runtime coordinator for knowledge distillation.

Responsibilities:
1. Build teacher models according to the student's model family.
2. Freeze teacher parameters (eval + requires_grad_(False)).
3. Register temporary collection hooks for feature-level distillation.
4. Run teacher forward passes and collect outputs/features during training.

The coordinator lives exclusively on the *trainer* -- it is never attached to
the student model object.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from pathlib import Path

from ultralytics.utils import LOGGER

from .schema import DistillConfig, FeatureMappingSpec, OutputTeacherSpec, TeacherSpec




_FEATURE_MAPPING_WARN_LIMIT = 20
_FEATURE_MAPPING_HARD_LIMIT = 50



_INPUT_SOURCE_MAP = {"RGB": "RGB", "X": "X", "DUAL": "Dual"}





class _FeatureCollector:


    def __init__(self):
        self.features: Dict[int, torch.Tensor] = {}
        self._handles: list = []

    def register(self, model: nn.Module, layer_indices: List[int]):






        self.clear()
        for idx in layer_indices:
            if idx < 0 or idx >= len(model.model):
                raise ValueError(
                    f"Layer index {idx} out of range for model with "
                    f"{len(model.model)} layers"
                )
            handle = model.model[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(_module, _input, output):
            self.features[layer_idx] = output
        return hook

    def reset(self):

        self.features.clear()

    def clear(self):

        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.features.clear()





class _TeacherRuntime:


    def __init__(self, spec: TeacherSpec, model: nn.Module, family: str):
        self.spec = spec
        self.model = model
        self.family = family
        self.collector = _FeatureCollector()

    def register_feature_hooks(self, layer_indices: List[int]):

        self.collector.register(self.model, layer_indices)

    @staticmethod
    def _build_teacher_batch(batch: dict, role: str) -> dict:














        img = batch["img"]
        out = dict(batch)

        if role == "rgb":
            out["img"] = img[:, :3]
        elif role == "x":
            out["img"] = img[:, 3:]
        elif role == "dual":
            out["img"] = img
        else:
            raise ValueError(
                f"Unsupported teacher role: '{role}'. "
                f"Must be one of 'rgb', 'x', 'dual'."
            )
        return out

    @staticmethod
    def _set_teacher_router(model: nn.Module, role: str):
















        router = getattr(model, 'mm_router', None)
        if router is None:
            return

        if role == "rgb":
            router.set_runtime_params("rgb")
        elif role == "x":
            router.set_runtime_params("x")
        elif role == "dual":
            router.set_runtime_params(None)

    @staticmethod
    def _check_input_channels(teacher_img: torch.Tensor, spec: TeacherSpec):
























        actual_ch = teacher_img.shape[1]

        if spec.role == "dual" and actual_ch <= 3:
            raise ValueError(
                f"Teacher '{spec.name}' (role=dual) expects a multi-channel input "
                f"but received {actual_ch}-ch tensor. The training batch may not "
                f"contain concatenated RGB+X data."
            )
        if spec.role == "rgb" and actual_ch != 3:
            raise ValueError(
                f"Teacher '{spec.name}' (role=rgb) expects 3-channel RGB input "
                f"but received {actual_ch}-ch tensor. Check that batch['img'] has "
                f"the expected [RGB|X] channel layout."
            )

    def forward(self, batch: dict) -> torch.Tensor:





















        self.collector.reset()


        teacher_batch = self._build_teacher_batch(batch, self.spec.role)
        teacher_img = teacher_batch["img"]


        self._check_input_channels(teacher_img, self.spec)


        self._set_teacher_router(self.model, self.spec.role)


        with torch.no_grad():
            preds = self.model.predict(teacher_img)


            if self.family == "rtdetrmm":


                if isinstance(preds, (tuple, list)) and len(preds) == 2:
                    preds = preds[1]
        return preds

    def get_features(self) -> Dict[int, torch.Tensor]:

        return dict(self.collector.features)

    def cleanup(self):

        self.collector.clear()





class DistillRuntime:










    def __init__(
        self,
        config: DistillConfig,
        mode: str,
        family: str,
        student_model: nn.Module,
        device: torch.device,
    ):








        self.config = config
        self.mode = mode
        self.family = family
        self.device = device


        self.use_feature = mode in ("feature", "both")
        self.use_output = mode in ("output", "both")


        self.feature_mappings: List[FeatureMappingSpec] = (
            self._compile_feature_mappings(
                raw_mappings=list(config.feature_mappings) if self.use_feature else [],
                student_model=student_model,
            )
        )
        self.output_teachers: List[OutputTeacherSpec] = (
            list(config.output_teachers) if self.use_output else []
        )


        self._teachers: Dict[str, _TeacherRuntime] = {}
        self._build_teachers(config, student_model)


        if self.use_feature:
            self._register_feature_hooks()

        LOGGER.info(
            f"DistillRuntime initialized: family={family}, mode={mode}, "
            f"teachers={list(self._teachers.keys())}, "
            f"feature_mappings={len(self.feature_mappings)}, "
            f"output_teachers={len(self.output_teachers)}"
        )



    def _compile_feature_mappings(
        self,
        raw_mappings: List[FeatureMappingSpec],
        student_model: nn.Module,
    ) -> List[FeatureMappingSpec]:


















        if not raw_mappings:
            return []

        compiled: List[FeatureMappingSpec] = []

        for i, raw in enumerate(raw_mappings):
            t_layer = raw.teacher_layer
            s_layer = raw.student_layer
            s_input = raw.student_input


            if s_layer is not None:
                if isinstance(t_layer, int) and isinstance(s_layer, int):

                    compiled.append(raw)

                elif isinstance(t_layer, tuple) and isinstance(s_layer, tuple):

                    t_range = list(range(t_layer[0], t_layer[1] + 1))
                    s_range = list(range(s_layer[0], s_layer[1] + 1))
                    if len(t_range) != len(s_range):
                        raise ValueError(
                            f"Feature mapping [{i}]: teacher range [{t_layer[0]}, {t_layer[1]}] "
                            f"has {len(t_range)} layers but student range [{s_layer[0]}, {s_layer[1]}] "
                            f"has {len(s_range)} layers -- lengths must match."
                        )
                    for tl, sl in zip(t_range, s_range):
                        compiled.append(FeatureMappingSpec(
                            teacher=raw.teacher,
                            teacher_layer=tl,
                            student_layer=sl,
                            student_input=None,
                            tap=raw.tap,
                        ))

                else:

                    raise ValueError(
                        f"Feature mapping [{i}]: teacher_layer and student_layer must be "
                        f"the same type (both int or both range), got "
                        f"teacher_layer={t_layer!r}, student_layer={s_layer!r}."
                    )

            elif s_input is not None:

                if isinstance(t_layer, int):

                    raise ValueError(
                        f"Feature mapping [{i}]: atomic teacher_layer={t_layer} cannot "
                        f"be used with student_input='{s_input}'. Use a range "
                        f"teacher_layer=[start, end] for auto-expansion."
                    )


                t_range = list(range(t_layer[0], t_layer[1] + 1))
                branch_layers = self._find_student_branch_layers(student_model, s_input)
                if len(t_range) != len(branch_layers):
                    raise ValueError(
                        f"Feature mapping [{i}]: teacher range [{t_layer[0]}, {t_layer[1]}] "
                        f"has {len(t_range)} layers but student '{s_input}' branch has "
                        f"{len(branch_layers)} layers ({branch_layers}) -- lengths must match."
                    )
                for tl, sl in zip(t_range, branch_layers):
                    compiled.append(FeatureMappingSpec(
                        teacher=raw.teacher,
                        teacher_layer=tl,
                        student_layer=sl,
                        student_input=None,
                        tap=raw.tap,
                    ))

            else:

                raise ValueError(
                    f"Feature mapping [{i}]: both student_layer and student_input are None. "
                    f"At least one must be provided."
                )


        total = len(compiled)
        if total > _FEATURE_MAPPING_HARD_LIMIT:
            raise ValueError(
                f"Feature mappings expanded to {total} atomic pairs, "
                f"exceeding hard limit {_FEATURE_MAPPING_HARD_LIMIT}. "
                f"Reduce teacher_layer range or use fewer mappings."
            )
        if total > _FEATURE_MAPPING_WARN_LIMIT:
            LOGGER.warning(
                f"Feature mappings expanded to {total} atomic pairs "
                f"(warning threshold={_FEATURE_MAPPING_WARN_LIMIT}). "
                f"This may significantly slow down training."
            )

        return compiled

    def _find_student_branch_layers(
        self, student_model: nn.Module, student_input: str
    ) -> list[int]:


















        mapped_source = _INPUT_SOURCE_MAP.get(student_input)
        if mapped_source is None:
            raise ValueError(
                f"Unknown student_input '{student_input}'. "
                f"Valid values: {list(_INPUT_SOURCE_MAP.keys())}."
            )


        start_idx = None
        for idx, m in enumerate(student_model.model):
            source = getattr(m, "_mm_input_source", None)
            if source == mapped_source:
                start_idx = idx
                break

        if start_idx is None:
            raise ValueError(
                f"No layer with _mm_input_source='{mapped_source}' found in student model. "
                f"student_input='{student_input}' branch does not exist in this model configuration."
            )


        branch_layers: list[int] = []
        for idx in range(start_idx, len(student_model.model)):
            m = student_model.model[idx]
            source = getattr(m, "_mm_input_source", None)

            if idx == start_idx:

                branch_layers.append(idx)
                continue


            if isinstance(m.f, list):
                break


            if source is not None and source != mapped_source:
                break



            if source == mapped_source:
                branch_layers.append(idx)
            elif m.f == -1:
                branch_layers.append(idx)
            else:

                break

        if not branch_layers:
            raise ValueError(
                f"student_input='{student_input}' branch starting at layer {start_idx} "
                f"yielded no contiguous layers."
            )

        return branch_layers



    def _build_teachers(self, config: DistillConfig, student_model: nn.Module):

        for spec in config.teachers:
            model = self._load_teacher_model(spec, student_model)
            model.eval()
            model.requires_grad_(False)
            model.to(self.device)
            self._teachers[spec.name] = _TeacherRuntime(spec, model, self.family)
            LOGGER.info(
                f"Teacher '{spec.name}' (role={spec.role}) loaded, frozen, "
                f"moved to {self.device}"
            )

    def _load_teacher_model(self, spec: TeacherSpec, student_model: nn.Module) -> nn.Module:













        weights_path = Path(spec.weights)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Teacher '{spec.name}' weights not found: {spec.weights}"
            )


        from ultralytics.nn.tasks import attempt_load_one_weight

        teacher_model, _ = attempt_load_one_weight(str(weights_path))


        if self.family == "yolomm":
            from ultralytics.nn.tasks import DetectionModel
            if not isinstance(teacher_model, DetectionModel):
                raise TypeError(
                    f"Teacher '{spec.name}' is not a DetectionModel; "
                    f"YOLOMM only supports same-family distillation"
                )
        elif self.family == "rtdetrmm":
            from ultralytics.nn.tasks import RTDETRDetectionModel
            if not isinstance(teacher_model, RTDETRDetectionModel):
                raise TypeError(
                    f"Teacher '{spec.name}' is not a RTDETRDetectionModel; "
                    f"RTDETRMM only supports same-family distillation"
                )
        else:
            raise ValueError(f"Unknown family: {self.family}")

        return teacher_model



    def _register_feature_hooks(self):


        teacher_layers: Dict[str, List[int]] = {}
        for m in self.feature_mappings:
            teacher_layers.setdefault(m.teacher, []).append(m.teacher_layer)

        for tname, layers in teacher_layers.items():
            unique_layers = sorted(set(layers))
            self._teachers[tname].register_feature_hooks(unique_layers)
            LOGGER.info(
                f"Teacher '{tname}': registered feature hooks at layers {unique_layers}"
            )



    def register_student_hooks(self, student_model: nn.Module) -> _FeatureCollector:





        collector = _FeatureCollector()
        if self.use_feature:
            student_layers = sorted(set(m.student_layer for m in self.feature_mappings))
            collector.register(student_model, student_layers)
            LOGGER.info(
                f"Student: registered feature hooks at layers {student_layers}"
            )
        return collector



    def run_teachers(self, batch: dict) -> Dict[str, "TeacherOutput"]:








        results: Dict[str, TeacherOutput] = {}
        for tname, trt in self._teachers.items():
            preds = trt.forward(batch)
            features = trt.get_features() if self.use_feature else {}
            results[tname] = TeacherOutput(
                name=tname,
                role=trt.spec.role,
                preds=preds,
                features=features,
            )
        return results



    def cleanup(self):

        for trt in self._teachers.values():
            trt.cleanup()
        self._teachers.clear()
        LOGGER.info("DistillRuntime cleaned up")

    @property
    def teacher_names(self) -> List[str]:
        return list(self._teachers.keys())





class TeacherOutput:


    __slots__ = ("name", "role", "preds", "features")

    def __init__(
        self,
        name: str,
        role: str,
        preds: torch.Tensor,
        features: Dict[int, torch.Tensor],
    ):
        self.name = name
        self.role = role
        self.preds = preds
        self.features = features

