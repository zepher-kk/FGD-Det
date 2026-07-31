from __future__ import annotations



"""
Family-level distillation adapters for YOLOMM and RTDETRMM.

Each adapter:
1. Standardises the student ``distill_forward()`` result into a student bundle.
2. Standardises teacher outputs + features into teacher bundles.
3. Delegates to the **family-specific** output distillation module for loss
   computation (NOT a shared flatten-MSE).
4. Delegates to the **family-specific** feature guidance/mask module for
   foreground-guided feature distillation.
5. Aggregates the final distillation loss contribution.

Architecture:
- ``_BaseDistillAdapter``: defines the ``compute_distill_loss`` API, handles
  feature distillation orchestration (guidance cache, mask generation, shared
  4-term loss via ``losses.py``).  Both ``_compute_output_loss`` and the
  family-specific ``_build_feature_guidance`` / ``_build_feature_mask`` are
  abstract -- subclasses MUST override.
- ``YOLOMMDetectDistillAdapter``: delegates output distillation to
  ``output_yolomm.compute_yolomm_output_kd``; delegates feature guidance to
  ``feature_yolomm.build_yolomm_feature_guidance`` /
  ``feature_yolomm.build_yolomm_feature_mask``.
- ``RTDETRMMDetectDistillAdapter``: delegates output distillation to
  ``output_rtdetrmm.compute_rtdetr_output_kd``; delegates feature guidance to
  ``feature_rtdetrmm.build_rtdetr_feature_guidance`` /
  ``feature_rtdetrmm.build_rtdetr_feature_mask``.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER

from .losses import compute_feature_distill_loss
from .runtime import TeacherOutput, DistillRuntime, _FeatureCollector
from .schema import DistillConfig, FeatureMappingSpec, OutputTeacherSpec

class ChannelAdapter(nn.Module):











    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.conv(x)

class _BaseDistillAdapter:












    def __init__(
        self,
        runtime: DistillRuntime,
        config: DistillConfig,
        student_model: nn.Module = None,
        distill_weight: float = 1.0,
        feature_weight: float = 1.0,
        output_weight: float = 1.0,
        trainer=None,
    ):
        self.runtime = runtime
        self.config = config
        self.student_model = student_model
        self.distill_weight = distill_weight
        self.feature_weight = feature_weight
        self.output_weight = output_weight
        self._trainer = trainer


        self._current_epoch: int = 0
        self._total_epochs: int = 100


        self._channel_adapters: Dict[str, ChannelAdapter] = {}

    def set_epoch_state(self, current_epoch: int, total_epochs: int):

        self._current_epoch = current_epoch
        self._total_epochs = total_epochs



    def _get_or_create_adapter(
        self,
        mapping_key: str,
        teacher_ch: int,
        student_ch: int,
        device: torch.device,
    ) -> Optional[ChannelAdapter]:














        if teacher_ch == student_ch:
            return None

        if mapping_key not in self._channel_adapters:
            adapter = ChannelAdapter(teacher_ch, student_ch).to(device)
            self._channel_adapters[mapping_key] = adapter
            LOGGER.info(
                f"Channel adapter created for {mapping_key}: "
                f"{teacher_ch} -> {student_ch} channels"
            )
            self._append_adapter_to_optimizer(adapter)

        return self._channel_adapters[mapping_key]

    def _append_adapter_to_optimizer(self, adapter: ChannelAdapter):





        if self._trainer is None or not hasattr(self._trainer, 'optimizer'):
            LOGGER.warning(
                "Channel adapter created but no trainer/optimizer available "
                "to register parameters. Adapter will NOT be optimised."
            )
            return
        optimizer = self._trainer.optimizer
        if optimizer is None:
            LOGGER.warning(
                "Channel adapter created but optimizer is None. "
                "Adapter will NOT be optimised."
            )
            return
        base_lr = optimizer.param_groups[0]['lr']
        optimizer.add_param_group({
            'params': list(adapter.parameters()),
            'lr': base_lr,
            'weight_decay': 0.0,
        })
        LOGGER.info(
            f"Channel adapter params appended to optimizer "
            f"(lr={base_lr:.6f}, weight_decay=0.0)"
        )

    def compute_distill_loss(
        self,
        student_preds,
        student_features: Dict[int, torch.Tensor],
        teacher_outputs: Dict[str, TeacherOutput],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:











        device = _get_device(student_preds)
        total_loss = torch.tensor(0.0, device=device)
        items: Dict[str, torch.Tensor] = {}


        if self.runtime.use_output:
            out_loss, out_items = self._compute_output_loss(
                student_preds, teacher_outputs, device
            )
            total_loss = total_loss + self.output_weight * out_loss

            items["d_out"] = out_loss.detach()

            items.update(out_items)


        if self.runtime.use_feature:
            feat_loss, feat_items = self._compute_feature_loss(
                student_features, teacher_outputs, device
            )
            total_loss = total_loss + self.feature_weight * feat_loss
            items["distill_feature"] = feat_loss.detach()
            items.update(feat_items)

        total_loss = total_loss * self.distill_weight
        return total_loss, items



    def _compute_output_loss(
        self,
        student_preds,
        teacher_outputs: Dict[str, TeacherOutput],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:









        raise NotImplementedError(
            f"{self.__class__.__name__} must override _compute_output_loss(). "
            f"Generic flatten-MSE has been retired from the output distillation "
            f"main path."
        )



    def _build_feature_guidance(
        self,
        teacher_name: str,
        teacher_output: TeacherOutput,
    ) -> object:













        raise NotImplementedError(
            f"{self.__class__.__name__} must override _build_feature_guidance()."
        )

    def _build_feature_mask(
        self,
        guidance,
        feature_shape: tuple,
    ) -> torch.Tensor:












        raise NotImplementedError(
            f"{self.__class__.__name__} must override _build_feature_mask()."
        )

    def _get_input_size(self) -> Optional[tuple]:




        return getattr(self.runtime, '_input_size', None)

    def _build_feature_guidance_cache(
        self,
        teacher_outputs: Dict[str, TeacherOutput],
    ) -> Dict[str, object]:











        cache: Dict[str, object] = {}
        for mapping in self.runtime.feature_mappings:
            t_name = mapping.teacher
            if t_name in cache:
                continue
            t_out = teacher_outputs.get(t_name)
            if t_out is None:
                continue
            cache[t_name] = self._build_feature_guidance(t_name, t_out)
        return cache

    def _compute_feature_loss(
        self,
        student_features: Dict[int, torch.Tensor],
        teacher_outputs: Dict[str, TeacherOutput],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:












        guidance_cache = self._build_feature_guidance_cache(teacher_outputs)


        unique_teachers = set(m.teacher for m in self.runtime.feature_mappings)
        multi_teacher = len(unique_teachers) > 1

        losses: List[torch.Tensor] = []
        agg_items: Dict[str, List[torch.Tensor]] = {}


        for mapping in self.runtime.feature_mappings:
            t_out = teacher_outputs.get(mapping.teacher)
            if t_out is None:
                continue
            t_feat = t_out.features.get(mapping.teacher_layer)
            s_feat = student_features.get(mapping.student_layer)
            if t_feat is None or s_feat is None:
                LOGGER.warning(
                    f"Missing feature for mapping {mapping.teacher}:"
                    f"T[{mapping.teacher_layer}]->S[{mapping.student_layer}], skipped"
                )
                continue



            if t_feat.dim() == 4 and s_feat.dim() == 4:
                mapping_key = (
                    f"{mapping.teacher}_T{mapping.teacher_layer}"
                    f"_S{mapping.student_layer}"
                )
                adapter = self._get_or_create_adapter(
                    mapping_key, t_feat.shape[1], s_feat.shape[1], t_feat.device,
                )
                if adapter is not None:
                    t_feat = adapter(t_feat.detach())


            guidance = guidance_cache.get(mapping.teacher)
            if guidance is None:
                LOGGER.warning(
                    f"No guidance available for teacher {mapping.teacher}, "
                    f"feature mapping skipped"
                )
                continue
            fg_mask = self._build_feature_mask(guidance, t_feat.shape)


            loss, sub_items = compute_feature_distill_loss(s_feat, t_feat, fg_mask)
            losses.append(loss)



            prefix = f"{mapping.teacher}/" if multi_teacher else ""
            for k, v in sub_items.items():
                if k == "total":
                    continue
                key = f"{prefix}distill_feature_{k}"
                agg_items.setdefault(key, []).append(v)


        if not losses:
            return torch.tensor(0.0, device=device), {}

        mean_loss = sum(losses) / len(losses)


        merged: Dict[str, torch.Tensor] = {}
        for k, v_list in agg_items.items():
            merged[k] = sum(v_list) / len(v_list)

        return mean_loss, merged





class YOLOMMDetectDistillAdapter(_BaseDistillAdapter):










    def _build_feature_guidance(self, teacher_name, teacher_output):

        from .feature_yolomm import build_yolomm_feature_guidance

        teacher_model = self.runtime._teachers[teacher_name].model
        return build_yolomm_feature_guidance(
            teacher_preds=teacher_output.preds,
            teacher_model=teacher_model,
        )

    def _build_feature_mask(self, guidance, feature_shape):

        from .feature_yolomm import build_yolomm_feature_mask

        return build_yolomm_feature_mask(
            guidance=guidance,
            feature_shape=feature_shape,
            input_size=self._get_input_size(),
        )

    def _compute_output_loss(
        self,
        student_preds,
        teacher_outputs: Dict[str, TeacherOutput],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        from .output_yolomm import compute_yolomm_output_kd

        all_losses: List[torch.Tensor] = []
        merged_items: Dict[str, torch.Tensor] = {}

        for ot_spec in self.runtime.output_teachers:
            t_out = teacher_outputs.get(ot_spec.teacher)
            if t_out is None:
                continue


            teacher_model = self.runtime._teachers[ot_spec.teacher].model

            loss, items = compute_yolomm_output_kd(
                student_preds=student_preds,
                teacher_preds=t_out.preds,
                student_model=self.student_model,
                teacher_model=teacher_model,
                current_epoch=self._current_epoch,
                total_epochs=self._total_epochs,
            )
            all_losses.append(loss)


            n_teachers = len(self.runtime.output_teachers)
            prefix = f"{ot_spec.teacher}/" if n_teachers > 1 else ""
            for k, v in items.items():
                merged_items[f"{prefix}{k}"] = v

        if not all_losses:
            return torch.tensor(0.0, device=device), {}
        return sum(all_losses) / len(all_losses), merged_items





class RTDETRMMDetectDistillAdapter(_BaseDistillAdapter):










    def _build_feature_guidance(self, teacher_name, teacher_output):

        from .feature_rtdetrmm import build_rtdetr_feature_guidance

        return build_rtdetr_feature_guidance(
            teacher_preds=teacher_output.preds,
            input_size=self._get_input_size(),
        )

    def _build_feature_mask(self, guidance, feature_shape):

        from .feature_rtdetrmm import build_rtdetr_feature_mask

        return build_rtdetr_feature_mask(
            guidance=guidance,
            feature_shape=feature_shape,
            input_size=self._get_input_size(),
        )

    def _compute_output_loss(
        self,
        student_preds,
        teacher_outputs: Dict[str, TeacherOutput],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        from .output_rtdetrmm import compute_rtdetr_output_kd

        all_losses: List[torch.Tensor] = []
        merged_items: Dict[str, torch.Tensor] = {}

        for ot_spec in self.runtime.output_teachers:
            t_out = teacher_outputs.get(ot_spec.teacher)
            if t_out is None:
                continue

            loss, items = compute_rtdetr_output_kd(
                student_preds=student_preds,
                teacher_preds=t_out.preds,
                current_epoch=self._current_epoch,
                total_epochs=self._total_epochs,
            )
            all_losses.append(loss)


            n_teachers = len(self.runtime.output_teachers)
            prefix = f"{ot_spec.teacher}/" if n_teachers > 1 else ""
            for k, v in items.items():
                merged_items[f"{prefix}{k}"] = v

        if not all_losses:
            return torch.tensor(0.0, device=device), {}
        return sum(all_losses) / len(all_losses), merged_items





def _get_device(preds) -> torch.device:

    if isinstance(preds, torch.Tensor):
        return preds.device
    if isinstance(preds, (tuple, list)):
        for p in preds:
            if isinstance(p, torch.Tensor):
                return p.device
    return torch.device("cpu")

