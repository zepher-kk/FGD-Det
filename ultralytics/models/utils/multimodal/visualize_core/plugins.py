from __future__ import annotations
"""Visualization method plugin stubs (to be implemented in later steps)."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import cv2
import torch

from .types import CoreVisualizationResult
from .registry import REGISTRY
from .preprocessor import Preprocessor
from .renderer import Renderer

class MethodPlugin:


    @staticmethod
    def run(
        *,
        model: Any,
        inputs: Dict[str, Any],
        layers: List[int],
        layer_names: List[str],
        save: bool,
        out_dir: Path,
        modality: str | None,
        family: str,
        **kwargs: Any,
    ) -> List[CoreVisualizationResult]:
        raise NotImplementedError

class HeatmapPlugin(MethodPlugin):
    @staticmethod
    def run(
        *,
        model: Any,
        inputs: Dict[str, Any],
        layers: List[int],
        layer_names: List[str],
        save: bool,
        out_dir: Path,
        modality: str | None,
        family: str,
        **kwargs: Any,
    ) -> List[CoreVisualizationResult]:

        from ultralytics.models.yolo.multimodal.visualize.heatmap import HeatmapVisualizer


        originals: Dict[str, np.ndarray] | np.ndarray
        if 'rgb' in inputs and 'x' in inputs:
            originals = {'rgb': inputs['rgb'], 'x': inputs['x']}
            inferred_modality = 'dual'
        elif 'rgb' in inputs:
            originals = inputs['rgb']
            inferred_modality = 'rgb'
        else:
            originals = inputs['x']
            inferred_modality = 'x'


        align_base = str(kwargs.get('align_base', 'rgb')).lower()
        ablation_fill = str(kwargs.get('ablation_fill', 'zeros')).lower()

        size_arg = kwargs.get('imgsz', kwargs.get('vis_imgsz', None))
        if size_arg is None:

            base_size = Preprocessor.model_input_size(model)
            auto_imgsz = bool(kwargs.get('auto_imgsz', True))
            if auto_imgsz:
                try:
                    cap = int(kwargs.get('imgsz_cap', 1280))
                except Exception:
                    raise ValueError(f"imgsz_cap 需要为整数，收到: {kwargs.get('imgsz_cap')!r}")
                if cap <= 0:
                    raise ValueError(f"imgsz_cap 必须为正整数，收到: {cap}")

                def _max_side(x: Any) -> int | None:
                    if isinstance(x, dict):
                        sides = []
                        for v in x.values():
                            s = _max_side(v)
                            if s is not None:
                                sides.append(s)
                        return max(sides) if sides else None
                    if not isinstance(x, np.ndarray) or x.size == 0:
                        return None
                    if x.ndim == 2:
                        return int(max(x.shape[0], x.shape[1]))
                    if x.ndim == 3:
                        return int(max(x.shape[0], x.shape[1]))
                    if x.ndim == 4:

                        if x.shape[-1] in (1, 3, 4, 6):
                            return int(max(x.shape[1], x.shape[2]))
                        return int(max(x.shape[2], x.shape[3]))
                    return None

                ms = _max_side(originals)
                if ms is not None and ms > base_size:
                    size = max(base_size, min(ms, cap))
                else:
                    size = base_size
            else:
                size = base_size
        else:
            try:
                size = int(size_arg)
            except Exception:
                raise ValueError(f"imgsz/vis_imgsz 需要为整数（例如 640/1280），收到: {size_arg!r}")
            if size <= 0:
                raise ValueError(f"imgsz/vis_imgsz 必须为正整数，收到: {size}")
        total_ch = Preprocessor.model_input_channels(model, default=6)
        x_expect = max(total_ch - 3, 0)
        fill_val = 0.0 if ablation_fill == 'zeros' else 0.5

        has_rgb = 'rgb' in inputs
        has_x = 'x' in inputs

        if has_rgb and has_x:
            hwc = Preprocessor.letterbox_dual_aligned(inputs['rgb'], inputs['x'], size=size, align_base=align_base)
            if hwc.ndim != 3 or hwc.shape[2] < 4:
                raise ValueError(f"预处理后的形状异常：{hwc.shape}，期望通道数 ≥4（RGB3 + Xch≥1）")
            if hwc.shape[2] != total_ch:
                raise ValueError(
                    f"输入通道数与模型不一致：预处理后={hwc.shape[2]}，模型期望(Dual)={total_ch}。"
                    f" 请检查数据 Xch / 模型配置 / modality 路由设定。"
                )
        elif has_rgb and not has_x:
            rgb_hwc = Preprocessor.letterbox_single(inputs['rgb'], size)
            if rgb_hwc.ndim == 2:
                rgb_hwc = rgb_hwc[:, :, None]
            if rgb_hwc.shape[2] == 1:
                rgb_hwc = np.repeat(rgb_hwc, 3, axis=2)
            elif rgb_hwc.shape[2] > 3:
                rgb_hwc = rgb_hwc[:, :, :3]
            x_hwc = (
                np.full((rgb_hwc.shape[0], rgb_hwc.shape[1], x_expect), fill_val, dtype=np.float32)
                if x_expect > 0
                else np.zeros((rgb_hwc.shape[0], rgb_hwc.shape[1], 0), dtype=np.float32)
            )
            hwc = np.concatenate([rgb_hwc, x_hwc], axis=2)
        elif has_x and not has_rgb:
            x_hwc = Preprocessor.letterbox_single(inputs['x'], size)
            if x_hwc.ndim == 2:
                x_hwc = x_hwc[:, :, None]
            if x_hwc.shape[2] == 1 and x_expect > 1:
                x_hwc = np.repeat(x_hwc, x_expect, axis=2)
            elif x_hwc.shape[2] < x_expect and x_hwc.shape[2] > 0:
                reps = int(np.ceil(x_expect / x_hwc.shape[2]))
                x_hwc = np.concatenate([x_hwc] * reps, axis=2)[:, :, :x_expect]
            elif x_hwc.shape[2] > x_expect:
                x_hwc = x_hwc[:, :, :x_expect]
            rgb_hwc = np.zeros((x_hwc.shape[0], x_hwc.shape[1], 3), dtype=np.float32)
            hwc = np.concatenate([rgb_hwc, x_hwc], axis=2)
        else:
            raise ValueError("未检测到有效的 RGB/X 输入。")

        input_tensor = torch.from_numpy(np.transpose(hwc, (2, 0, 1))).unsqueeze(0).float()


        heat_layers = [str(i) for i in layers]
        alg = kwargs.get('alg', 'gradcam')

        vis = HeatmapVisualizer(model)
        results = vis.visualize(
            images=originals,
            layers=heat_layers,
            alg=alg,
            preprocessed_input=input_tensor,
            original_images=originals,
            **{k: v for k, v in kwargs.items() if k != 'alg'}
        )

        out: List[CoreVisualizationResult] = []
        blend_alpha = float(kwargs.get('blend_alpha', 0.5))
        cmap = kwargs.get('colormap', 'jet')
        cmap_map = {
            'jet': cv2.COLORMAP_JET,
            'turbo': cv2.COLORMAP_TURBO,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'inferno': cv2.COLORMAP_INFERNO,
            'magma': cv2.COLORMAP_MAGMA,
            'plasma': cv2.COLORMAP_PLASMA,
        }
        cmap_name = str(cmap).lower()
        if cmap_name not in cmap_map:
            raise ValueError(f"不支持的 colormap: {cmap}，可选：{', '.join(sorted(cmap_map.keys()))}")
        cmap_cv2 = cmap_map[cmap_name]

        layout = str(kwargs.get('layout', 'panel')).lower().strip()
        if layout not in {'overlay', 'panel', 'both'}:
            raise ValueError(f"layout 参数非法：{layout}，可选：overlay|panel|both")
        panel_scale = float(kwargs.get('panel_scale', 1.0))
        panel_title = bool(kwargs.get('panel_title', True))


        overlay_req = kwargs.get('overlay', None)
        has_rgb = 'rgb' in inputs
        has_x = 'x' in inputs
        overlay_base: str
        if overlay_req is None:

            overlay_base = 'dual' if (has_rgb and has_x) else ('rgb' if has_rgb else ('x' if has_x else 'rgb'))
        else:
            overlay_base = str(overlay_req).lower().strip()
            if overlay_base not in {'rgb', 'x', 'dual'}:
                raise ValueError(f"overlay 参数非法：{overlay_req}，可选：rgb|x|dual")

            if overlay_base == 'rgb' and (not has_rgb) and has_x:
                overlay_base = 'x'

        if overlay_base == 'x' and not has_x:
            raise ValueError("overlay='x' 需要提供 X 模态输入")
        if overlay_base == 'dual' and not (has_rgb and has_x):
            raise ValueError("overlay='dual' 需要同时提供 RGB 与 X 输入")


        export_components = bool(kwargs.get('export_components', True))
        export_panel = bool(kwargs.get('export_panel', True))
        export_scale_arg = kwargs.get('export_scale', kwargs.get('save_scale', None))
        if export_scale_arg is None:
            export_scale = 1.0
        else:
            try:
                export_scale = float(export_scale_arg)
            except Exception:
                raise ValueError(f"export_scale/save_scale 需要为数字，收到: {export_scale_arg!r}")
            if not np.isfinite(export_scale) or export_scale <= 0:
                raise ValueError(f"export_scale/save_scale 必须为正数，收到: {export_scale}")

        img_key = kwargs.get('img_key', None)

        for li, r in zip(layers, results):
            originals = getattr(r, 'original_image', None)
            heatmaps = getattr(r, 'heatmap', None)
            overlays = getattr(r, 'overlay', None)
            if originals is None or heatmaps is None:
                raise ValueError("HeatmapVisualizer 返回结果缺少 original_image/heatmap 字段，无法生成可视化输出。")


            layer_subdir = f"layer{int(li):03d}"

            assets: Dict[str, np.ndarray] = {}


            if overlay_base == 'dual':
                if not isinstance(originals, dict) or not isinstance(heatmaps, dict):
                    raise ValueError("overlay='dual' 时期望 HeatmapResult 返回 dict 形式的 original_image/heatmap。")
                if overlays is None or not isinstance(overlays, dict):

                    overlays = Renderer.heat_overlay_multimodal(originals, heatmaps, alpha=blend_alpha, colormap=cmap_cv2)

                need_panel = layout in {'panel', 'both'}
                panel_rgb = None
                panel_x = None
                panel_dual = None
                if need_panel:
                    panel_rgb = Renderer.heat_triptych(
                        originals['rgb'],
                        heatmaps['rgb'],
                        overlays.get('rgb'),
                        alpha=blend_alpha,
                        colormap=cmap_cv2,
                        scale=panel_scale,
                        title=panel_title,
                    )
                    panel_x = Renderer.heat_triptych(
                        originals['x'],
                        heatmaps['x'],
                        overlays.get('x'),
                        alpha=blend_alpha,
                        colormap=cmap_cv2,
                        scale=panel_scale,
                        title=panel_title,
                    )

                    if panel_rgb.shape[1] != panel_x.shape[1]:
                        raise ValueError(
                            f"RGB/X 三联图宽度不一致，无法拼接：rgb={panel_rgb.shape} x={panel_x.shape}。"
                            f"请确保两路输入分辨率一致，或使用 align_base/统一数据预处理后再可视化。"
                        )
                    sep = np.ones((8, panel_rgb.shape[1], 3), dtype=np.uint8) * 255
                    panel_dual = np.vstack([panel_rgb, sep, panel_x])

                if export_components:
                    for mk in ('rgb', 'x'):
                        if mk not in originals or mk not in heatmaps:
                            continue
                        ov_m = overlays.get(mk)
                        if ov_m is None:
                            ov_m = Renderer.heat_overlay(originals[mk], heatmaps[mk], alpha=blend_alpha, colormap=cmap_cv2)
                        assets[f"{mk}/original"] = originals[mk]
                        assets[f"{mk}/heatmap"] = heatmaps[mk]
                        assets[f"{mk}/overlay"] = ov_m
                        if export_panel:

                            if mk == 'rgb' and panel_rgb is not None:
                                assets[f"{mk}/panel"] = panel_rgb
                            elif mk == 'x' and panel_x is not None:
                                assets[f"{mk}/panel"] = panel_x
                            else:
                                assets[f"{mk}/panel"] = Renderer.heat_triptych(
                                    originals[mk],
                                    heatmaps[mk],
                                    ov_m,
                                    alpha=blend_alpha,
                                    colormap=cmap_cv2,
                                    scale=panel_scale,
                                    title=panel_title,
                                )

                    if export_panel:
                        pr = assets.get("rgb/panel", panel_rgb)
                        px = assets.get("x/panel", panel_x)
                        if pr is not None and px is not None:
                            if pr.shape[1] != px.shape[1]:
                                raise ValueError(
                                    f"RGB/X 导出 panel 宽度不一致，无法拼接：rgb={pr.shape} x={px.shape}。"
                                    f"请确保两路输入分辨率一致，或仅导出单路素材。"
                                )
                            sep = np.ones((8, pr.shape[1], 3), dtype=np.uint8) * 255
                            assets["dual/panel"] = np.vstack([pr, sep, px])

                if layout == 'overlay':
                    data = {'rgb': overlays['rgb'], 'x': overlays['x']}
                elif layout == 'panel':
                    data = panel_dual
                else:
                    data = {
                        'dual_panel': panel_dual,
                        'rgb_panel': panel_rgb,
                        'x_panel': panel_x,
                        'rgb_original': originals['rgb'],
                        'x_original': originals['x'],
                        'rgb_heatmap': heatmaps['rgb'],
                        'x_heatmap': heatmaps['x'],
                        'rgb_overlay': overlays['rgb'],
                        'x_overlay': overlays['x'],
                    }
            else:

                if isinstance(originals, dict):
                    if overlay_base not in originals or overlay_base not in heatmaps:
                        raise ValueError(f"叠加底图 {overlay_base} 在输入中不可用")
                    orig = originals[overlay_base]
                    hm = heatmaps[overlay_base]
                    ov = overlays.get(overlay_base) if isinstance(overlays, dict) else None
                else:
                    orig = originals
                    hm = heatmaps
                    ov = overlays
                if ov is None:
                    ov = Renderer.heat_overlay(orig, hm, alpha=blend_alpha, colormap=cmap_cv2)

                if export_components:

                    if isinstance(originals, dict) and isinstance(heatmaps, dict):
                        for mk in ('rgb', 'x'):
                            if mk not in originals or mk not in heatmaps:
                                continue
                            ov_m = overlays.get(mk) if isinstance(overlays, dict) else None
                            if ov_m is None:
                                ov_m = Renderer.heat_overlay(originals[mk], heatmaps[mk], alpha=blend_alpha, colormap=cmap_cv2)
                            assets[f"{mk}/original"] = originals[mk]
                            assets[f"{mk}/heatmap"] = heatmaps[mk]
                            assets[f"{mk}/overlay"] = ov_m
                            if export_panel:
                                assets[f"{mk}/panel"] = Renderer.heat_triptych(
                                    originals[mk],
                                    heatmaps[mk],
                                    ov_m,
                                    alpha=blend_alpha,
                                    colormap=cmap_cv2,
                                    scale=panel_scale,
                                    title=panel_title,
                                )
                    else:
                        mk = 'rgb' if overlay_base == 'rgb' else 'x'
                        assets[f"{mk}/original"] = orig
                        assets[f"{mk}/heatmap"] = hm
                        assets[f"{mk}/overlay"] = ov
                        if export_panel:
                            assets[f"{mk}/panel"] = Renderer.heat_triptych(
                                orig,
                                hm,
                                ov,
                                alpha=blend_alpha,
                                colormap=cmap_cv2,
                                scale=panel_scale,
                                title=panel_title,
                            )

                if layout == 'overlay':
                    data = ov
                elif layout == 'panel':
                    data = Renderer.heat_triptych(
                        orig,
                        hm,
                        ov,
                        alpha=blend_alpha,
                        colormap=cmap_cv2,
                        scale=panel_scale,
                        title=panel_title,
                    )
                else:
                    data = {
                        'panel': Renderer.heat_triptych(
                            orig,
                            hm,
                            ov,
                            alpha=blend_alpha,
                            colormap=cmap_cv2,
                            scale=panel_scale,
                            title=panel_title,
                        ),
                        'original': orig,
                        'heatmap': hm,
                        'overlay': ov,
                    }

            meta = {
                'method': 'heat',
                'layer_idx': li,

                'modality': overlay_base,
                'family': family,
                'algorithm': alg,
                'alpha': blend_alpha,
                'colormap': cmap,
                'layout': layout,
                'panel_scale': panel_scale,
                'subdir': layer_subdir,
            }
            if isinstance(img_key, str) and len(img_key) > 0:
                meta['img_key'] = img_key
            out.append(CoreVisualizationResult(type='heat', data=data, meta=meta))


            if export_components and assets:
                asset_meta = dict(meta)
                if export_scale != 1.0:
                    asset_meta['save_scale'] = float(export_scale)
                out.append(CoreVisualizationResult(type='heat_assets', data=assets, meta=asset_meta))
        return out

class FeatureMapPlugin(MethodPlugin):
    @staticmethod
    def run(
        *,
        model: Any,
        inputs: Dict[str, Any],
        layers: List[int],
        layer_names: List[str],
        save: bool,
        out_dir: Path,
        modality: str | None,
        family: str,
        **kwargs: Any,
    ) -> List[CoreVisualizationResult]:











        align_base = str(kwargs.get('align_base', 'rgb')).lower()
        metric = str(kwargs.get('metric', 'sum')).lower()
        top_k = int(kwargs.get('top_k', 8))
        normalize = str(kwargs.get('normalize', 'minmax')).lower()
        colormap = str(kwargs.get('colormap', 'gray')).lower()
        split = bool(kwargs.get('split', False))


        has_rgb = 'rgb' in inputs
        has_x = 'x' in inputs
        if modality is None or str(modality).lower() == 'auto':
            m = ('dual' if (has_rgb and has_x) else ('rgb' if has_rgb else 'x'))
        else:
            m = str(modality).lower()
        ablation_fill = str(kwargs.get('ablation_fill', 'zeros')).lower()




        size = Preprocessor.model_input_size(model)
        total_ch = Preprocessor.model_input_channels(model, default=6)
        x_expect = max(total_ch - 3, 0)
        fill_val = 0.0 if ablation_fill == 'zeros' else 0.5

        if has_rgb and has_x:
            hwc = Preprocessor.letterbox_dual_aligned(inputs['rgb'], inputs['x'], size=size, align_base=align_base)
            if hwc.ndim != 3 or hwc.shape[2] < 4:
                raise ValueError(f"预处理后的形状异常：{hwc.shape}，期望通道数 ≥4（RGB3 + Xch≥1）")
            if hwc.shape[2] != total_ch:
                raise ValueError(
                    f"输入通道数与模型不一致：预处理后={hwc.shape[2]}，模型期望(Dual)={total_ch}。"
                    f" 请检查数据 Xch / 模型配置 / modality 路由设定。"
                )
            rgb_hwc = hwc[:, :, :3]
            x_hwc = hwc[:, :, 3:]
        elif has_rgb and not has_x:
            rgb_hwc = Preprocessor.letterbox_single(inputs['rgb'], size)
            if rgb_hwc.ndim == 2:
                rgb_hwc = rgb_hwc[:, :, None]
            if rgb_hwc.shape[2] == 1:
                rgb_hwc = np.repeat(rgb_hwc, 3, axis=2)
            elif rgb_hwc.shape[2] > 3:
                rgb_hwc = rgb_hwc[:, :, :3]
            x_hwc = (
                np.full((rgb_hwc.shape[0], rgb_hwc.shape[1], x_expect), fill_val, dtype=np.float32)
                if x_expect > 0
                else np.zeros((rgb_hwc.shape[0], rgb_hwc.shape[1], 0), dtype=np.float32)
            )
        elif has_x and not has_rgb:
            x_hwc = Preprocessor.letterbox_single(inputs['x'], size)
            if x_hwc.ndim == 2:
                x_hwc = x_hwc[:, :, None]
            if x_hwc.shape[2] == 1 and x_expect > 1:
                x_hwc = np.repeat(x_hwc, x_expect, axis=2)
            elif x_hwc.shape[2] < x_expect:
                reps = int(np.ceil(x_expect / x_hwc.shape[2]))
                x_hwc = np.concatenate([x_hwc] * reps, axis=2)[:, :, :x_expect]
            elif x_hwc.shape[2] > x_expect:
                x_hwc = x_hwc[:, :, :x_expect]
            rgb_hwc = np.zeros((x_hwc.shape[0], x_hwc.shape[1], 3), dtype=np.float32)
        else:
            raise ValueError("未检测到有效的 RGB/X 输入。")

        hwc = np.concatenate([rgb_hwc, x_hwc], axis=2)


        nchw = np.transpose(hwc, (2, 0, 1))[None, ...].astype(np.float32)
        inp = torch.from_numpy(nchw)




        feats: Dict[int, torch.Tensor] = {}
        handles = []

        def _first_tensor(x: Any) -> torch.Tensor | None:
            if isinstance(x, torch.Tensor):
                return x
            if isinstance(x, (list, tuple)):
                for t in x:
                    if isinstance(t, torch.Tensor):
                        return t
            return None

        for li in layers:
            if not hasattr(model, 'model') or li < 0 or li >= len(model.model):
                raise ValueError(f"层索引越界：{li}")
            mod = model.model[li]

            def _hook_closure(idx: int):
                def _hook(module, inputs_, output):
                    t = _first_tensor(output)
                    if t is not None:
                        feats[idx] = t.detach()
                return _hook

            handles.append(mod.register_forward_hook(_hook_closure(li)))


        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        inp = inp.to(device)
        with torch.no_grad():
            _ = model(inp)


        for h in handles:
            try:
                h.remove()
            except Exception:
                pass




        def _score_channel_map(t: torch.Tensor) -> torch.Tensor:

            if metric == 'sum':
                return t.abs().sum(dim=(1, 2))
            elif metric == 'var':
                return t.var(dim=(1, 2))
            else:
                raise ValueError(f"不支持的 metric: {metric}（仅 'sum'|'var'）")

        def _norm_uint8(arr: np.ndarray) -> np.ndarray:
            if normalize == 'minmax':
                a_min, a_max = float(arr.min()), float(arr.max())
                if a_max > a_min:
                    out = (arr - a_min) / (a_max - a_min)
                else:
                    out = np.zeros_like(arr)
                return (out * 255.0).astype(np.uint8)
            else:
                raise ValueError(f"不支持的 normalize: {normalize}（基础版仅 'minmax'）")

        def _cv2_colormap(name: str) -> int | None:
            name = str(name).lower()
            if name in {'gray', 'grey', 'grayscale', 'none'}:
                return None
            m = {
                'jet': cv2.COLORMAP_JET,
                'viridis': cv2.COLORMAP_VIRIDIS,
                'inferno': cv2.COLORMAP_INFERNO,
                'magma': cv2.COLORMAP_MAGMA,
                'plasma': cv2.COLORMAP_PLASMA,
            }
            return m.get(name, cv2.COLORMAP_VIRIDIS)

        def _render_grid(feat_list: List[tuple[int, float, np.ndarray]], cell: tuple[int, int] = (128, 128)) -> np.ndarray:

            if not feat_list:
                return np.zeros((256, 256, 3), dtype=np.uint8)
            n = len(feat_list)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            h, w = cell
            pad = 5
            label_h = 22
            canvas = np.ones((rows * (h + label_h + pad) + pad, cols * (w + pad) + pad, 3), dtype=np.uint8) * 255

            cmap = _cv2_colormap(colormap)

            for i, (ch, sc, fm) in enumerate(feat_list):
                r, c = i // cols, i % cols
                y = pad + r * (h + label_h + pad)
                x = pad + c * (w + pad)
                u8 = _norm_uint8(fm)
                u8 = cv2.resize(u8, (w, h), interpolation=cv2.INTER_AREA)
                if cmap is None:
                    rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.applyColorMap(u8, cmap)
                    try:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass
                canvas[y:y + h, x:x + w] = rgb
                label = f"ch:{ch} score:{sc:.1f}"
                cv2.putText(canvas, label, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            return canvas

        modality_meta = {
            ('rgb', True): 'rgb_ablate',
            ('x', True): 'x_ablate',
        }.get((m, not (has_rgb and has_x)), m)

        results: List[CoreVisualizationResult] = []
        for li in layers:
            t = feats.get(li, None)
            if t is None:

                continue
            if t.dim() == 4:

                b, c, h_, w_ = t.shape

                tt = t[0]
            elif t.dim() == 3:

                tt = t
                c = tt.shape[0]
            else:

                continue

            scores = _score_channel_map(tt)
            k = int(min(max(1, top_k), int(scores.numel())))
            top_idx = torch.topk(scores, k).indices.cpu().numpy().tolist()

            feat_list: List[tuple[int, float, np.ndarray]] = []
            tiles_imgs: List[np.ndarray] = []
            tiles_channels: List[int] = []
            tiles_scores: List[float] = []

            def _render_tile(fm: np.ndarray, cell: tuple[int, int] = (128, 128)) -> np.ndarray:
                u8 = _norm_uint8(fm)
                u8 = cv2.resize(u8, (cell[1], cell[0]), interpolation=cv2.INTER_AREA)
                cmap_code = _cv2_colormap(colormap)
                if cmap_code is None:
                    return cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.applyColorMap(u8, cmap_code)
                    try:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass
                    return rgb

            for ch in top_idx:
                fmap = tt[ch].detach().cpu().float().numpy()
                sc = float(scores[ch].detach().cpu().item())
                feat_list.append((int(ch), sc, fmap))
                if split:
                    tiles_imgs.append(_render_tile(fmap))
                    tiles_channels.append(int(ch))
                    tiles_scores.append(sc)

            grid = _render_grid(feat_list)
            meta = {
                'method': 'feature',
                'layer_idx': li,
                'modality': modality_meta,
                'family': family,
                'metric': metric,
                'top_k': top_k,
                'normalize': normalize,
                'align_base': align_base,
            }

            img_key = kwargs.get('img_key', inputs.get('img_key') if isinstance(inputs, dict) else None)
            if img_key is not None:
                meta['img_key'] = img_key
            results.append(CoreVisualizationResult(type='feature', data=grid, meta=meta))

            if split and tiles_imgs:
                tiles_meta = {
                    **meta,
                    'channels': tiles_channels,
                    'scores': tiles_scores,
                    'subdir': f'layer{li}',
                }
                if img_key is not None:
                    tiles_meta['img_key'] = img_key
                results.append(CoreVisualizationResult(type='feature_tiles', data=tiles_imgs, meta=tiles_meta))

        return results


REGISTRY.register('heat', HeatmapPlugin)
REGISTRY.register('feature', FeatureMapPlugin)

