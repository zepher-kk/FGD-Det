from __future__ import annotations
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
RT-DETR MultiModal predictor module.

This module provides the RTDETRMMPredictor class for inference with multi-modal RT-DETR models
supporting RGB+X modality inputs. Architecture strictly follows YOLOMM's successful pattern.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import LOGGER, DEFAULT_CFG, ops, colorstr
from ultralytics.engine.results import Results


class RTDETRMMPredictor(BasePredictor):










































    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):









        super().__init__(cfg, overrides, _callbacks)
        

        self.modality = getattr(self.args, 'modality', None)
        

        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        

        self.rgb_source = None
        self.x_source = None
        self.input_mode = None
        

        LOGGER.info(f"RTDETRMMPredictor initialized: modality={self.modality}, "
                   f"dual={self.is_dual_modal}, single={self.is_single_modal}")




    def _get_mm_router(self):

        m = getattr(self, "model", None)
        if m is None:
            return None

        if hasattr(m, "mm_router") and m.mm_router is not None:
            return m.mm_router

        if hasattr(m, "pt") and getattr(m, "pt", False) and hasattr(m, "model"):
            inner = getattr(m, "model", None)
            if inner is not None and hasattr(inner, "mm_router") and inner.mm_router is not None:
                return inner.mm_router
        return None

    def _set_runtime_modality_for_router(self):

        mm_router = self._get_mm_router()
        if self._dual_input_detected:
            if mm_router:
                mm_router.set_runtime_params(None)
            return

        if not mm_router:
            raise RuntimeError(
                "检测到单模态输入但未找到多模态路由器(mm_router)。"
                "请确保使用多模态权重并在 PyTorch/AutoBackend(PyTorch) 后端运行。"
            )
        mm_router.set_runtime_params(
            self.modality,
            strategy=getattr(self.args, "ablation_strategy", None),
            seed=getattr(self.args, "seed", None),
        )

    def _get_dual_channels(self) -> int:

        mm_router = self._get_mm_router()
        try:
            if mm_router and hasattr(mm_router, "INPUT_SOURCES"):
                return int(mm_router.INPUT_SOURCES.get("Dual", 6))
        except Exception:
            pass
        return 6

    def _parse_inference_input(self, source):





















        import numpy as np
        from PIL import Image
        from pathlib import Path
        

        input_info = {
            'input_type': type(source).__name__,
            'is_batch': False,
            'source_count': 1,
            'modality_mode': 'dual' if self.is_dual_modal else f'single_{self.modality}',
            'inference_format': None,
            'validation_passed': False
        }
        
        try:

            LOGGER.debug(f"解析推理输入: 类型={input_info['input_type']}, 模态模式={input_info['modality_mode']}")
            

            if isinstance(source, torch.Tensor):
                input_info['inference_format'] = 'preprocessed_tensor'
                input_info['tensor_shape'] = list(source.shape)
                
                if source.dim() == 4 and source.shape[1] == 6:
                    LOGGER.debug("检测到6通道预处理tensor，直接使用")
                    input_info['validation_passed'] = True
                    return source, input_info
                else:
                    LOGGER.warning(f"Tensor维度不符合预期: {source.shape}，将重新处理")
            

            elif isinstance(source, (list, tuple)):
                input_info['source_count'] = len(source)
                
                if len(source) == 2 and self.is_dual_modal:

                    input_info['inference_format'] = 'dual_modal'
                    rgb_source, x_source = source
                    

                    rgb_info = self._analyze_single_source(rgb_source, 'rgb')
                    x_info = self._analyze_single_source(x_source, 'x_modal')
                    
                    input_info['rgb_source'] = rgb_info
                    input_info['x_source'] = x_info
                    input_info['validation_passed'] = True
                    
                    return source, input_info
                    
                elif len(source) == 1 and self.is_single_modal:

                    input_info['inference_format'] = 'single_modal'
                    single_source = source[0]
                    LOGGER.debug(f"单模态输入(列表包装): {type(single_source)}")
                    

                    source_info = self._analyze_single_source(single_source, self.modality)
                    input_info.update(source_info)
                    input_info['validation_passed'] = True
                    
                    return single_source, input_info
                    
                else:

                    if self.is_dual_modal:
                        raise ValueError(f"双模态推理需要2个输入源，但接收到{len(source)}个")
                    else:

                        single_source = source[0]
                        LOGGER.warning(f"单模态推理接收到{len(source)}个输入，使用第一个: {single_source}")
                        return self._parse_inference_input(single_source)
            

            else:
                if self.is_dual_modal:
                    raise ValueError(
                        f"双模态推理需要列表格式输入 [rgb_source, x_source]，"
                        f"但接收到单个源: {type(source)}"
                    )
                

                input_info['inference_format'] = 'single_modal'
                source_info = self._analyze_single_source(source, self.modality)
                input_info.update(source_info)
                input_info['validation_passed'] = True
                
                return source, input_info
                
        except Exception as e:
            input_info['validation_passed'] = False
            input_info['error'] = str(e)
            LOGGER.error(f"输入解析失败: {e}")
            raise
        
        finally:

            self._log_input_analysis(input_info)

    def _analyze_single_source(self, source, modality_hint=None):











        import numpy as np
        from PIL import Image
        from pathlib import Path
        
        analysis = {
            'source_type': 'unknown',
            'path': None,
            'exists': False,
            'format': None,
            'modality_hint': modality_hint
        }
        
        if isinstance(source, (str, Path)):

            path = Path(source)
            analysis['source_type'] = 'file_path'
            analysis['path'] = str(path)
            analysis['exists'] = path.exists()
            analysis['format'] = path.suffix.lower() if path.suffix else 'no_extension'
            
            if not analysis['exists']:
                raise FileNotFoundError(f"输入文件不存在: {path}")
                
        elif isinstance(source, Image.Image):

            analysis['source_type'] = 'pil_image'
            analysis['format'] = source.format or 'unknown'
            analysis['mode'] = source.mode
            analysis['size'] = source.size
            
        elif isinstance(source, np.ndarray):

            analysis['source_type'] = 'numpy_array'
            analysis['shape'] = source.shape
            analysis['dtype'] = str(source.dtype)
            
        elif isinstance(source, torch.Tensor):

            analysis['source_type'] = 'torch_tensor'
            analysis['shape'] = list(source.shape)
            analysis['dtype'] = str(source.dtype)
            analysis['device'] = str(source.device)
            
        else:
            analysis['source_type'] = f'unsupported_{type(source).__name__}'
            
        return analysis

    def _log_input_analysis(self, input_info):







        LOGGER.debug("=== RTDETRMM输入解析分析报告 ===")
        LOGGER.debug(f"输入类型: {input_info['input_type']}")
        LOGGER.debug(f"推理格式: {input_info['inference_format']}")
        LOGGER.debug(f"模态模式: {input_info['modality_mode']}")
        LOGGER.debug(f"源数量: {input_info['source_count']}")
        LOGGER.debug(f"验证通过: {input_info['validation_passed']}")
        
        if 'rgb_source' in input_info:
            LOGGER.debug(f"RGB源信息: {input_info['rgb_source']}")
            
        if 'x_source' in input_info:
            LOGGER.debug(f"X模态源信息: {input_info['x_source']}")
            
        if 'error' in input_info:
            LOGGER.debug(f"错误信息: {input_info['error']}")
            
        LOGGER.debug("=== 分析报告结束 ===")

    def _update_modality_state(self):




        current_modality = getattr(self.args, 'modality', None)
        if current_modality != self.modality:
            LOGGER.debug(f"RTDETRMMPredictor: 动态更新模态状态 {self.modality} → {current_modality}")
            self.modality = current_modality
            self.is_dual_modal = self.modality is None
            self.is_single_modal = self.modality is not None

    def preprocess(self, im):







        self._update_modality_state()


        if isinstance(im, (list, tuple)) and len(im) == 2:
            self._dual_input_detected = True
            self.input_mode = 'dual'
        else:
            self._dual_input_detected = False
            self.input_mode = f'single_{self.modality}' if self.modality else 'single'


        self._set_runtime_modality_for_router()


        if isinstance(im, (list, tuple)) and len(im) == 2:
            return self._process_dual_modality(im)


        return super().preprocess(im)

    def _process_dual_modality(self, im):












        if isinstance(im, torch.Tensor) and im.shape[1] == 6:
            LOGGER.debug("输入已为6通道tensor，直接返回")
            return im
        

        rgb_images, x_images = self._parse_dual_modal_input(im)
        

        rgb_tensor = super().preprocess(rgb_images)
        x_tensor = super().preprocess(x_images)
        

        rgb_tensor, x_tensor = self._align_tensor_dimensions(rgb_tensor, x_tensor)
        

        combined_tensor = torch.cat([x_tensor, rgb_tensor], dim=1)
        
        LOGGER.debug(f"双模态预处理完成: {combined_tensor.shape}")
        return combined_tensor

    def _parse_dual_modal_input(self, im):










        if isinstance(im, (list, tuple)):
            if len(im) == 2:

                rgb_source, x_source = im
                LOGGER.debug(f"解析标准双模态输入: RGB={type(rgb_source)}, X={type(x_source)}")
                

                rgb_images = self._load_image_source(rgb_source)
                x_images = self._load_image_source(x_source)
                
                return rgb_images, x_images
                
            else:

                raise ValueError(
                    f"双模态推理需要包含2个元素的列表输入 [rgb_source, x_source]，"
                    f"但接收到: {type(im)} with {len(im)} 元素"
                )
        else:

            raise ValueError(
                f"双模态推理需要列表格式输入 [rgb_source, x_source]，"
                f"但接收到单个源: {type(im)}"
            )

    def _load_image_source(self, source):










        import cv2
        import numpy as np
        from PIL import Image
        from pathlib import Path
        
        LOGGER.debug(f"加载图像源: 类型={type(source)}")
        

        if isinstance(source, (str, Path)):
            source_path = Path(source)
            

            if not source_path.exists():
                raise FileNotFoundError(f"图像文件不存在: {source_path}")
            

            img = cv2.imread(str(source_path))
            if img is None:
                raise ValueError(f"无法加载图像: {source_path}")
            

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            LOGGER.debug(f"成功加载图像: {source_path}")
            return [img]
            

        elif isinstance(source, Image.Image):
            LOGGER.debug("处理PIL图像输入")
            if source.mode != "RGB":
                source = source.convert("RGB")
            img = np.asarray(source)
            return [img]
            

        elif isinstance(source, np.ndarray):
            LOGGER.debug(f"处理numpy数组输入: shape={source.shape}")
            
            if source.ndim == 3:

                return [source]
            elif source.ndim == 4:

                if source.shape[1] == 3 or source.shape[1] == 1:

                    images = [img.transpose(1, 2, 0) for img in source]
                else:

                    images = list(source)
                return images
            else:
                raise ValueError(f"不支持的numpy数组维度: {source.shape}")
                

        elif isinstance(source, torch.Tensor):
            LOGGER.debug(f"处理torch.Tensor输入: shape={source.shape}")
            

            if source.device != torch.device('cpu'):
                source = source.cpu()
            source_np = source.numpy()
            

            return self._load_image_source(source_np)
            

        elif isinstance(source, (list, tuple)):
            LOGGER.debug(f"处理列表输入: 长度={len(source)}")
            
            all_images = []
            for i, item in enumerate(source):
                try:
                    loaded = self._load_image_source(item)
                    all_images.extend(loaded)
                except Exception as e:
                    LOGGER.error(f"加载列表项[{i}]失败: {e}")
                    raise
            
            LOGGER.debug(f"列表加载完成: 总计{len(all_images)}张图像")
            return all_images
            
        else:
            raise TypeError(f"不支持的图像源类型: {type(source)}")

    def _align_tensor_dimensions(self, tensor1, tensor2):











        import torch.nn.functional as F
        
        if tensor1.shape[2:] == tensor2.shape[2:]:

            return tensor1, tensor2
        

        h1, w1 = tensor1.shape[2:]
        h2, w2 = tensor2.shape[2:]
        

        target_h = min(h1, h2)
        target_w = min(w1, w2)
        target_size = (target_h, target_w)
        
        LOGGER.debug(f"对齐tensor维度到: {target_size}")
        

        if (h1, w1) != target_size:
            tensor1 = F.interpolate(tensor1, size=target_size, mode='bilinear', align_corners=False)
        
        if (h2, w2) != target_size:
            tensor2 = F.interpolate(tensor2, size=target_size, mode='bilinear', align_corners=False)
        
        return tensor1, tensor2

    def _process_single_modality(self, im):






        raise RuntimeError(
            "Single-modal preprocessing-side filling is disabled. "
            "Pass 3-channel input and rely on MultiModalRouter via modality runtime params."
        )

    def _generate_modality_filling(self, source_tensor: torch.Tensor, 
                                   source_modality: str, target_modality: str) -> torch.Tensor:






        from ultralytics.nn.mm.filling import generate_modality_filling
        return generate_modality_filling(source_tensor, source_modality, target_modality)

    def _finalize_tensor(self, tensor):













        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"期望torch.Tensor输出，但得到: {type(tensor)}")
        
        if tensor.dim() != 4:
            raise ValueError(f"期望4维tensor [B, C, H, W]，但得到维度: {tensor.dim()}")
        
        if tensor.shape[1] != 6:
            raise ValueError(f"期望6通道tensor，但得到: {tensor.shape[1]}通道")
        

        if hasattr(self, 'device') and self.device != tensor.device:
            LOGGER.debug(f"转移tensor到设备: {self.device}")
            tensor = tensor.to(self.device)
        

        if tensor.dtype != torch.float32:
            LOGGER.debug(f"转换tensor数据类型: {tensor.dtype} -> torch.float32")
            tensor = tensor.float()
        
        return tensor

    def _log_debug_info(self, im, exception):








        LOGGER.debug("=== 多模态预处理调试信息 ===")
        LOGGER.debug(f"模态设置: modality={self.modality}, is_dual_modal={self.is_dual_modal}")
        LOGGER.debug(f"输入类型: {type(im)}")
        
        if isinstance(im, (list, tuple)):
            LOGGER.debug(f"列表输入长度: {len(im)}")
            for i, item in enumerate(im):
                LOGGER.debug(f"  项目[{i}]: {type(item)} - {item}")
        elif isinstance(im, torch.Tensor):
            LOGGER.debug(f"Tensor形状: {im.shape}")
            LOGGER.debug(f"Tensor设备: {im.device}")
            LOGGER.debug(f"Tensor数据类型: {im.dtype}")
        else:
            LOGGER.debug(f"输入内容: {im}")
        
        LOGGER.debug(f"异常类型: {type(exception).__name__}")
        LOGGER.debug(f"异常信息: {str(exception)}")
        LOGGER.debug("=== 调试信息结束 ===")

    def stream_inference(self, source=None, model=None, *args, **kwargs):















        if self.args.verbose:
            LOGGER.info("")


        if not self.model:
            self.setup_model(model)

        with self._lock:

            self.setup_source(source if source is not None else self.args.source)


            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)


            if not self.done_warmup:
                model_channels = self._get_dual_channels()
                LOGGER.info(f"RTDETRMMPredictor: 使用 {model_channels} 通道进行模型预热")
                self.model.warmup(
                    imgsz=(
                        1 if getattr(self.model, "pt", False) or getattr(self.model, "triton", False) else self.dataset.bs,
                        model_channels,
                        *self.imgsz,
                    )
                )
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                try:
                    self._mm_current_im0s = im0s
                except Exception:
                    self._mm_current_im0s = None


                with profilers[0]:
                    im = self.preprocess(im0s)


                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds
                        continue


                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")


                n = len(im0s)
                results_count = len(self.results)
                


                if results_count != n:
                    LOGGER.debug(f"多模态推理: 输入{n}张图像，生成{results_count}个结果")
                
                for i in range(results_count):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / results_count,
                        "inference": profilers[1].dt * 1e3 / results_count,
                        "postprocess": profilers[2].dt * 1e3 / results_count,
                    }
                    

                    if results_count < n:

                        result_path = Path(paths[0])
                        result_string = s[0] if s else ""
                        

                        if len(paths) > 1:
                            modality_info = f"({len(paths)}模态输入)"
                            result_string = f"{result_string} {modality_info}" if result_string else modality_info
                    else:

                        result_path = Path(paths[i])
                        result_string = s[i] if i < len(s) else ""
                    
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:

                        result_string += self.write_results(i, result_path, im, s)
                    

                    if i < len(s):
                        s[i] = result_string
                    elif len(s) == 0:
                        s = [result_string]


                if self.args.verbose:

                    valid_strings = [s_item for s_item in s[:results_count] if s_item]
                    if valid_strings:
                        LOGGER.info("\n".join(valid_strings))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results


        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()


        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)
            display_ch = self._get_dual_channels()
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), display_ch, *im.shape[2:])}"
                % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")
    
    def postprocess(self, preds, img, orig_imgs):















        self._orig_imgs_cache = orig_imgs
        

        results = self._postprocess_rtdetr(preds, img, orig_imgs)
        


        if self.is_dual_modal and isinstance(orig_imgs, list) and len(orig_imgs) == 2 and len(results) == 2:

            results = results[:1]
            LOGGER.debug("RTDETRMMPredictor: 双模态输入合并为单个结果对象")
        
        return results

    def _postprocess_rtdetr(self, preds, img, orig_imgs):











        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)
            idx = max_score.squeeze(-1) > self.args.conf
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):



        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=x) for x in im]
    
    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list) -> str:



















        self._update_modality_state()


        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


        orig_save = getattr(self.args, 'save', False)
        try:
            self.args.save = False
            string = super().write_results(i, p, im, s)
        finally:
            self.args.save = orig_save


        if not orig_save:
            return string


        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            ensure_batch_idx_long,
        )
        from ultralytics.models.utils.multimodal.vis import clip_boxes_norm_xywh as _clip_norm_xywh


        def _resolve_x_modality_strict():

            if self.is_single_modal and self.modality and str(self.modality).lower() == 'rgb':
                return 'rgb'
            return 'x'


        def _np_to_tensor3ch(img_np: np.ndarray) -> torch.Tensor:
            if img_np is None:
                raise RuntimeError("缺少原始图像用于可视化背景")
            if img_np.ndim == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            else:
                raise RuntimeError(f"不支持的原始图像形状: {img_np.shape}")
            t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            return t.unsqueeze(0)


        def _get_orig_modal_tensors():
            if hasattr(self, '_mm_current_im0s') and self._mm_current_im0s is not None:
                im0s = self._mm_current_im0s
                if isinstance(im0s, (list, tuple)):
                    if len(im0s) == 2:
                        return _np_to_tensor3ch(im0s[0]), _np_to_tensor3ch(im0s[1])
                    elif len(im0s) == 1:
                        if self.modality and str(self.modality).lower() == 'rgb':
                            return _np_to_tensor3ch(im0s[0]), None
                        else:
                            return None, _np_to_tensor3ch(im0s[0])

                if self.modality and str(self.modality).lower() == 'rgb':
                    return _np_to_tensor3ch(im0s), None
                else:
                    return None, _np_to_tensor3ch(im0s)

            if not hasattr(self, '_orig_imgs_cache') or self._orig_imgs_cache is None:
                raise RuntimeError("未找到原始图像用于可视化背景")
            oi = self._orig_imgs_cache
            if isinstance(oi, (list, tuple)):
                if len(oi) == 2:
                    return _np_to_tensor3ch(oi[0]), _np_to_tensor3ch(oi[1])
                elif len(oi) == 1:
                    if self.modality and str(self.modality).lower() == 'rgb':
                        return _np_to_tensor3ch(oi[0]), None
                    else:
                        return None, _np_to_tensor3ch(oi[0])
            if self.modality and str(self.modality).lower() == 'rgb':
                return _np_to_tensor3ch(oi), None
            else:
                return None, _np_to_tensor3ch(oi)


        result = self.results[i]
        base = p.stem


        rgb_tensor, x_tensor = _get_orig_modal_tensors()


        n_boxes = 0 if result.boxes is None else len(result.boxes)
        if n_boxes:
            cls_ids = result.boxes.cls
            confs = getattr(result.boxes, 'conf', None)
            boxes_norm = result.boxes.xywhn
            if not isinstance(cls_ids, torch.Tensor):
                cls_ids = torch.as_tensor(cls_ids, dtype=torch.long)
            else:
                cls_ids = cls_ids.long()
            if confs is not None and not isinstance(confs, torch.Tensor):
                confs = torch.as_tensor(confs, dtype=torch.float32)
            batch_idx = ensure_batch_idx_long(torch.zeros(cls_ids.shape[0]))
        else:
            cls_ids = torch.zeros((0,), dtype=torch.long)
            confs = torch.zeros((0,), dtype=torch.float32)
            boxes_norm = torch.zeros((0, 4), dtype=torch.float32)
            batch_idx = ensure_batch_idx_long(torch.zeros((0,), dtype=torch.long))

        names = getattr(self.model, 'names', {})
        x_modality = _resolve_x_modality_strict()


        def _norm_for_target(boxes_xywh_norm: torch.Tensor) -> torch.Tensor:
            if boxes_xywh_norm is None or boxes_xywh_norm.numel() == 0:
                return torch.zeros((0, 4), dtype=torch.float32)
            return _clip_norm_xywh(boxes_xywh_norm, 0.0, 1.0, 0.0, 1.0)


        if self.is_single_modal:
            if str(self.modality).lower() == 'rgb':
                if rgb_tensor is None:
                    raise RuntimeError("期望RGB原图用于可视化，但缓存缺失")
                boxes_norm_rgb = _norm_for_target(boxes_norm)
                fname_rgb = self.save_dir / f"pred_{base}_labels_rgb.jpg"
                plot_images(rgb_tensor, batch_idx, cls_ids, boxes_norm_rgb, confs=confs,
                            paths=[str(p)], fname=fname_rgb, names=names)
            else:
                if x_tensor is None:
                    raise RuntimeError("期望X原图用于可视化，但缓存缺失")
                boxes_norm_x = _norm_for_target(boxes_norm)
                fname_x = self.save_dir / f"pred_{base}_labels_{x_modality}.jpg"
                plot_images(x_tensor, batch_idx, cls_ids, boxes_norm_x, confs=confs,
                            paths=[str(p.with_name(f"{base}_{x_modality}{p.suffix}"))],
                            fname=fname_x, names=names)
            return string


        if rgb_tensor is None or x_tensor is None:

            LOGGER.warning("RTDETRMM write_results: missing one modality in dual-modal visualization; skip custom visualization.")
            return string

        Hr, Wr = int(rgb_tensor.shape[-2]), int(rgb_tensor.shape[-1])
        boxes_norm_rgb = _norm_for_target(boxes_norm)
        fname_rgb = self.save_dir / f"pred_{base}_labels_rgb.jpg"
        plot_images(rgb_tensor, batch_idx, cls_ids, boxes_norm_rgb, confs=confs,
                    paths=[str(p)], fname=fname_rgb, names=names)


        Hx, Wx = int(x_tensor.shape[-2]), int(x_tensor.shape[-1])
        boxes_norm_x = _norm_for_target(boxes_norm)
        fname_x = self.save_dir / f"pred_{base}_labels_{x_modality}.jpg"
        plot_images(x_tensor, batch_idx, cls_ids, boxes_norm_x, confs=confs,
                    paths=[str(p.with_name(f"{base}_{x_modality}{p.suffix}"))],
                    fname=fname_x, names=names)


        if (Hr, Wr) != (Hx, Wx):
            x_tensor_resized = torch.nn.functional.interpolate(x_tensor, size=(Hr, Wr), mode='bilinear', align_corners=False)
        else:
            x_tensor_resized = x_tensor
        side = concat_side_by_side(rgb_tensor, x_tensor_resized)
        batch_dup, cls_dup, boxes_dup, confs_dup = duplicate_bboxes_for_side_by_side(
            batch_idx, cls_ids, boxes_norm_rgb, confs
        )
        fname_mm = self.save_dir / f"pred_{base}_labels_multimodal.jpg"
        plot_images(side, batch_dup, cls_dup, boxes_dup, confs=confs_dup,
                    paths=[str(p.with_name(f"{base}_multimodal{p.suffix}"))],
                    fname=fname_mm, names=names)

        return string
    















    
    def _separate_modalities(self, tensor: torch.Tensor) -> tuple:









        if tensor.dim() == 3:

            x_tensor = tensor[:3]
            rgb_tensor = tensor[3:]
        else:

            x_tensor = tensor[:, :3]
            rgb_tensor = tensor[:, 3:]
            
        return rgb_tensor, x_tensor
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:












        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.dim() == 4:

            tensor = tensor[0]
            

        tensor = tensor.cpu()
        

        img = tensor.permute(1, 2, 0).numpy()
        

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        

        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        

        if img.shape[2] == 1:
            img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
            
        return img
    
    def _plot_on_image(self, result, img: np.ndarray, modality_name: str) -> np.ndarray:












        img_copy = img.copy()
        

        annotated = result.plot(
            img=img_copy,
            line_width=self.args.line_width,
            boxes=self.args.show_boxes,
            conf=self.args.show_conf,
            labels=self.args.show_labels
        )
        

        h, w = annotated.shape[:2]
        label_bg_color = (0, 0, 0)
        label_text_color = (255, 255, 255)
        

        cv2.rectangle(annotated, (10, 10), (150, 40), label_bg_color, -1)
        cv2.putText(annotated, f"{modality_name} Modality", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_text_color, 2)
        
        return annotated
    
    def _create_multimodal_comparison(self, rgb_img: np.ndarray, x_img: np.ndarray) -> np.ndarray:











        h1, w1 = rgb_img.shape[:2]
        h2, w2 = x_img.shape[:2]
        
        if h1 != h2:

            target_h = max(h1, h2)
            if h1 < target_h:
                scale = target_h / h1
                new_w1 = int(w1 * scale)
                rgb_img = cv2.resize(rgb_img, (new_w1, target_h))
            else:
                scale = target_h / h2
                new_w2 = int(w2 * scale)
                x_img = cv2.resize(x_img, (new_w2, target_h))
        

        gap = 10
        combined_width = rgb_img.shape[1] + x_img.shape[1] + gap
        combined_height = max(rgb_img.shape[0], x_img.shape[0])
        

        combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        

        combined[:rgb_img.shape[0], :rgb_img.shape[1]] = rgb_img
        combined[:x_img.shape[0], rgb_img.shape[1] + gap:] = x_img
        

        title_height = 50
        final_img = np.zeros((combined_height + title_height, combined_width, 3), dtype=np.uint8)
        final_img[title_height:] = combined
        

        cv2.putText(final_img, "RT-DETR Multi-Modal Detection Results", 
                    (combined_width // 2 - 200, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return final_img

