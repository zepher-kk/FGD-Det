from __future__ import annotations
# Ultralytics YOLO, AGPL-3.0 license

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr, ops
from ultralytics.data.build import load_inference_source
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.nn.mm import MultiModalSourceMatcher
from tqdm import tqdm
from copy import deepcopy

    

class MultiModalDetectionPredictor(DetectionPredictor):




























    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):








        super().__init__(cfg, overrides, _callbacks)
        


        self.modality = getattr(self.args, 'modality', None)
        

        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        

        self.rgb_source = None
        self.x_source = None
        self.input_mode = None
        



    def __call__(
        self,
        source=None,
        model=None,
        stream: bool = False,
        rgb_source=None,
        x_source=None,
        *args,
        **kwargs
    ):


















        runtime_modality = kwargs.pop("modality", None)
        if runtime_modality is not None:
            self.modality = runtime_modality

            if hasattr(self, "args") and self.args is not None:
                setattr(self.args, "modality", runtime_modality)
        else:

            self.modality = getattr(self.args, "modality", None)

        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None





        if rgb_source is not None or x_source is not None:

            if rgb_source is not None and x_source is not None:

                if self.is_single_modal:
                    raise ValueError(
                        "检测到双输入 rgb_source+x_source，但当前为单模态模式（已显式指定 modality）。"
                        "请去掉 modality 以启用双模态，或仅提供单一路输入。"
                    )
                combined_source = [rgb_source, x_source]

            elif rgb_source is not None:

                if self.is_dual_modal:
                    raise ValueError(
                        "仅提供 rgb_source 时必须显式指定 modality='rgb'（或在 args 中设置）。"
                        "否则会被视为双模态模式并要求 [rgb_source, x_source]。"
                    )
                if str(self.modality).lower() != "rgb":
                    raise ValueError(
                        f"仅提供 rgb_source，但 modality={self.modality!r} 不匹配。"
                        "请设置 modality='rgb'，或改为使用 x_source 并指定对应 X 模态名称。"
                    )
                combined_source = rgb_source

            else:

                if self.is_dual_modal:
                    raise ValueError(
                        "仅提供 x_source 时必须显式指定 modality 为 X 模态名称（如 'thermal'/'depth' 等）。"
                        "否则会被视为双模态模式并要求 [rgb_source, x_source]。"
                    )
                if str(self.modality).lower() == "rgb":
                    raise ValueError(
                        "仅提供 x_source 但 modality='rgb' 语义不一致。"
                        "请将 modality 设置为 X 模态名称，或改为传 rgb_source。"
                    )
                combined_source = x_source

        elif source is not None:

            combined_source = source
        else:
            combined_source = self.args.source

        if combined_source is None:
            raise ValueError(
                "未提供推理输入源：请传 source，或传 rgb_source/x_source（双模态需同时提供）。"
            )


        parsed_source, input_info = self._parse_inference_input(combined_source)


        if input_info.get('is_batch') and input_info.get('matched_pairs'):

            self.stream = False
            if not self.model:
                self.setup_model(model)
            return self._batch_inference_with_progress(
                input_info['matched_pairs'],
                save_subdir=True
            )
        else:

            self.stream = stream
            if stream:
                return self.stream_inference(parsed_source, model, *args, **kwargs)
            else:
                return list(self.stream_inference(parsed_source, model, *args, **kwargs))




    def _get_mm_router(self):

        m = getattr(self, "model", None)
        if m is None:
            return None


        for key in ("mm_router", "multimodal_router"):
            if hasattr(m, key):
                obj = getattr(m, key)
                if obj is not None:
                    return obj


        if hasattr(m, "pt") and getattr(m, "pt", False) and hasattr(m, "model"):
            inner = getattr(m, "model", None)
            if inner is None:
                return None

            for key in ("mm_router", "multimodal_router"):
                if hasattr(inner, key):
                    obj = getattr(inner, key)
                    if obj is not None:
                        return obj

            try:
                cfg = getattr(inner, "yaml", None)
                if isinstance(cfg, dict):
                    from ultralytics.nn.mm import MultiModalConfigParser, MultiModalRouter
                    model_config = MultiModalConfigParser().parse_config(cfg)
                    if model_config.get("has_multimodal_layers", False):
                        router = MultiModalRouter(model_config, verbose=False)

                        setattr(inner, "multimodal_router", router)
                        setattr(inner, "mm_router", router)
                        return router
            except Exception:
                pass

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

    def _detect_input_type(self, source) -> str:









        from PIL import Image

        if source is None:
            return 'none'
        if isinstance(source, torch.Tensor):
            return 'tensor'
        if isinstance(source, np.ndarray):
            return 'array'
        if isinstance(source, Image.Image):
            return 'pil'
        if isinstance(source, (list, tuple)):
            return 'list'
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                return 'directory'
            elif path.is_file() or path.exists():
                return 'file'
            else:
                return 'path_not_exists'
        return 'unknown'

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


            if isinstance(source, (list, tuple)) and len(source) == 2:

                rgb_type = self._detect_input_type(source[0])
                x_type = self._detect_input_type(source[1])


                if rgb_type == 'directory' and x_type == 'directory':
                    input_info['inference_format'] = 'directory_batch'
                    strict_match = getattr(self.args, 'strict_match', True)
                    matcher = MultiModalSourceMatcher(source[0], source[1], strict_match=strict_match)
                    matched_pairs = matcher.match()
                    input_info['batch_size'] = len(matched_pairs)
                    input_info['is_batch'] = True
                    input_info['matched_pairs'] = matched_pairs
                    input_info['validation_passed'] = True
                    LOGGER.info(f"目录批量推理: 匹配到 {len(matched_pairs)} 对图片")
                    return matched_pairs, input_info


                elif rgb_type == 'list' and x_type == 'list':
                    input_info['inference_format'] = 'list_batch'
                    strict_match = getattr(self.args, 'strict_match', True)
                    matched_pairs = MultiModalSourceMatcher.match_lists(source[0], source[1], strict_match=strict_match)
                    input_info['batch_size'] = len(matched_pairs)
                    input_info['is_batch'] = True
                    input_info['matched_pairs'] = matched_pairs
                    input_info['validation_passed'] = True
                    LOGGER.info(f"列表批量推理: {len(matched_pairs)} 对图片")
                    return matched_pairs, input_info


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

                if self.is_single_modal and len(source) == 2:
                    raise ValueError("单模态模式下不接受双输入 [rgb, x]；请仅提供单一路径/图像，或去掉 modality 参数。")
                
                if len(source) == 2 and self.is_dual_modal:

                    input_info['inference_format'] = 'dual_modal_list'
                    rgb_source, x_source = source
                    

                    rgb_info = self._analyze_single_source(rgb_source, 'rgb')
                    x_info = self._analyze_single_source(x_source, 'x_modal')
                    
                    input_info['rgb_source'] = rgb_info
                    input_info['x_source'] = x_info
                    input_info['validation_passed'] = True
                    

                    return source, input_info
                    
                elif len(source) == 1 and self.is_single_modal:

                    input_info['inference_format'] = 'single_modal_list'
                    single_source = source[0]
                    LOGGER.debug(f"单模态输入(列表包装): {type(single_source)}")
                    

                    source_info = self._analyze_single_source(single_source, self.modality)
                    input_info.update(source_info)
                    input_info['validation_passed'] = True
                    

                    return single_source, input_info
                elif len(source) > 2:

                    input_info['inference_format'] = 'batch_inference'
                    input_info['is_batch'] = True
                    
                    if self.is_dual_modal:

                        if len(source) % 2 != 0:
                            raise ValueError(f"双模态批量推理需要偶数个输入源，但接收到{len(source)}个")
                        

                        pairs = [(source[i], source[i+1]) for i in range(0, len(source), 2)]
                        input_info['batch_size'] = len(pairs)
                        LOGGER.info(f"双模态批量推理: {input_info['batch_size']}对图像")
                        input_info['validation_passed'] = True
                        return pairs, input_info
                    else:

                        input_info['batch_size'] = len(source)
                        LOGGER.info(f"单模态批量推理: {input_info['batch_size']}张图像")
                        input_info['validation_passed'] = True
                        return source, input_info
                        
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
                

                input_info['inference_format'] = 'single_modal_source'
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






        LOGGER.debug("=== 输入解析分析报告 ===")
        LOGGER.debug(f"输入类型: {input_info['input_type']}")
        LOGGER.debug(f"推理格式: {input_info['inference_format']}")
        LOGGER.debug(f"模态模式: {input_info['modality_mode']}")
        LOGGER.debug(f"源数量: {input_info['source_count']}")
        LOGGER.debug(f"批量推理: {input_info['is_batch']}")
        LOGGER.debug(f"验证通过: {input_info['validation_passed']}")
        
        if 'batch_size' in input_info:
            LOGGER.debug(f"批量大小: {input_info['batch_size']}")
            
        if 'rgb_source' in input_info:
            LOGGER.debug(f"RGB源信息: {input_info['rgb_source']}")
            
        if 'x_source' in input_info:
            LOGGER.debug(f"X模态源信息: {input_info['x_source']}")
            
        if 'error' in input_info:
            LOGGER.debug(f"错误信息: {input_info['error']}")

        LOGGER.debug("=== 分析报告结束 ===")

    def _batch_inference_with_progress(
        self,
        matched_pairs: List[Tuple[Path, Path]],
        save_subdir: bool = True
    ) -> List:










        results = []
        failed = []


        original_save_dir = self.save_dir

        for rgb_path, x_path in tqdm(matched_pairs, desc="Batch inference", unit="pair"):
            pair_stem = rgb_path.stem

            try:

                if save_subdir and self.args.save:
                    pair_save_dir = original_save_dir / pair_stem
                    pair_save_dir.mkdir(parents=True, exist_ok=True)
                    self.save_dir = pair_save_dir


                pair_result = self._infer_single_pair(rgb_path, x_path)
                results.extend(pair_result if isinstance(pair_result, list) else [pair_result])

            except Exception as e:
                LOGGER.warning(f"推理失败，跳过: {pair_stem} - {e}")
                failed.append((rgb_path, x_path, str(e)))
                continue
            finally:

                self.save_dir = original_save_dir


        total = len(matched_pairs)
        success = total - len(failed)
        if failed:
            LOGGER.warning(f"批量推理完成: {success}/{total} 成功，{len(failed)} 失败")
            for rgb_path, x_path, error in failed[:5]:
                LOGGER.warning(f"  - {rgb_path.stem}: {error}")
            if len(failed) > 5:
                LOGGER.warning(f"  ... 还有 {len(failed) - 5} 个失败")
        else:
            LOGGER.info(f"批量推理完成: {total}/{total} 全部成功")

        return results

    def _infer_single_pair(self, rgb_path: Path, x_path: Path) -> List:











        source = [str(rgb_path), str(x_path)]


        self.seen = 0
        self.batch = None


        self.setup_source(source)


        results = []
        for result in self.stream_inference(source):
            results.append(result)

        return results

    def preprocess(self, im):




















        try:

            LOGGER.debug(f"开始多模态预处理: modality={self.modality}, input_type={type(im)}")
            

            parsed_source, input_info = self._parse_inference_input(im)
            

            if isinstance(parsed_source, torch.Tensor) and parsed_source.dim() == 4 and parsed_source.shape[1] == 6:
                LOGGER.debug("输入已为6通道tensor，进行格式验证后直接返回")
                return self._finalize_tensor(parsed_source)
            

            if input_info['inference_format'] in ['dual_modal_list', 'batch_inference'] and self.is_dual_modal:
                result_tensor = self._process_dual_modality(parsed_source)
            elif input_info['inference_format'] in ['single_modal_source', 'single_modal_list'] and self.is_single_modal:
                result_tensor = self._process_single_modality(parsed_source)
            else:

                if self.is_dual_modal:
                    result_tensor = self._process_dual_modality(parsed_source)
                else:
                    result_tensor = self._process_single_modality(parsed_source)
            

            final_tensor = self._finalize_tensor(result_tensor)
            
            LOGGER.debug(f"多模态预处理完成: shape={final_tensor.shape}, device={final_tensor.device}")
            return final_tensor
            
        except Exception as e:

            error_msg = f"多模态预处理失败: {str(e)}"
            LOGGER.error(error_msg)
            self._log_debug_info(im, e)
            raise RuntimeError(error_msg) from e
    
    def _process_dual_modality(self, im):













        if isinstance(im, torch.Tensor) and im.shape[1] == 6:
            LOGGER.debug("输入已为6通道tensor，直接返回")
            return im
        

        rgb_images, x_images = self._parse_dual_modal_input(im)
        

        rgb_tensor = super().preprocess(rgb_images)
        x_tensor = super().preprocess(x_images)


        rgb_tensor, x_tensor = self._align_tensor_dimensions(rgb_tensor, x_tensor)



        combined_tensor = torch.cat([rgb_tensor, x_tensor], dim=1)

        LOGGER.debug(f"双模态预处理完成: {combined_tensor.shape}")
        return combined_tensor
    
    def _parse_dual_modal_input(self, im):















        if isinstance(im, (list, tuple)):

            if len(im) > 2 and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in im):

                LOGGER.debug(f"解析批量双模态输入: {len(im)}对图像")
                
                rgb_sources = []
                x_sources = []
                
                for i, (rgb_source, x_source) in enumerate(im):
                    try:

                        rgb_data, rgb_meta = self._integrate_with_load_inference_source(rgb_source)
                        x_data, x_meta = self._integrate_with_load_inference_source(x_source)
                        
                        rgb_sources.append(rgb_data)
                        x_sources.append(x_data)
                        
                        LOGGER.debug(f"批量[{i}] RGB: {rgb_meta.get('dataset_type', 'direct')}, "
                                   f"X: {x_meta.get('dataset_type', 'direct')}")
                        
                    except Exception as e:
                        LOGGER.error(f"批量双模态输入[{i}]处理失败: {e}")
                        raise
                
                return rgb_sources, x_sources
                
            elif len(im) == 2:

                rgb_source, x_source = im
                LOGGER.debug(f"解析标准双模态输入: RGB={type(rgb_source)}, X={type(x_source)}")
                

                rgb_data, rgb_meta = self._integrate_with_load_inference_source(rgb_source)
                x_data, x_meta = self._integrate_with_load_inference_source(x_source)
                


                

                if hasattr(rgb_data, '__iter__') and hasattr(rgb_data, 'source_type'):

                    rgb_images = self._extract_images_from_dataset(rgb_data)
                else:

                    rgb_images = rgb_data if isinstance(rgb_data, list) else [rgb_data]
                
                if hasattr(x_data, '__iter__') and hasattr(x_data, 'source_type'):

                    x_images = self._extract_images_from_dataset(x_data)
                else:

                    x_images = x_data if isinstance(x_data, list) else [x_data]
                
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
    
    def _extract_images_from_dataset(self, dataset):









        images = []
        
        try:
            for batch_idx, batch in enumerate(dataset):
                if isinstance(batch, (list, tuple)):

                    if len(batch) > 1:
                        batch_images = batch[1]
                        
                        if isinstance(batch_images, torch.Tensor):

                            batch_np = batch_images.cpu().numpy()
                            
                            if batch_np.ndim == 4:
                                for img in batch_np:

                                    images.append(img.transpose(1, 2, 0))
                            elif batch_np.ndim == 3:
                                images.append(batch_np.transpose(1, 2, 0))
                        
                        elif isinstance(batch_images, np.ndarray):

                            if batch_images.ndim == 4:
                                images.extend(list(batch_images))
                            elif batch_images.ndim == 3:
                                images.append(batch_images)
                

                if batch_idx == 0:
                    break
                    
        except Exception as e:
            LOGGER.error(f"从数据集提取图像失败: {e}")
            raise
        
        if not images:
            raise ValueError("无法从数据集中提取图像")
        
        LOGGER.debug(f"从数据集成功提取{len(images)}张图像")
        return images
    
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
            

            try:
                dataset = load_inference_source(source_path)
                LOGGER.debug(f"使用load_inference_source加载: {source_path}")
                

                images = []
                for batch in dataset:
                    if isinstance(batch, (list, tuple)):

                        if len(batch) > 1 and hasattr(batch[1], 'shape'):

                            batch_images = batch[1]
                            if isinstance(batch_images, torch.Tensor):

                                batch_images = batch_images.cpu().numpy()
                            
                            LOGGER.debug(f"load_inference_source返回的数据格式: {batch_images.shape}, dtype={batch_images.dtype}")
                            if batch_images.ndim == 4:
                                for i, img in enumerate(batch_images):
                                    LOGGER.debug(f"批处理图像[{i}]格式: {img.shape}")

                                    if img.shape[0] in [1, 3]:
                                        LOGGER.debug(f"检测到CHW格式，执行transpose")
                                        images.append(img.transpose(1, 2, 0))
                                    else:
                                        LOGGER.debug(f"检测到HWC格式，直接使用")
                                        images.append(img)
                            elif batch_images.ndim == 3:
                                LOGGER.debug(f"单张图像格式: {batch_images.shape}")

                                if batch_images.shape[0] in [1, 3]:
                                    LOGGER.debug(f"检测到CHW格式，执行transpose")
                                    images.append(batch_images.transpose(1, 2, 0))
                                else:
                                    LOGGER.debug(f"检测到HWC格式，直接使用")
                                    images.append(batch_images)
                    break
                
                if images:
                    LOGGER.debug(f"通过load_inference_source成功加载{len(images)}张图像")
                    return images
                    
            except Exception as e:
                LOGGER.warning(f"load_inference_source加载失败，使用备用方法: {e}")
            

            img = cv2.imread(str(source_path))
            if img is None:
                raise ValueError(f"无法加载图像: {source_path}")
            

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            LOGGER.debug(f"使用OpenCV备用方法成功加载: {source_path}")
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
            

        elif hasattr(source, '__iter__') and hasattr(source, 'source_type'):
            LOGGER.debug("处理数据集对象输入")
            
            images = []
            for batch in source:
                if isinstance(batch, (list, tuple)) and len(batch) > 1:
                    batch_images = batch[1]
                    if isinstance(batch_images, torch.Tensor):

                        loaded = self._load_image_source(batch_images)
                        images.extend(loaded)
                break
            
            return images
            
        else:
            raise TypeError(f"不支持的图像源类型: {type(source)}")

    def _integrate_with_load_inference_source(self, source):












        from ultralytics.data.build import check_source
        
        try:

            checked_source, webcam, screenshot, from_img, in_memory, tensor = check_source(source)
            

            source_metadata = {
                'original_source': source,
                'checked_source': checked_source,
                'is_webcam': webcam,
                'is_screenshot': screenshot,
                'from_img': from_img,
                'in_memory': in_memory,
                'is_tensor': tensor
            }
            
            LOGGER.debug(f"源检查结果: webcam={webcam}, screenshot={screenshot}, "
                        f"from_img={from_img}, in_memory={in_memory}, tensor={tensor}")
            

            if tensor:

                return checked_source, source_metadata
                
            elif in_memory or from_img:

                loaded_images = self._load_image_source(checked_source)
                return loaded_images, source_metadata
                
            else:

                dataset = load_inference_source(checked_source)
                source_metadata['dataset_type'] = type(dataset).__name__
                return dataset, source_metadata
                
        except Exception as e:
            LOGGER.warning(f"load_inference_source集成失败，使用标准加载: {e}")

            loaded_images = self._load_image_source(source)
            source_metadata = {
                'original_source': source,
                'fallback_used': True,
                'error': str(e)
            }
            return loaded_images, source_metadata
    
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
    
    def _validate_input_modality_consistency(self, im):









        if self.is_dual_modal:

            if not isinstance(im, (list, tuple)) or len(im) != 2:
                if not (isinstance(im, torch.Tensor) and im.shape[1] == 6):
                    raise ValueError(
                        f"双模态推理需要包含2个元素的列表输入 [rgb_source, x_source] "
                        f"或6通道tensor，但接收到: {type(im)}"
                    )
        else:

            if isinstance(im, (list, tuple)) and len(im) > 1:
                LOGGER.warning(
                    f"单模态推理模式({self.modality})接收到多个输入源，将仅使用第一个: {im[0]}"
                )
    
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
    
    def get_preprocessing_info(self):






        return {
            'modality': self.modality,
            'is_dual_modal': self.is_dual_modal,
            'is_single_modal': self.is_single_modal,
            'supported_modalities': list(self.SUPPORTED_MODALITIES),
            'expected_input_channels': 6,
            'device': getattr(self, 'device', 'not_set')
        }

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
                        result_string += self.write_results(i, result_path, im, result_string)
                    

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
        

        results = super().postprocess(preds, img, orig_imgs)
        

        if self.is_dual_modal and hasattr(self, '_dual_input_detected') and self._dual_input_detected:

            if hasattr(self, 'batch') and self.batch and len(self.batch[0]) == 2 and len(results) > 1:


                return [results[0]]
        

        return results
    
    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list) -> str:

















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
            if not hasattr(self, '_orig_imgs_cache') or self._orig_imgs_cache is None:
                raise RuntimeError("未找到原始图像缓存，无法生成以原图为背景的可视化")
            oi = self._orig_imgs_cache
            rgb_t, x_t = None, None
            if isinstance(oi, (list, tuple)):
                if len(oi) == 2:
                    rgb_t = _np_to_tensor3ch(oi[0])
                    x_t = _np_to_tensor3ch(oi[1])
                elif len(oi) == 1:
                    if self.modality and str(self.modality).lower() == 'rgb':
                        rgb_t = _np_to_tensor3ch(oi[0])
                    else:
                        x_t = _np_to_tensor3ch(oi[0])
                else:
                    raise RuntimeError(f"原始图像数量异常: {len(oi)}")
            else:

                if self.modality and str(self.modality).lower() == 'rgb':
                    rgb_t = _np_to_tensor3ch(oi)
                else:
                    x_t = _np_to_tensor3ch(oi)
            return rgb_t, x_t


        def _reproject_to_target_norm(boxes_xywh_px: torch.Tensor, orig_hw: tuple[int, int], target_h: int, target_w: int) -> torch.Tensor:
            if boxes_xywh_px is None or boxes_xywh_px.numel() == 0:
                return torch.zeros((0, 4), dtype=torch.float32)
            oh, ow = float(orig_hw[0]), float(orig_hw[1])
            sx, sy = float(target_w) / ow, float(target_h) / oh
            b = boxes_xywh_px.clone().float()
            b[:, 0] *= sx
            b[:, 2] *= sx
            b[:, 1] *= sy
            b[:, 3] *= sy
            b[:, 0] /= float(target_w)
            b[:, 2] /= float(target_w)
            b[:, 1] /= float(target_h)
            b[:, 3] /= float(target_h)
            return _clip_norm_xywh(b, 0.0, 1.0, 0.0, 1.0)


        def _resolve_x_modality_strict():

            if self.is_single_modal and self.modality and self.modality.lower() == 'rgb':
                return 'rgb'
            return 'x'


        result = self.results[i]
        base = p.stem


        rgb_tensor, x_tensor = _get_orig_modal_tensors()


        n_boxes = 0 if result.boxes is None else len(result.boxes)
        if n_boxes:
            cls_ids = result.boxes.cls.long()
            boxes_px = result.boxes.xywh
            orig_h, orig_w = result.boxes.orig_shape
            confs = getattr(result.boxes, 'conf', None)

            if not isinstance(cls_ids, torch.Tensor):
                cls_ids = torch.as_tensor(cls_ids, dtype=torch.long)
            if boxes_px is not None and not isinstance(boxes_px, torch.Tensor):
                boxes_px = torch.as_tensor(boxes_px, dtype=torch.float32)
            if confs is not None and not isinstance(confs, torch.Tensor):
                confs = torch.as_tensor(confs, dtype=torch.float32)
            batch_idx = ensure_batch_idx_long(torch.zeros(cls_ids.shape[0]))
        else:

            cls_ids = torch.zeros((0,), dtype=torch.long)
            boxes_px = torch.zeros((0, 4), dtype=torch.float32)
            orig_h, orig_w = 1, 1
            confs = torch.zeros((0,), dtype=torch.float32)
            batch_idx = ensure_batch_idx_long(torch.zeros((0,), dtype=torch.long))

        names = getattr(self.model, 'names', {})


        x_modality = _resolve_x_modality_strict()


        if self.is_single_modal:
            if self.modality.lower() == 'rgb':

                if rgb_tensor is None:
                    raise RuntimeError("期望RGB原图用于可视化，但缓存缺失")
                Ht, Wt = int(rgb_tensor.shape[-2]), int(rgb_tensor.shape[-1])
                boxes_norm_rgb = _reproject_to_target_norm(boxes_px, (orig_h, orig_w), Ht, Wt)
                fname_rgb = self.save_dir / f"pred_{base}_labels_rgb.jpg"
                plot_images(rgb_tensor, batch_idx, cls_ids, boxes_norm_rgb, confs=confs,
                            paths=[str(p)], fname=fname_rgb, names=names)
            else:

                if x_tensor is None:
                    raise RuntimeError("期望X原图用于可视化，但缓存缺失")
                Ht, Wt = int(x_tensor.shape[-2]), int(x_tensor.shape[-1])
                boxes_norm_x = _reproject_to_target_norm(boxes_px, (orig_h, orig_w), Ht, Wt)
                fname_x = self.save_dir / f"pred_{base}_labels_{x_modality}.jpg"
                plot_images(x_tensor, batch_idx, cls_ids, boxes_norm_x, confs=confs,
                            paths=[str(p.with_name(f"{base}_{x_modality}{p.suffix}"))],
                            fname=fname_x, names=names)
            return string



        if rgb_tensor is None or x_tensor is None:
            raise RuntimeError("双模态可视化需要RGB与X原图，但缓存缺失")
        Hr, Wr = int(rgb_tensor.shape[-2]), int(rgb_tensor.shape[-1])
        boxes_norm_rgb = _reproject_to_target_norm(boxes_px, (orig_h, orig_w), Hr, Wr)
        fname_rgb = self.save_dir / f"pred_{base}_labels_rgb.jpg"
        plot_images(rgb_tensor, batch_idx, cls_ids, boxes_norm_rgb, confs=confs,
                    paths=[str(p)], fname=fname_rgb, names=names)


        Hx, Wx = int(x_tensor.shape[-2]), int(x_tensor.shape[-1])
        boxes_norm_x = _reproject_to_target_norm(boxes_px, (orig_h, orig_w), Hx, Wx)
        fname_x = self.save_dir / f"pred_{base}_labels_{x_modality}.jpg"
        plot_images(x_tensor, batch_idx, cls_ids, boxes_norm_x, confs=confs,
                    paths=[str(p.with_name(f"{base}_{x_modality}{p.suffix}"))],
                    fname=fname_x, names=names)


        if (Hr, Wr) != (Hx, Wx):

            x_tensor_resized = torch.nn.functional.interpolate(x_tensor, size=(Hr, Wr), mode='bilinear', align_corners=False)
        else:
            x_tensor_resized = x_tensor
        side = concat_side_by_side(rgb_tensor, x_tensor_resized)
        batch_dup, cls_dup, boxes_dup, confs_dup = duplicate_bboxes_for_side_by_side(batch_idx, cls_ids, boxes_norm_rgb, confs)
        fname_mm = self.save_dir / f"pred_{base}_labels_multimodal.jpg"
        plot_images(side, batch_dup, cls_dup, boxes_dup, confs=confs_dup,
                    paths=[str(p.with_name(f"{base}_multimodal{p.suffix}"))],
                    fname=fname_mm, names=names)

        return string
    
































    
























    
    def _separate_modalities(self, tensor: torch.Tensor) -> tuple:









        if tensor.dim() == 3:


            rgb_tensor = tensor[:3]
            x_tensor = tensor[3:]
        else:


            rgb_tensor = tensor[:, :3]
            x_tensor = tensor[:, 3:]

        return rgb_tensor, x_tensor
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:












        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.dim() == 4:

            tensor = tensor[0]
            

        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        

        img = tensor.numpy()
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = img.transpose(1, 2, 0)
        


        if img.shape[2] == 3:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
        

        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        

        if img.ndim == 2 or img.shape[2] == 1:
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
        

        title = "Multi-Modal Detection Results"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        title_x = (combined_width - title_size[0]) // 2
        cv2.putText(combined, title, (title_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        return combined
    
    def preprocess(self, im):







        self.modality = getattr(self.args, 'modality', None)
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None


        if isinstance(im, (list, tuple)) and len(im) == 2:
            self._dual_input_detected = True
            self.input_mode = 'dual'
        else:
            self._dual_input_detected = False
            self.input_mode = f'single_{self.modality}' if self.modality else 'single'


        if self.is_single_modal and isinstance(im, (list, tuple)) and len(im) == 2:
            raise ValueError(
                "单模态模式下不接受双输入 [rgb, x]；请仅提供单一路径/图像，或去掉 modality 参数。"
            )


        self._set_runtime_modality_for_router()


        if isinstance(im, (list, tuple)) and len(im) == 2:
            return self._process_dual_modality(im)


        return super().preprocess(im)

