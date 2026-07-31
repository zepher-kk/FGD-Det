from __future__ import annotations




import torch
import torch.nn as nn
from ultralytics.utils import LOGGER
from .filling import generate_modality_filling, adapt_xch


class MultiModalRouter:













    

    runtime_modality = None
    runtime_strategy = None
    runtime_seed = None

    def __init__(self, config_dict=None, verbose=True):


        x_channels = 3
        self.x_modality_type = 'unknown'
        if config_dict and 'dataset_config' in config_dict:
            dataset_config = config_dict['dataset_config']
            x_channels = dataset_config.get('Xch', 3)
            self.x_modality_type = dataset_config.get('x_modality', 'unknown')
        
        self.INPUT_SOURCES = {
            'RGB': 3,
            'X': x_channels,
            'Dual': 3 + x_channels
        }
        

        self.has_multimodal_config = self._detect_multimodal_config(config_dict)
        
        self.verbose = verbose
        self.original_spatial_size = None
        self.original_inputs = {}


        self.runtime_modality = None
        self.runtime_strategy = None
        self.runtime_seed = None
        
        if self.verbose:
            LOGGER.info("MultiModal: router initialized")
            LOGGER.info(f"MultiModal: RGB=3ch, X={self.x_modality_type}({x_channels}ch), Dual={3 + x_channels}ch")
            LOGGER.info(f"MultiModal: multimodal_config_detected={self.has_multimodal_config}")

    def set_runtime_params(self, modality: str | None, strategy: str | None = None, seed: int | None = None):

        self.runtime_modality = (modality.lower() if isinstance(modality, str) else None)
        self.runtime_strategy = strategy
        self.runtime_seed = seed


    def __setstate__(self, state):

        self.__dict__.update(state)
        if 'runtime_modality' not in self.__dict__:
            self.runtime_modality = None
        if 'runtime_strategy' not in self.__dict__:
            self.runtime_strategy = None
        if 'runtime_seed' not in self.__dict__:
            self.runtime_seed = None



        self.original_spatial_size = None
        self.original_inputs = {'RGB': None, 'X': None, 'Dual': None}

    def __getstate__(self):








        state = dict(self.__dict__)
        state['original_spatial_size'] = None
        state['original_inputs'] = {'RGB': None, 'X': None, 'Dual': None}
        return state

    def _ensure_runtime_defaults(self):

        if not hasattr(self, 'runtime_modality'):
            self.runtime_modality = None
        if not hasattr(self, 'runtime_strategy'):
            self.runtime_strategy = None
        if not hasattr(self, 'runtime_seed'):
            self.runtime_seed = None
    
    def parse_layer_config(self, layer_config, layer_index, ch, verbose=True):
















        if len(layer_config) > 5:
            raise ValueError(
                f"Layer {layer_index}: layer_config has {len(layer_config)} fields, but only 5 are supported "
                f"(from, repeats, module, args, input_source). "
                f"The 6th-field HOOK system has been removed. "
                f"Please update your model YAML to remove the 6th field."
            )


        if len(layer_config) >= 5:
            f, n, m, args, mm_input_source = layer_config[:5]
        else:
            f, n, m, args = layer_config[:4]
            mm_input_source = None
            
        mm_attributes = {}
        

        if mm_input_source and mm_input_source in self.INPUT_SOURCES:

            c1 = self.INPUT_SOURCES[mm_input_source]
            

            mm_attributes = {
                '_mm_input_source': mm_input_source,
                '_mm_layer_index': layer_index,
                '_mm_version': 'v1.0',
                '_mm_x_modality': self.x_modality_type
            }
            

            if mm_input_source == 'X' and f == -1:
                mm_attributes['_mm_new_input_start'] = True

                mm_attributes['_mm_spatial_reset'] = True

                if verbose:
                    LOGGER.info(f"MultiModal Layer {layer_index}: X模态新输入起点 (from=-1被重定向)")
                    LOGGER.info(f"MultiModal Layer {layer_index}: 空间重置标记已设置 (尺寸将从输入动态获取)")
            
            if verbose:
                if mm_input_source == 'RGB':
                    LOGGER.info(f"MultiModal Layer {layer_index}: '{m.__name__ if hasattr(m, '__name__') else m}' ← RGB模态输入 ({c1}通道)")
                elif mm_input_source == 'X':
                    LOGGER.info(f"MultiModal Layer {layer_index}: '{m.__name__ if hasattr(m, '__name__') else m}' ← X模态({self.x_modality_type})输入 ({c1}通道)")
                else:
                    LOGGER.info(f"MultiModal Layer {layer_index}: '{m.__name__ if hasattr(m, '__name__') else m}' ← RGB+X双模态输入 ({c1}通道)")
        else:


            if isinstance(f, list):
                if len(f) == 1:
                    f_idx = f[0]
                    c1 = ch[f_idx] if f_idx != -1 else ch[-1]
                else:

                    c1 = sum(ch[i] if i != -1 else ch[-1] for i in f)
            else:
                c1 = ch[f] if f != -1 else ch[-1]
            
        return c1, mm_input_source, mm_attributes
    
    def setup_multimodal_routing(self, x, profile=False):











        self._ensure_runtime_defaults()

        routing_enabled = False
        input_sources = None
        

        expected_dual_channels = self.INPUT_SOURCES['Dual']
        x_channels = self.INPUT_SOURCES['X']
        is_dual_channel_input = x.shape[1] == expected_dual_channels
        is_multimodal_config = self.has_multimodal_config
        
        if is_dual_channel_input:
            routing_enabled = True
            self.original_spatial_size = (x.shape[2], x.shape[3])
            rgb = x[:, :3, :, :]
            xmod = x[:, 3:3 + x_channels, :, :]

            rm = self.runtime_modality
            if rm is None:
                dual = x
            elif rm == 'rgb':
                filled_x = generate_modality_filling(rgb, 'rgb', 'x', strategy=self.runtime_strategy)
                filled_x = adapt_xch(filled_x, x_channels)

                dual = torch.cat([rgb, filled_x], dim=1)
                xmod = filled_x
            else:
                filled_rgb = generate_modality_filling(xmod, 'x', 'rgb', strategy=self.runtime_strategy)
                filled_rgb = adapt_xch(filled_rgb, 3)

                dual = torch.cat([filled_rgb, xmod], dim=1)
                rgb = filled_rgb

            input_sources = {
                'RGB': rgb,
                'X': xmod,
                'Dual': dual,
            }
            self.cache_original_inputs(input_sources)
                           
        elif is_multimodal_config and x.shape[1] == 3:
            routing_enabled = True
            self.original_spatial_size = (x.shape[2], x.shape[3])
            rm = self.runtime_modality
            if rm is None:

                rgb = x
                xmod = x.clone()
                dual = x
            elif rm == 'rgb':
                rgb = x
                xmod = generate_modality_filling(rgb, 'rgb', 'x', strategy=self.runtime_strategy)
                xmod = adapt_xch(xmod, x_channels)

                dual = torch.cat([rgb, xmod], dim=1)
            else:
                xmod = x
                if xmod.shape[1] != x_channels:
                    xmod = adapt_xch(xmod, x_channels)
                rgb = generate_modality_filling(xmod, 'x', 'rgb', strategy=self.runtime_strategy)
                rgb = adapt_xch(rgb, 3)

                dual = torch.cat([rgb, xmod], dim=1)

            input_sources = {
                'RGB': rgb,
                'X': xmod,
                'Dual': dual,
            }
            self.cache_original_inputs(input_sources)
        
        return routing_enabled, input_sources
    
    def route_layer_input(self, x, module, input_sources, profile=False):












        if not hasattr(module, '_mm_input_source'):
            return None
            

        if not input_sources:
            if profile:
                LOGGER.warning(f"⚠️  MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - 输入源不可用")
            return None
            
        mm_input_source = module._mm_input_source



        if hasattr(module, '_mm_new_input_start') and module._mm_new_input_start:

            if 'X' not in input_sources:
                if profile:
                    LOGGER.warning(f"⚠️  MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                                  f"X模态新输入起点需要X输入源")
                return None
                
            routed_x = input_sources['X']


            expected_x_channels = self.INPUT_SOURCES['X']
            if routed_x.shape[1] != expected_x_channels:
                if profile:
                    LOGGER.error(f"❌ MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                                f"X模态新输入起点期望{expected_x_channels}通道，但接收到{routed_x.shape[1]}通道")
                    LOGGER.error("MultiModal: 当前输入源状态:")
                    for k, v in input_sources.items():
                        LOGGER.error(f"   {k}: {v.shape}")
                return None

            if profile:
                x_modality = getattr(module, '_mm_x_modality', 'unknown')
                LOGGER.info(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - X模态({x_modality})新输入起点"
                )
                LOGGER.info(f"MultiModal: 输入切换 {x.shape} → {routed_x.shape}")
        else:

            if mm_input_source not in input_sources:
                if profile:
                    LOGGER.warning(f"⚠️  MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                                  f"请求的模态 '{mm_input_source}' 不存在于输入源中")
                return None
                
            routed_x = input_sources[mm_input_source]

            if profile:
                LOGGER.info(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 路由到 '{mm_input_source}' - 输入形状: {x.shape} → {routed_x.shape}"
                )
        

        if routed_x is None:
            if profile:
                LOGGER.warning(f"⚠️  MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                              f"路由结果为None")
            return None
            
        return routed_x
    
    def set_module_attributes(self, module, mm_attributes):

        for attr_name, attr_value in mm_attributes.items():
            setattr(module, attr_name, attr_value)
            
    def get_original_spatial_size(self):

        return self.original_spatial_size
    
    def cache_original_inputs(self, input_sources):







        self.original_inputs = {
            'RGB': input_sources['RGB'] if 'RGB' in input_sources else None,
            'X': input_sources['X'] if 'X' in input_sources else None,
            'Dual': input_sources['Dual'] if 'Dual' in input_sources else None
        }
        
    def get_original_x_input(self, target_size=None):









        if 'X' not in self.original_inputs or self.original_inputs['X'] is None:
            return None
            
        x_input = self.original_inputs['X']
        


        if target_size and target_size != x_input.shape[2:4]:

            pass
            
        return x_input
        
    def reset_spatial_input(self, x, module, mm_input_sources, profile=False):












        if not hasattr(module, '_mm_new_input_start') or not module._mm_new_input_start:
            return x
            

        if not mm_input_sources or 'X' not in mm_input_sources:
            if profile:
                LOGGER.warning(
                    f"⚠️ MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置失败 - 缺少X模态输入源"
                )
            return x
            

        if self.original_spatial_size is None:
            if profile:
                LOGGER.warning(
                    f"⚠️ MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置失败 - 无法获取原始尺寸"
                )
            return x
            

        reset_x = mm_input_sources['X']
        
        if profile:
            LOGGER.info(f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置完成")
            LOGGER.info(f"MultiModal: 尺寸重置 {x.shape} → {reset_x.shape}")
            
        return reset_x

    def update_dataset_config(self, dataset_config):






        if not dataset_config:
            return


        if 'Xch' in dataset_config:
            x_channels = int(dataset_config['Xch'])
            self.INPUT_SOURCES['X'] = x_channels
            self.INPUT_SOURCES['Dual'] = 3 + x_channels
            if self.verbose:
                LOGGER.info(f"MultiModal: 更新X模态通道数为 {x_channels}")
                LOGGER.info(f"MultiModal: 更新后路由配置: RGB(3ch), X({x_channels}ch), Dual({3 + x_channels}ch)")


        x_mod = None

        if isinstance(dataset_config, dict):
            x_mod = dataset_config.get('x_modality', None)

            if not x_mod:
                mods = dataset_config.get('modality_used') or dataset_config.get('models')
                if isinstance(mods, (list, tuple)):
                    non_rgb = [m for m in mods if isinstance(m, str) and m.lower() != 'rgb']
                    if non_rgb:
                        x_mod = non_rgb[0]

            if not x_mod:
                mod_map = dataset_config.get('modality') or dataset_config.get('modalities')
                if isinstance(mod_map, dict):
                    non_rgb = [k for k in mod_map.keys() if isinstance(k, str) and k.lower() != 'rgb']
                    if non_rgb:
                        x_mod = non_rgb[0]

        if x_mod:
            self.x_modality_type = str(x_mod)
    
    def _detect_multimodal_config(self, config_dict):









        if not config_dict:
            return False


        all_layers = config_dict.get('backbone', []) + config_dict.get('head', [])

        for layer_config in all_layers:
            if len(layer_config) >= 5:
                mm_input_source = layer_config[4]
                if mm_input_source in self.INPUT_SOURCES:
                    return True

        return False

