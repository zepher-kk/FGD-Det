# Ultralytics Multimodal Config Parser
# Universal YAML configuration parsing for RGB+X architectures
# Version: v1.0

from ultralytics.utils import LOGGER


class MultiModalConfigParser:
    """
    Universal Multimodal Configuration Parser

    Handles YAML configuration parsing for both YOLO and RTDETR
    with RGB+X multimodal extensions (5-field layer config only).
    """

    def __init__(self):
        self.supported_input_sources = ['RGB', 'X', 'Dual']

    def validate_config_format(self, config):
        """Validate multimodal configuration format correctness"""

        rgb_layers = []
        x_layers = []
        dual_layers = []

        for section in ['backbone', 'head']:
            for i, layer_config in enumerate(config.get(section, [])):
                if len(layer_config) == 5:  # Has 5th field
                    input_source = layer_config[4]
                    if input_source == 'RGB':
                        rgb_layers.append(i)
                    elif input_source == 'X':
                        x_layers.append(i)
                    elif input_source == 'Dual':
                        dual_layers.append(i)

        LOGGER.info("MultiModal: config validation complete")
        LOGGER.info(f"MultiModal: RGB routed={len(rgb_layers)}, X routed={len(x_layers)}, Dual routed={len(dual_layers)}")

        return {
            'rgb_layers': rgb_layers,
            'x_layers': x_layers,
            'dual_layers': dual_layers,
            'total_routing_layers': len(rgb_layers) + len(x_layers) + len(dual_layers)
        }

    def extract_multimodal_info(self, config):
        """Extract multimodal information from configuration"""

        # Get X modality type from dataset config
        x_modality_type = config.get('dataset_config', {}).get('x_modality', 'unknown')

        # Count multimodal layers
        mm_layer_count = 0
        for section in ['backbone', 'head']:
            for layer_config in config.get(section, []):
                if len(layer_config) >= 5 and layer_config[4] in self.supported_input_sources:
                    mm_layer_count += 1

        return {
            'x_modality_type': x_modality_type,
            'mm_layer_count': mm_layer_count,
            'supports_multimodal': mm_layer_count > 0
        }

    def parse_config(self, config: dict) -> dict:
        """
        Build a minimal multimodal model_config dict for MultiModalRouter.

        Detects whether YAML has any 5th-field multimodal routing markers.
        Returns original config with helper flags added; this function keeps
        backward-compatibility by not enforcing any schema beyond what's required
        by the current router implementation.
        """
        has_mm = False
        input_layers = []
        for section in ['backbone', 'head']:
            for i, layer_config in enumerate(config.get(section, [])):
                if len(layer_config) >= 5 and layer_config[4] in self.supported_input_sources:
                    has_mm = True
                    input_layers.append((section, i))
        out = dict(config)
        out['has_multimodal_layers'] = has_mm
        out['input_layers'] = input_layers
        return out
