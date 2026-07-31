from __future__ import annotations
"""
Visualization exceptions for YOLOMM multimodal model.
"""

class LayerNotSpecifiedError(ValueError):

    
    def __init__(self, model_layers=None, message=None):







        if message is None:
            if model_layers is not None:
                message = (
                    f"layers parameter is required for visualization.\n"
                    f"The model has {model_layers} layers (valid indices: 0-{model_layers-1}).\n"
                    f"Please specify which layers to visualize.\n"
                    f"Example usage:\n"
                    f"  - Single layer: model.vis(source='image.jpg', layers=[5])\n"
                    f"  - Multiple layers: model.vis(source='image.jpg', layers=[3, 5, 7])\n"
                    f"  - Range of layers: model.vis(source='image.jpg', layers=list(range(3, 8)))"
                )
            else:
                message = (
                    "layers parameter is required for visualization.\n"
                    "Please specify which layers to visualize as a list of layer indices.\n"
                    "Example: model.vis(source='image.jpg', layers=[5])"
                )
        
        super().__init__(message)
        self.model_layers = model_layers

class EmptyLayersError(ValueError):

    
    def __init__(self):
        message = (
            "layers parameter cannot be an empty list.\n"
            "Please specify at least one layer index to visualize.\n"
            "Example: model.vis(source='image.jpg', layers=[5])"
        )
        super().__init__(message)

class InvalidLayerIndexError(ValueError):

    
    def __init__(self, invalid_indices, valid_range):







        message = (
            f"Invalid layer indices: {invalid_indices}\n"
            f"Valid layer indices are in range [{valid_range[0]}, {valid_range[1]}].\n"
            f"Please check your layer indices and try again."
        )
        super().__init__(message)
        self.invalid_indices = invalid_indices
        self.valid_range = valid_range
