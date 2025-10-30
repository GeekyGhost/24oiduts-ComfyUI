"""
LCARS Base Node Framework
--------------------------
Easy-to-use base classes for creating custom ComfyUI nodes
with beautiful LCARS + Hitchhiker's Guide aesthetics.

Makes node creation as simple as defining inputs and a process function!
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Callable
from abc import ABC, abstractmethod


class LCARSBaseNode(ABC):
    """
    Base class for all LCARS-themed nodes.
    Provides common utilities and standardizes the interface.

    To create a new node, inherit from this class and implement:
    - define_inputs(): Return INPUT_TYPES dict
    - process(): Your main processing logic

    Optionally override:
    - define_outputs(): Customize RETURN_TYPES and RETURN_NAMES
    - category(): Set custom category path
    """

    # Default outputs (can be overridden)
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"

    @classmethod
    @abstractmethod
    def define_inputs(cls) -> Dict[str, Any]:
        """
        Define node inputs. Return a dict with 'required' and 'optional' keys.

        Example:
            return {
                "required": {
                    "image": ("IMAGE",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                },
                "optional": {
                    "mask": ("MASK",),
                }
            }
        """
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """ComfyUI required method - calls define_inputs()"""
        return cls.define_inputs()

    @classmethod
    def define_outputs(cls) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """
        Override to customize outputs.
        Returns: (RETURN_TYPES, RETURN_NAMES)

        Example:
            return (("IMAGE", "MASK"), ("image", "mask"))
        """
        return (cls.RETURN_TYPES, cls.RETURN_NAMES)

    @classmethod
    def category(cls) -> str:
        """Override to set custom category. Default is Studio42/LCARS"""
        return "Studio42/LCARS"

    @property
    def CATEGORY(self):
        """ComfyUI required property"""
        return self.category()

    @abstractmethod
    def process(self, **kwargs) -> Tuple[Any, ...]:
        """
        Main processing logic. Receives all inputs as kwargs.
        Return outputs as tuple matching RETURN_TYPES.

        Example:
            def process(self, image, strength, mask=None):
                # Your processing here
                result = self.apply_effect(image, strength)
                return (result,)
        """
        pass

    def execute(self, **kwargs) -> Tuple[Any, ...]:
        """
        ComfyUI calls this. Wraps process() with error handling.
        Don't override this unless you need custom execution logic.
        """
        try:
            return self.process(**kwargs)
        except Exception as e:
            print(f"❌ Error in {self.__class__.__name__}: {str(e)}")
            raise

    # ==================== Utility Methods ====================

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor, batch_index: int = 0) -> Image.Image:
        """
        Convert ComfyUI tensor to PIL Image.

        Args:
            tensor: Shape (B, H, W, C) in range [0, 1]
            batch_index: Which image in batch to extract

        Returns:
            PIL Image in RGB or RGBA mode
        """
        img = tensor[batch_index].cpu().numpy()
        img = (img * 255).astype(np.uint8)

        if img.shape[-1] == 4:
            return Image.fromarray(img, 'RGBA')
        elif img.shape[-1] == 3:
            return Image.fromarray(img, 'RGB')
        elif img.shape[-1] == 1:
            return Image.fromarray(img.squeeze(-1), 'L')
        else:
            raise ValueError(f"Unsupported channel count: {img.shape[-1]}")

    @staticmethod
    def pil_to_tensor(pil_img: Image.Image, add_batch: bool = True) -> torch.Tensor:
        """
        Convert PIL Image to ComfyUI tensor.

        Args:
            pil_img: PIL Image
            add_batch: If True, adds batch dimension (B=1)

        Returns:
            Tensor shape (B, H, W, C) or (H, W, C) in range [0, 1]
        """
        # Convert to RGB if needed
        if pil_img.mode not in ['RGB', 'RGBA', 'L']:
            pil_img = pil_img.convert('RGB')

        # To numpy
        img = np.array(pil_img).astype(np.float32) / 255.0

        # Ensure 3D
        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        # To tensor
        tensor = torch.from_numpy(img)

        # Add batch dimension if requested
        if add_batch:
            tensor = tensor.unsqueeze(0)

        return tensor

    @staticmethod
    def batch_process(tensor: torch.Tensor,
                      process_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Process each image in a batch.

        Args:
            tensor: Input tensor (B, H, W, C)
            process_fn: Function that takes (H, W, C) and returns processed (H, W, C)

        Returns:
            Processed batch tensor (B, H, W, C)
        """
        batch_size = tensor.shape[0]
        results = []

        for i in range(batch_size):
            single = tensor[i]
            processed = process_fn(single)
            results.append(processed)

        return torch.stack(results, dim=0)

    @staticmethod
    def mask_to_tensor(mask: torch.Tensor, channels: int = 1) -> torch.Tensor:
        """
        Convert mask to image tensor format.

        Args:
            mask: Shape (B, H, W) in range [0, 1]
            channels: Number of channels (1, 3, or 4)

        Returns:
            Tensor shape (B, H, W, C)
        """
        # Add channel dimension
        mask = mask.unsqueeze(-1)

        # Repeat for RGB/RGBA if needed
        if channels > 1:
            mask = mask.repeat(1, 1, 1, channels)

        return mask

    @staticmethod
    def tensor_to_mask(tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert image tensor to mask.

        Args:
            tensor: Shape (B, H, W, C) in range [0, 1]

        Returns:
            Mask shape (B, H, W) in range [0, 1]
        """
        # If grayscale, just remove channel
        if tensor.shape[-1] == 1:
            return tensor.squeeze(-1)

        # If RGB, convert to grayscale
        # Standard luminance formula
        weights = torch.tensor([0.299, 0.587, 0.114], device=tensor.device)
        mask = (tensor * weights).sum(dim=-1)

        return mask


class LCARSImageNode(LCARSBaseNode):
    """
    Base class for image processing nodes.
    Convenience class with image-specific defaults.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def category(cls) -> str:
        return "Studio42/LCARS/Image"


class LCARSGeneratorNode(LCARSBaseNode):
    """
    Base class for generator nodes (no image input required).
    Perfect for procedural generation.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def category(cls) -> str:
        return "Studio42/LCARS/Generators"

    @staticmethod
    def create_blank_tensor(width: int, height: int,
                           channels: int = 3,
                           batch_size: int = 1,
                           fill_value: float = 0.0) -> torch.Tensor:
        """
        Create a blank tensor.

        Args:
            width: Image width
            height: Image height
            channels: Number of channels (1, 3, or 4)
            batch_size: Batch size
            fill_value: Initial value (0-1)

        Returns:
            Tensor shape (B, H, W, C) filled with fill_value
        """
        return torch.full((batch_size, height, width, channels),
                         fill_value,
                         dtype=torch.float32)


class LCARSModelNode(LCARSBaseNode):
    """
    Base class for AI model nodes (Qwen, Wan, etc.).
    Provides model loading and caching utilities.
    """

    # Class variable for model cache
    _model_cache: Dict[str, Any] = {}

    @classmethod
    def category(cls) -> str:
        return "Studio42/LCARS/Models"

    @classmethod
    def load_model(cls, model_name: str,
                   load_fn: Callable[[], Any],
                   force_reload: bool = False) -> Any:
        """
        Load model with caching.

        Args:
            model_name: Unique identifier for this model
            load_fn: Function that loads the model (called if not cached)
            force_reload: If True, reload even if cached

        Returns:
            Loaded model
        """
        cache_key = f"{cls.__name__}_{model_name}"

        if force_reload or cache_key not in cls._model_cache:
            print(f"🔄 Loading model: {model_name}")
            model = load_fn()
            cls._model_cache[cache_key] = model
            print(f"✅ Model loaded: {model_name}")
        else:
            print(f"📦 Using cached model: {model_name}")

        return cls._model_cache[cache_key]

    @classmethod
    def unload_model(cls, model_name: str):
        """Unload a specific model from cache"""
        cache_key = f"{cls.__name__}_{model_name}"
        if cache_key in cls._model_cache:
            del cls._model_cache[cache_key]
            print(f"🗑️  Unloaded model: {model_name}")

    @classmethod
    def clear_cache(cls):
        """Clear all cached models"""
        cls._model_cache.clear()
        print("🗑️  Cleared all model cache")


class LCARSComboNode(LCARSBaseNode):
    """
    Base class for nodes that combine multiple inputs/operations.
    """

    @classmethod
    def category(cls) -> str:
        return "Studio42/LCARS/Combo"

    @staticmethod
    def blend_images(image1: torch.Tensor,
                    image2: torch.Tensor,
                    alpha: float = 0.5,
                    mode: str = "normal") -> torch.Tensor:
        """
        Blend two images with various modes.

        Args:
            image1: Base image (B, H, W, C)
            image2: Blend image (B, H, W, C)
            alpha: Blend factor (0-1)
            mode: Blend mode (normal, multiply, screen, overlay, add)

        Returns:
            Blended image (B, H, W, C)
        """
        if mode == "normal":
            return image1 * (1 - alpha) + image2 * alpha

        elif mode == "multiply":
            blended = image1 * image2
            return image1 * (1 - alpha) + blended * alpha

        elif mode == "screen":
            blended = 1 - (1 - image1) * (1 - image2)
            return image1 * (1 - alpha) + blended * alpha

        elif mode == "overlay":
            mask = image1 < 0.5
            blended = torch.where(
                mask,
                2 * image1 * image2,
                1 - 2 * (1 - image1) * (1 - image2)
            )
            return image1 * (1 - alpha) + blended * alpha

        elif mode == "add":
            blended = torch.clamp(image1 + image2, 0, 1)
            return image1 * (1 - alpha) + blended * alpha

        else:
            raise ValueError(f"Unknown blend mode: {mode}")


# ==================== Helper Functions ====================

def create_simple_node(name: str,
                      category: str,
                      inputs: Dict[str, Any],
                      outputs: Tuple[Tuple[str, ...], Tuple[str, ...]],
                      process_fn: Callable) -> type:
    """
    Factory function to create a simple node class quickly.

    Args:
        name: Node class name
        category: Category path
        inputs: INPUT_TYPES dict
        outputs: (RETURN_TYPES, RETURN_NAMES)
        process_fn: Processing function

    Returns:
        Node class ready to register

    Example:
        MyNode = create_simple_node(
            name="MyCustomNode",
            category="Studio42/LCARS/Custom",
            inputs={
                "required": {
                    "image": ("IMAGE",),
                    "factor": ("FLOAT", {"default": 1.0}),
                }
            },
            outputs=(("IMAGE",), ("result",)),
            process_fn=lambda self, image, factor: (image * factor,)
        )
    """

    return type(name, (LCARSBaseNode,), {
        "define_inputs": classmethod(lambda cls: inputs),
        "define_outputs": classmethod(lambda cls: outputs),
        "category": classmethod(lambda cls: category),
        "RETURN_TYPES": outputs[0],
        "RETURN_NAMES": outputs[1],
        "process": process_fn,
    })


def register_node(node_class: type, display_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Helper to create node registration dicts.

    Args:
        node_class: The node class to register
        display_name: Optional display name (defaults to class name)

    Returns:
        Dict with NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS

    Example:
        mappings = register_node(MyNode, "✨ My Awesome Node")
        NODE_CLASS_MAPPINGS.update(mappings["class"])
        NODE_DISPLAY_NAME_MAPPINGS.update(mappings["display"])
    """
    class_name = node_class.__name__
    display = display_name or class_name

    return {
        "class": {class_name: node_class},
        "display": {class_name: display}
    }


# ==================== Example Node Template ====================

class ExampleLCARSNode(LCARSImageNode):
    """
    Example node showing how simple it is to create custom nodes!
    This node brightens or darkens an image.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    @classmethod
    def define_outputs(cls):
        return (("IMAGE",), ("image",))

    def process(self, image, brightness, mask=None):
        """Simply multiply image by brightness factor"""

        result = image * brightness

        # Apply mask if provided
        if mask is not None:
            mask_img = self.mask_to_tensor(mask, channels=image.shape[-1])
            result = image * (1 - mask_img) + result * mask_img

        # Clamp to valid range
        result = torch.clamp(result, 0.0, 1.0)

        return (result,)


# Export
__all__ = [
    'LCARSBaseNode',
    'LCARSImageNode',
    'LCARSGeneratorNode',
    'LCARSModelNode',
    'LCARSComboNode',
    'create_simple_node',
    'register_node',
    'ExampleLCARSNode',
]
