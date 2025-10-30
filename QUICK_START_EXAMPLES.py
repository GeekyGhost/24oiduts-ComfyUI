"""
🚀 LCARS ComfyUI - Quick Start Examples
========================================

This file contains ready-to-use examples for creating custom nodes.
Copy and paste these examples to get started quickly!

DON'T PANIC - Creating nodes is easy! 🌟
"""

# ==================== EXAMPLE 1: Simple Brightness Adjuster ====================
"""
This is the simplest possible image processing node.
It just multiplies the image by a brightness value.
"""

from lcars_base_node import LCARSImageNode
import torch


class SimpleBrightness(LCARSImageNode):
    """☀️ Simple Brightness Adjuster"""

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }

    def process(self, image, brightness):
        """Multiply image by brightness factor"""
        result = image * brightness
        result = torch.clamp(result, 0.0, 1.0)
        return (result,)


# ==================== EXAMPLE 2: Color Tint Node ====================
"""
This node adds a color tint to an image.
Shows how to work with individual RGB channels.
"""

from lcars_base_node import LCARSImageNode
import torch


class ColorTint(LCARSImageNode):
    """🎨 Color Tint"""

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "red_tint": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "green_tint": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "blue_tint": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }

    def process(self, image, red_tint, green_tint, blue_tint):
        """Apply color tint to each channel"""
        result = image.clone()
        result[:, :, :, 0] *= red_tint   # Red channel
        result[:, :, :, 1] *= green_tint # Green channel
        result[:, :, :, 2] *= blue_tint  # Blue channel
        result = torch.clamp(result, 0.0, 1.0)
        return (result,)


# ==================== EXAMPLE 3: Solid Color Generator ====================
"""
This is a generator node - it creates images from scratch.
No image input needed!
"""

from lcars_base_node import LCARSGeneratorNode


class SolidColorGenerator(LCARSGeneratorNode):
    """🌈 Solid Color Generator"""

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "red": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "green": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blue": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def process(self, width, height, red, green, blue):
        """Create a solid color image"""
        result = self.create_blank_tensor(width, height, channels=3)
        result[:, :, :, 0] = red
        result[:, :, :, 1] = green
        result[:, :, :, 2] = blue
        return (result,)


# ==================== EXAMPLE 4: Text Processor ====================
"""
This node works with text instead of images.
Shows how to process non-image data.
"""

from lcars_base_node import LCARSBaseNode


class TextProcessor(LCARSBaseNode):
    """📝 Text Processor"""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello World"}),
                "operation": (["Uppercase", "Lowercase", "Reverse", "Length"], {
                    "default": "Uppercase"
                }),
            }
        }

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Text"

    def process(self, text, operation):
        """Process text based on operation"""
        if operation == "Uppercase":
            result = text.upper()
        elif operation == "Lowercase":
            result = text.lower()
        elif operation == "Reverse":
            result = text[::-1]
        else:  # Length
            result = f"Length: {len(text)} characters"

        return (result,)


# ==================== EXAMPLE 5: Image Mixer ====================
"""
This node combines two images.
Shows how to work with multiple image inputs.
"""

from lcars_base_node import LCARSComboNode
import torch


class ImageMixer(LCARSComboNode):
    """🔀 Image Mixer"""

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "mix_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "blend_mode": (["Normal", "Multiply", "Screen", "Overlay"], {
                    "default": "Normal"
                }),
            }
        }

    def process(self, image1, image2, mix_ratio, blend_mode):
        """Mix two images together"""
        result = self.blend_images(image1, image2, mix_ratio, blend_mode)
        return (result,)


# ==================== EXAMPLE 6: Using PIL for Advanced Processing ====================
"""
This example shows how to use PIL (Python Imaging Library) for more complex operations.
"""

from lcars_base_node import LCARSImageNode
from PIL import Image, ImageFilter
import torch


class BlurNode(LCARSImageNode):
    """💫 Blur Effect"""

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
            }
        }

    def process(self, image, blur_radius):
        """Apply Gaussian blur using PIL"""
        # Convert tensor to PIL
        pil_image = self.tensor_to_pil(image)

        # Apply blur
        if blur_radius > 0:
            blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        else:
            blurred = pil_image

        # Convert back to tensor
        result = self.pil_to_tensor(blurred)

        return (result,)


# ==================== EXAMPLE 7: Batch Processing ====================
"""
This example shows how to process each image in a batch differently.
"""

from lcars_base_node import LCARSImageNode
import torch


class GradientBrightness(LCARSImageNode):
    """📊 Gradient Brightness (Batch)"""

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_brightness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "end_brightness": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 2.0}),
            }
        }

    def process(self, images, start_brightness, end_brightness):
        """Apply gradually changing brightness across batch"""
        batch_size = images.shape[0]
        results = []

        for i in range(batch_size):
            # Calculate brightness for this frame
            t = i / max(batch_size - 1, 1)
            brightness = start_brightness + (end_brightness - start_brightness) * t

            # Apply brightness
            frame = images[i] * brightness
            frame = torch.clamp(frame, 0.0, 1.0)
            results.append(frame)

        result = torch.stack(results, dim=0)
        return (result,)


# ==================== EXAMPLE 8: Using the Template Generator ====================
"""
The easiest way to create nodes - use the template generator!
"""

from lcars_node_template_generator import quick_image_node

# Generate a saturation adjuster node
saturation_code = quick_image_node(
    name="SaturationAdjuster",
    emoji="🎨",
    description="Adjust image saturation",
    extra_inputs={
        "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
    },
    process_code="""
        # Convert to HSV, adjust saturation, convert back
        import colorsys
        # (Simplified example - actual implementation would be more complex)
        result = image * saturation
        result = torch.clamp(result, 0.0, 1.0)
    """,
    output_path="saturation_adjuster.py"  # Will save to file
)


# ==================== EXAMPLE 9: Multiple Outputs ====================
"""
This node returns multiple outputs.
"""

from lcars_base_node import LCARSImageNode
import torch


class SplitRGB(LCARSImageNode):
    """🔴🟢🔵 Split RGB Channels"""

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("red", "green", "blue")

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    def process(self, image):
        """Split image into RGB channels"""
        # Create grayscale images for each channel
        red = image[:, :, :, 0:1].repeat(1, 1, 1, 3)
        green = image[:, :, :, 1:2].repeat(1, 1, 1, 3)
        blue = image[:, :, :, 2:3].repeat(1, 1, 1, 3)

        return (red, green, blue)


# ==================== EXAMPLE 10: Optional Inputs ====================
"""
This node has both required and optional inputs.
"""

from lcars_base_node import LCARSImageNode
import torch


class MaskedBrightness(LCARSImageNode):
    """💡 Masked Brightness"""

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 2.0}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    def process(self, image, brightness, mask=None):
        """Adjust brightness, optionally using a mask"""
        result = image * brightness
        result = torch.clamp(result, 0.0, 1.0)

        # If mask provided, blend original and result
        if mask is not None:
            mask_img = self.mask_to_tensor(mask, channels=3)
            result = image * (1 - mask_img) + result * mask_img

        return (result,)


# ==================== HOW TO REGISTER YOUR NODES ====================
"""
To make your nodes available in ComfyUI, add this at the end of your file:
"""

NODE_CLASS_MAPPINGS = {
    "SimpleBrightness": SimpleBrightness,
    "ColorTint": ColorTint,
    "SolidColorGenerator": SolidColorGenerator,
    "TextProcessor": TextProcessor,
    "ImageMixer": ImageMixer,
    "BlurNode": BlurNode,
    "GradientBrightness": GradientBrightness,
    "SplitRGB": SplitRGB,
    "MaskedBrightness": MaskedBrightness,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleBrightness": "☀️ Simple Brightness",
    "ColorTint": "🎨 Color Tint",
    "SolidColorGenerator": "🌈 Solid Color Generator",
    "TextProcessor": "📝 Text Processor",
    "ImageMixer": "🔀 Image Mixer",
    "BlurNode": "💫 Blur Effect",
    "GradientBrightness": "📊 Gradient Brightness",
    "SplitRGB": "🔴🟢🔵 Split RGB",
    "MaskedBrightness": "💡 Masked Brightness",
}


# ==================== TIPS & TRICKS ====================
"""
💡 TIPS:

1. **Use descriptive emojis** in your display names - makes nodes easy to find!

2. **Always clamp image outputs** to 0.0-1.0 range:
   result = torch.clamp(result, 0.0, 1.0)

3. **Handle batch processing** when needed:
   for i in range(image.shape[0]):
       frame = image[i]
       # process frame

4. **Use helper methods** from base classes:
   - self.tensor_to_pil(tensor)
   - self.pil_to_tensor(pil_img)
   - self.mask_to_tensor(mask)

5. **Set good defaults** for parameters - users appreciate sensible starting values!

6. **Add helpful descriptions** in docstrings - they help users understand your node.

7. **Test with different image sizes** - make sure your node works with various resolutions.

8. **Use graceful error handling** - print helpful error messages if something goes wrong.

9. **Organize nodes in categories** - use the category() method to group related nodes.

10. **Share your nodes!** - The community loves new creative tools!


🎨 STYLING TIPS:

- Nodes automatically get LCARS styling
- Use orange (#ff9900) as the primary color
- Purple for Qwen nodes, Blue for Wan nodes
- Emojis make nodes stand out in the menu


🚀 NEXT STEPS:

1. Copy one of these examples
2. Modify it for your use case
3. Save as a .py file in custom_nodes directory
4. Restart ComfyUI
5. Your node appears in the menu!


📚 LEARN MORE:

- Check LCARS_README.md for full documentation
- Explore lcars_base_node.py for all available methods
- Look at lcars_qwen_nodes.py, lcars_wan_nodes.py for advanced examples
- Use lcars_node_template_generator.py for quick generation


🎉 HAVE FUN CREATING!

Remember: DON'T PANIC - Node creation is easy and fun! 🌟
"""

# ==================== END OF EXAMPLES ====================

if __name__ == "__main__":
    print("🚀 LCARS ComfyUI Quick Start Examples")
    print("=" * 70)
    print("\n✨ This file contains 10 ready-to-use node examples!")
    print("\nTo use them:")
    print("1. Copy an example class")
    print("2. Save to a new .py file")
    print("3. Add NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS")
    print("4. Restart ComfyUI")
    print("\n💡 Or use the template generator for even easier creation:")
    print("   from lcars_node_template_generator import quick_image_node")
    print("\n🎉 Happy creating!")
