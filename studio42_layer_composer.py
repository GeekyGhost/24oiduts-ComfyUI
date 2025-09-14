import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import math
import logging
from typing import Tuple, Optional, List
import cv2

# Configure logging
logger = logging.getLogger(__name__)

class Studio42LayerComposer:
    """
    🎬 Studio42 Layer Composer - ComfyUI Compatible Version
    
    Professional layer composition with proper ComfyUI IMAGE/MASK handling.
    
    FIXED: Rotation transparency issues - no more black boxes when rotating!
    - Uses direct pixel coordinates (x_position, y_position) like other Studio42 nodes
    - (0,0) is top-left corner, same as PatchLift/PatchDrop
    - Proper RGBA handling for rotation transparency
    
    Features:
    - IMAGE format: [B,H,W,C] where C=3 (RGB only)
    - MASK format: [B,H,W] (single channel, 0-1 values)
    - Direct pixel positioning matching other Studio42 nodes
    - Professional blend modes and transformations
    - Rotation with proper transparency (no black boxes!)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        blend_modes = [
            "normal", "multiply", "screen", "overlay", "soft_light", 
            "hard_light", "color_dodge", "color_burn", "darken", 
            "lighten", "difference", "exclusion", "add", "subtract",
            "divide", "linear_burn", "linear_dodge"
        ]
        
        position_modes = [
            "custom", "center", "top_left", "top_center", "top_right",
            "middle_left", "middle_right", "bottom_left", 
            "bottom_center", "bottom_right"
        ]
        
        fit_modes = [
            "none", "fit_width", "fit_height", "fit_both", "fill", "stretch"
        ]
        
        return {
            "required": {
                "background": ("IMAGE",),
                "foreground": ("IMAGE",),
                "blend_mode": (blend_modes, {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # FIXED: Use same coordinate system as other Studio42 nodes
                "foreground_mask": ("MASK",),
                
                # Position - now matches PatchLift/PatchDrop coordinate system
                "x_position": ("INT", {"default": 0, "min": -4000, "max": 4000, "step": 1,
                              "tooltip": "X coordinate (pixels from left, 0 = left edge)"}),
                "y_position": ("INT", {"default": 0, "min": -4000, "max": 4000, "step": 1,
                              "tooltip": "Y coordinate (pixels from top, 0 = top edge)"}),
                "position_mode": (position_modes, {"default": "custom",
                                "tooltip": "Positioning mode - 'custom' uses x/y coordinates directly"}),
                
                # Transform and scaling
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01}),
                "fit_mode": (fit_modes, {"default": "none"}),
                "maintain_aspect": ("BOOLEAN", {"default": True}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                
                # Anchor point (for rotation and scaling center)
                "anchor_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                           "tooltip": "Rotation/scale anchor X (0.0=left, 0.5=center, 1.0=right)"}),
                "anchor_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                           "tooltip": "Rotation/scale anchor Y (0.0=top, 0.5=center, 1.0=bottom)"}),
                
                # Flipping
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "flip_vertical": ("BOOLEAN", {"default": False}),
                
                # Edge processing
                "feather_edges": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "anti_aliasing": ("BOOLEAN", {"default": True}),
                
                # Clipping
                "clip_to_background": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composite_image", "composite_mask")
    FUNCTION = "compose_layers"
    CATEGORY = "🎬 Studio42/Composition"
    
    def compose_layers(self, background, foreground, blend_mode="normal", opacity=1.0,
                      foreground_mask=None, x_position=0, y_position=0, position_mode="custom",
                      scale_x=1.0, scale_y=1.0, fit_mode="none", maintain_aspect=True, 
                      rotation=0.0, anchor_x=0.5, anchor_y=0.5, flip_horizontal=False, 
                      flip_vertical=False, feather_edges=0, anti_aliasing=True, 
                      clip_to_background=True):
        
        try:
            # Get batch sizes
            batch_size_bg = background.shape[0]
            batch_size_fg = foreground.shape[0]
            max_batch_size = max(batch_size_bg, batch_size_fg)
            
            # Handle foreground mask batch size if provided
            if foreground_mask is not None:
                batch_size_mask = foreground_mask.shape[0]
                max_batch_size = max(max_batch_size, batch_size_mask)
            
            results = []
            masks = []
            
            # Process each frame
            for i in range(max_batch_size):
                # Get current frame (cycle if batch sizes don't match)
                bg_tensor = background[i % batch_size_bg]
                fg_tensor = foreground[i % batch_size_fg]
                
                # Get mask if provided (ComfyUI MASK format: [H,W])
                fg_mask_tensor = None
                if foreground_mask is not None:
                    fg_mask_tensor = foreground_mask[i % foreground_mask.shape[0]]
                
                # Convert ComfyUI tensors to PIL Images
                bg_pil = self._tensor_to_pil(bg_tensor)
                fg_pil = self._tensor_to_pil(fg_tensor)
                
                # Convert mask tensor to PIL if provided
                fg_mask_pil = None
                if fg_mask_tensor is not None:
                    fg_mask_pil = self._mask_tensor_to_pil(fg_mask_tensor)
                
                # Perform single layer composition
                composite, comp_mask = self._compose_single_layer(
                    bg_pil, fg_pil, fg_mask_pil, x_position, y_position, position_mode,
                    scale_x, scale_y, fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
                    flip_horizontal, flip_vertical, blend_mode, opacity, feather_edges, 
                    anti_aliasing, clip_to_background
                )
                
                # Convert back to ComfyUI tensors
                composite_tensor = self._pil_to_tensor(composite)
                mask_tensor = self._pil_to_mask_tensor(comp_mask)
                
                results.append(composite_tensor)
                masks.append(mask_tensor)
            
            # Stack results
            composite_result = torch.cat(results, dim=0)
            mask_result = torch.stack(masks, dim=0)
            
            logger.info(f"✅ Layer composition completed: {max_batch_size} frames, mode: {blend_mode}")
            
            return (composite_result, mask_result)
            
        except Exception as e:
            logger.error(f"❌ Layer composition failed: {e}")
            # Return background as fallback with full opacity mask
            batch_size = background.shape[0]
            fallback_mask = torch.ones(batch_size, background.shape[1], background.shape[2], dtype=torch.float32)
            return (background, fallback_mask)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI IMAGE tensor [H,W,C] to PIL Image (RGB)"""
        # Convert to numpy and scale to 0-255
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Ensure RGB format (ComfyUI IMAGE is always RGB with C=3)
        if len(np_img.shape) == 2:
            np_img = np.stack([np_img, np_img, np_img], axis=-1)
        elif np_img.shape[-1] == 4:  # Should not happen in ComfyUI IMAGE, but handle it
            np_img = np_img[:, :, :3]
        
        return Image.fromarray(np_img, 'RGB')
    
    def _mask_tensor_to_pil(self, mask_tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI MASK tensor [H,W] to PIL Image (L)"""
        # Convert to numpy and scale to 0-255
        mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(mask_np, 'L')
    
    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to ComfyUI IMAGE tensor [1,H,W,C]"""
        # Ensure RGB mode
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Convert to numpy and normalize to 0-1
        np_img = np.array(pil_img).astype(np.float32) / 255.0
        
        # Add batch dimension and convert to tensor
        return torch.from_numpy(np_img).unsqueeze(0)
    
    def _pil_to_mask_tensor(self, pil_mask: Image.Image) -> torch.Tensor:
        """Convert PIL mask to ComfyUI MASK tensor [H,W]"""
        # Convert to grayscale if needed
        if pil_mask.mode != 'L':
            pil_mask = pil_mask.convert('L')
        
        # Convert to numpy and normalize to 0-1
        mask_np = np.array(pil_mask).astype(np.float32) / 255.0
        
        # Convert to tensor (no batch dimension for individual mask)
        return torch.from_numpy(mask_np)
    
    def _compose_single_layer(self, background: Image.Image, foreground: Image.Image,
                             foreground_mask: Optional[Image.Image], x_position: int, y_position: int,
                             position_mode: str, scale_x: float, scale_y: float, fit_mode: str,
                             maintain_aspect: bool, rotation: float, anchor_x: float, anchor_y: float,
                             flip_horizontal: bool, flip_vertical: bool, blend_mode: str, opacity: float,
                             feather_edges: int, anti_aliasing: bool, clip_to_background: bool) -> Tuple[Image.Image, Image.Image]:
        """Compose a single foreground layer onto background"""
        
        # Get dimensions
        bg_w, bg_h = background.size
        
        # Transform foreground layer
        transformed_fg, transformed_mask = self._transform_layer(
            foreground, foreground_mask, (bg_w, bg_h), scale_x, scale_y, 
            fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
            flip_horizontal, flip_vertical, anti_aliasing
        )
        
        # Calculate final position using direct coordinates (like PatchDrop)
        final_x, final_y = self._calculate_position_fixed(
            transformed_fg.size, (bg_w, bg_h), position_mode, x_position, y_position
        )
        
        # Create composition canvas
        canvas = Image.new('RGBA', (bg_w, bg_h), (0, 0, 0, 0))
        composite_mask = Image.new('L', (bg_w, bg_h), 0)
        
        # Prepare foreground with alpha
        fg_rgba = transformed_fg.convert('RGBA') if transformed_fg.mode != 'RGBA' else transformed_fg.copy()
        
        if transformed_mask:
            # Apply mask to alpha channel
            if feather_edges > 0:
                transformed_mask = self._feather_mask(transformed_mask, feather_edges)
            fg_rgba.putalpha(transformed_mask)
        elif opacity < 1.0:
            # Apply opacity to existing alpha channel
            if fg_rgba.mode == 'RGBA':
                existing_alpha = fg_rgba.split()[3]
                # Multiply existing alpha by opacity
                alpha_array = np.array(existing_alpha, dtype=np.float32) * opacity
                new_alpha = Image.fromarray(alpha_array.astype(np.uint8), mode='L')
                fg_rgba.putalpha(new_alpha)
            else:
                # Create new alpha channel with opacity
                alpha = Image.new('L', fg_rgba.size, int(255 * opacity))
                fg_rgba.putalpha(alpha)
        # If no mask and opacity is 1.0, preserve existing alpha from rotation
        
        # Paste foreground onto canvas with proper bounds checking
        if clip_to_background:
            # Clip to background bounds
            paste_x = max(0, min(final_x, bg_w))
            paste_y = max(0, min(final_y, bg_h))
            
            # Adjust foreground if it extends beyond bounds
            clip_left = max(0, -final_x)
            clip_top = max(0, -final_y)
            clip_right = min(fg_rgba.width, bg_w - final_x)
            clip_bottom = min(fg_rgba.height, bg_h - final_y)
            
            if clip_left < clip_right and clip_top < clip_bottom:
                fg_clipped = fg_rgba.crop((clip_left, clip_top, clip_right, clip_bottom))
                canvas.paste(fg_clipped, (paste_x, paste_y), fg_clipped)
                
                # Update composite mask
                if fg_clipped.mode == 'RGBA':
                    fg_alpha = fg_clipped.split()[3]
                    composite_mask.paste(fg_alpha, (paste_x, paste_y))
        else:
            # Allow positioning outside background bounds
            canvas.paste(fg_rgba, (final_x, final_y), fg_rgba)
            if fg_rgba.mode == 'RGBA':
                fg_alpha = fg_rgba.split()[3]
                composite_mask.paste(fg_alpha, (final_x, final_y))
        
        # Apply blend mode
        bg_rgba = background.convert('RGBA')
        if blend_mode == "normal":
            result = Image.alpha_composite(bg_rgba, canvas)
        else:
            result = self._apply_blend_mode(bg_rgba, canvas, blend_mode)
        
        # FIXED: Convert result back to RGB preserving transparency properly
        # Instead of using black background, preserve the original background
        result_rgb = background.copy()
        if result.mode == 'RGBA':
            # Only paste where there's actual content (non-transparent areas)
            alpha_mask = result.split()[3]
            result_rgb.paste(result.convert('RGB'), mask=alpha_mask)
        else:
            result_rgb = result.convert('RGB')
        
        return result_rgb, composite_mask
    
    def _calculate_position_fixed(self, layer_size: Tuple[int, int], canvas_size: Tuple[int, int],
                                 position_mode: str, x_position: int, y_position: int) -> Tuple[int, int]:
        """Calculate position using same coordinate system as PatchLift/PatchDrop"""
        
        layer_w, layer_h = layer_size
        canvas_w, canvas_h = canvas_size
        
        if position_mode == "custom":
            # Direct pixel coordinates - same as PatchLift/PatchDrop
            return (x_position, y_position)
        
        # For other modes, calculate base position then apply as offset
        position_map = {
            "top_left": (0, 0),
            "top_center": ((canvas_w - layer_w) // 2, 0),
            "top_right": (canvas_w - layer_w, 0),
            "middle_left": (0, (canvas_h - layer_h) // 2),
            "center": ((canvas_w - layer_w) // 2, (canvas_h - layer_h) // 2),
            "middle_right": (canvas_w - layer_w, (canvas_h - layer_h) // 2),
            "bottom_left": (0, canvas_h - layer_h),
            "bottom_center": ((canvas_w - layer_w) // 2, canvas_h - layer_h),
            "bottom_right": (canvas_w - layer_w, canvas_h - layer_h),
        }
        
        base_x, base_y = position_map.get(position_mode, (0, 0))
        
        # Add position offset to base position
        final_x = base_x + x_position
        final_y = base_y + y_position
        
        return (final_x, final_y)
    
    def _transform_layer(self, layer: Image.Image, mask: Optional[Image.Image], 
                        canvas_size: Tuple[int, int], scale_x: float, scale_y: float,
                        fit_mode: str, maintain_aspect: bool, rotation: float,
                        anchor_x: float, anchor_y: float, flip_horizontal: bool, 
                        flip_vertical: bool, anti_aliasing: bool) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Apply transformations to layer and mask"""
        
        # Store original mode to preserve RGBA if present
        original_mode = layer.mode
        
        # Apply fitting first
        if fit_mode != "none":
            layer, mask = self._apply_fitting(layer, mask, canvas_size, fit_mode)
        else:
            # Apply manual scaling
            if scale_x != 1.0 or scale_y != 1.0:
                if maintain_aspect:
                    scale = min(scale_x, scale_y)
                    scale_x = scale_y = scale
                
                new_size = (int(layer.width * scale_x), int(layer.height * scale_y))
                resample = Image.Resampling.LANCZOS if anti_aliasing else Image.Resampling.NEAREST
                
                layer = layer.resize(new_size, resample)
                if mask:
                    mask = mask.resize(new_size, resample)
        
        # Apply flipping
        if flip_horizontal:
            layer = layer.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if mask:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        if flip_vertical:
            layer = layer.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if mask:
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        # Apply rotation around anchor point (this will convert to RGBA for transparency)
        if abs(rotation) > 0.01:
            layer, mask = self._rotate_around_anchor(layer, mask, rotation, anchor_x, anchor_y, anti_aliasing)
        
        return layer, mask
    
    def _rotate_around_anchor(self, layer: Image.Image, mask: Optional[Image.Image], 
                             rotation: float, anchor_x: float, anchor_y: float, 
                             anti_aliasing: bool) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Rotate image around specified anchor point with proper transparency handling"""
        
        # Skip rotation if angle is too small
        if abs(rotation) < 0.01:
            return layer, mask
        
        # Calculate anchor point in pixels
        anchor_px = int(layer.width * anchor_x)
        anchor_py = int(layer.height * anchor_y)
        
        resample = Image.Resampling.BICUBIC if anti_aliasing else Image.Resampling.NEAREST
        
        # FIXED: Convert to RGBA before rotation to ensure proper transparency
        layer_rgba = layer.convert('RGBA') if layer.mode != 'RGBA' else layer.copy()
        
        # Rotate with transparent fill (RGBA ensures this works properly)
        layer_rotated = layer_rgba.rotate(
            rotation, 
            expand=True, 
            resample=resample, 
            fillcolor=(0, 0, 0, 0)  # Fully transparent
        )
        
        # Handle mask rotation
        mask_rotated = None
        if mask:
            # Ensure mask is grayscale
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            mask_rotated = mask.rotate(
                rotation, 
                expand=True, 
                resample=resample, 
                fillcolor=0  # Black fill for mask (transparent areas)
            )
        
        return layer_rotated, mask_rotated
    
    def _apply_fitting(self, layer: Image.Image, mask: Optional[Image.Image], 
                      canvas_size: Tuple[int, int], fit_mode: str) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Apply fitting modes to layer and mask preserving transparency"""
        layer_w, layer_h = layer.size
        canvas_w, canvas_h = canvas_size
        
        if fit_mode == "fit_width":
            scale = canvas_w / layer_w
            new_size = (canvas_w, int(layer_h * scale))
        elif fit_mode == "fit_height":
            scale = canvas_h / layer_h
            new_size = (int(layer_w * scale), canvas_h)
        elif fit_mode == "fit_both":
            scale = min(canvas_w / layer_w, canvas_h / layer_h)
            new_size = (int(layer_w * scale), int(layer_h * scale))
        elif fit_mode == "fill":
            scale = max(canvas_w / layer_w, canvas_h / layer_h)
            new_size = (int(layer_w * scale), int(layer_h * scale))
        elif fit_mode == "stretch":
            new_size = canvas_size
        else:
            return layer, mask
        
        # Preserve original mode during resize
        resample_method = Image.Resampling.LANCZOS
        layer = layer.resize(new_size, resample_method)
        if mask:
            mask = mask.resize(new_size, resample_method)
        
        return layer, mask
    
    def _feather_mask(self, mask: Image.Image, feather_pixels: int) -> Image.Image:
        """Apply feathering to mask edges"""
        if feather_pixels <= 0:
            return mask
        
        # Convert to numpy for OpenCV processing
        mask_np = np.array(mask)
        
        # Apply Gaussian blur for feathering
        kernel_size = feather_pixels * 2 + 1
        blurred = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), feather_pixels / 3)
        
        return Image.fromarray(blurred, mode='L')
    
    def _apply_blend_mode(self, background: Image.Image, foreground: Image.Image, 
                         blend_mode: str) -> Image.Image:
        """Apply blend modes using PIL and numpy operations"""
        
        # Convert to numpy arrays for mathematical operations
        bg_array = np.array(background, dtype=np.float32) / 255.0
        fg_array = np.array(foreground, dtype=np.float32) / 255.0
        
        # Extract RGB and alpha channels
        bg_rgb = bg_array[:, :, :3]
        fg_rgb = fg_array[:, :, :3]
        fg_alpha = fg_array[:, :, 3:4] if fg_array.shape[2] == 4 else np.ones_like(fg_rgb[:, :, 0:1])
        
        # Apply blend mode
        if blend_mode == "multiply":
            blended = bg_rgb * fg_rgb
        elif blend_mode == "screen":
            blended = 1 - (1 - bg_rgb) * (1 - fg_rgb)
        elif blend_mode == "overlay":
            blended = np.where(bg_rgb < 0.5,
                              2 * bg_rgb * fg_rgb,
                              1 - 2 * (1 - bg_rgb) * (1 - fg_rgb))
        elif blend_mode == "soft_light":
            blended = np.where(fg_rgb < 0.5,
                              bg_rgb - (1 - 2 * fg_rgb) * bg_rgb * (1 - bg_rgb),
                              bg_rgb + (2 * fg_rgb - 1) * (np.sqrt(np.maximum(bg_rgb, 1e-10)) - bg_rgb))
        elif blend_mode == "hard_light":
            blended = np.where(fg_rgb < 0.5,
                              2 * bg_rgb * fg_rgb,
                              1 - 2 * (1 - bg_rgb) * (1 - fg_rgb))
        elif blend_mode == "color_dodge":
            blended = np.where(fg_rgb >= 1.0, 1.0, 
                              np.minimum(1.0, bg_rgb / np.maximum(1e-10, 1 - fg_rgb)))
        elif blend_mode == "color_burn":
            blended = np.where(fg_rgb <= 0.0, 0.0, 
                              1 - np.minimum(1.0, (1 - bg_rgb) / np.maximum(1e-10, fg_rgb)))
        elif blend_mode == "darken":
            blended = np.minimum(bg_rgb, fg_rgb)
        elif blend_mode == "lighten":
            blended = np.maximum(bg_rgb, fg_rgb)
        elif blend_mode == "difference":
            blended = np.abs(bg_rgb - fg_rgb)
        elif blend_mode == "exclusion":
            blended = bg_rgb + fg_rgb - 2 * bg_rgb * fg_rgb
        elif blend_mode == "add":
            blended = np.minimum(1.0, bg_rgb + fg_rgb)
        elif blend_mode == "subtract":
            blended = np.maximum(0.0, bg_rgb - fg_rgb)
        elif blend_mode == "divide":
            blended = np.minimum(1.0, bg_rgb / np.maximum(1e-10, fg_rgb))
        elif blend_mode == "linear_burn":
            blended = np.maximum(0.0, bg_rgb + fg_rgb - 1)
        elif blend_mode == "linear_dodge":
            blended = np.minimum(1.0, bg_rgb + fg_rgb)
        else:  # normal
            blended = fg_rgb
        
        # Alpha blend the result
        result_rgb = bg_rgb * (1 - fg_alpha) + blended * fg_alpha
        
        # Combine with background alpha
        bg_alpha = bg_array[:, :, 3:4] if bg_array.shape[2] == 4 else np.ones_like(bg_rgb[:, :, 0:1])
        result_alpha = bg_alpha * (1 - fg_alpha) + fg_alpha
        
        # Combine RGB and alpha
        result_array = np.concatenate([result_rgb, result_alpha], axis=2)
        
        # Convert back to PIL Image
        result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array, 'RGBA')


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Studio42LayerComposer": Studio42LayerComposer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Studio42LayerComposer": "🎬 Studio42 Layer Composer"
}
