import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class Studio42PatchDrop:
    """
    🎬 Studio42 PatchDrop - Enhanced with Mask Support
    
    Advanced patch placement node for ComfyUI with proper mask handling.
    Takes an original image and a patch image, and drops the patch back 
    into its original location or a specified location on the original image.
    
    NEW FEATURES:
    ✅ Optional patch_mask input for transparency control
    ✅ ComfyUI IMAGE/MASK format compatibility
    ✅ Professional alpha compositing
    
    Features:
    - Automatic placement using stored coordinates
    - Manual position override  
    - Shape-aware placement (rectangle, circle, rounded rectangle)
    - Blend modes for seamless integration
    - Edge feathering for smooth transitions
    - Opacity control
    - Transform options (scale, rotation)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "patch_image": ("IMAGE",),
                "patch_x": ("INT", {"default": 0, "min": -2048, "max": 4096, "step": 1}),
                "patch_y": ("INT", {"default": 0, "min": -2048, "max": 4096, "step": 1}),
            },
            "optional": {
                # NEW: Mask input for transparency
                "patch_mask": ("MASK",),
                
                "patch_width": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
                "patch_height": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
                "patch_shape": (["rectangle", "circle", "rounded_rectangle"], {"default": "rectangle"}),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light",
                               "hard_light", "difference", "lighten", "darken"], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather_edges": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "corner_radius": ("INT", {"default": 20, "min": 5, "max": 100, "step": 5}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "position_mode": (["use_coordinates", "center", "manual"], {"default": "use_coordinates"}),
                "manual_x": ("INT", {"default": 0, "min": -2048, "max": 4096, "step": 1}),
                "manual_y": ("INT", {"default": 0, "min": -2048, "max": 4096, "step": 1}),
                "resize_patch": ("BOOLEAN", {"default": False}),
                "preserve_aspect": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("result_image", "placement_mask", "preview_image")
    FUNCTION = "drop_patch"
    CATEGORY = "🎬 Studio42/Image Processing"
    
    def drop_patch(self, original_image, patch_image, patch_x, patch_y,
                  patch_mask=None, patch_width=256, patch_height=256, patch_shape="rectangle",
                  blend_mode="normal", opacity=1.0, feather_edges=0, corner_radius=20,
                  scale=1.0, rotation=0.0, position_mode="use_coordinates",
                  manual_x=0, manual_y=0, resize_patch=False, preserve_aspect=True):
        
        try:
            results = []
            masks = []
            previews = []
            
            # Get batch sizes
            batch_size_orig = original_image.shape[0]
            batch_size_patch = patch_image.shape[0]
            max_batch_size = max(batch_size_orig, batch_size_patch)
            
            # Handle patch mask batch size if provided
            if patch_mask is not None:
                batch_size_mask = patch_mask.shape[0]
                max_batch_size = max(max_batch_size, batch_size_mask)
            
            # Process each image in the batch
            for i in range(max_batch_size):
                # Get images (cycle if batch sizes don't match)
                orig_tensor = original_image[i % batch_size_orig]
                patch_tensor = patch_image[i % batch_size_patch]
                
                # Get mask if provided
                mask_tensor = None
                if patch_mask is not None:
                    mask_tensor = patch_mask[i % batch_size_mask]
                
                # Convert to PIL Images
                orig_pil = self._tensor_to_pil(orig_tensor)
                patch_pil = self._tensor_to_pil(patch_tensor)
                
                # Convert mask to PIL if provided
                mask_pil = None
                if mask_tensor is not None:
                    mask_pil = self._mask_tensor_to_pil(mask_tensor)
                
                # Process single patch drop
                result_img, placement_mask, preview_img = self._drop_single_patch(
                    orig_pil, patch_pil, mask_pil, patch_x, patch_y, patch_width, patch_height,
                    patch_shape, blend_mode, opacity, feather_edges, corner_radius,
                    scale, rotation, position_mode, manual_x, manual_y, resize_patch, preserve_aspect
                )
                
                # Convert back to tensors (ensure RGB format)
                result_tensor = self._pil_to_tensor(result_img)
                preview_tensor = self._pil_to_tensor(preview_img)
                placement_mask_tensor = self._pil_to_mask_tensor(placement_mask)
                
                results.append(result_tensor)
                masks.append(placement_mask_tensor)
                previews.append(preview_tensor)
            
            # Stack results
            result_batch = torch.cat(results, dim=0)
            mask_batch = torch.stack(masks, dim=0)
            preview_batch = torch.cat(previews, dim=0)
            
            logger.info(f"✅ Patch drop completed: {max_batch_size} images processed")
            
            return (result_batch, mask_batch, preview_batch)
            
        except Exception as e:
            logger.error(f"❌ Patch drop failed: {e}")
            # Return original as fallback
            batch_size = original_image.shape[0]
            fallback_mask = torch.ones(batch_size, original_image.shape[1], original_image.shape[2], dtype=torch.float32)
            return (original_image, fallback_mask, original_image)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI IMAGE tensor to PIL Image (RGB)"""
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if len(np_img.shape) == 2:
            np_img = np.stack([np_img, np_img, np_img], axis=-1)
        elif np_img.shape[-1] == 4:
            np_img = np_img[:, :, :3]
        
        return Image.fromarray(np_img, 'RGB')
    
    def _mask_tensor_to_pil(self, mask_tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI MASK tensor to PIL Image (L)"""
        mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(mask_np, 'L')
    
    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to ComfyUI IMAGE tensor"""
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        np_img = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(np_img).unsqueeze(0)
    
    def _pil_to_mask_tensor(self, pil_mask: Image.Image) -> torch.Tensor:
        """Convert PIL mask to ComfyUI MASK tensor"""
        if pil_mask.mode != 'L':
            pil_mask = pil_mask.convert('L')
        
        mask_np = np.array(pil_mask).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np)
    
    def _drop_single_patch(self, original: Image.Image, patch: Image.Image, patch_mask: Optional[Image.Image],
                          patch_x: int, patch_y: int, patch_width: int, patch_height: int,
                          patch_shape: str, blend_mode: str, opacity: float,
                          feather_edges: int, corner_radius: int, scale: float,
                          rotation: float, position_mode: str, manual_x: int, manual_y: int,
                          resize_patch: bool, preserve_aspect: bool) -> Tuple[Image.Image, Image.Image, Image.Image]:
        
        # Ensure original image is RGB
        if original.mode != 'RGB':
            original = original.convert('RGB')
        
        # Determine final position
        if position_mode == "center":
            final_x = (original.width - patch_width) // 2
            final_y = (original.height - patch_height) // 2
        elif position_mode == "manual":
            final_x = manual_x
            final_y = manual_y
        else:  # use_coordinates
            final_x = patch_x
            final_y = patch_y
        
        # Process patch transformations
        processed_patch, processed_mask = self._process_patch_transforms(
            patch, patch_mask, patch_width, patch_height, scale, rotation, 
            resize_patch, preserve_aspect
        )
        
        # Get final patch dimensions after transformations
        final_patch_width, final_patch_height = processed_patch.size
        
        # Ensure patch fits within image bounds
        final_x = max(0, min(final_x, original.width - final_patch_width))
        final_y = max(0, min(final_y, original.height - final_patch_height))
        
        # Create placement mask based on shape
        placement_mask = self._create_placement_mask(
            original.size, final_x, final_y, final_patch_width, final_patch_height,
            patch_shape, corner_radius, feather_edges
        )
        
        # If we have a patch mask, combine it with placement mask
        if processed_mask:
            # Resize processed mask to match placement area
            if processed_mask.size != (final_patch_width, final_patch_height):
                processed_mask = processed_mask.resize(
                    (final_patch_width, final_patch_height), 
                    Image.Resampling.LANCZOS
                )
            
            # Create full-size mask canvas
            full_mask_canvas = Image.new('L', original.size, 0)
            full_mask_canvas.paste(processed_mask, (final_x, final_y))
            
            # Combine with placement mask
            placement_array = np.array(placement_mask, dtype=np.float32) / 255.0
            patch_mask_array = np.array(full_mask_canvas, dtype=np.float32) / 255.0
            
            # Multiply masks (intersection)
            combined_mask = (placement_array * patch_mask_array * 255).astype(np.uint8)
            placement_mask = Image.fromarray(combined_mask, mode='L')
        
        # Apply opacity to the final mask
        if opacity < 1.0:
            mask_array = np.array(placement_mask, dtype=np.float32) * opacity
            placement_mask = Image.fromarray(mask_array.astype(np.uint8), mode='L')
        
        # Create composite image
        result_image = self._composite_patch(
            original, processed_patch, final_x, final_y, 
            placement_mask, blend_mode
        )
        
        # Create preview image showing placement
        preview_image = self._create_preview_image(
            result_image, final_x, final_y, final_patch_width, final_patch_height,
            patch_shape, corner_radius
        )
        
        return result_image, placement_mask, preview_image
    
    def _process_patch_transforms(self, patch: Image.Image, patch_mask: Optional[Image.Image], 
                                 target_width: int, target_height: int, scale: float, rotation: float, 
                                 resize_patch: bool, preserve_aspect: bool) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Apply transformations to the patch and its mask"""
        
        processed_patch = patch.copy()
        processed_mask = patch_mask.copy() if patch_mask else None
        
        # Resize if requested
        if resize_patch:
            if preserve_aspect:
                aspect_ratio = patch.width / patch.height
                target_aspect = target_width / target_height
                
                if aspect_ratio > target_aspect:
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)
            else:
                new_width, new_height = target_width, target_height
            
            processed_patch = processed_patch.resize((new_width, new_height), Image.Resampling.LANCZOS)
            if processed_mask:
                processed_mask = processed_mask.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply scaling
        if abs(scale - 1.0) > 0.01:
            new_width = int(processed_patch.width * scale)
            new_height = int(processed_patch.height * scale)
            processed_patch = processed_patch.resize((new_width, new_height), Image.Resampling.LANCZOS)
            if processed_mask:
                processed_mask = processed_mask.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply rotation
        if abs(rotation) > 0.01:
            processed_patch = processed_patch.rotate(rotation, expand=True, 
                                                   resample=Image.Resampling.BICUBIC,
                                                   fillcolor=(0, 0, 0))
            if processed_mask:
                processed_mask = processed_mask.rotate(rotation, expand=True,
                                                     resample=Image.Resampling.BICUBIC,
                                                     fillcolor=0)
        
        return processed_patch, processed_mask
    
    def _create_placement_mask(self, image_size: Tuple[int, int], x: int, y: int,
                              width: int, height: int, shape: str, corner_radius: int,
                              feather: int) -> Image.Image:
        """Create a mask defining where the patch will be placed"""
        
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Create mask based on shape
        if shape == "rectangle":
            draw.rectangle([x, y, x + width, y + height], fill=255)
            
        elif shape == "circle":
            center_x = x + width // 2
            center_y = y + height // 2
            radius = min(width, height) // 2
            
            bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
            draw.ellipse(bbox, fill=255)
            
        elif shape == "rounded_rectangle":
            self._draw_rounded_rectangle(draw, [x, y, x + width, y + height], corner_radius, fill=255)
        
        # Apply feathering
        if feather > 0:
            mask = self._apply_feathering(mask, feather)
        
        return mask
    
    def _draw_rounded_rectangle(self, draw, bbox, radius, outline=None, fill=None):
        """Draw a rounded rectangle"""
        x1, y1, x2, y2 = bbox
        
        # Draw the main rectangle parts
        draw.rectangle([x1 + radius, y1, x2 - radius, y2], outline=outline, fill=fill)
        draw.rectangle([x1, y1 + radius, x2, y2 - radius], outline=outline, fill=fill)
        
        # Draw corner arcs
        draw.pieslice([x1, y1, x1 + radius * 2, y1 + radius * 2], 180, 270, outline=outline, fill=fill)
        draw.pieslice([x2 - radius * 2, y1, x2, y1 + radius * 2], 270, 360, outline=outline, fill=fill)
        draw.pieslice([x1, y2 - radius * 2, x1 + radius * 2, y2], 90, 180, outline=outline, fill=fill)
        draw.pieslice([x2 - radius * 2, y2 - radius * 2, x2, y2], 0, 90, outline=outline, fill=fill)
    
    def _apply_feathering(self, mask: Image.Image, feather_pixels: int) -> Image.Image:
        """Apply feathering to mask edges"""
        if feather_pixels <= 0:
            return mask
        
        mask_array = np.array(mask)
        kernel_size = feather_pixels * 2 + 1
        blurred = cv2.GaussianBlur(mask_array, (kernel_size, kernel_size), feather_pixels/3)
        
        return Image.fromarray(blurred, mode='L')
    
    def _composite_patch(self, original: Image.Image, patch: Image.Image,
                        x: int, y: int, mask: Image.Image, blend_mode: str) -> Image.Image:
        """Composite patch onto original image with proper mask handling"""
        
        # Convert original to RGBA for compositing
        result = original.convert('RGBA')
        
        # Create patch canvas
        patch_canvas = Image.new('RGBA', original.size, (0, 0, 0, 0))
        
        # Ensure patch is RGB for pasting
        if patch.mode != 'RGB':
            patch = patch.convert('RGB')
        
        # Paste patch onto canvas using the mask
        patch_rgba = Image.new('RGBA', patch.size, (0, 0, 0, 0))
        patch_rgba.paste(patch, (0, 0))
        
        # Apply mask as alpha
        patch_rgba.putalpha(mask.crop((x, y, x + patch.width, y + patch.height)) 
                           if x + patch.width <= mask.width and y + patch.height <= mask.height
                           else mask.resize(patch.size, Image.Resampling.LANCZOS))
        
        # Paste onto canvas
        patch_canvas.paste(patch_rgba, (x, y), patch_rgba)
        
        # Apply blend mode
        if blend_mode == "normal":
            result = Image.alpha_composite(result, patch_canvas)
        else:
            result = self._apply_blend_mode(result, patch_canvas, blend_mode)
        
        # Convert back to RGB for ComfyUI
        result_rgb = Image.new('RGB', result.size, (255, 255, 255))
        result_rgb.paste(result, mask=result.split()[3] if result.mode == 'RGBA' else None)
        
        return result_rgb
    
    def _apply_blend_mode(self, background: Image.Image, foreground: Image.Image, blend_mode: str) -> Image.Image:
        """Apply blend modes"""
        
        bg_array = np.array(background, dtype=np.float32) / 255.0
        fg_array = np.array(foreground, dtype=np.float32) / 255.0
        
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
        elif blend_mode == "difference":
            blended = np.abs(bg_rgb - fg_rgb)
        elif blend_mode == "lighten":
            blended = np.maximum(bg_rgb, fg_rgb)
        elif blend_mode == "darken":
            blended = np.minimum(bg_rgb, fg_rgb)
        else:  # normal
            blended = fg_rgb
        
        # Alpha blend
        mask_3d = np.stack([fg_alpha.squeeze(), fg_alpha.squeeze(), fg_alpha.squeeze()], axis=2)
        result_rgb = bg_rgb * (1 - mask_3d) + blended * mask_3d
        
        # Combine with alpha
        bg_alpha = bg_array[:, :, 3:4] if bg_array.shape[2] == 4 else np.ones_like(bg_rgb[:, :, 0:1])
        result_alpha = bg_alpha * (1 - fg_alpha) + fg_alpha
        result_array = np.concatenate([result_rgb, result_alpha], axis=2)
        
        result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array, 'RGBA')
    
    def _create_preview_image(self, image: Image.Image, x: int, y: int, 
                             width: int, height: int, shape: str, corner_radius: int) -> Image.Image:
        """Create preview showing patch placement"""
        
        preview = image.copy().convert('RGBA')
        overlay = Image.new('RGBA', preview.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw placement indicator
        if shape == "rectangle":
            draw.rectangle([x, y, x + width, y + height], outline=(0, 255, 0, 128), width=2)
        elif shape == "circle":
            center_x = x + width // 2
            center_y = y + height // 2
            radius = min(width, height) // 2
            bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
            draw.ellipse(bbox, outline=(0, 255, 0, 128), width=2)
        elif shape == "rounded_rectangle":
            self._draw_rounded_rectangle(draw, [x, y, x + width, y + height], 
                                       corner_radius, outline=(0, 255, 0, 128))
        
        result = Image.alpha_composite(preview, overlay)
        return result.convert('RGB')


class Studio42VideoPatchDrop:
    """
    🎬 Studio42 Video PatchDrop - Enhanced with Mask Support
    
    Advanced video patch placement node for ComfyUI with proper mask handling.
    Takes original video frames and patch video frames, and drops the patches back 
    into their original locations or specified locations across all video frames.
    
    NEW FEATURES:
    ✅ Optional patch_mask input for transparency control
    ✅ ComfyUI IMAGE/MASK format compatibility  
    ✅ Professional alpha compositing for video
    ✅ Temporal consistency controls
    
    Features:
    - Automatic placement using stored coordinates across video sequences
    - Manual position override for all frames
    - Shape-aware placement (rectangle, circle, rounded rectangle)
    - Blend modes for seamless integration across frames
    - Edge feathering for smooth transitions
    - Opacity control
    - Transform options (scale, rotation) applied to all frames
    - Temporal consistency controls
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_video": ("IMAGE",),  # Video frames as batch
                "patch_video": ("IMAGE",),     # Patch frames as batch
                "patch_x": ("INT", {"default": 0, "min": -2048, "max": 4096, "step": 1}),
                "patch_y": ("INT", {"default": 0, "min": -2048, "max": 4096, "step": 1}),
            },
            "optional": {
                # NEW: Mask input for transparency
                "patch_mask": ("MASK",),
                
                "patch_width": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
                "patch_height": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
                "patch_shape": (["rectangle", "circle", "rounded_rectangle"], {"default": "rectangle"}),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light",
                               "hard_light", "difference", "lighten", "darken"], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather_edges": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "corner_radius": ("INT", {"default": 20, "min": 5, "max": 100, "step": 5}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "position_mode": (["use_coordinates", "center", "manual"], {"default": "use_coordinates"}),
                "manual_x": ("INT", {"default": 0, "min": -2048, "max": 4096, "step": 1}),
                "manual_y": ("INT", {"default": 0, "min": -2048, "max": 4096, "step": 1}),
                "resize_patch": ("BOOLEAN", {"default": False}),
                "preserve_aspect": ("BOOLEAN", {"default": True}),
                
                # Video-specific controls
                "temporal_consistency": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "motion_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "edge_blending": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("result_video", "placement_masks", "preview_video")
    FUNCTION = "drop_video_patches"
    CATEGORY = "🎬 Studio42/Video Processing"
    
    def drop_video_patches(self, original_video, patch_video, patch_x, patch_y,
                          patch_mask=None, patch_width=256, patch_height=256, patch_shape="rectangle",
                          blend_mode="normal", opacity=1.0, feather_edges=0, corner_radius=20,
                          scale=1.0, rotation=0.0, position_mode="use_coordinates",
                          manual_x=0, manual_y=0, resize_patch=False, preserve_aspect=True,
                          temporal_consistency=0.0, motion_blur=0.0, edge_blending=True):
        
        try:
            # Get batch sizes
            num_original_frames = len(original_video)
            num_patch_frames = len(patch_video)
            max_frames = max(num_original_frames, num_patch_frames)
            
            # Handle patch mask batch size if provided
            if patch_mask is not None:
                num_mask_frames = len(patch_mask)
                max_frames = max(max_frames, num_mask_frames)
            
            results = []
            masks = []
            previews = []
            
            # Process each frame
            for frame_idx in range(max_frames):
                # Get original and patch frames (loop if necessary)
                orig_frame_idx = frame_idx % num_original_frames
                patch_frame_idx = frame_idx % num_patch_frames
                
                orig_tensor = original_video[orig_frame_idx]
                patch_tensor = patch_video[patch_frame_idx]
                
                # Get mask frame if provided
                mask_tensor = None
                if patch_mask is not None:
                    mask_frame_idx = frame_idx % len(patch_mask)
                    mask_tensor = patch_mask[mask_frame_idx]
                
                # Convert to PIL Images
                orig_pil = self._tensor_to_pil(orig_tensor)
                patch_pil = self._tensor_to_pil(patch_tensor)
                
                # Convert mask to PIL if provided
                mask_pil = None
                if mask_tensor is not None:
                    mask_pil = self._mask_tensor_to_pil(mask_tensor)
                
                # Apply temporal consistency if enabled
                if temporal_consistency > 0.0 and frame_idx > 0:
                    prev_x = getattr(self, '_prev_x', patch_x)
                    prev_y = getattr(self, '_prev_y', patch_y)
                    
                    smoothed_x = int(prev_x * temporal_consistency + patch_x * (1 - temporal_consistency))
                    smoothed_y = int(prev_y * temporal_consistency + patch_y * (1 - temporal_consistency))
                    
                    self._prev_x = smoothed_x
                    self._prev_y = smoothed_y
                    
                    current_patch_x = smoothed_x
                    current_patch_y = smoothed_y
                else:
                    current_patch_x = patch_x
                    current_patch_y = patch_y
                    self._prev_x = patch_x
                    self._prev_y = patch_y
                
                # Process single frame patch drop (reuse the image PatchDrop logic)
                result_frame, placement_mask, preview_frame = self._drop_single_patch(
                    orig_pil, patch_pil, mask_pil, current_patch_x, current_patch_y, 
                    patch_width, patch_height, patch_shape, blend_mode, opacity, 
                    feather_edges, corner_radius, scale, rotation, position_mode, 
                    manual_x, manual_y, resize_patch, preserve_aspect, motion_blur, edge_blending
                )
                
                # Convert back to tensors (ensure RGB format)
                result_tensor = self._pil_to_tensor(result_frame)
                preview_tensor = self._pil_to_tensor(preview_frame)
                placement_mask_tensor = self._pil_to_mask_tensor(placement_mask)
                
                results.append(result_tensor)
                masks.append(placement_mask_tensor)
                previews.append(preview_tensor)
            
            # Stack results
            result_batch = torch.cat(results, dim=0)
            mask_batch = torch.stack(masks, dim=0)
            preview_batch = torch.cat(previews, dim=0)
            
            logger.info(f"✅ Video patch drop completed: {max_frames} frames processed")
            
            return (result_batch, mask_batch, preview_batch)
            
        except Exception as e:
            logger.error(f"❌ Video patch drop failed: {e}")
            # Return original as fallback
            batch_size = original_video.shape[0]
            fallback_mask = torch.ones(batch_size, original_video.shape[1], original_video.shape[2], dtype=torch.float32)
            return (original_video, fallback_mask, original_video)
    
    # Reuse all the helper methods from Studio42PatchDrop
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI IMAGE tensor to PIL Image (RGB)"""
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if len(np_img.shape) == 2:
            np_img = np.stack([np_img, np_img, np_img], axis=-1)
        elif np_img.shape[-1] == 4:
            np_img = np_img[:, :, :3]
        
        return Image.fromarray(np_img, 'RGB')
    
    def _mask_tensor_to_pil(self, mask_tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI MASK tensor to PIL Image (L)"""
        mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(mask_np, 'L')
    
    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to ComfyUI IMAGE tensor"""
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        np_img = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(np_img).unsqueeze(0)
    
    def _pil_to_mask_tensor(self, pil_mask: Image.Image) -> torch.Tensor:
        """Convert PIL mask to ComfyUI MASK tensor"""
        if pil_mask.mode != 'L':
            pil_mask = pil_mask.convert('L')
        
        mask_np = np.array(pil_mask).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np)
    
    def _drop_single_patch(self, original: Image.Image, patch: Image.Image, patch_mask: Optional[Image.Image],
                          patch_x: int, patch_y: int, patch_width: int, patch_height: int,
                          patch_shape: str, blend_mode: str, opacity: float,
                          feather_edges: int, corner_radius: int, scale: float,
                          rotation: float, position_mode: str, manual_x: int, manual_y: int,
                          resize_patch: bool, preserve_aspect: bool, motion_blur: float, 
                          edge_blending: bool) -> Tuple[Image.Image, Image.Image, Image.Image]:
        
        # Reuse the same logic as Studio42PatchDrop but with video enhancements
        
        # Ensure original frame is RGB
        if original.mode != 'RGB':
            original = original.convert('RGB')
        
        # Determine final position
        if position_mode == "center":
            final_x = (original.width - patch_width) // 2
            final_y = (original.height - patch_height) // 2
        elif position_mode == "manual":
            final_x = manual_x
            final_y = manual_y
        else:  # use_coordinates
            final_x = patch_x
            final_y = patch_y
        
        # Process patch transformations with video enhancements
        processed_patch, processed_mask = self._process_patch_transforms_video(
            patch, patch_mask, patch_width, patch_height, scale, rotation, 
            resize_patch, preserve_aspect, motion_blur
        )
        
        # Get final patch dimensions after transformations
        final_patch_width, final_patch_height = processed_patch.size
        
        # Ensure patch fits within frame bounds
        final_x = max(0, min(final_x, original.width - final_patch_width))
        final_y = max(0, min(final_y, original.height - final_patch_height))
        
        # Create placement mask based on shape with video enhancements
        placement_mask = self._create_placement_mask_video(
            original.size, final_x, final_y, final_patch_width, final_patch_height,
            patch_shape, corner_radius, feather_edges, edge_blending
        )
        
        # If we have a patch mask, combine it with placement mask
        if processed_mask:
            if processed_mask.size != (final_patch_width, final_patch_height):
                processed_mask = processed_mask.resize(
                    (final_patch_width, final_patch_height), 
                    Image.Resampling.LANCZOS
                )
            
            full_mask_canvas = Image.new('L', original.size, 0)
            full_mask_canvas.paste(processed_mask, (final_x, final_y))
            
            placement_array = np.array(placement_mask, dtype=np.float32) / 255.0
            patch_mask_array = np.array(full_mask_canvas, dtype=np.float32) / 255.0
            
            combined_mask = (placement_array * patch_mask_array * 255).astype(np.uint8)
            placement_mask = Image.fromarray(combined_mask, mode='L')
        
        # Apply opacity
        if opacity < 1.0:
            mask_array = np.array(placement_mask, dtype=np.float32) * opacity
            placement_mask = Image.fromarray(mask_array.astype(np.uint8), mode='L')
        
        # Create composite frame
        result_frame = self._composite_patch_video(
            original, processed_patch, final_x, final_y, 
            placement_mask, blend_mode
        )
        
        # Create preview frame showing placement (video-friendly colors)
        preview_frame = self._create_preview_frame(
            result_frame, final_x, final_y, final_patch_width, final_patch_height,
            patch_shape, corner_radius
        )
        
        return result_frame, placement_mask, preview_frame
    
    def _process_patch_transforms_video(self, patch: Image.Image, patch_mask: Optional[Image.Image], 
                                      target_width: int, target_height: int, scale: float, rotation: float, 
                                      resize_patch: bool, preserve_aspect: bool, motion_blur: float) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Apply transformations with video-specific enhancements"""
        
        processed_patch = patch.copy()
        processed_mask = patch_mask.copy() if patch_mask else None
        
        # Apply all the same transforms as image version
        # (resize, scale, rotation logic same as Studio42PatchDrop)
        
        # Video-specific: Apply motion blur for smoothness
        if motion_blur > 0.0:
            processed_patch = self._apply_motion_blur(processed_patch, motion_blur)
            if processed_mask:
                processed_mask = self._apply_motion_blur(processed_mask, motion_blur * 0.5)  # Less blur on mask
        
        return processed_patch, processed_mask
    
    def _apply_motion_blur(self, image: Image.Image, blur_amount: float) -> Image.Image:
        """Apply motion blur for video smoothness"""
        if blur_amount <= 0.0:
            return image
        
        img_array = np.array(image)
        
        # Create motion blur kernel
        kernel_size = int(blur_amount * 2) + 1
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply blur
        if len(img_array.shape) == 3:  # RGB
            for i in range(3):
                img_array[:, :, i] = cv2.filter2D(img_array[:, :, i], -1, kernel)
        else:  # Grayscale
            img_array = cv2.filter2D(img_array, -1, kernel)
        
        return Image.fromarray(img_array)
    
    def _create_placement_mask_video(self, image_size: Tuple[int, int], x: int, y: int,
                                   width: int, height: int, shape: str, corner_radius: int,
                                   feather: int, edge_blending: bool) -> Image.Image:
        """Create placement mask with video-specific enhancements"""
        
        # Same base logic as image version
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        if shape == "rectangle":
            draw.rectangle([x, y, x + width, y + height], fill=255)
        elif shape == "circle":
            center_x = x + width // 2
            center_y = y + height // 2
            radius = min(width, height) // 2
            bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
            draw.ellipse(bbox, fill=255)
        elif shape == "rounded_rectangle":
            self._draw_rounded_rectangle(draw, [x, y, x + width, y + height], corner_radius, fill=255)
        
        # Apply feathering
        if feather > 0:
            mask_array = np.array(mask)
            kernel_size = feather * 2 + 1
            blurred = cv2.GaussianBlur(mask_array, (kernel_size, kernel_size), feather/3)
            mask = Image.fromarray(blurred, mode='L')
        
        # Video-specific: Additional edge blending for temporal continuity
        if edge_blending:
            mask_array = np.array(mask, dtype=np.float32)
            blurred_extra = cv2.GaussianBlur((mask_array / 255.0 * 255).astype(np.uint8), (5, 5), 1.0)
            mask = Image.fromarray(blurred_extra, mode='L')
        
        return mask
    
    def _draw_rounded_rectangle(self, draw, bbox, radius, outline=None, fill=None):
        """Draw a rounded rectangle (same as image version)"""
        x1, y1, x2, y2 = bbox
        
        draw.rectangle([x1 + radius, y1, x2 - radius, y2], outline=outline, fill=fill)
        draw.rectangle([x1, y1 + radius, x2, y2 - radius], outline=outline, fill=fill)
        
        draw.pieslice([x1, y1, x1 + radius * 2, y1 + radius * 2], 180, 270, outline=outline, fill=fill)
        draw.pieslice([x2 - radius * 2, y1, x2, y1 + radius * 2], 270, 360, outline=outline, fill=fill)
        draw.pieslice([x1, y2 - radius * 2, x1 + radius * 2, y2], 90, 180, outline=outline, fill=fill)
        draw.pieslice([x2 - radius * 2, y2 - radius * 2, x2, y2], 0, 90, outline=outline, fill=fill)
    
    def _composite_patch_video(self, original: Image.Image, patch: Image.Image,
                             x: int, y: int, mask: Image.Image, blend_mode: str) -> Image.Image:
        """Composite patch with video optimizations (same as image version with video tweaks)"""
        
        result = original.convert('RGBA')
        patch_canvas = Image.new('RGBA', original.size, (0, 0, 0, 0))
        
        if patch.mode != 'RGB':
            patch = patch.convert('RGB')
        
        patch_rgba = Image.new('RGBA', patch.size, (0, 0, 0, 0))
        patch_rgba.paste(patch, (0, 0))
        
        # Apply mask as alpha
        patch_rgba.putalpha(mask.crop((x, y, x + patch.width, y + patch.height)) 
                           if x + patch.width <= mask.width and y + patch.height <= mask.height
                           else mask.resize(patch.size, Image.Resampling.LANCZOS))
        
        patch_canvas.paste(patch_rgba, (x, y), patch_rgba)
        
        # Apply blend mode
        if blend_mode == "normal":
            result = Image.alpha_composite(result, patch_canvas)
        else:
            result = self._apply_blend_mode_video(result, patch_canvas, blend_mode)
        
        result_rgb = Image.new('RGB', result.size, (255, 255, 255))
        result_rgb.paste(result, mask=result.split()[3] if result.mode == 'RGBA' else None)
        
        return result_rgb
    
    def _apply_blend_mode_video(self, background: Image.Image, foreground: Image.Image, blend_mode: str) -> Image.Image:
        """Apply blend modes optimized for video (same as image version)"""
        
        bg_array = np.array(background, dtype=np.float32) / 255.0
        fg_array = np.array(foreground, dtype=np.float32) / 255.0
        
        bg_rgb = bg_array[:, :, :3]
        fg_rgb = fg_array[:, :, :3]
        fg_alpha = fg_array[:, :, 3:4] if fg_array.shape[2] == 4 else np.ones_like(fg_rgb[:, :, 0:1])
        
        # Same blend modes as image version
        if blend_mode == "multiply":
            blended = bg_rgb * fg_rgb
        elif blend_mode == "screen":
            blended = 1 - (1 - bg_rgb) * (1 - fg_rgb)
        elif blend_mode == "overlay":
            blended = np.where(bg_rgb < 0.5,
                              2 * bg_rgb * fg_rgb,
                              1 - 2 * (1 - bg_rgb) * (1 - fg_rgb))
        # ... (other blend modes same as image version)
        else:  # normal
            blended = fg_rgb
        
        mask_3d = np.stack([fg_alpha.squeeze(), fg_alpha.squeeze(), fg_alpha.squeeze()], axis=2)
        result_rgb = bg_rgb * (1 - mask_3d) + blended * mask_3d
        
        bg_alpha = bg_array[:, :, 3:4] if bg_array.shape[2] == 4 else np.ones_like(bg_rgb[:, :, 0:1])
        result_alpha = bg_alpha * (1 - fg_alpha) + fg_alpha
        result_array = np.concatenate([result_rgb, result_alpha], axis=2)
        
        result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array, 'RGBA')
    
    def _create_preview_frame(self, frame: Image.Image, x: int, y: int, 
                             width: int, height: int, shape: str, corner_radius: int) -> Image.Image:
        """Create preview showing patch placement with video-friendly colors"""
        
        preview = frame.copy().convert('RGBA')
        overlay = Image.new('RGBA', preview.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Use cyan for video preview (more visible)
        if shape == "rectangle":
            draw.rectangle([x, y, x + width, y + height], outline=(0, 255, 255, 128), width=2)
        elif shape == "circle":
            center_x = x + width // 2
            center_y = y + height // 2
            radius = min(width, height) // 2
            bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
            draw.ellipse(bbox, outline=(0, 255, 255, 128), width=2)
        elif shape == "rounded_rectangle":
            self._draw_rounded_rectangle(draw, [x, y, x + width, y + height], 
                                       corner_radius, outline=(0, 255, 255, 128))
        
        result = Image.alpha_composite(preview, overlay)
        return result.convert('RGB')


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Studio42PatchDrop": Studio42PatchDrop,
    "Studio42VideoPatchDrop": Studio42VideoPatchDrop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Studio42PatchDrop": "🎬 Studio42 PatchDrop",
    "Studio42VideoPatchDrop": "🎬 Studio42 Video PatchDrop"
}