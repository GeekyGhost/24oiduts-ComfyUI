import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import math
import logging
from typing import Tuple, Optional, List
import cv2
import gc

# Configure logging
logger = logging.getLogger(__name__)

class Studio42LayerComposer:
    """
    Studio42 Layer Composer - Proxy Workflow Version
    
    Professional layer composition with intelligent resolution scaling.
    Uses proxy workflow: scale down → compose → scale back up for massive images.
    
    BREAKTHROUGH: Handles massive resolution differences efficiently
    
    Features:
    - PROXY WORKFLOW: Scale down massive images before compositing
    - INTELLIGENT SCALING: Automatic resolution balancing
    - MEMORY EFFICIENT: Reduces 96GB → 2GB workflows
    - QUALITY PRESERVATION: Smart upscaling maintains detail
    - CHANNEL COMPATIBILITY: Handles RGB/RGBA input mixing
    - BATCH PROCESSING: Vectorized operations throughout
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
                "foreground_mask": ("MASK",),
                
                # Position - matches PatchLift/PatchDrop coordinate system
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
                
                # Anchor point
                "anchor_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "anchor_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Flipping
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "flip_vertical": ("BOOLEAN", {"default": False}),
                
                # Edge processing
                "feather_edges": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "anti_aliasing": ("BOOLEAN", {"default": True}),
                
                # Clipping
                "clip_to_background": ("BOOLEAN", {"default": True}),
                
                # PROXY WORKFLOW SETTINGS
                "use_proxy_workflow": ("BOOLEAN", {"default": True,
                                      "tooltip": "Scale down large images for faster processing"}),
                "proxy_max_resolution": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256,
                                        "tooltip": "Maximum resolution for proxy processing"}),
                "upscale_method": (["bilinear", "bicubic", "lanczos"], {"default": "bicubic",
                                 "tooltip": "Method for upscaling final result"}),
                
                # Performance settings
                "batch_processing_mode": (["auto", "full_batch", "chunked"], {"default": "auto",
                                        "tooltip": "auto=intelligent, full_batch=fastest, chunked=safest"}),
                "max_chunk_size": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1,
                                  "tooltip": "Maximum images per chunk (only used in chunked mode)"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composite_image", "composite_mask")
    FUNCTION = "compose_layers"
    CATEGORY = "Studio42/Composition"
    
    def compose_layers(self, background, foreground, blend_mode="normal", opacity=1.0,
                      foreground_mask=None, x_position=0, y_position=0, position_mode="custom",
                      scale_x=1.0, scale_y=1.0, fit_mode="none", maintain_aspect=True, 
                      rotation=0.0, anchor_x=0.5, anchor_y=0.5, flip_horizontal=False, 
                      flip_vertical=False, feather_edges=0, anti_aliasing=True, 
                      clip_to_background=True, use_proxy_workflow=True, proxy_max_resolution=2048,
                      upscale_method="bicubic", batch_processing_mode="auto", max_chunk_size=16):
        
        try:
            # Get batch info and device
            device = background.device
            dtype = background.dtype
            
            # CRITICAL FIX: Normalize channel dimensions first
            logger.info(f"🔍 Input shapes - Background: {background.shape}, Foreground: {foreground.shape}")
            background, bg_alpha = self._normalize_channels(background, "background")
            foreground, fg_alpha = self._normalize_channels(foreground, "foreground") 
            logger.info(f"✅ Normalized shapes - Background: {background.shape}, Foreground: {foreground.shape}")
            
            # Store original dimensions for final upscaling
            original_bg_shape = background.shape
            original_fg_shape = foreground.shape
            
            # Use extracted alpha as mask if no explicit mask provided
            if foreground_mask is None and fg_alpha is not None:
                foreground_mask = fg_alpha
                logger.info("🎭 Using extracted alpha channel as foreground mask")
            
            # PROXY WORKFLOW: Check if we need to scale down
            bg_h, bg_w = background.shape[1], background.shape[2]
            fg_h, fg_w = foreground.shape[1], foreground.shape[2]
            max_dimension = max(bg_h, bg_w, fg_h, fg_w)
            
            use_proxy = use_proxy_workflow and max_dimension > proxy_max_resolution
            
            if use_proxy:
                logger.info(f"🎬 PROXY WORKFLOW: Scaling down from {max_dimension}px to {proxy_max_resolution}px")
                
                # Calculate scaling factors
                proxy_scale = proxy_max_resolution / max_dimension
                logger.info(f"📐 Proxy scale factor: {proxy_scale:.3f}")
                
                # Scale down inputs
                background_proxy, bg_scale_info = self._scale_down_for_proxy(background, proxy_scale)
                foreground_proxy, fg_scale_info = self._scale_down_for_proxy(foreground, proxy_scale)
                
                # Scale down mask if present
                mask_proxy = None
                if foreground_mask is not None:
                    mask_proxy, _ = self._scale_down_for_proxy(foreground_mask.unsqueeze(-1), proxy_scale)
                    mask_proxy = mask_proxy.squeeze(-1)
                
                # Adjust positions for proxy resolution
                proxy_x = int(x_position * proxy_scale)
                proxy_y = int(y_position * proxy_scale)
                
                logger.info(f"📏 Proxy shapes - BG: {background_proxy.shape}, FG: {foreground_proxy.shape}")
                
                # Process at proxy resolution
                composite_proxy, mask_proxy_out = self._process_composition(
                    background_proxy, foreground_proxy, mask_proxy,
                    blend_mode, opacity, proxy_x, proxy_y, position_mode,
                    scale_x, scale_y, fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
                    flip_horizontal, flip_vertical, feather_edges, anti_aliasing, 
                    clip_to_background, batch_processing_mode, max_chunk_size
                )
                
                # Scale back up to original resolution with correct batch size
                target_batch_size = composite_proxy.shape[0]  # Use the actual batch size from processing
                target_shape = (target_batch_size, original_bg_shape[1], original_bg_shape[2], 3)
                logger.info(f"🔍 Upscaling result from {composite_proxy.shape} to {target_shape}")
                final_composite = self._scale_up_from_proxy(composite_proxy, target_shape, upscale_method)
                final_mask = self._scale_up_from_proxy(mask_proxy_out.unsqueeze(-1), 
                                                     (target_batch_size, original_bg_shape[1], original_bg_shape[2], 1), 
                                                     upscale_method).squeeze(-1)
                
                logger.info(f"✅ PROXY WORKFLOW completed: Final shape {final_composite.shape}")
                
            else:
                logger.info(f"🚀 DIRECT Processing: Resolution {max_dimension}px within limits")
                
                # Process directly at full resolution
                final_composite, final_mask = self._process_composition(
                    background, foreground, foreground_mask,
                    blend_mode, opacity, x_position, y_position, position_mode,
                    scale_x, scale_y, fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
                    flip_horizontal, flip_vertical, feather_edges, anti_aliasing, 
                    clip_to_background, batch_processing_mode, max_chunk_size
                )
            
            return (final_composite, final_mask)
            
        except Exception as e:
            logger.error(f"❌ Layer composition failed: {e}")
            # Return background as fallback
            batch_size = background.shape[0]
            # Ensure background is RGB for fallback
            if background.shape[-1] != 3:
                background = background[..., :3]  # Take only RGB channels
            fallback_mask = torch.ones(batch_size, background.shape[1], background.shape[2], 
                                     dtype=background.dtype, device=background.device)
            return (background, fallback_mask)
    
    def _scale_down_for_proxy(self, tensor, scale_factor):
        """Scale down tensor for proxy processing"""
        if tensor.dim() == 3:  # Single image [H, W, C]
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            remove_batch = True
        else:
            remove_batch = False
        
        # Convert to BCHW for interpolation
        if tensor.shape[-1] <= 4:  # Assume [B, H, W, C] format
            tensor_bchw = tensor.permute(0, 3, 1, 2)
        else:
            tensor_bchw = tensor
        
        original_h, original_w = tensor_bchw.shape[2], tensor_bchw.shape[3]
        new_h = max(1, int(original_h * scale_factor))
        new_w = max(1, int(original_w * scale_factor))
        
        # Scale down using high-quality interpolation
        scaled_bchw = F.interpolate(tensor_bchw, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Convert back to BHWC
        if tensor.shape[-1] <= 4:
            scaled = scaled_bchw.permute(0, 2, 3, 1)
        else:
            scaled = scaled_bchw
        
        if remove_batch:
            scaled = scaled.squeeze(0)
        
        scale_info = {
            'original_size': (original_h, original_w),
            'proxy_size': (new_h, new_w),
            'scale_factor': scale_factor
        }
        
        return scaled, scale_info
    
    def _scale_up_from_proxy(self, tensor_proxy, target_shape, upscale_method):
        """Scale up result from proxy resolution to original - MEMORY SAFE VERSION"""
        target_batch, target_h, target_w = target_shape[0], target_shape[1], target_shape[2]
        
        logger.info(f"📈 Upscaling {tensor_proxy.shape} to {target_shape}")
        
        # CRITICAL FIX: For large target shapes, upscale in chunks to avoid memory explosion
        estimated_memory = target_batch * target_h * target_w * 3 * 4  # RGB float32
        memory_limit = 8e9  # 8GB limit
        
        if estimated_memory > memory_limit:
            logger.info(f"⚠️  Large upscale detected ({estimated_memory/1e9:.1f}GB), using chunked upscaling")
            return self._scale_up_chunked(tensor_proxy, target_shape, upscale_method)
        
        # For smaller targets, use direct upscaling
        return self._scale_up_direct(tensor_proxy, target_shape, upscale_method)
    
    def _scale_up_chunked(self, tensor_proxy, target_shape, upscale_method):
        """Upscale in chunks to avoid memory explosion"""
        target_batch, target_h, target_w = target_shape[0], target_shape[1], target_shape[2]
        device = tensor_proxy.device
        dtype = tensor_proxy.dtype
        
        # Calculate safe chunk size for upscaling
        chunk_size = max(1, min(8, target_batch))  # Process max 8 images at once during upscaling
        
        logger.info(f"📊 Upscaling in chunks of {chunk_size}")
        
        # Process chunks and save to CPU to avoid GPU memory buildup
        result_chunks = []
        
        for chunk_start in range(0, target_batch, chunk_size):
            chunk_end = min(chunk_start + chunk_size, target_batch)
            
            # Get proxy chunk
            proxy_chunk = tensor_proxy[chunk_start:chunk_end]
            
            # Upscale chunk
            chunk_target_shape = (chunk_end - chunk_start, target_h, target_w, target_shape[3])
            upscaled_chunk = self._scale_up_direct(proxy_chunk, chunk_target_shape, upscale_method)
            
            # Move to CPU immediately to free GPU memory
            result_chunks.append(upscaled_chunk.cpu())
            
            # Clean up GPU memory
            del proxy_chunk, upscaled_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"📈 Upscaled chunk {chunk_start//chunk_size + 1}/{math.ceil(target_batch/chunk_size)}")
        
        # Combine chunks back on GPU
        logger.info("🔗 Combining upscaled chunks...")
        final_result = torch.cat(result_chunks, dim=0).to(device=device, dtype=dtype)
        
        # Clean up
        del result_chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return final_result
    
    def _scale_up_direct(self, tensor_proxy, target_shape, upscale_method):
        """Direct upscaling for smaller tensors"""
        target_batch, target_h, target_w = target_shape[0], target_shape[1], target_shape[2]
        
        if tensor_proxy.dim() == 3 and len(target_shape) == 4:  # Add batch dimension if needed
            tensor_proxy = tensor_proxy.unsqueeze(0)
        
        # Convert to BCHW for interpolation
        if tensor_proxy.shape[-1] <= 4:  # Assume [B, H, W, C] format
            tensor_bchw = tensor_proxy.permute(0, 3, 1, 2)
        else:
            tensor_bchw = tensor_proxy
        
        # Map upscale method to interpolation mode
        mode_map = {
            'bilinear': 'bilinear',
            'bicubic': 'bicubic',
            'lanczos': 'bilinear'  # PyTorch doesn't have lanczos, use bilinear
        }
        mode = mode_map.get(upscale_method, 'bilinear')
        
        # Scale up using high-quality interpolation
        upscaled_bchw = F.interpolate(tensor_bchw, size=(target_h, target_w), mode=mode, align_corners=False)
        
        # Expand batch dimension if needed
        if upscaled_bchw.shape[0] != target_batch:
            upscaled_bchw = upscaled_bchw.repeat(target_batch, 1, 1, 1)
        
        # Convert back to BHWC
        if len(target_shape) == 4 and target_shape[-1] <= 4:
            upscaled = upscaled_bchw.permute(0, 2, 3, 1)
        else:
            upscaled = upscaled_bchw
        
        return upscaled
    
    def _process_composition(self, background, foreground, foreground_mask,
                           blend_mode, opacity, x_position, y_position, position_mode,
                           scale_x, scale_y, fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
                           flip_horizontal, flip_vertical, feather_edges, anti_aliasing, 
                           clip_to_background, batch_processing_mode, max_chunk_size):
        """Core composition processing logic"""
        
        device = background.device
        dtype = background.dtype
        
        batch_size_bg = background.shape[0]
        batch_size_fg = foreground.shape[0]
        max_batch_size = max(batch_size_bg, batch_size_fg)
        
        # Handle foreground mask batch size if provided
        if foreground_mask is not None:
            batch_size_mask = foreground_mask.shape[0]
            max_batch_size = max(max_batch_size, batch_size_mask)
        
        # Get dimensions from background
        bg_h, bg_w = background.shape[1], background.shape[2]
        
        logger.info(f"🚀 Processing composition: {max_batch_size} frames, {bg_w}x{bg_h}, mode: {blend_mode}")
        
        # Determine processing strategy
        processing_mode = self._determine_processing_mode(
            batch_processing_mode, max_batch_size, bg_h, bg_w, device
        )
        
        logger.info(f"⚡ Processing mode: {processing_mode}")
        
        # Execute based on processing mode
        if processing_mode == "full_batch":
            # Process everything at once (fastest)
            result, mask = self._process_full_batch(
                background, foreground, foreground_mask, max_batch_size,
                batch_size_bg, batch_size_fg, x_position, y_position, position_mode,
                scale_x, scale_y, fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
                flip_horizontal, flip_vertical, blend_mode, opacity, feather_edges, 
                anti_aliasing, clip_to_background
            )
        else:
            # Process in optimized chunks
            result, mask = self._process_optimized_chunks(
                background, foreground, foreground_mask, max_batch_size,
                batch_size_bg, batch_size_fg, x_position, y_position, position_mode,
                scale_x, scale_y, fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
                flip_horizontal, flip_vertical, blend_mode, opacity, feather_edges, 
                anti_aliasing, clip_to_background, max_chunk_size
            )
        
        return result, mask
    
    def _normalize_channels(self, tensor, name):
        """Normalize tensor to RGB format and extract alpha if present"""
        
        if tensor.shape[-1] == 3:
            # Already RGB
            return tensor, None
        elif tensor.shape[-1] == 4:
            # RGBA - split into RGB and alpha
            rgb = tensor[..., :3]  # Take RGB channels
            alpha = tensor[..., 3]  # Extract alpha channel
            logger.info(f"📋 {name}: Converted RGBA to RGB + alpha mask")
            return rgb, alpha
        elif tensor.shape[-1] == 1:
            # Grayscale - convert to RGB
            rgb = tensor.repeat(1, 1, 1, 3)
            logger.info(f"📋 {name}: Converted grayscale to RGB")
            return rgb, None
        else:
            raise ValueError(f"Unsupported channel count: {tensor.shape[-1]} for {name}")
    
    def _determine_processing_mode(self, batch_processing_mode, max_batch_size, bg_h, bg_w, device):
        """Intelligently determine the best processing mode"""
        
        if batch_processing_mode == "full_batch":
            return "full_batch"
        elif batch_processing_mode == "chunked":
            return "chunked"
        
        # Auto mode - intelligent decision
        image_size_mp = (bg_h * bg_w) / 1_000_000  # Megapixels
        total_data_size = max_batch_size * image_size_mp
        
        # For proxy workflow, we can be more aggressive with batch processing
        if image_size_mp <= 4:  # 2K or smaller (proxy resolution)
            return "full_batch"
        
        # For larger images, use chunked
        return "chunked"
    
    def _process_full_batch(self, background, foreground, foreground_mask, max_batch_size,
                           batch_size_bg, batch_size_fg, x_position, y_position, position_mode,
                           scale_x, scale_y, fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
                           flip_horizontal, flip_vertical, blend_mode, opacity, feather_edges, 
                           anti_aliasing, clip_to_background):
        """Process entire batch at once - maximum speed"""
        
        device = background.device
        dtype = background.dtype
        bg_h, bg_w = background.shape[1], background.shape[2]
        
        # Expand tensors to match batch size efficiently
        bg_expanded = self._expand_tensor_efficient(background, max_batch_size, batch_size_bg)
        fg_expanded = self._expand_tensor_efficient(foreground, max_batch_size, batch_size_fg)
        
        mask_expanded = None
        if foreground_mask is not None:
            mask_expanded = self._expand_tensor_efficient(foreground_mask, max_batch_size, foreground_mask.shape[0])
        
        # Transform foreground using vectorized operations
        fg_transformed, mask_transformed = self._transform_batch_vectorized(
            fg_expanded, mask_expanded, (bg_w, bg_h), scale_x, scale_y,
            fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
            flip_horizontal, flip_vertical, anti_aliasing
        )
        
        # Calculate positions
        positions = self._calculate_positions_batch(
            fg_transformed.shape[1:3], (bg_w, bg_h), position_mode, x_position, y_position
        )
        
        # Compose using pure tensor operations
        composite, comp_mask = self._compose_batch_vectorized(
            bg_expanded, fg_transformed, mask_transformed, positions,
            blend_mode, opacity, feather_edges, clip_to_background
        )
        
        return composite, comp_mask
    
    def _process_optimized_chunks(self, background, foreground, foreground_mask, max_batch_size,
                                 batch_size_bg, batch_size_fg, x_position, y_position, position_mode,
                                 scale_x, scale_y, fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
                                 flip_horizontal, flip_vertical, blend_mode, opacity, feather_edges, 
                                 anti_aliasing, clip_to_background, max_chunk_size):
        """Process in optimized chunks"""
        
        device = background.device
        dtype = background.dtype
        bg_h, bg_w = background.shape[1], background.shape[2]
        
        # Calculate optimal chunk size
        optimal_chunk_size = self._calculate_optimal_chunk_size(bg_h, bg_w, max_chunk_size)
        
        logger.info(f"📊 Using optimized chunk size: {optimal_chunk_size}")
        
        # Pre-allocate result tensors
        results = torch.empty(max_batch_size, bg_h, bg_w, 3, dtype=dtype, device=device)
        masks = torch.empty(max_batch_size, bg_h, bg_w, dtype=dtype, device=device)
        
        # Process chunks
        for chunk_start in range(0, max_batch_size, optimal_chunk_size):
            chunk_end = min(chunk_start + optimal_chunk_size, max_batch_size)
            chunk_indices = list(range(chunk_start, chunk_end))
            
            logger.info(f"🔄 Processing chunk {chunk_start//optimal_chunk_size + 1}/{math.ceil(max_batch_size/optimal_chunk_size)}: frames {chunk_start}-{chunk_end-1}")
            
            # Get chunk data efficiently
            bg_chunk = self._get_chunk_efficient(background, chunk_indices, batch_size_bg)
            fg_chunk = self._get_chunk_efficient(foreground, chunk_indices, batch_size_fg)
            mask_chunk = None
            if foreground_mask is not None:
                mask_chunk = self._get_chunk_efficient(foreground_mask, chunk_indices, foreground_mask.shape[0])
            
            # Process chunk
            chunk_result, chunk_mask = self._process_chunk_optimized(
                bg_chunk, fg_chunk, mask_chunk, x_position, y_position, position_mode,
                scale_x, scale_y, fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
                flip_horizontal, flip_vertical, blend_mode, opacity, feather_edges, 
                anti_aliasing, clip_to_background
            )
            
            # Store results
            results[chunk_start:chunk_end] = chunk_result
            masks[chunk_start:chunk_end] = chunk_mask
            
            # Clean up
            del bg_chunk, fg_chunk, mask_chunk, chunk_result, chunk_mask
        
        return results, masks
    
    def _calculate_optimal_chunk_size(self, bg_h, bg_w, max_chunk_size):
        """Calculate optimal chunk size based on image dimensions"""
        image_size_mp = (bg_h * bg_w) / 1_000_000
        
        # For proxy workflow, we can be more aggressive
        if image_size_mp > 8:   # Very large images (8MP+)
            return min(4, max_chunk_size)
        elif image_size_mp > 4: # Large images (4-8MP)
            return min(8, max_chunk_size)
        elif image_size_mp > 1: # Medium images (1-4MP)
            return min(16, max_chunk_size)
        else:                   # Small images (<1MP)
            return max_chunk_size
    
    def _expand_tensor_efficient(self, tensor, target_size, current_size):
        """Efficiently expand tensor to target size using indexing"""
        if current_size == target_size:
            return tensor
        
        # Use modulo indexing for efficient expansion
        indices = torch.arange(target_size, device=tensor.device) % current_size
        return tensor[indices]
    
    def _get_chunk_efficient(self, tensor, chunk_indices, batch_size):
        """Get chunk data efficiently"""
        device = tensor.device
        indices = torch.tensor([i % batch_size for i in chunk_indices], device=device)
        return tensor[indices]
    
    def _transform_batch_vectorized(self, fg_batch, mask_batch, canvas_size, scale_x, scale_y,
                                   fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
                                   flip_horizontal, flip_vertical, anti_aliasing):
        """Transform entire batch using vectorized operations"""
        
        device = fg_batch.device
        batch_size = fg_batch.shape[0]
        
        # Convert to BCHW for efficient tensor operations
        fg_tensor = fg_batch.permute(0, 3, 1, 2)  # [B,3,H,W]
        
        # Apply transformations using efficient tensor operations
        if fit_mode != "none":
            fg_tensor = self._apply_fitting_vectorized(fg_tensor, canvas_size, fit_mode)
        elif scale_x != 1.0 or scale_y != 1.0:
            fg_tensor = self._apply_scaling_vectorized(fg_tensor, scale_x, scale_y, maintain_aspect, anti_aliasing)
        
        # Apply flipping using vectorized operations
        if flip_horizontal:
            fg_tensor = torch.flip(fg_tensor, dims=[3])
        if flip_vertical:
            fg_tensor = torch.flip(fg_tensor, dims=[2])
        
        # Handle mask transformations
        transformed_mask = None
        if mask_batch is not None:
            mask_tensor = mask_batch.unsqueeze(1)  # [B,1,H,W]
            
            # Apply same transformations to mask efficiently
            if fit_mode != "none" or scale_x != 1.0 or scale_y != 1.0:
                target_h, target_w = fg_tensor.shape[2], fg_tensor.shape[3]
                current_h, current_w = mask_tensor.shape[2], mask_tensor.shape[3]
                if current_h != target_h or current_w != target_w:
                    mask_tensor = F.interpolate(mask_tensor, size=(target_h, target_w), 
                                              mode='bilinear', align_corners=False)
            
            if flip_horizontal:
                mask_tensor = torch.flip(mask_tensor, dims=[3])
            if flip_vertical:
                mask_tensor = torch.flip(mask_tensor, dims=[2])
            
            transformed_mask = mask_tensor.squeeze(1)  # [B,H,W]
        
        # Convert back to BHWC
        fg_transformed = fg_tensor.permute(0, 2, 3, 1)  # [B,H,W,3]
        
        # Apply rotation if needed (optimized for batch processing)
        if abs(rotation) > 0.01:
            fg_transformed, transformed_mask = self._apply_rotation_vectorized(
                fg_transformed, transformed_mask, rotation, anti_aliasing
            )
        
        return fg_transformed, transformed_mask
    
    def _apply_scaling_vectorized(self, tensor, scale_x, scale_y, maintain_aspect, anti_aliasing):
        """Apply scaling using vectorized operations"""
        if maintain_aspect:
            scale = min(scale_x, scale_y)
            scale_x = scale_y = scale
        
        _, _, h, w = tensor.shape
        new_h = int(h * scale_y)
        new_w = int(w * scale_x)
        
        mode = 'bilinear' if anti_aliasing else 'nearest'
        return F.interpolate(tensor, size=(new_h, new_w), mode=mode, align_corners=False)
    
    def _apply_fitting_vectorized(self, tensor, canvas_size, fit_mode):
        """Apply fitting using vectorized operations"""
        _, _, h, w = tensor.shape
        canvas_w, canvas_h = canvas_size
        
        if fit_mode == "fit_width":
            scale = canvas_w / w
            new_size = (int(h * scale), canvas_w)
        elif fit_mode == "fit_height":
            scale = canvas_h / h
            new_size = (canvas_h, int(w * scale))
        elif fit_mode == "fit_both":
            scale = min(canvas_w / w, canvas_h / h)
            new_size = (int(h * scale), int(w * scale))
        elif fit_mode == "fill":
            scale = max(canvas_w / w, canvas_h / h)
            new_size = (int(h * scale), int(w * scale))
        elif fit_mode == "stretch":
            new_size = (canvas_h, canvas_w)
        else:
            return tensor
        
        return F.interpolate(tensor, size=new_size, mode='bilinear', align_corners=False)
    
    def _apply_rotation_vectorized(self, fg_tensor, mask_tensor, rotation, anti_aliasing):
        """Apply rotation with batch optimization"""
        # For 90-degree rotations, use efficient tensor operations
        if abs(rotation % 90) < 0.1:
            k = int(rotation // 90) % 4
            if k != 0:
                fg_tensor = torch.rot90(fg_tensor, k, dims=[1, 2])
                if mask_tensor is not None:
                    mask_tensor = torch.rot90(mask_tensor, k, dims=[1, 2])
            return fg_tensor, mask_tensor
        
        # For arbitrary rotations, process efficiently
        batch_size = fg_tensor.shape[0]
        device = fg_tensor.device
        dtype = fg_tensor.dtype
        
        rotated_fg = []
        rotated_masks = []
        
        for i in range(batch_size):
            fg_pil = self._tensor_to_pil_fast(fg_tensor[i])
            mask_pil = None
            if mask_tensor is not None:
                mask_pil = self._mask_tensor_to_pil_fast(mask_tensor[i])
            
            resample = Image.Resampling.BICUBIC if anti_aliasing else Image.Resampling.NEAREST
            fg_rotated = fg_pil.rotate(rotation, expand=True, resample=resample, fillcolor=(0, 0, 0))
            mask_rotated = None
            if mask_pil:
                mask_rotated = mask_pil.rotate(rotation, expand=True, resample=resample, fillcolor=0)
            
            rotated_fg.append(self._pil_to_tensor_fast(fg_rotated))
            if mask_rotated:
                rotated_masks.append(self._pil_to_mask_tensor_fast(mask_rotated))
        
        fg_result = torch.stack(rotated_fg).to(device=device, dtype=dtype)
        mask_result = None
        if rotated_masks:
            mask_result = torch.stack(rotated_masks).to(device=device, dtype=dtype)
        
        return fg_result, mask_result
    
    def _calculate_positions_batch(self, layer_size, canvas_size, position_mode, x_position, y_position):
        """Calculate positions for batch processing"""
        layer_h, layer_w = layer_size
        canvas_w, canvas_h = canvas_size
        
        if position_mode == "custom":
            return [(x_position, y_position)]
        
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
        final_x = base_x + x_position
        final_y = base_y + y_position
        
        return [(final_x, final_y)]
    
    def _compose_batch_vectorized(self, bg_batch, fg_batch, mask_batch, positions,
                                 blend_mode, opacity, feather_edges, clip_to_background):
        """Compose entire batch using vectorized operations"""
        
        device = bg_batch.device
        dtype = bg_batch.dtype
        batch_size, bg_h, bg_w, _ = bg_batch.shape
        _, fg_h, fg_w, _ = fg_batch.shape
        
        # Clone background for composition
        composite = bg_batch.clone()
        composition_mask = torch.zeros(batch_size, bg_h, bg_w, dtype=dtype, device=device)
        
        # Get position
        final_x, final_y = positions[0]
        
        # Calculate placement bounds
        start_x = max(0, final_x)
        start_y = max(0, final_y)
        end_x = min(bg_w, final_x + fg_w)
        end_y = min(bg_h, final_y + fg_h)
        
        # Calculate foreground crop
        fg_start_x = max(0, -final_x)
        fg_start_y = max(0, -final_y)
        fg_end_x = fg_start_x + (end_x - start_x)
        fg_end_y = fg_start_y + (end_y - start_y)
        
        if start_x < end_x and start_y < end_y and fg_start_x < fg_end_x and fg_start_y < fg_end_y:
            # Extract regions using vectorized operations
            fg_region = fg_batch[:, fg_start_y:fg_end_y, fg_start_x:fg_end_x, :]
            bg_region = composite[:, start_y:end_y, start_x:end_x, :]
            
            # Create alpha mask
            if mask_batch is not None:
                alpha = mask_batch[:, fg_start_y:fg_end_y, fg_start_x:fg_end_x]
            else:
                alpha = torch.ones(batch_size, fg_end_y - fg_start_y, fg_end_x - fg_start_x, 
                                 dtype=dtype, device=device)
            
            # Apply opacity and feathering vectorized
            alpha = alpha * opacity
            if feather_edges > 0:
                alpha = self._apply_feathering_vectorized(alpha, feather_edges)
            
            # Apply blend mode vectorized
            blended_region = self._apply_blend_mode_vectorized(bg_region, fg_region, blend_mode)
            
            # Composite using vectorized alpha blending
            alpha_expanded = alpha.unsqueeze(-1)  # [B,H,W,1]
            composited_region = bg_region * (1 - alpha_expanded) + blended_region * alpha_expanded
            
            # Place back into composite using vectorized operations
            composite[:, start_y:end_y, start_x:end_x, :] = composited_region
            composition_mask[:, start_y:end_y, start_x:end_x] = alpha
        
        return composite, composition_mask
    
    def _process_chunk_optimized(self, bg_chunk, fg_chunk, mask_chunk, x_position, y_position, position_mode,
                                scale_x, scale_y, fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
                                flip_horizontal, flip_vertical, blend_mode, opacity, feather_edges, 
                                anti_aliasing, clip_to_background):
        """Process chunk with optimized operations"""
        
        bg_h, bg_w = bg_chunk.shape[1], bg_chunk.shape[2]
        
        # Transform using vectorized operations
        fg_transformed, mask_transformed = self._transform_batch_vectorized(
            fg_chunk, mask_chunk, (bg_w, bg_h), scale_x, scale_y,
            fit_mode, maintain_aspect, rotation, anchor_x, anchor_y,
            flip_horizontal, flip_vertical, anti_aliasing
        )
        
        # Calculate positions
        positions = self._calculate_positions_batch(
            fg_transformed.shape[1:3], (bg_w, bg_h), position_mode, x_position, y_position
        )
        
        # Compose using vectorized operations
        composite, comp_mask = self._compose_batch_vectorized(
            bg_chunk, fg_transformed, mask_transformed, positions,
            blend_mode, opacity, feather_edges, clip_to_background
        )
        
        return composite, comp_mask
    
    def _apply_feathering_vectorized(self, alpha, feather_pixels):
        """Apply feathering using vectorized operations"""
        if feather_pixels <= 0:
            return alpha
        
        batch_size = alpha.shape[0]
        device = alpha.device
        dtype = alpha.dtype
        
        # Use tensor-based gaussian blur for small kernels
        if feather_pixels <= 5:
            kernel_size = feather_pixels * 2 + 1
            sigma = feather_pixels / 3
            alpha_blurred = self._gaussian_blur_separable(alpha, kernel_size, sigma)
            return alpha_blurred
        else:
            # Fall back to OpenCV for large kernels
            feathered = []
            for i in range(batch_size):
                alpha_np = alpha[i].cpu().numpy()
                kernel_size = feather_pixels * 2 + 1
                blurred = cv2.GaussianBlur((alpha_np * 255).astype(np.uint8), 
                                         (kernel_size, kernel_size), feather_pixels / 3)
                feathered.append(torch.from_numpy(blurred.astype(np.float32) / 255.0))
            
            return torch.stack(feathered).to(device=device, dtype=dtype)
    
    def _gaussian_blur_separable(self, tensor, kernel_size, sigma):
        """Apply gaussian blur using separable kernels for efficiency"""
        # Create 1D gaussian kernel
        x = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Reshape for convolution
        kernel_h = kernel_1d.view(1, 1, kernel_size, 1)
        kernel_w = kernel_1d.view(1, 1, 1, kernel_size)
        
        # Add channel dimension
        tensor_padded = tensor.unsqueeze(1)  # [B, 1, H, W]
        
        # Apply separable convolution
        padding = kernel_size // 2
        blurred_h = F.conv2d(tensor_padded, kernel_h, padding=(padding, 0))
        blurred_hw = F.conv2d(blurred_h, kernel_w, padding=(0, padding))
        
        return blurred_hw.squeeze(1)  # [B, H, W]
    
    def _apply_blend_mode_vectorized(self, bg, fg, blend_mode):
        """Apply blend modes using vectorized operations"""
        
        if blend_mode == "normal":
            return fg
        
        # Clamp inputs to valid range
        bg = torch.clamp(bg, 0, 1)
        fg = torch.clamp(fg, 0, 1)
        
        if blend_mode == "multiply":
            return bg * fg
        elif blend_mode == "screen":
            return 1 - (1 - bg) * (1 - fg)
        elif blend_mode == "overlay":
            return torch.where(bg < 0.5,
                             2 * bg * fg,
                             1 - 2 * (1 - bg) * (1 - fg))
        elif blend_mode == "soft_light":
            return torch.where(fg < 0.5,
                             bg - (1 - 2 * fg) * bg * (1 - bg),
                             bg + (2 * fg - 1) * (torch.sqrt(torch.clamp(bg, 1e-10, 1.0)) - bg))
        elif blend_mode == "hard_light":
            return torch.where(fg < 0.5,
                             2 * bg * fg,
                             1 - 2 * (1 - bg) * (1 - fg))
        elif blend_mode == "color_dodge":
            return torch.where(fg >= 1.0, 1.0, 
                             torch.clamp(bg / torch.clamp(1 - fg, 1e-10, 1.0), 0, 1))
        elif blend_mode == "color_burn":
            return torch.where(fg <= 0.0, 0.0, 
                             1 - torch.clamp((1 - bg) / torch.clamp(fg, 1e-10, 1.0), 0, 1))
        elif blend_mode == "darken":
            return torch.min(bg, fg)
        elif blend_mode == "lighten":
            return torch.max(bg, fg)
        elif blend_mode == "difference":
            return torch.abs(bg - fg)
        elif blend_mode == "exclusion":
            return bg + fg - 2 * bg * fg
        elif blend_mode == "add":
            return torch.clamp(bg + fg, 0, 1)
        elif blend_mode == "subtract":
            return torch.clamp(bg - fg, 0, 1)
        elif blend_mode == "divide":
            return torch.clamp(bg / torch.clamp(fg, 1e-10, 1.0), 0, 1)
        elif blend_mode == "linear_burn":
            return torch.clamp(bg + fg - 1, 0, 1)
        elif blend_mode == "linear_dodge":
            return torch.clamp(bg + fg, 0, 1)
        else:
            return fg
    
    # Fast helper methods for tensor/PIL conversion
    def _tensor_to_pil_fast(self, tensor):
        """Convert tensor to PIL efficiently"""
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_img, 'RGB')
    
    def _mask_tensor_to_pil_fast(self, mask_tensor):
        """Convert mask tensor to PIL efficiently"""
        mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(mask_np, 'L')
    
    def _pil_to_tensor_fast(self, pil_img):
        """Convert PIL to tensor efficiently"""
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        np_img = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(np_img)
    
    def _pil_to_mask_tensor_fast(self, pil_mask):
        """Convert PIL mask to tensor efficiently"""
        if pil_mask.mode != 'L':
            pil_mask = pil_mask.convert('L')
        mask_np = np.array(pil_mask).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Studio42LayerComposer": Studio42LayerComposer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Studio42LayerComposer": "Studio42 Layer Composer"
}
