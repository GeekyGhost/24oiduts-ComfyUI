import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import folder_paths
import cv2
from typing import Tuple, Optional, List
import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Suppress specific timm deprecation warnings
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Importing from timm.models.registry is deprecated.*", category=FutureWarning)

class Studio42VideoPatchLiftLoader:
    """
    🎬 Studio42 Video PatchLift Loader - Enhanced with Mask Output
    
    Advanced video loader with patch selection capabilities for ComfyUI.
    Loads video files and allows interactive selection of patches/regions to extract from each frame.
    
    NEW FEATURES:
    ✅ Mask output for extracted patches
    ✅ Better ComfyUI integration
    ✅ Enhanced error handling
    
    Features:
    - Load video files (mp4, webm, mkv, avi, mov, gif)
    - Visual grid preview with selection indicator
    - Frame sampling and selection controls
    - Patch extraction from video frames with masks
    - Outputs: Full video frames, Preview with selection overlay, Extracted patch sequence, Patch masks
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) 
                if os.path.isfile(os.path.join(input_dir, f)) and 
                f.lower().endswith(('.mp4', '.webm', '.mkv', '.avi', '.mov', '.gif'))]
        
        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
                "patch_x": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 1}),
                "patch_y": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 1}),
                "patch_width": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
                "patch_height": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
                "patch_shape": (["rectangle", "circle", "rounded_rectangle"], {"default": "rectangle"}),
            },
            "optional": {
                # Video loading controls
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0, "step": 0.1}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                
                # Preview controls
                "grid_size": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "grid_opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "selection_color": ("STRING", {"default": "#FF0000"}),
                "selection_thickness": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "show_coordinates": ("BOOLEAN", {"default": True}),
                "corner_radius": ("INT", {"default": 20, "min": 5, "max": 100, "step": 5}),
                "maintain_aspect": ("BOOLEAN", {"default": False}),
                "preview_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.25}),
            }
        }
    
    # UPDATED: Added MASK output
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("video_frames", "preview_frames", "patch_frames", "patch_masks", "patch_x", "patch_y", "patch_width", "patch_height")
    FUNCTION = "load_video_and_patch"
    CATEGORY = "🎬 Studio42/Video Processing"
    
    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        """Check if video file has changed"""
        try:
            video_path = folder_paths.get_annotated_filepath(video)
            if os.path.exists(video_path):
                return str(os.path.getmtime(video_path))
        except:
            pass
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        """Validate that the video file exists"""
        try:
            if not folder_paths.exists_annotated_filepath(video):
                return f"Invalid video file: {video}"
        except:
            return f"Could not validate video file: {video}"
        return True
    
    def load_video_and_patch(self, video, patch_x, patch_y, patch_width, patch_height, patch_shape,
                            frame_load_cap=0, skip_first_frames=0, select_every_nth=1, force_rate=0.0,
                            custom_width=0, custom_height=0, grid_size=32, grid_opacity=0.3, 
                            selection_color="#FF0000", selection_thickness=3, show_coordinates=True,
                            corner_radius=20, maintain_aspect=False, preview_scale=1.0):
        
        try:
            # Load the video
            video_path = folder_paths.get_annotated_filepath(video)
            
            # Load video frames using OpenCV
            video_frames = self._load_video_opencv(
                video_path, frame_load_cap, skip_first_frames, select_every_nth,
                force_rate, custom_width, custom_height
            )
            
            if not video_frames:
                raise ValueError(f"Failed to load video: {video}")
            
            # Get frame dimensions
            frame_height, frame_width = video_frames[0].shape[:2]
            
            # Adjust patch coordinates to stay within frame bounds
            patch_x = max(0, min(patch_x, frame_width - patch_width))
            patch_y = max(0, min(patch_y, frame_height - patch_height))
            
            # Handle aspect ratio maintenance
            if maintain_aspect:
                aspect_ratio = patch_width / patch_height
                if patch_y + patch_height > frame_height:
                    patch_height = frame_height - patch_y
                    patch_width = int(patch_height * aspect_ratio)
            
            # Process each frame
            preview_frames = []
            patch_frames = []
            patch_masks = []
            
            for frame_np in video_frames:
                # Convert to PIL for processing
                frame_pil = Image.fromarray((frame_np * 255).astype(np.uint8))
                
                # Create preview with overlay
                preview_pil = self._create_preview_frame(
                    frame_pil, patch_x, patch_y, patch_width, patch_height, patch_shape,
                    grid_size, grid_opacity, selection_color, selection_thickness,
                    show_coordinates, corner_radius, preview_scale
                )
                
                # Extract patch and mask
                patch_pil, mask_pil = self._extract_patch_and_mask_from_frame(
                    frame_pil, patch_x, patch_y, patch_width, patch_height, 
                    patch_shape, corner_radius
                )
                
                # Convert back to numpy
                preview_frames.append(np.array(preview_pil).astype(np.float32) / 255.0)
                
                # Ensure patch is RGB before converting to numpy
                if patch_pil.mode != 'RGB':
                    patch_pil = patch_pil.convert('RGB')
                patch_frames.append(np.array(patch_pil).astype(np.float32) / 255.0)
                
                # Convert mask to numpy
                if mask_pil.mode != 'L':
                    mask_pil = mask_pil.convert('L')
                patch_masks.append(np.array(mask_pil).astype(np.float32) / 255.0)
            
            # Convert to tensors
            video_tensor = torch.from_numpy(np.stack(video_frames, axis=0))
            preview_tensor = torch.from_numpy(np.stack(preview_frames, axis=0))
            patch_tensor = torch.from_numpy(np.stack(patch_frames, axis=0))
            mask_tensor = torch.from_numpy(np.stack(patch_masks, axis=0))
            
            logger.info(f"✅ Video patch lift completed: {len(video_frames)} frames processed")
            
            return (video_tensor, preview_tensor, patch_tensor, mask_tensor, patch_x, patch_y, patch_width, patch_height)
            
        except Exception as e:
            logger.error(f"❌ Video patch lift failed: {e}")
            # Create fallback tensors
            fallback_image = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
            fallback_mask = torch.zeros(1, 512, 512, dtype=torch.float32)
            return (fallback_image, fallback_image, fallback_image, fallback_mask, 0, 0, 256, 256)
    
    def _load_video_opencv(self, video_path: str, frame_load_cap: int, skip_first_frames: int,
                          select_every_nth: int, force_rate: float, custom_width: int, custom_height: int) -> List[np.ndarray]:
        """Load video using OpenCV with enhanced error handling"""
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frames = []
            frame_count = 0
            frames_added = 0
            
            # Skip initial frames
            for _ in range(skip_first_frames):
                ret = cap.grab()
                if not ret:
                    break
                frame_count += 1
            
            # Load frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every nth frame
                if frame_count % select_every_nth == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize if custom dimensions specified
                    if custom_width > 0 and custom_height > 0:
                        frame = cv2.resize(frame, (custom_width, custom_height), interpolation=cv2.INTER_LANCZOS4)
                    elif custom_width > 0:
                        height, width = frame.shape[:2]
                        aspect_ratio = height / width
                        new_height = int(custom_width * aspect_ratio)
                        frame = cv2.resize(frame, (custom_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                    elif custom_height > 0:
                        height, width = frame.shape[:2]
                        aspect_ratio = width / height
                        new_width = int(custom_height * aspect_ratio)
                        frame = cv2.resize(frame, (new_width, custom_height), interpolation=cv2.INTER_LANCZOS4)
                    
                    # Convert to float32 and normalize
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                    frames_added += 1
                    
                    # Check frame cap
                    if frame_load_cap > 0 and frames_added >= frame_load_cap:
                        break
                
                frame_count += 1
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames were loaded from the video")
            
            return frames
            
        except Exception as e:
            logger.error(f"OpenCV video loading failed: {e}")
            raise
    
    def _create_preview_frame(self, frame: Image.Image, patch_x: int, patch_y: int, 
                            patch_width: int, patch_height: int, patch_shape: str,
                            grid_size: int, grid_opacity: float, selection_color: str,
                            selection_thickness: int, show_coordinates: bool,
                            corner_radius: int, preview_scale: float) -> Image.Image:
        """Create preview frame with grid overlay and selection indicator"""
        
        # Scale frame if requested
        preview_frame = frame.copy()
        if preview_scale != 1.0:
            new_width = int(frame.width * preview_scale)
            new_height = int(frame.height * preview_scale)
            preview_frame = preview_frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Scale patch coordinates accordingly
            scaled_patch_x = int(patch_x * preview_scale)
            scaled_patch_y = int(patch_y * preview_scale)
            scaled_patch_width = int(patch_width * preview_scale)
            scaled_patch_height = int(patch_height * preview_scale)
        else:
            scaled_patch_x = patch_x
            scaled_patch_y = patch_y
            scaled_patch_width = patch_width
            scaled_patch_height = patch_height
        
        # Convert to RGBA for transparency effects
        preview_frame = preview_frame.convert('RGBA')
        
        # Create overlay layer
        overlay = Image.new('RGBA', preview_frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw grid
        if grid_size > 0:
            grid_color = (128, 128, 128, int(255 * grid_opacity))
            scaled_grid_size = int(grid_size * preview_scale)
            
            # Vertical lines
            for x in range(0, preview_frame.width, scaled_grid_size):
                draw.line([(x, 0), (x, preview_frame.height)], fill=grid_color, width=1)
            
            # Horizontal lines
            for y in range(0, preview_frame.height, scaled_grid_size):
                draw.line([(0, y), (preview_frame.width, y)], fill=grid_color, width=1)
        
        # Parse selection color
        try:
            if selection_color.startswith('#'):
                color_hex = selection_color[1:]
                r = int(color_hex[0:2], 16)
                g = int(color_hex[2:4], 16)
                b = int(color_hex[4:6], 16)
                sel_color = (r, g, b, 255)
            else:
                sel_color = (255, 0, 0, 255)  # Default red
        except:
            sel_color = (255, 0, 0, 255)  # Default red
        
        # Draw selection indicator based on shape
        if patch_shape == "rectangle":
            bbox = [scaled_patch_x, scaled_patch_y, 
                   scaled_patch_x + scaled_patch_width, scaled_patch_y + scaled_patch_height]
            
            for i in range(selection_thickness):
                draw.rectangle([bbox[0] - i, bbox[1] - i, bbox[2] + i, bbox[3] + i], 
                             outline=sel_color, fill=None)
                
        elif patch_shape == "circle":
            center_x = scaled_patch_x + scaled_patch_width // 2
            center_y = scaled_patch_y + scaled_patch_height // 2
            radius = min(scaled_patch_width, scaled_patch_height) // 2
            
            bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
            
            for i in range(selection_thickness):
                draw.ellipse([bbox[0] - i, bbox[1] - i, bbox[2] + i, bbox[3] + i], 
                           outline=sel_color, fill=None)
                
        elif patch_shape == "rounded_rectangle":
            bbox = [scaled_patch_x, scaled_patch_y, 
                   scaled_patch_x + scaled_patch_width, scaled_patch_y + scaled_patch_height]
            
            scaled_corner_radius = int(corner_radius * preview_scale)
            
            for i in range(selection_thickness):
                self._draw_rounded_rectangle(draw, [bbox[0] - i, bbox[1] - i, bbox[2] + i, bbox[3] + i], 
                                           scaled_corner_radius, outline=sel_color)
        
        # Add coordinate text if requested
        if show_coordinates:
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            coord_text = f"Frame: ({patch_x}, {patch_y}) - {patch_width}x{patch_height}"
            text_bbox = draw.textbbox((0, 0), coord_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = scaled_patch_x
            text_y = max(0, scaled_patch_y - text_height - 5)
            
            # Draw text background
            draw.rectangle([text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2], 
                         fill=(0, 0, 0, 180))
            
            # Draw text
            draw.text((text_x, text_y), coord_text, fill=(255, 255, 255, 255), font=font)
        
        # Composite overlay with frame
        result = Image.alpha_composite(preview_frame, overlay)
        return result.convert('RGB')
    
    def _extract_patch_and_mask_from_frame(self, frame: Image.Image, patch_x: int, patch_y: int,
                                         patch_width: int, patch_height: int, patch_shape: str,
                                         corner_radius: int) -> Tuple[Image.Image, Image.Image]:
        """Extract patch and mask from frame - UPDATED to return both"""
        
        if patch_shape == "rectangle":
            # Extract rectangular patch normally
            patch = frame.crop((patch_x, patch_y, patch_x + patch_width, patch_y + patch_height))
            # Create full opacity mask for rectangle
            mask = Image.new('L', (patch_width, patch_height), 255)
            return patch, mask
            
        elif patch_shape == "circle":
            # Extract circular patch
            circle_size = min(patch_width, patch_height)
            circle_x = patch_x + (patch_width - circle_size) // 2
            circle_y = patch_y + (patch_height - circle_size) // 2
            
            # Extract the square region containing the circle
            circle_crop = frame.crop((circle_x, circle_y, circle_x + circle_size, circle_y + circle_size))
            
            # Create circular mask
            mask = Image.new('L', (circle_size, circle_size), 0)
            mask_draw = ImageDraw.Draw(mask)
            center = circle_size // 2
            radius = circle_size // 2
            mask_draw.ellipse([center - radius, center - radius, 
                              center + radius, center + radius], fill=255)
            
            return circle_crop, mask
                
        elif patch_shape == "rounded_rectangle":
            # Extract rounded rectangle patch
            rect_crop = frame.crop((patch_x, patch_y, patch_x + patch_width, patch_y + patch_height))
            
            # Create rounded rectangle mask
            mask = Image.new('L', (patch_width, patch_height), 0)
            mask_draw = ImageDraw.Draw(mask)
            self._draw_rounded_rectangle(mask_draw, [0, 0, patch_width, patch_height], 
                                       corner_radius, fill=255)
            
            return rect_crop, mask
        
        # Fallback to rectangle
        patch = frame.crop((patch_x, patch_y, patch_x + patch_width, patch_y + patch_height))
        mask = Image.new('L', (patch_width, patch_height), 255)
        return patch, mask
    
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


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Studio42VideoPatchLiftLoader": Studio42VideoPatchLiftLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Studio42VideoPatchLiftLoader": "🎬 Studio42 Video PatchLift Loader"
}