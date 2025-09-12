import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import folder_paths
from typing import Tuple, Optional
import warnings

# Suppress specific timm deprecation warnings
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Importing from timm.models.registry is deprecated.*", category=FutureWarning)

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available for PatchLift Loader. Some features will use fallback methods.")

class Studio42PatchLiftLoader:
    """
    Advanced image loader with patch selection capabilities.
    Loads an image and allows interactive selection of patches/regions to extract.
    Features:
    - Load image like standard LoadImage node
    - Visual grid preview with selection indicator
    - Adjustable selection box position (X, Y)
    - Customizable selection box size (Width, Height)  
    - Shape options (rectangle/circle)
    - Outputs: Full image, Preview with selection overlay, Extracted patch
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) 
                if os.path.isfile(os.path.join(input_dir, f)) and 
                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
        
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "patch_x": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 1}),
                "patch_y": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 1}),
                "patch_width": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
                "patch_height": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
                "patch_shape": (["rectangle", "circle", "rounded_rectangle"], {"default": "rectangle"}),
            },
            "optional": {
                "grid_size": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "grid_opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "selection_color": ("COLOR",),
                "selection_thickness": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "show_coordinates": ("BOOLEAN", {"default": True}),
                "corner_radius": ("INT", {"default": 20, "min": 5, "max": 100, "step": 5}),
                "maintain_aspect": ("BOOLEAN", {"default": False}),
                "preview_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.25}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "preview_image", "patch_image", "patch_x", "patch_y", "patch_width", "patch_height")
    FUNCTION = "load_and_patch"
    CATEGORY = "Studio42/Image Loading"
    
    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        """Check if image file has changed"""
        image_path = folder_paths.get_annotated_filepath(image)
        if os.path.exists(image_path):
            return str(os.path.getmtime(image_path))
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        """Validate that the image file exists"""
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
    
    def load_and_patch(self, image, patch_x, patch_y, patch_width, patch_height, patch_shape,
                      grid_size=32, grid_opacity=0.3, selection_color="#FF0000", 
                      selection_thickness=3, show_coordinates=True, corner_radius=20,
                      maintain_aspect=False, preview_scale=1.0):
        
        # Load the image
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Adjust patch coordinates to stay within image bounds
        patch_x = max(0, min(patch_x, img_width - patch_width))
        patch_y = max(0, min(patch_y, img_height - patch_height))
        
        # Handle aspect ratio maintenance
        if maintain_aspect:
            aspect_ratio = patch_width / patch_height
            # Adjust height to maintain aspect ratio
            patch_height = int(patch_width / aspect_ratio)
            # Ensure height doesn't exceed image bounds
            if patch_y + patch_height > img_height:
                patch_height = img_height - patch_y
                patch_width = int(patch_height * aspect_ratio)
        
        # Create preview image with grid and selection overlay
        preview_img = self._create_preview_image(
            img, patch_x, patch_y, patch_width, patch_height, patch_shape,
            grid_size, grid_opacity, selection_color, selection_thickness,
            show_coordinates, corner_radius, preview_scale
        )
        
        # Extract the patch
        patch_img = self._extract_patch(
            img, patch_x, patch_y, patch_width, patch_height, patch_shape, corner_radius
        )
        
        # Convert images to tensors
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        preview_tensor = torch.from_numpy(np.array(preview_img).astype(np.float32) / 255.0).unsqueeze(0)
        
        # Ensure patch is RGB before tensor conversion
        if patch_img.mode != 'RGB':
            patch_img = patch_img.convert('RGB')
        patch_tensor = torch.from_numpy(np.array(patch_img).astype(np.float32) / 255.0).unsqueeze(0)
        
        return (img_tensor, preview_tensor, patch_tensor, patch_x, patch_y, patch_width, patch_height)
    
    def _create_preview_image(self, img: Image.Image, patch_x: int, patch_y: int, 
                            patch_width: int, patch_height: int, patch_shape: str,
                            grid_size: int, grid_opacity: float, selection_color: int,
                            selection_thickness: int, show_coordinates: bool,
                            corner_radius: int, preview_scale: float) -> Image.Image:
        """Create preview image with grid overlay and selection indicator"""
        
        # Scale image if requested
        preview_img = img.copy()
        if preview_scale != 1.0:
            new_width = int(img.width * preview_scale)
            new_height = int(img.height * preview_scale)
            preview_img = preview_img.resize((new_width, new_height), Image.LANCZOS)
            
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
        preview_img = preview_img.convert('RGBA')
        
        # Create overlay layer
        overlay = Image.new('RGBA', preview_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw grid
        if grid_size > 0:
            grid_color = (128, 128, 128, int(255 * grid_opacity))
            scaled_grid_size = int(grid_size * preview_scale)
            
            # Vertical lines
            for x in range(0, preview_img.width, scaled_grid_size):
                draw.line([(x, 0), (x, preview_img.height)], fill=grid_color, width=1)
            
            # Horizontal lines  
            for y in range(0, preview_img.height, scaled_grid_size):
                draw.line([(0, y), (preview_img.width, y)], fill=grid_color, width=1)
        
        # Parse selection color from ComfyUI COLOR widget (comes as hex like "#FF0000")
        try:
            if isinstance(selection_color, str) and selection_color.startswith('#'):
                # Remove # and convert hex to RGB
                color_hex = selection_color[1:]
                r = int(color_hex[0:2], 16)
                g = int(color_hex[2:4], 16)  
                b = int(color_hex[4:6], 16)
                sel_color = (r, g, b, 255)
            else:
                # Fallback if not hex format
                sel_color = (255, 0, 0, 255)  # Default red
        except:
            sel_color = (255, 0, 0, 255)  # Default red
        
        # Draw selection indicator based on shape
        if patch_shape == "rectangle":
            # Rectangle selection
            bbox = [
                scaled_patch_x, 
                scaled_patch_y,
                scaled_patch_x + scaled_patch_width, 
                scaled_patch_y + scaled_patch_height
            ]
            
            # Draw selection rectangle
            for i in range(selection_thickness):
                draw.rectangle([
                    bbox[0] - i, bbox[1] - i,
                    bbox[2] + i, bbox[3] + i
                ], outline=sel_color, fill=None)
                
        elif patch_shape == "circle":
            # Circle selection
            center_x = scaled_patch_x + scaled_patch_width // 2
            center_y = scaled_patch_y + scaled_patch_height // 2
            radius = min(scaled_patch_width, scaled_patch_height) // 2
            
            bbox = [
                center_x - radius,
                center_y - radius, 
                center_x + radius,
                center_y + radius
            ]
            
            # Draw selection circle
            for i in range(selection_thickness):
                draw.ellipse([
                    bbox[0] - i, bbox[1] - i,
                    bbox[2] + i, bbox[3] + i
                ], outline=sel_color, fill=None)
                
        elif patch_shape == "rounded_rectangle":
            # Rounded rectangle selection
            bbox = [
                scaled_patch_x,
                scaled_patch_y,
                scaled_patch_x + scaled_patch_width,
                scaled_patch_y + scaled_patch_height
            ]
            
            scaled_corner_radius = int(corner_radius * preview_scale)
            
            # Draw rounded rectangle (simplified approach)
            for i in range(selection_thickness):
                self._draw_rounded_rectangle(draw, [
                    bbox[0] - i, bbox[1] - i,
                    bbox[2] + i, bbox[3] + i
                ], scaled_corner_radius, outline=sel_color)
        
        # Add coordinate text if requested
        if show_coordinates:
            try:
                # Try to use a default font
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
            
            coord_text = f"({patch_x}, {patch_y}) - {patch_width}x{patch_height}"
            text_bbox = draw.textbbox((0, 0), coord_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text near the selection
            text_x = scaled_patch_x
            text_y = max(0, scaled_patch_y - text_height - 5)
            
            # Draw text background
            draw.rectangle([
                text_x - 2, text_y - 2,
                text_x + text_width + 2, text_y + text_height + 2
            ], fill=(0, 0, 0, 180))
            
            # Draw text
            draw.text((text_x, text_y), coord_text, fill=(255, 255, 255, 255), font=font)
        
        # Composite overlay with image
        result = Image.alpha_composite(preview_img, overlay)
        return result.convert('RGB')
    
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
    
    def _extract_patch(self, img: Image.Image, patch_x: int, patch_y: int,
                      patch_width: int, patch_height: int, patch_shape: str,
                      corner_radius: int) -> Image.Image:
        """Extract the selected patch from the image with true shape-based cropping"""
        
        if patch_shape == "rectangle":
            # Extract rectangular patch normally
            patch = img.crop((patch_x, patch_y, patch_x + patch_width, patch_y + patch_height))
            return patch
            
        elif patch_shape == "circle":
            # True circular extraction with chroma green background
            # Use square dimensions based on the smaller of width/height for perfect circle
            circle_size = min(patch_width, patch_height)
            
            # Calculate centered circle position
            circle_x = patch_x + (patch_width - circle_size) // 2
            circle_y = patch_y + (patch_height - circle_size) // 2
            
            # Extract the square region containing the circle
            circle_crop = img.crop((circle_x, circle_y, circle_x + circle_size, circle_y + circle_size))
            
            # Create chroma green background canvas
            chroma_green = (0, 255, 0)
            result_canvas = Image.new('RGB', (circle_size, circle_size), chroma_green)
            
            # Create circular mask
            mask = Image.new('L', (circle_size, circle_size), 0)
            mask_draw = ImageDraw.Draw(mask)
            
            center = circle_size // 2
            radius = circle_size // 2
            
            mask_draw.ellipse([center - radius, center - radius, 
                              center + radius, center + radius], fill=255)
            
            # Composite circular content onto chroma green background
            circle_crop_rgba = circle_crop.convert('RGBA')
            circle_crop_rgba.putalpha(mask)
            
            result_canvas = result_canvas.convert('RGBA')
            result_canvas = Image.alpha_composite(result_canvas, circle_crop_rgba)
            
            return result_canvas.convert('RGB')
                
        elif patch_shape == "rounded_rectangle":
            # True rounded rectangle extraction with chroma green background
            # Extract the rectangular region
            rect_crop = img.crop((patch_x, patch_y, patch_x + patch_width, patch_y + patch_height))
            
            # Create chroma green background canvas
            chroma_green = (0, 255, 0)
            result_canvas = Image.new('RGB', (patch_width, patch_height), chroma_green)
            
            # Create rounded rectangle mask
            mask = Image.new('L', (patch_width, patch_height), 0)
            mask_draw = ImageDraw.Draw(mask)
            
            self._draw_rounded_rectangle(mask_draw, [0, 0, patch_width, patch_height], 
                                       corner_radius, fill=255)
            
            # Composite rounded rectangle content onto chroma green background
            rect_crop_rgba = rect_crop.convert('RGBA')
            rect_crop_rgba.putalpha(mask)
            
            result_canvas = result_canvas.convert('RGBA')
            result_canvas = Image.alpha_composite(result_canvas, rect_crop_rgba)
            
            return result_canvas.convert('RGB')
        
        # Fallback to rectangle if unknown shape
        return img.crop((patch_x, patch_y, patch_x + patch_width, patch_y + patch_height))
    
    def _apply_rounded_corners(self, img: Image.Image, radius: int) -> Image.Image:
        """Apply rounded corners to an image and convert to RGB"""
        # Create mask for rounded corners
        mask = Image.new('L', img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # Draw rounded rectangle mask
        self._draw_rounded_rectangle(mask_draw, [0, 0, img.width, img.height], radius, fill=255)
        
        # Apply mask to image
        img_rgba = img.convert('RGBA')
        img_rgba.putalpha(mask)
        
        # Composite onto black background and convert to RGB
        background = Image.new('RGB', img.size, (0, 0, 0))
        result = Image.alpha_composite(background.convert('RGBA'), img_rgba).convert('RGB')
        
        return result

# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Studio42PatchLiftLoader": Studio42PatchLiftLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Studio42PatchLiftLoader": "Studio42 PatchLift Loader"
}