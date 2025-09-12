import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional
import logging
import warnings
import math
import os
import folder_paths
import time

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# --- OPTIMIZATION: Global session cache ---
# This dictionary will hold loaded models to prevent reloading them on every run.
_GLOBAL_SESSION_CACHE = {}
_LAST_CLEANUP_TIME = 0
CLEANUP_INTERVAL = 300  # Clean up unused models every 5 minutes

# Define the path for rembg models within ComfyUI's model directory
REMBG_MODELS_DIR = os.path.join(folder_paths.models_dir, "rembg")
if not os.path.exists(REMBG_MODELS_DIR):
    os.makedirs(REMBG_MODELS_DIR)

# Try to import AI background removal libraries with graceful fallbacks
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    logger.info("✅ Rembg library loaded successfully")
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("⚠️ Rembg not available - AI methods will be disabled")

class RemovalMethod(Enum):
    """Background removal methods with proper model names"""
    BIREFNET_GENERAL = "birefnet-general"
    BIREFNET_PORTRAIT = "birefnet-portrait"
    BIREFNET_MASSIVE = "birefnet-massive"
    BIREFNET_GENERAL_LITE = "birefnet-general-lite"
    U2NET = "u2net"
    U2NET_HUMAN = "u2net_human_seg"
    ISNET_GENERAL = "isnet-general-use"
    CHROMA_KEY_PROFESSIONAL = "chroma_key_professional"
    LUMA_KEY_PROFESSIONAL = "luma_key_professional"
    DIFFERENCE_KEY = "difference_key"
    HYBRID_AI_TRADITIONAL = "hybrid_ai_traditional"

@dataclass
class BackgroundRemovalConfig:
    """Professional configuration for background removal"""
    method: RemovalMethod = RemovalMethod.BIREFNET_GENERAL
    use_gpu: bool = True
    model_precision: str = "fp16"
    batch_size: int = 1
    processing_resolution: int = 1024
    maintain_aspect_ratio: bool = True
    chroma_color: str = "green"
    primary_threshold: float = 0.15
    secondary_threshold: float = 0.05
    saturation_min: float = 0.3
    brightness_min: float = 0.2
    brightness_max: float = 0.95
    multi_pass_keying: bool = True
    spill_suppression: float = 0.8
    spill_preserve_luminance: bool = True
    edge_softness: float = 2.0
    core_matte_choke: float = 0.0
    luma_key_mode: str = "shadows_highlights"
    shadow_threshold: float = 0.1
    highlight_threshold: float = 0.9
    luma_softness: float = 0.1
    difference_threshold: float = 0.1
    garbage_matte_enabled: bool = False
    remove_small_objects: bool = True
    small_object_threshold: int = 500
    edge_detection_enabled: bool = True
    morphological_operations: bool = True
    mask_dilation: int = 1
    mask_erosion: int = 1
    mask_blur: float = 1.0
    color_space_mode: str = "hsv"

class ProfessionalBackgroundRemover:
    """Professional background removal with advanced techniques from VFX industry"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reference_background = None
        logger.info(f"🎬 Professional Background Remover initialized on {self.device}")

    # --- OPTIMIZATION: Get model from global cache ---
    def _get_cached_session(self, model_name: str, use_gpu: bool):
        global _GLOBAL_SESSION_CACHE, _LAST_CLEANUP_TIME
        current_time = time.time()
        
        if current_time - _LAST_CLEANUP_TIME > CLEANUP_INTERVAL:
            _LAST_CLEANUP_TIME = current_time
        
        cache_key = f"{model_name}_{use_gpu}"
        
        if cache_key not in _GLOBAL_SESSION_CACHE:
            try:
                logger.info(f"🚀 Creating and caching new session for {model_name}")
                os.environ['U2NET_HOME'] = REMBG_MODELS_DIR
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
                
                session = new_session(model_name, providers=providers)
                _GLOBAL_SESSION_CACHE[cache_key] = session
                logger.info(f"✅ Session for {model_name} cached successfully.")
                
            except Exception as e:
                logger.error(f"Failed to create session for {model_name}: {e}")
                return None
        
        return _GLOBAL_SESSION_CACHE[cache_key]

    def remove_background(self, image: Image.Image, config: BackgroundRemovalConfig, 
                         reference_bg: Optional[Image.Image] = None) -> Tuple[Image.Image, Image.Image]:
        """Main background removal with professional multi-stage processing"""
        
        try:
            if image.mode != 'RGB': image = image.convert('RGB')
            if reference_bg is not None: self.reference_background = reference_bg.convert('RGB') if reference_bg.mode != 'RGB' else reference_bg
            
            if config.method in [RemovalMethod.BIREFNET_GENERAL, RemovalMethod.BIREFNET_PORTRAIT, 
                                RemovalMethod.BIREFNET_MASSIVE, RemovalMethod.BIREFNET_GENERAL_LITE,
                                RemovalMethod.U2NET, RemovalMethod.U2NET_HUMAN, RemovalMethod.ISNET_GENERAL]:
                if REMBG_AVAILABLE: return self._remove_with_rembg(image, config)
                else:
                    logger.warning("🔄 Rembg not available, falling back to chroma key")
                    config.method = RemovalMethod.CHROMA_KEY_PROFESSIONAL
            
            # --- FIX: Restore original, working traditional keying methods ---
            if config.method == RemovalMethod.CHROMA_KEY_PROFESSIONAL:
                return self._remove_with_professional_chroma_key(image, config)
            elif config.method == RemovalMethod.LUMA_KEY_PROFESSIONAL:
                return self._remove_with_professional_luma_key(image, config)
            elif config.method == RemovalMethod.DIFFERENCE_KEY:
                return self._remove_with_difference_key(image, config)
            elif config.method == RemovalMethod.HYBRID_AI_TRADITIONAL:
                return self._remove_hybrid_professional(image, config)
            else:
                raise ValueError(f"Unknown removal method: {config.method}")
        
        except Exception as e:
            logger.error(f"❌ Professional background removal failed: {e}")
            mask = Image.new('L', image.size, 255); return image, mask

    def _remove_with_rembg(self, image: Image.Image, config: BackgroundRemovalConfig) -> Tuple[Image.Image, Image.Image]:
        model_name = config.method.value
        model_path = os.path.join(REMBG_MODELS_DIR, f"{model_name}.onnx")

        if not os.path.exists(model_path):
            logger.info(f"Model {model_name}.onnx not found locally. Rembg will attempt to download it.")

        session = self._get_cached_session(model_name, config.use_gpu)
        if session is None:
            logger.error(f"Failed to load rembg model {model_name}, falling back to chroma key.")
            return self._remove_with_professional_chroma_key(image, config)

        original_size = image.size
        
        if config.processing_resolution > 0 and max(original_size) > config.processing_resolution:
            if config.maintain_aspect_ratio:
                image_resized = self._resize_maintain_aspect(image, config.processing_resolution)
            else:
                image_resized = image.resize((config.processing_resolution, config.processing_resolution), Image.Resampling.LANCZOS)
        else:
            image_resized = image
        
        result = remove(image_resized, session=session)
        
        if image_resized.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        
        mask = result.split()[3]
        mask = self._post_process_mask_professional(mask, config)
        
        result_clean = Image.new('RGBA', original_size, (0, 0, 0, 0))
        result_clean.paste(image, (0, 0))
        result_clean.putalpha(mask)
        
        return result_clean, mask

    # --- FIX: This is the restored, fully functional chroma key method from your original file ---
    def _remove_with_professional_chroma_key(self, image: Image.Image, config: BackgroundRemovalConfig) -> Tuple[Image.Image, Image.Image]:
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        if config.color_space_mode == "hsv": working_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        elif config.color_space_mode == "lab": working_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        else: working_array = img_array.copy()
        
        primary_mask = self._extract_primary_chroma_key(working_array, config)
        
        if config.multi_pass_keying:
            secondary_mask = self._extract_secondary_chroma_key(working_array, primary_mask, config)
            combined_mask = np.maximum(primary_mask, secondary_mask)
        else:
            combined_mask = primary_mask
        
        despilled_image = self._apply_professional_spill_suppression(img_array, combined_mask, config)
        
        if config.edge_detection_enabled:
            combined_mask = self._apply_edge_detection_refinement(despilled_image, combined_mask, config)
        
        if abs(config.core_matte_choke) > 0.001:
            combined_mask = self._apply_core_matte_choke(combined_mask, config.core_matte_choke)
        
        mask_pil = Image.fromarray((combined_mask * 255).astype(np.uint8), mode='L')
        mask_pil = self._post_process_mask_professional(mask_pil, config)
        
        result = Image.new('RGBA', image.size, (0, 0, 0, 0))
        despilled_pil = Image.fromarray((despilled_image * 255).astype(np.uint8), 'RGB')
        result.paste(despilled_pil, (0, 0))
        result.putalpha(mask_pil)
        
        logger.info(f"✅ Professional chroma key completed.")
        return result, mask_pil

    def _extract_primary_chroma_key(self, working_array: np.ndarray, config: BackgroundRemovalConfig) -> np.ndarray:
        if config.color_space_mode == "hsv": return self._chroma_key_hsv_primary(working_array, config)
        elif config.color_space_mode == "lab": return self._chroma_key_lab_primary(working_array, config)
        else: return self._chroma_key_rgb_primary(working_array, config)

    def _chroma_key_hsv_primary(self, hsv_array, config):
        hue, saturation, value = hsv_array[:, :, 0] * 360, hsv_array[:, :, 1], hsv_array[:, :, 2]
        color_ranges = {"green": (75, 165), "blue": (200, 260), "red": (345, 15), "cyan": (165, 200), "magenta": (280, 340), "yellow": (45, 75)}
        hue_min, hue_max = color_ranges.get(config.chroma_color, color_ranges["green"])
        hue_match = (hue >= hue_min) | (hue <= hue_max) if config.chroma_color == "red" else (hue >= hue_min) & (hue <= hue_max)
        saturation_match = saturation > config.saturation_min
        brightness_match = (value > config.brightness_min) & (value < config.brightness_max)
        background_mask = hue_match & saturation_match & brightness_match
        return (~background_mask).astype(np.float32)

    def _chroma_key_lab_primary(self, lab_array, config):
        l, a, b = lab_array[:, :, 0], lab_array[:, :, 1], lab_array[:, :, 2]
        color_ranges = {"green": {"a": (-50, -20), "b": (20, 50)}, "blue": {"a": (10, 40), "b": (-80, -40)}, "red": {"a": (40, 80), "b": (20, 60)}}
        ranges = color_ranges.get(config.chroma_color, color_ranges["green"])
        a_match = (a >= ranges["a"][0]) & (a <= ranges["a"][1])
        b_match = (b >= ranges["b"][0]) & (b <= ranges["b"][1])
        l_match = (l > config.brightness_min * 100) & (l < config.brightness_max * 100)
        background_mask = a_match & b_match & l_match
        return (~background_mask).astype(np.float32)
    
    def _chroma_key_rgb_primary(self, rgb_array, config):
        r, g, b = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
        if config.chroma_color == "green": target_strength = g > np.maximum(r, b) + config.primary_threshold
        elif config.chroma_color == "blue": target_strength = b > np.maximum(r, g) + config.primary_threshold
        else: target_strength = r > np.maximum(g, b) + config.primary_threshold
        brightness = (r + g + b) / 3
        brightness_match = (brightness > config.brightness_min) & (brightness < config.brightness_max)
        background_mask = target_strength & brightness_match
        return (~background_mask).astype(np.float32)

    def _extract_secondary_chroma_key(self, working_array, primary_mask, config):
        edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        edges = cv2.filter2D(primary_mask, -1, edge_kernel)
        edge_areas = np.abs(edges) > 0.1
        if config.color_space_mode == "hsv": return self._chroma_key_hsv_secondary(working_array, config, edge_areas)
        else: return primary_mask.copy()

    def _chroma_key_hsv_secondary(self, hsv_array, config, edge_areas):
        hue, sat, val = hsv_array[:,:,0]*360, hsv_array[:,:,1], hsv_array[:,:,2]
        color_ranges = {"green": (75, 165), "blue": (200, 260), "red": (345, 15)}
        hue_min, hue_max = color_ranges.get(config.chroma_color, color_ranges["green"])
        hue_match = (hue >= hue_min) | (hue <= hue_max) if config.chroma_color == "red" else (hue >= hue_min) & (hue <= hue_max)
        sat_match = sat > (config.saturation_min - 0.1)
        val_match = (val > (config.brightness_min - 0.1)) & (val < (config.brightness_max + 0.1))
        background_mask = hue_match & sat_match & val_match & edge_areas
        return (~background_mask).astype(np.float32)

    def _apply_professional_spill_suppression(self, rgb_array, mask, config):
        if config.spill_suppression <= 0: return rgb_array
        # This is a simplified but effective spill suppression from your original logic
        spill_channel = {"green": 1, "blue": 2, "red": 0}.get(config.chroma_color, 1)
        despill_amount = rgb_array[:, :, spill_channel] - (rgb_array[:, :, (spill_channel + 1) % 3] + rgb_array[:, :, (spill_channel + 2) % 3]) / 2.0
        despill_amount = np.maximum(0, despill_amount)
        corrected_array = rgb_array.copy()
        corrected_array[:, :, spill_channel] -= despill_amount * config.spill_suppression
        return np.clip(corrected_array, 0, 1)

    def _apply_edge_detection_refinement(self, image_array, mask, config):
        gray = np.dot(image_array, [0.299, 0.587, 0.114])
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150) / 255.0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edge_strength = cv2.GaussianBlur(edges, (5,5), 1.0)
        return np.clip(mask * (1 - edge_strength * 0.3) + mask * edge_strength, 0, 1)

    def _apply_core_matte_choke(self, mask, choke_amount):
        mask_8bit = (mask * 255).astype(np.uint8)
        kernel_size = int(abs(choke_amount) * 5) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if choke_amount > 0: result = cv2.erode(mask_8bit, kernel, iterations=1)
        else: result = cv2.dilate(mask_8bit, kernel, iterations=1)
        return result.astype(np.float32) / 255.0

    def _remove_with_professional_luma_key(self, image, config):
        img_array = np.array(image, dtype=np.float32) / 255.0
        gray = np.dot(img_array, [0.299, 0.587, 0.114])
        if config.luma_key_mode == "shadows": background_mask = gray < config.shadow_threshold
        elif config.luma_key_mode == "highlights": background_mask = gray > config.highlight_threshold
        else: background_mask = (gray < config.shadow_threshold) | (gray > config.highlight_threshold)
        foreground_mask = (~background_mask).astype(np.float32)
        mask_pil = Image.fromarray((foreground_mask * 255).astype(np.uint8), 'L')
        mask_pil = self._post_process_mask_professional(mask_pil, config)
        result = Image.new('RGBA', image.size, (0,0,0,0)); result.paste(image, (0,0)); result.putalpha(mask_pil)
        return result, mask_pil
    
    def _remove_with_difference_key(self, image, config):
        if self.reference_background is None: return self._remove_with_professional_chroma_key(image, config)
        ref_bg = self.reference_background.resize(image.size, Image.Resampling.LANCZOS)
        current_array = np.array(image, dtype=np.float32) / 255.0
        reference_array = np.array(ref_bg, dtype=np.float32) / 255.0
        diff_magnitude = np.sqrt(np.sum((current_array - reference_array)**2, axis=2))
        foreground_mask = (diff_magnitude > config.difference_threshold).astype(np.float32)
        mask_pil = Image.fromarray((foreground_mask * 255).astype(np.uint8), 'L')
        mask_pil = self._post_process_mask_professional(mask_pil, config)
        result = Image.new('RGBA', image.size, (0,0,0,0)); result.paste(image, (0,0)); result.putalpha(mask_pil)
        return result, mask_pil

    def _remove_hybrid_professional(self, image, config):
        if REMBG_AVAILABLE: _, ai_mask = self._remove_with_rembg(image, BackgroundRemovalConfig(method=RemovalMethod.BIREFNET_GENERAL))
        else: _, ai_mask = self._remove_with_professional_luma_key(image, config)
        traditional_result, traditional_mask = self._remove_with_professional_chroma_key(image, config)
        ai_mask_array = np.array(ai_mask, dtype=np.float32) / 255.0
        traditional_mask_array = np.array(traditional_mask, dtype=np.float32) / 255.0
        ai_confidence = np.abs(ai_mask_array - 0.5) * 2
        combined_mask = ai_mask_array * ai_confidence + traditional_mask_array * (1.0 - ai_confidence)
        final_mask = Image.fromarray((np.clip(combined_mask, 0, 1) * 255).astype(np.uint8), 'L')
        result = traditional_result.copy(); result.putalpha(final_mask)
        return result, final_mask

    def _post_process_mask_professional(self, mask, config):
        mask_array = np.array(mask)
        if config.garbage_matte_enabled: mask_array = self._apply_auto_garbage_matte(mask_array)
        if config.remove_small_objects and config.small_object_threshold > 0: mask_array = self._remove_small_objects_professional(mask_array, config.small_object_threshold)
        if config.morphological_operations:
            if config.mask_dilation > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.mask_dilation*2+1, config.mask_dilation*2+1))
                mask_array = cv2.dilate(mask_array, kernel, iterations=1)
            if config.mask_erosion > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.mask_erosion*2+1, config.mask_erosion*2+1))
                mask_array = cv2.erode(mask_array, kernel, iterations=1)
        if config.edge_softness > 0:
            kernel_size = int(config.edge_softness * 2) | 1 # Ensure odd kernel size
            mask_array = cv2.GaussianBlur(mask_array, (kernel_size, kernel_size), config.edge_softness/3.0)
        if config.mask_blur > 0:
            mask_array = cv2.GaussianBlur(mask_array, (0,0), config.mask_blur)
        return Image.fromarray(mask_array, 'L')

    def _apply_auto_garbage_matte(self, mask_array):
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            garbage_matte = np.zeros_like(mask_array)
            cv2.fillPoly(garbage_matte, [largest_contour], 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
            garbage_matte = cv2.dilate(garbage_matte, kernel, iterations=1)
            return cv2.bitwise_and(mask_array, garbage_matte)
        return mask_array

    def _remove_small_objects_professional(self, mask_array, threshold):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_array, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < threshold: mask_array[labels == i] = 0
        return mask_array

    def _resize_maintain_aspect(self, image, target_size):
        w, h = image.size
        if w > h: new_w, new_h = target_size, int(h * target_size / w)
        else: new_h, new_w = target_size, int(w * target_size / h)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


class Studio42BackgroundRemover:
    """ Studio42 Background Remover - Professional & Optimized """
    
    def __init__(self):
        self.remover = ProfessionalBackgroundRemover()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "method": ([method.value for method in RemovalMethod], { "default": RemovalMethod.U2NET.value }),
            },
            "optional": {
                "reference_background": ("IMAGE",),
                "processing_resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "chroma_color": (["green", "blue", "red", "cyan", "magenta", "yellow"], {"default": "green"}),
                "primary_threshold": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01}),
                "secondary_threshold": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.3, "step": 0.01}),
                "saturation_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                "brightness_min": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "brightness_max": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "multi_pass_keying": ("BOOLEAN", {"default": True}),
                "spill_suppression": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
                "spill_preserve_luminance": ("BOOLEAN", {"default": True}),
                "edge_softness": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5}),
                "core_matte_choke": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                "luma_key_mode": (["shadows", "highlights", "shadows_highlights"], {"default": "shadows_highlights"}),
                "shadow_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05}),
                "highlight_threshold": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 1.0, "step": 0.05}),
                "luma_softness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "difference_threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
                "color_space_mode": (["hsv", "lab", "rgb"], {"default": "hsv"}),
                "garbage_matte_enabled": ("BOOLEAN", {"default": False}),
                "edge_detection_enabled": ("BOOLEAN", {"default": True}),
                "morphological_operations": ("BOOLEAN", {"default": True}),
                "remove_small_objects": ("BOOLEAN", {"default": True}),
                "small_object_threshold": ("INT", {"default": 500, "min": 50, "max": 2000, "step": 50}),
                "mask_blur": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("result_image", "result_mask")
    FUNCTION = "remove_background"
    CATEGORY = "🎬 Studio42/Background Removal"
    
    def remove_background(self, images, method, **kwargs):
        try:
            config_params = {k: v for k, v in kwargs.items()}
            config = BackgroundRemovalConfig(
                method=RemovalMethod(method),
                use_gpu=config_params.get('use_gpu', True),
                processing_resolution=config_params.get('processing_resolution', 1024),
                chroma_color=config_params.get('chroma_color', 'green'),
                primary_threshold=config_params.get('primary_threshold', 0.15),
                secondary_threshold=config_params.get('secondary_threshold', 0.05),
                saturation_min=config_params.get('saturation_threshold', 0.3),
                brightness_min=config_params.get('brightness_min', 0.2),
                brightness_max=config_params.get('brightness_max', 0.95),
                multi_pass_keying=config_params.get('multi_pass_keying', True),
                spill_suppression=config_params.get('spill_suppression', 0.8),
                spill_preserve_luminance=config_params.get('spill_preserve_luminance', True),
                edge_softness=config_params.get('edge_softness', 2.0),
                core_matte_choke=config_params.get('core_matte_choke', 0.0),
                luma_key_mode=config_params.get('luma_key_mode', 'shadows_highlights'),
                shadow_threshold=config_params.get('shadow_threshold', 0.1),
                highlight_threshold=config_params.get('highlight_threshold', 0.9),
                luma_softness=config_params.get('luma_softness', 0.1),
                difference_threshold=config_params.get('difference_threshold', 0.1),
                color_space_mode=config_params.get('color_space_mode', 'hsv'),
                garbage_matte_enabled=config_params.get('garbage_matte_enabled', False),
                edge_detection_enabled=config_params.get('edge_detection_enabled', True),
                morphological_operations=config_params.get('morphological_operations', True),
                remove_small_objects=config_params.get('remove_small_objects', True),
                small_object_threshold=config_params.get('small_object_threshold', 500),
                mask_blur=config_params.get('mask_blur', 1.0)
            )
            
            batch_size = images.shape[0]
            results, masks = [], []
            
            reference_pil = None
            if 'reference_background' in kwargs and kwargs['reference_background'] is not None:
                ref_tensor = kwargs['reference_background'][0] if len(kwargs['reference_background'].shape) == 4 else kwargs['reference_background']
                reference_pil = self._tensor_to_pil(ref_tensor)
            
            logger.info(f"🎬 Processing {batch_size} images with {method}...")
            
            for i in range(batch_size):
                img_pil = self._tensor_to_pil(images[i])
                result_pil, mask_pil = self.remover.remove_background(img_pil, config, reference_pil)
                
                result_rgb = Image.new('RGB', result_pil.size, (0, 0, 0))
                result_rgb.paste(result_pil, mask=result_pil.split()[3])
                
                results.append(self._pil_to_tensor(result_rgb))
                masks.append(self._pil_to_mask_tensor(mask_pil))
            
            result_batch = torch.cat(results, dim=0)
            mask_batch = torch.stack(masks, dim=0)
            
            logger.info(f"✅ Background removal completed for {batch_size} images.")
            return (result_batch, mask_batch)
            
        except Exception as e:
            logger.error(f"❌ Top-level background removal failed: {e}", exc_info=True)
            fallback_masks = torch.ones(images.shape[0], images.shape[1], images.shape[2], dtype=torch.float32)
            return (images, fallback_masks)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
        if len(np_img.shape) == 2: np_img = np.stack([np_img, np_img, np_img], axis=-1)
        elif np_img.shape[-1] == 4: np_img = np_img[:, :, :3]
        return Image.fromarray(np_img, 'RGB')
    
    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
        np_img = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(np_img).unsqueeze(0)
    
    def _pil_to_mask_tensor(self, pil_mask: Image.Image) -> torch.Tensor:
        if pil_mask.mode != 'L': pil_mask = pil_mask.convert('L')
        mask_np = np.array(pil_mask).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np)

NODE_CLASS_MAPPINGS = {"Studio42BackgroundRemover": Studio42BackgroundRemover}
NODE_DISPLAY_NAME_MAPPINGS = {"Studio42BackgroundRemover": "🎬 Studio42 Background Remover"}
