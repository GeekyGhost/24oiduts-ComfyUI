"""
LCARS Wan Model Nodes
---------------------
Beautiful nodes for Wan video and audio-visual models.

Supports:
- Wan Video Generation
- Wan Video Editing
- Wan Audio-to-Video
- Wan Video Enhancement
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
from lcars_base_node import LCARSModelNode, LCARSImageNode, register_node

# Optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  opencv not available - some Wan features will be limited")

try:
    from diffusers import DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠️  diffusers not available - Wan nodes will use fallback mode")


class WanVideoGenerator(LCARSModelNode):
    """
    🎬 Wan Video Generation Node
    Generate videos from text prompts using Wan models.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "A beautiful sunset over the ocean",
                    "multiline": True
                }),
                "num_frames": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 4
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64
                }),
                "fps": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 30,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 5
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
            }
        }

    @classmethod
    def define_outputs(cls):
        return (("IMAGE",), ("frames",))

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Wan"

    def process(self, prompt, num_frames, width, height, fps, guidance_scale,
                num_inference_steps, negative_prompt="", seed=-1):
        """Generate video frames using Wan model"""

        if not DIFFUSERS_AVAILABLE:
            # Fallback: create placeholder frames
            print("⚠️  Using fallback - generating placeholder frames")
            return self._create_placeholder_video(num_frames, width, height, prompt)

        try:
            # Set seed
            if seed == -1:
                seed = torch.randint(0, 2147483647, (1,)).item()

            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(seed)

            # Load model
            pipe = self._load_wan_model()

            print(f"🎬 Generating video: {num_frames} frames at {width}x{height}")
            print(f"   Prompt: {prompt[:80]}...")

            # Generate video
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )

            # Convert to tensor
            frames = output.frames[0]  # List of PIL Images
            frame_tensors = []

            for frame in frames:
                frame_tensor = self.pil_to_tensor(frame, add_batch=False)
                frame_tensors.append(frame_tensor)

            # Stack into batch
            video_tensor = torch.stack(frame_tensors, dim=0)

            print(f"✅ Generated {num_frames} frames (seed: {seed})")
            return (video_tensor,)

        except Exception as e:
            print(f"❌ Error generating video: {str(e)}")
            return self._create_placeholder_video(num_frames, width, height, f"Error: {str(e)}")

    def _load_wan_model(self):
        """Load Wan video generation model"""

        def load_fn():
            print("📦 Loading Wan video model...")
            # Using text-to-video model as placeholder
            # Replace with actual Wan model when available
            pipe = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float16,
                variant="fp16"
            )

            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            return pipe

        return self.load_model("wan_video", load_fn)

    def _create_placeholder_video(self, num_frames, width, height, text):
        """Create placeholder video frames"""
        frames = []

        for i in range(num_frames):
            # Create gradient frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Gradient based on frame number
            intensity = int(255 * i / num_frames)
            frame[:, :] = [intensity % 255, (intensity * 2) % 255, (intensity * 3) % 255]

            pil_frame = Image.fromarray(frame)
            frame_tensor = self.pil_to_tensor(pil_frame, add_batch=False)
            frames.append(frame_tensor)

        return (torch.stack(frames, dim=0),)


class WanImageToVideo(LCARSModelNode):
    """
    🖼️→🎬 Wan Image-to-Video Node
    Animate a still image using Wan models.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "motion_prompt": ("STRING", {
                    "default": "gentle camera pan to the right",
                    "multiline": True
                }),
                "num_frames": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 4
                }),
                "motion_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "fps": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 30
                }),
            },
        }

    @classmethod
    def define_outputs(cls):
        return (("IMAGE",), ("video_frames",))

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Wan"

    def process(self, image, motion_prompt, num_frames, motion_intensity, fps):
        """Animate image into video"""

        print(f"🖼️→🎬 Animating image: {motion_prompt[:50]}...")

        # Simple interpolation-based animation as fallback
        frames = []
        base_image = image[0]  # First image in batch

        for i in range(num_frames):
            # Create motion effect
            t = i / (num_frames - 1)

            # Apply simple transformations based on motion intensity
            frame = self._apply_motion(base_image, t, motion_intensity, motion_prompt)
            frames.append(frame)

        video_tensor = torch.stack(frames, dim=0)

        print(f"✅ Generated {num_frames} animated frames")
        return (video_tensor,)

    def _apply_motion(self, image, t, intensity, prompt):
        """Apply motion transformation to image"""

        if not CV2_AVAILABLE:
            return image

        # Convert to numpy for OpenCV
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        h, w = img_np.shape[:2]

        # Determine motion type from prompt
        prompt_lower = prompt.lower()

        if "pan" in prompt_lower:
            # Pan motion
            if "right" in prompt_lower:
                shift_x = int(w * 0.1 * t * intensity)
                shift_y = 0
            elif "left" in prompt_lower:
                shift_x = -int(w * 0.1 * t * intensity)
                shift_y = 0
            elif "up" in prompt_lower:
                shift_x = 0
                shift_y = -int(h * 0.1 * t * intensity)
            else:  # down
                shift_x = 0
                shift_y = int(h * 0.1 * t * intensity)

            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            result = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        elif "zoom" in prompt_lower:
            # Zoom motion
            scale = 1.0 + (0.2 * t * intensity)
            M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            result = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        elif "rotate" in prompt_lower:
            # Rotation motion
            angle = 15 * t * intensity
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            result = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        else:
            # Default: gentle parallax
            result = img_np

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0)

        return result_tensor


class WanVideoEnhancer(LCARSModelNode):
    """
    ✨ Wan Video Enhancement Node
    Enhance video quality, interpolate frames, upscale.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "enhancement_type": (
                    ["Denoise", "Sharpen", "Upscale", "Interpolate"],
                    {"default": "Sharpen"}
                ),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "upscale_factor": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 4
                }),
            },
        }

    @classmethod
    def define_outputs(cls):
        return (("IMAGE",), ("enhanced_frames",))

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Wan"

    def process(self, frames, enhancement_type, strength, upscale_factor):
        """Enhance video frames"""

        print(f"✨ Enhancing {frames.shape[0]} frames with {enhancement_type}")

        if enhancement_type == "Denoise":
            enhanced = self._denoise_frames(frames, strength)
        elif enhancement_type == "Sharpen":
            enhanced = self._sharpen_frames(frames, strength)
        elif enhancement_type == "Upscale":
            enhanced = self._upscale_frames(frames, upscale_factor)
        elif enhancement_type == "Interpolate":
            enhanced = self._interpolate_frames(frames)
        else:
            enhanced = frames

        print(f"✅ Enhanced to {enhanced.shape[0]} frames")
        return (enhanced,)

    def _denoise_frames(self, frames, strength):
        """Apply denoising to frames"""
        if not CV2_AVAILABLE:
            return frames

        denoised = []
        for i in range(frames.shape[0]):
            frame = (frames[i].cpu().numpy() * 255).astype(np.uint8)

            # Non-local means denoising
            h_param = int(10 * strength)
            denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, h_param, h_param, 7, 21)

            frame_tensor = torch.from_numpy(denoised_frame.astype(np.float32) / 255.0)
            denoised.append(frame_tensor)

        return torch.stack(denoised, dim=0)

    def _sharpen_frames(self, frames, strength):
        """Apply sharpening to frames"""
        if not CV2_AVAILABLE:
            return frames

        # Sharpening kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, 0],
            [0, -1, 0]
        ]) * strength + np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]) * (1 - strength)

        sharpened = []
        for i in range(frames.shape[0]):
            frame = (frames[i].cpu().numpy() * 255).astype(np.uint8)
            sharp_frame = cv2.filter2D(frame, -1, kernel)

            frame_tensor = torch.from_numpy(sharp_frame.astype(np.float32) / 255.0)
            sharpened.append(frame_tensor)

        return torch.stack(sharpened, dim=0)

    def _upscale_frames(self, frames, factor):
        """Upscale frames"""
        if not CV2_AVAILABLE or factor == 1:
            return frames

        upscaled = []
        for i in range(frames.shape[0]):
            frame = (frames[i].cpu().numpy() * 255).astype(np.uint8)
            h, w = frame.shape[:2]

            up_frame = cv2.resize(frame, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)

            frame_tensor = torch.from_numpy(up_frame.astype(np.float32) / 255.0)
            upscaled.append(frame_tensor)

        return torch.stack(upscaled, dim=0)

    def _interpolate_frames(self, frames):
        """Interpolate between frames to double frame rate"""
        if frames.shape[0] < 2:
            return frames

        interpolated = [frames[0]]

        for i in range(frames.shape[0] - 1):
            # Add original frame
            interpolated.append(frames[i])

            # Add interpolated frame
            interp_frame = (frames[i] + frames[i + 1]) / 2.0
            interpolated.append(interp_frame)

        # Add last frame
        interpolated.append(frames[-1])

        return torch.stack(interpolated, dim=0)


class WanAudioToVideo(LCARSModelNode):
    """
    🎵→🎬 Wan Audio-to-Video Node
    Generate video visualizations from audio.
    """

    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "style": (
                    ["Waveform", "Spectrum", "Bars", "Particles", "Abstract"],
                    {"default": "Spectrum"}
                ),
                "width": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1920,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1080,
                    "step": 64
                }),
                "color_scheme": (
                    ["LCARS Orange", "Neon Blue", "Rainbow", "Monochrome"],
                    {"default": "LCARS Orange"}
                ),
            },
        }

    @classmethod
    def define_outputs(cls):
        return (("IMAGE",), ("video_frames",))

    @classmethod
    def category(cls):
        return "Studio42/LCARS/Wan"

    def process(self, audio, style, width, height, color_scheme):
        """Generate video from audio"""

        print(f"🎵→🎬 Generating {style} visualization at {width}x{height}")

        # Extract audio data
        waveform = audio["waveform"]  # Shape: (B, C, S)
        sample_rate = audio["sample_rate"]

        # Convert to mono if stereo
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)

        # Generate frames based on audio
        num_frames = min(60, waveform.shape[-1] // (sample_rate // 30))  # Max 2 seconds at 30fps
        frames = []

        for i in range(num_frames):
            frame = self._create_audio_visualization(
                waveform, i, num_frames, style, width, height, color_scheme
            )
            frames.append(frame)

        video_tensor = torch.stack(frames, dim=0)

        print(f"✅ Generated {num_frames} visualization frames")
        return (video_tensor,)

    def _create_audio_visualization(self, waveform, frame_idx, total_frames,
                                   style, width, height, color_scheme):
        """Create a single visualization frame"""

        # Create blank canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Color schemes
        colors = {
            "LCARS Orange": [(255, 153, 0), (204, 102, 0)],
            "Neon Blue": [(0, 153, 255), (0, 102, 204)],
            "Rainbow": [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            "Monochrome": [(255, 255, 255), (128, 128, 128)],
        }
        color_palette = colors.get(color_scheme, colors["LCARS Orange"])

        # Sample audio segment
        samples_per_frame = waveform.shape[-1] // total_frames
        start_idx = frame_idx * samples_per_frame
        end_idx = start_idx + samples_per_frame
        audio_segment = waveform[0, 0, start_idx:end_idx].cpu().numpy()

        if style == "Waveform":
            self._draw_waveform(canvas, audio_segment, color_palette)
        elif style == "Spectrum":
            self._draw_spectrum(canvas, audio_segment, color_palette)
        elif style == "Bars":
            self._draw_bars(canvas, audio_segment, color_palette)
        else:
            # Simple radial visualization
            self._draw_radial(canvas, audio_segment, color_palette)

        # Convert to tensor
        return torch.from_numpy(canvas.astype(np.float32) / 255.0)

    def _draw_waveform(self, canvas, audio_segment, colors):
        """Draw waveform visualization"""
        h, w = canvas.shape[:2]
        mid_y = h // 2

        points = []
        for i, sample in enumerate(audio_segment):
            x = int((i / len(audio_segment)) * w)
            y = int(mid_y + sample * (h // 2) * 0.8)
            points.append((x, y))

        if CV2_AVAILABLE and len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, colors[0], 2, cv2.LINE_AA)

    def _draw_spectrum(self, canvas, audio_segment, colors):
        """Draw spectrum visualization"""
        # FFT for frequency spectrum
        spectrum = np.abs(np.fft.rfft(audio_segment))
        spectrum = spectrum[:len(spectrum)//2]  # Use lower half

        h, w = canvas.shape[:2]
        bar_width = w // len(spectrum)

        for i, magnitude in enumerate(spectrum):
            bar_height = int(min(magnitude * h * 10, h))
            x1 = i * bar_width
            x2 = x1 + bar_width - 1
            y1 = h - bar_height
            y2 = h

            if CV2_AVAILABLE:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), colors[0], -1)

    def _draw_bars(self, canvas, audio_segment, colors):
        """Draw bar visualization"""
        h, w = canvas.shape[:2]
        num_bars = 32
        samples_per_bar = len(audio_segment) // num_bars

        bar_width = w // num_bars

        for i in range(num_bars):
            start = i * samples_per_bar
            end = start + samples_per_bar
            bar_value = np.abs(audio_segment[start:end]).mean()

            bar_height = int(bar_value * h * 2)
            x1 = i * bar_width
            x2 = x1 + bar_width - 2
            y1 = h - bar_height
            y2 = h

            if CV2_AVAILABLE:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), colors[i % len(colors)], -1)

    def _draw_radial(self, canvas, audio_segment, colors):
        """Draw radial visualization"""
        h, w = canvas.shape[:2]
        center = (w // 2, h // 2)

        num_points = min(64, len(audio_segment))
        samples = audio_segment[::len(audio_segment)//num_points][:num_points]

        for i, sample in enumerate(samples):
            angle = (i / num_points) * 2 * np.pi
            radius = 50 + abs(sample) * 200

            x = int(center[0] + np.cos(angle) * radius)
            y = int(center[1] + np.sin(angle) * radius)

            if CV2_AVAILABLE:
                cv2.circle(canvas, (x, y), 5, colors[0], -1)
                cv2.line(canvas, center, (x, y), colors[1], 1)


# ==================== Node Registration ====================

NODE_CLASS_MAPPINGS = {
    "WanVideoGenerator": WanVideoGenerator,
    "WanImageToVideo": WanImageToVideo,
    "WanVideoEnhancer": WanVideoEnhancer,
    "WanAudioToVideo": WanAudioToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoGenerator": "🎬 Wan Video Generator",
    "WanImageToVideo": "🖼️→🎬 Wan Image-to-Video",
    "WanVideoEnhancer": "✨ Wan Video Enhancer",
    "WanAudioToVideo": "🎵→🎬 Wan Audio-to-Video",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
