# 🚀 Studio42 LCARS Edition - ComfyUI

**Version 3.0.0 - The Future is Now!**

A stunning, futuristic update to ComfyUI featuring LCARS (Star Trek) + Hitchhiker's Guide to the Galaxy aesthetics, with integrated Qwen/Wan AI models and powerful procedural generation nodes.

---

## ✨ What's New in LCARS Edition?

### 🎨 **Futuristic UI Theme**
- **LCARS-inspired design** with sleek rounded corners
- **Orange glow effects** on all nodes
- **Sci-fi color schemes**: Orange, Blue Neon, Rainbow
- **Hitchhiker's Guide tooltips**: "DON'T PANIC" styling
- Custom fonts and animations

### 🤖 **Qwen AI Model Nodes**
Five powerful nodes for language and vision AI:
- **Qwen Text Generator**: Advanced text generation
- **Qwen Vision-Language**: Image understanding and Q&A
- **Qwen Image Caption**: Detailed image descriptions
- **Qwen Chat**: Multi-turn conversations with memory
- **Qwen Image-to-Text**: Generate and overlay text on images

### 🎬 **Wan Video Model Nodes**
Four nodes for video generation and processing:
- **Wan Video Generator**: Text-to-video generation
- **Wan Image-to-Video**: Animate still images
- **Wan Video Enhancer**: Denoise, sharpen, upscale, interpolate
- **Wan Audio-to-Video**: Create visualizations from audio

### 🌫️ **Procedural Generation Nodes**
Four powerful generators:
- **Noise Generator**: Perlin, Simplex, White, Pink, Blue, Fractal noise
- **Pattern Generator**: Grid, Hexagon, Voronoi, Circles, Waves, Spiral, LCARS patterns
- **Gradient Generator**: Linear, Radial, Angular, Diamond, Spiral, Conic gradients
- **Fractal Generator**: Mandelbrot, Julia, Burning Ship, Newton fractals

### 🎯 **Easy Custom Node Creation**
- Simple base classes (`LCARSImageNode`, `LCARSGeneratorNode`, etc.)
- Template generator system
- Quick creation functions
- Interactive node builder

---

## 📦 Installation

### Basic Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/GeekyGhost/24oiduts-ComfyUI.git
cd 24oiduts-ComfyUI
pip install -r requirements.txt
```

### Optional Dependencies

**For Qwen AI Models:**
```bash
pip install transformers torch
```

**For Wan Video Models:**
```bash
pip install diffusers opencv-python
```

**For Full Features:**
```bash
pip install transformers diffusers opencv-python ffmpeg-python torchaudio soundfile scipy
```

---

## 🎨 Using the LCARS Theme

The LCARS theme is automatically applied to all nodes. Customize it by editing:

```
web/css/lcars-theme.css
```

### Key Theme Features:
- **Node colors**: Defined by type (Qwen=Purple, Wan=Blue, Procedural=Tan)
- **Glow effects**: Adjustable spread and intensity
- **Rounded corners**: 20px default radius
- **Orange accent color**: `#ff9900`
- **Animations**: Scanning effects, pulse animations, wire flow

---

## 🤖 Qwen Nodes Guide

### 1. Qwen Text Generator
Generate text using state-of-the-art language models.

**Inputs:**
- `prompt`: Your text prompt
- `model_size`: Qwen-7B, Qwen-14B, Qwen-72B, Qwen2-7B
- `max_tokens`: Length of generation (1-2048)
- `temperature`: Creativity (0.1-2.0)
- `top_p`: Diversity (0.0-1.0)
- `system_prompt`: Optional system instructions

**Outputs:**
- `text`: Generated text

**Example:**
```
Prompt: "Write a sci-fi story about LCARS interfaces"
Model: Qwen2-7B
Temperature: 0.8
```

### 2. Qwen Vision-Language
Ask questions about images.

**Inputs:**
- `image`: Input image
- `question`: What do you want to know?
- `model_variant`: Qwen-VL or Qwen-VL-Chat
- `max_tokens`: Response length

**Outputs:**
- `answer`: Text response
- `annotated_image`: Original image (for chaining)

**Example:**
```
Image: [Your image]
Question: "What objects are in this image and where are they located?"
```

### 3. Qwen Image Caption
Generate detailed captions for images.

**Inputs:**
- `image`: Input image
- `caption_style`: Detailed, Brief, Creative, Technical
- `language`: English, Chinese, Auto

**Outputs:**
- `caption`: Generated caption

### 4. Qwen Chat
Multi-turn conversation with memory.

**Inputs:**
- `message`: Your message
- `conversation_id`: Unique ID for conversation thread
- `model_size`: Qwen2-7B or Qwen-14B
- `clear_history`: Reset conversation
- `system_prompt`: AI personality/instructions

**Outputs:**
- `response`: AI response
- `history`: Full conversation history

**Example:**
```
Message 1: "Hello! I'm working on a ComfyUI workflow."
Message 2: "Can you suggest some creative node combinations?"
Conversation ID: "my_session"
```

### 5. Qwen Image-to-Text
Generate text from images and optionally overlay it.

**Inputs:**
- `image`: Input image
- `prompt`: What text to generate
- `overlay_text`: Whether to draw text on image
- `text_color`: White, Black, Orange, Green

**Outputs:**
- `image`: Annotated image (if overlay enabled)
- `text`: Generated text

---

## 🎬 Wan Nodes Guide

### 1. Wan Video Generator
Generate videos from text prompts.

**Inputs:**
- `prompt`: Video description
- `num_frames`: 4-64 frames
- `width/height`: Resolution (256-1024)
- `fps`: Frame rate (1-30)
- `guidance_scale`: Prompt adherence (1-20)
- `num_inference_steps`: Quality (10-100)
- `negative_prompt`: What to avoid
- `seed`: Reproducibility (-1 for random)

**Outputs:**
- `frames`: Video frames as image batch

**Example:**
```
Prompt: "A futuristic spaceship flying through a nebula"
Frames: 16
Resolution: 512x512
FPS: 8
```

### 2. Wan Image-to-Video
Animate still images with motion.

**Inputs:**
- `image`: Starting image
- `motion_prompt`: Type of motion (pan, zoom, rotate)
- `num_frames`: Animation length
- `motion_intensity`: Effect strength (0-1)
- `fps`: Playback speed

**Outputs:**
- `video_frames`: Animated frames

**Motion Types:**
- "pan right/left/up/down"
- "zoom in/out"
- "rotate clockwise/counterclockwise"

### 3. Wan Video Enhancer
Improve video quality.

**Inputs:**
- `frames`: Input video frames
- `enhancement_type`: Denoise, Sharpen, Upscale, Interpolate
- `strength`: Effect intensity
- `upscale_factor`: 1-4x

**Outputs:**
- `enhanced_frames`: Processed frames

**Enhancement Types:**
- **Denoise**: Remove noise/grain
- **Sharpen**: Increase detail
- **Upscale**: Increase resolution
- **Interpolate**: Double frame rate

### 4. Wan Audio-to-Video
Create visual effects from audio.

**Inputs:**
- `audio`: Audio input
- `style`: Waveform, Spectrum, Bars, Particles, Abstract
- `width/height`: Resolution
- `color_scheme`: LCARS Orange, Neon Blue, Rainbow, Monochrome

**Outputs:**
- `video_frames`: Visualization frames

---

## 🌫️ Procedural Nodes Guide

### 1. Procedural Noise Generator
Generate various noise patterns.

**Inputs:**
- `width/height`: Resolution
- `noise_type`: Perlin, Simplex, White, Pink, Blue, Fractal
- `scale`: Pattern frequency (0.001-0.1)
- `octaves`: Detail levels (1-8)
- `persistence`: Amplitude falloff (0-1)
- `lacunarity`: Frequency multiplier (1-4)
- `seed`: Random seed

**Outputs:**
- `image`: Noise pattern

**Noise Types:**
- **Perlin**: Smooth, natural-looking
- **Simplex**: Similar to Perlin, faster
- **White**: Pure random
- **Pink**: 1/f noise, natural
- **Blue**: High-frequency
- **Fractal**: Inverted Perlin

### 2. Procedural Pattern Generator
Create geometric patterns.

**Inputs:**
- `width/height`: Resolution
- `pattern_type`: Grid, Hexagon, Voronoi, Circles, Waves, Spiral, LCARS
- `scale`: Pattern size
- `complexity`: Detail level
- `color_scheme`: Orange Glow, Blue Neon, Rainbow, Monochrome
- `seed`: Random seed

**Outputs:**
- `image`: Pattern

**Pattern Types:**
- **Grid**: Simple grid lines
- **Hexagon**: Hexagonal tiling
- **Voronoi**: Cellular pattern
- **Circles**: Random circles
- **Waves**: Sine wave patterns
- **Spiral**: Spiral curves
- **LCARS**: Star Trek interface style

### 3. Procedural Gradient Generator
Create beautiful gradients.

**Inputs:**
- `width/height`: Resolution
- `gradient_type`: Linear, Radial, Angular, Diamond, Spiral, Conic
- `color1_r/g/b`: Start color (0-1)
- `color2_r/g/b`: End color (0-1)
- `angle`: Rotation (0-360°)

**Outputs:**
- `image`: Gradient

**Gradient Types:**
- **Linear**: Straight line blend
- **Radial**: Center to edges
- **Angular**: Around a point
- **Diamond**: Diamond-shaped
- **Spiral**: Spiral pattern
- **Conic**: Circular sweep

### 4. Procedural Fractal Generator
Generate mathematical fractals.

**Inputs:**
- `width/height`: Resolution
- `fractal_type`: Mandelbrot, Julia, Burning Ship, Newton
- `max_iterations`: Detail (10-500)
- `zoom`: Magnification (0.1-10)
- `center_x/y`: Pan position (-2 to 2)
- `julia_c_real/imag`: Julia set parameters

**Outputs:**
- `image`: Fractal

**Fractal Types:**
- **Mandelbrot**: Classic fractal
- **Julia**: Julia sets
- **Burning Ship**: Warped Mandelbrot
- **Newton**: Newton's method visualization

---

## 🎯 Creating Custom Nodes

### Method 1: Using Base Classes

```python
from lcars_base_node import LCARSImageNode

class MyAwesomeNode(LCARSImageNode):
    @classmethod
    def define_inputs(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }

    def process(self, image, strength):
        result = image * strength
        result = torch.clamp(result, 0.0, 1.0)
        return (result,)
```

### Method 2: Template Generator

```python
from lcars_node_template_generator import quick_image_node

code = quick_image_node(
    name="MyBrightnessNode",
    emoji="☀️",
    description="Adjust image brightness",
    extra_inputs={
        "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0})
    },
    process_code="result = image * brightness\nresult = torch.clamp(result, 0.0, 1.0)",
    output_path="my_brightness_node.py"
)
```

### Method 3: Interactive Generator

```python
from lcars_node_template_generator import interactive_generate

interactive_generate()
```

Follow the prompts to create your node!

### Available Base Classes

- **`LCARSBaseNode`**: Generic base class
- **`LCARSImageNode`**: For image processing
- **`LCARSGeneratorNode`**: For generators (no input image)
- **`LCARSModelNode`**: For AI models (with caching)
- **`LCARSComboNode`**: For combining multiple inputs

### Utility Methods

All base classes include:
- `tensor_to_pil(tensor)`: Convert to PIL Image
- `pil_to_tensor(pil_img)`: Convert to tensor
- `batch_process(tensor, fn)`: Process batch
- `mask_to_tensor(mask)`: Mask to image
- `tensor_to_mask(tensor)`: Image to mask

Generator classes include:
- `create_blank_tensor(w, h, channels, fill)`: Create blank image

Model classes include:
- `load_model(name, load_fn)`: Load with caching
- `unload_model(name)`: Clear from cache

---

## 🎨 Customizing the LCARS Theme

Edit `web/css/lcars-theme.css` to customize:

### Color Scheme
```css
:root {
    --lcars-orange: #ff9900;
    --lcars-orange-dark: #cc6600;
    --lcars-orange-glow: rgba(255, 153, 0, 0.6);
    /* Change these to your preference */
}
```

### Glow Effect
```css
.comfy-node {
    box-shadow:
        0 0 var(--glow-spread) var(--node-glow),  /* Change glow-spread */
        0 4px 20px rgba(0, 0, 0, 0.5);
}
```

### Border Radius
```css
:root {
    --border-radius: 20px;  /* Adjust roundness */
}
```

---

## 📊 Node Categories

All nodes are organized into categories:

- **Studio42/LCARS/Qwen**: AI language and vision models
- **Studio42/LCARS/Wan**: Video generation and enhancement
- **Studio42/LCARS/Procedural**: Noise, patterns, gradients, fractals
- **Studio42/LCARS/Generators**: Procedural generators
- **Studio42/LCARS/Image**: Image processing
- **Studio42/LCARS/Custom**: Your custom nodes

---

## 🚀 Example Workflows

### Workflow 1: AI Image Captioning + Text Overlay
1. **Load Image** → Qwen Image Caption
2. **Qwen Image Caption** → Qwen Image-to-Text (caption as prompt)
3. **Qwen Image-to-Text** (overlay_text=True) → Save Image

### Workflow 2: Procedural Background + Video Generation
1. **Procedural Pattern Generator** (LCARS pattern) → image
2. **Wan Image-to-Video** (motion_prompt="pan right") → frames
3. **Wan Video Enhancer** (Interpolate) → smooth video
4. **Save Video**

### Workflow 3: Audio Visualization
1. **Load Audio** → audio
2. **Wan Audio-to-Video** (style=Spectrum) → frames
3. **Wan Video Enhancer** (Sharpen) → enhanced
4. **Save Video**

### Workflow 4: Fractal + Gradient Blend
1. **Procedural Fractal** (Mandelbrot) → fractal
2. **Procedural Gradient** (Radial, orange) → gradient
3. **Layer Composer** (blend mode=Overlay) → result

---

## 🐛 Troubleshooting

### Qwen Nodes Not Loading
```bash
pip install transformers torch
```

### Wan Nodes Not Loading
```bash
pip install diffusers opencv-python
```

### "Module not found" Error
```bash
cd ComfyUI/custom_nodes/24oiduts-ComfyUI
pip install -r requirements.txt
```

### Models Too Large for GPU
Edit the node code to use CPU:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",  # Change from "auto"
)
```

### Custom Nodes Not Appearing
1. Check your node file is in the custom_nodes directory
2. Ensure it exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
3. Restart ComfyUI

---

## 📚 Resources

- **ComfyUI Documentation**: https://github.com/comfyanonymous/ComfyUI
- **Qwen Models**: https://huggingface.co/Qwen
- **Diffusers**: https://huggingface.co/docs/diffusers
- **LCARS Design**: https://www.lcarsdisplay.com/

---

## 🤝 Contributing

We welcome contributions! To add your own nodes:

1. Use the template generator
2. Follow the LCARS styling guidelines
3. Include documentation
4. Submit a pull request

---

## 📄 License

This project inherits the license from the original Studio42 ComfyUI suite.

---

## 🎉 Credits

- **Original Studio42 Suite**: Advanced image/audio/video processing
- **LCARS Design**: Inspired by Star Trek interfaces
- **Hitchhiker's Guide**: "DON'T PANIC" philosophy
- **Qwen**: Alibaba Cloud's language models
- **ComfyUI**: The amazing node-based interface

---

## 🚀 DON'T PANIC - You're Ready!

Your LCARS ComfyUI installation is complete. Enjoy creating with futuristic style! 🌟

For help, check the examples in `lcars_node_template_generator.py` or explore the base classes in `lcars_base_node.py`.

**The future is now. Make something amazing!** ✨
