# 🎬 Studio42 Image, Audio, and Video Editing Suite for ComfyUI

This is a work in progress and not recommended for use at this time. 

**Professional-grade custom nodes for advanced image and video editing workflows in ComfyUI**

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/your-repo/Studio42-ComfyUI)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🌟 Overview

Studio42 is a comprehensive suite of advanced custom nodes that brings professional-grade image and video editing capabilities to ComfyUI. Designed for efficiency, quality, and creative flexibility, this suite provides cutting-edge background removal, layer composition, and patch manipulation tools used in modern VFX and content creation workflows.

## 🎯 Key Features

### 🎨 Professional Background Removal
- **Multi-method AI processing** with BiRefNet, U2Net, and ISNet models
- **Industry-standard keying** with professional chroma key, luma key, and difference key
- **Advanced spill suppression** with luminance preservation
- **Multi-stage processing** for complex scenes and challenging backgrounds
- **Batch processing** with automatic memory optimization

### 🎭 Advanced Layer Composition
- **Precise positioning** with pixel-perfect coordinate control
- **Professional transformations** including scaling, rotation, and anchor points
- **Comprehensive blend modes** (Normal, Multiply, Screen, Overlay, Soft Light, etc.)
- **Edge processing** with feathering and anti-aliasing
- **Real-time preview** with visual feedback

### 🎯 Interactive Patch Processing
- **Visual patch selection** with grid overlay and shape tools
- **Multi-format support** for images and video sequences
- **Shape-aware extraction** (Rectangle, Circle, Rounded Rectangle)
- **Intelligent placement** with automatic and manual positioning
- **Temporal consistency** for video workflows

## 📦 Node Catalog

### 1. 🎬 Studio42 Background Remover
*Professional multi-method background removal with VFX-quality results*

**AI Methods:**
- `birefnet-general` - Best general purpose (recommended)
- `birefnet-portrait` - Optimized for human portraits
- `birefnet-massive` - Highest accuracy, largest dataset
- `birefnet-general-lite` - Lightweight version for faster processing
- `u2net` - Classic reliable model
- `u2net_human_seg` - Human segmentation specialist
- `isnet-general-use` - ISNet general model

**Professional Traditional Methods:**
- `chroma_key_professional` - Multi-stage chroma keying with spill suppression
- `luma_key_professional` - Advanced luminance-based removal
- `difference_key` - Background difference detection
- `hybrid_ai_traditional` - AI + traditional combined approach

**Advanced Features:**
- Professional spill suppression with luminance preservation
- Multi-stage processing pipeline (After Effects/Nuke quality)
- Edge detection and refinement
- Core matte adjustment (choke/expand)
- Morphological operations for cleanup
- Support for HSV, LAB, and RGB color spaces

### 2. 🎭 Studio42 Layer Composer
*Professional layer composition with full creative control*

**Positioning & Transforms:**
- Direct pixel coordinates (matches other Studio42 nodes)
- Preset positioning modes (center, corners, edges)
- Independent X/Y scaling with aspect ratio control
- 360-degree rotation with custom anchor points
- Horizontal/vertical flipping

**Blend Modes:**
- Normal, Multiply, Screen, Overlay
- Soft Light, Hard Light, Color Dodge, Color Burn
- Darken, Lighten, Difference, Exclusion
- Add, Subtract, Divide, Linear Burn, Linear Dodge

**Quality Controls:**
- Edge feathering for smooth transitions
- Anti-aliasing for high-quality scaling
- Clipping control for boundary handling
- Fit modes (fit width/height/both, fill, stretch)

### 3. 🎯 Studio42 PatchLift Loader
*Interactive image loading with visual patch selection*

**Visual Selection:**
- Grid overlay with adjustable size and opacity
- Real-time coordinate display
- Customizable selection colors and thickness
- Shape-based selection (Rectangle, Circle, Rounded Rectangle)

**Smart Extraction:**
- True shape-based cropping with masks
- Chroma green background for non-rectangular shapes
- Aspect ratio maintenance options
- Preview scaling for detailed work

### 4. 🎯 Studio42 PatchDrop
*Intelligent patch placement with advanced integration*

**Placement Control:**
- Automatic coordinate-based placement
- Manual position override
- Center and preset positioning modes
- Boundary clipping and overflow handling

**Transform & Effects:**
- Scale and rotation with quality resampling
- Edge feathering for seamless integration
- Multiple blend modes for creative effects
- Opacity control with professional alpha compositing

**Mask Support:**
- Optional patch mask input for transparency control
- Shape-aware placement masks
- Combined mask processing for complex compositions

### 5. 🎥 Studio42 Video PatchLift Loader
*Advanced video loading with frame-by-frame patch selection*

**Video Loading:**
- Support for MP4, WebM, MKV, AVI, MOV, GIF formats
- Frame sampling controls (skip, select every nth)
- Custom resolution and frame rate override
- Batch frame processing with memory optimization

**Patch Selection:**
- Consistent patch extraction across video sequences
- Visual overlay with frame indicators
- Temporal coordinate display
- **NEW:** Mask output for extracted patches

### 6. 🎥 Studio42 Video PatchDrop
*Professional video patch placement with temporal consistency*

**Video Processing:**
- Frame-by-frame patch placement
- Temporal consistency controls for smooth motion
- Motion blur for natural integration
- Edge blending for seamless transitions

**Advanced Features:**
- Position smoothing across frames
- Video-optimized blend modes
- Temporal mask consistency
- Professional preview with visual indicators

## 🚀 Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager in your ComfyUI interface
2. Search for "Studio42"
3. Click **Install**
4. Restart ComfyUI

### Method 2: Manual Installation
```bash
# Navigate to ComfyUI custom nodes directory
cd ComfyUI/custom_nodes/

# Clone the repository
git clone https://github.com/your-repo/Studio42-ComfyUI.git

# Install dependencies
pip install -r Studio42-ComfyUI/requirements.txt

# Restart ComfyUI
```

### Method 3: Git Installation in ComfyUI
```bash
# In ComfyUI custom_nodes directory
git clone https://github.com/your-repo/Studio42-ComfyUI.git
```

## 📋 Dependencies

### Core Requirements (Always Needed)
```txt
Pillow>=9.5.0
opencv-python>=4.8.0
numpy>=1.24.0
```

### AI Background Removal (Optional - Graceful Fallback)
```bash
# For Rembg 2.0 with GPU acceleration
pip install rembg[gpu]

# For BiRefNet models
pip install transformers>=4.30.0 timm

# For GPU support (choose one)
pip install onnxruntime-gpu>=1.15.0      # NVIDIA
pip install onnxruntime-rocm>=1.15.0     # AMD
```

### Video Processing (Required for Video Features)
```bash
pip install ffmpeg-python>=0.2.0
```

### Performance Recommendations
- **CPU-only:** All traditional methods work perfectly
- **4GB+ VRAM:** Recommended for AI background removal
- **8GB+ VRAM:** Optimal for BiRefNet HR models and batch processing
- **System FFmpeg:** Install system FFmpeg for optimal video performance

## 🎨 Professional Workflows

### 1. Advanced Background Replacement
```
Image → Studio42 Background Remover (BiRefNet-General)
     → Studio42 Layer Composer (Soft Light blend)
     → Professional result with natural integration
```

**Settings for Best Results:**
- Method: `birefnet-general` or `birefnet-portrait` for people
- Enable `multi_pass_keying` for complex edges
- Use `spill_suppression: 0.8` for color bleeding removal
- Apply `edge_softness: 2.0` for smooth transitions

### 2. Selective Image Enhancement
```
Image → Studio42 PatchLift Loader (select area)
     → Apply effects/filters to patch
     → Studio42 PatchDrop (overlay blend, feathered edges)
     → Seamlessly integrated enhancement
```

**Pro Tips:**
- Use `feather_edges: 5-15` for natural blending
- Try `overlay` or `soft_light` blend modes for enhancements
- Enable `resize_patch: false` to maintain original quality

### 3. Professional Chroma Key (Green Screen)
```
Green Screen Image → Studio42 Background Remover
```

**Professional Settings:**
- Method: `chroma_key_professional`
- Color: `green`
- Primary threshold: `0.15` (adjust for your green screen)
- Enable `spill_suppression: 0.8`
- Enable `spill_preserve_luminance: true`
- Set `edge_softness: 2.0` for natural edges

### 4. Video Patch Processing
```
Video → Studio42 Video PatchLift Loader
     → Apply temporal effects
     → Studio42 Video PatchDrop (with temporal consistency)
     → Professional video result
```

**Video Settings:**
- Enable `temporal_consistency: 0.3` for smooth motion
- Use `motion_blur: 1.0` for natural movement
- Enable `edge_blending: true` for seamless integration

### 5. Batch Background Removal
```
Multiple Images → Studio42 Background Remover
```

**Batch Optimization:**
- Start with `batch_size: 4` and adjust based on VRAM
- Use `use_fp16: true` for memory efficiency
- Consider `processing_resolution: 1024` for speed vs quality balance

## ⚙️ Advanced Configuration

### Background Remover Pro Settings

**For Portraits:**
```
Method: birefnet-portrait
Multi-pass keying: true
Spill suppression: 0.8
Edge softness: 2.0
Color space: hsv
```

**For Product Photography:**
```
Method: chroma_key_professional
Chroma color: green (or blue)
Primary threshold: 0.12
Spill suppression: 0.9
Edge detection: true
```

**For Complex Scenes:**
```
Method: hybrid_ai_traditional
Use both AI and traditional methods
Enable garbage matte
Morphological operations: true
```

### Layer Composer Pro Settings

**For Realistic Compositing:**
```
Blend mode: soft_light or overlay
Opacity: 0.85-0.95
Feather edges: 3-8 pixels
Anti-aliasing: true
```

**For Creative Effects:**
```
Blend mode: screen, multiply, or difference
Experiment with anchor points for rotation
Use fit modes for automatic sizing
```

## 🛠️ Troubleshooting

### Common Issues

**Memory Issues:**
- Reduce `batch_size` parameter (try 2 or 1)
- Enable `use_fp16: true` for AI methods
- Use traditional methods for low-VRAM systems
- Close other applications using GPU memory

**AI Models Not Loading:**
- Check internet connection (models download automatically)
- Verify sufficient disk space (models can be 100MB+)
- Traditional methods will activate automatically as fallback
- Try restarting ComfyUI after first model download

**Video Processing Issues:**
- Install system FFmpeg for best compatibility
- Check video format compatibility (MP4/WebM recommended)
- For large videos, consider reducing resolution or frame rate
- Use `frame_load_cap` to limit memory usage

**Installation Problems:**
- Ensure Python environment matches ComfyUI
- Install dependencies in the correct virtual environment
- Check for conflicting package versions
- Try manual dependency installation

### Performance Optimization

**For Speed:**
- Use traditional methods (chroma/luma key)
- Reduce `processing_resolution`
- Disable advanced features for batch processing
- Use smaller `batch_size` to prevent memory swapping

**For Quality:**
- Use BiRefNet models for AI processing
- Enable multi-pass keying
- Use higher `processing_resolution`
- Enable edge detection and morphological operations

**For Memory Efficiency:**
- Enable `use_fp16: true`
- Use batch processing with appropriate `batch_size`
- Traditional methods use minimal memory
- Close other GPU applications

## 🔧 Technical Details

### Image Format Support
- **Input:** RGB, RGBA, Grayscale (auto-converted to RGB)
- **Output:** RGB for images, separate alpha masks
- **Internal:** Professional RGBA compositing
- **Bit Depth:** 8-bit input/output, 32-bit float internal processing

### Video Format Support
- **Containers:** MP4, WebM, MKV, AVI, MOV, GIF
- **Codecs:** H.264, H.265, VP8, VP9, AV1 (depends on system FFmpeg)
- **Features:** Frame sampling, resolution override, frame rate control

### Color Space Support
- **RGB:** Standard processing mode
- **HSV:** Recommended for chroma keying
- **LAB:** Professional color-accurate processing
- **Auto-conversion:** Seamless between color spaces

### Performance Characteristics
- **Memory Usage:** Optimized for batch processing
- **GPU Acceleration:** CUDA and ROCm support
- **CPU Fallback:** All features work without GPU
- **Threading:** Multi-threaded where applicable

## 📚 Examples & Tutorials

### Example Workflows
Check the `examples/` directory for sample workflows demonstrating:
- Professional green screen removal
- Advanced layer composition techniques
- Video patch processing workflows
- Batch processing setups

### Community Resources
- **Documentation:** [Wiki](https://github.com/your-repo/Studio42-ComfyUI/wiki)
- **Community Forum:** [Discussions](https://github.com/your-repo/Studio42-ComfyUI/discussions)
- **Bug Reports:** [Issues](https://github.com/your-repo/Studio42-ComfyUI/issues)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Pull Requests. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
git clone https://github.com/your-repo/Studio42-ComfyUI.git
cd Studio42-ComfyUI
pip install -r requirements.txt -r requirements-dev.txt
```

## 📞 Support

For support and updates:
- 🐛 **Bug Reports:** [GitHub Issues](https://github.com/your-repo/Studio42-ComfyUI/issues)
- 💬 **Community:** [ComfyUI Community Forums](https://github.com/comfyanonymous/ComfyUI/discussions)
- 📖 **Documentation:** [Wiki](https://github.com/your-repo/Studio42-ComfyUI/wiki)
- 🎥 **Video Tutorials:** [YouTube Channel](https://youtube.com/your-channel)

## 🙏 Acknowledgments

- **ComfyUI Team** for the excellent node-based interface
- **Rembg Contributors** for the background removal models
- **BiRefNet Team** for state-of-the-art segmentation
- **OpenCV Community** for computer vision tools
- **PIL/Pillow Team** for image processing capabilities

---

**Studio42 Image, Audio, and Video Editing Suite** - *Professional tools for the modern creative workflow.*


🎬 *Transform your ComfyUI workflows with professional-grade editing capabilities*
