"""
Studio42 Image, Audio, and Video Editing Suite for ComfyUI
Advanced nodes for background removal, layer composition, patch manipulation, and audio processing.
Now with LCARS-themed futuristic UI, Qwen/Wan models, and procedural generators!

Author: Studio42
Version: 3.0.0 - LCARS Edition
"""

# Import all node mappings
from .studio42_bg_remover import NODE_CLASS_MAPPINGS as BG_REMOVER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BG_REMOVER_DISPLAY
from .studio42_layer_composer import NODE_CLASS_MAPPINGS as LAYER_COMPOSER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LAYER_COMPOSER_DISPLAY
from .studio42_patchlift_loader import NODE_CLASS_MAPPINGS as PATCHLIFT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PATCHLIFT_DISPLAY
from .studio42_patchdrop import NODE_CLASS_MAPPINGS as PATCHDROP_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PATCHDROP_DISPLAY

# Import audio processing nodes
from .studio42_audio_loader import NODE_CLASS_MAPPINGS as AUDIO_LOADER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as AUDIO_LOADER_DISPLAY
from .studio42_audio_mixer import NODE_CLASS_MAPPINGS as AUDIO_MIXER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as AUDIO_MIXER_DISPLAY

# Gracefully import video nodes
try:
    from .studio42_video_patchlift_loader import NODE_CLASS_MAPPINGS as VIDEO_PATCHLIFT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VIDEO_PATCHLIFT_DISPLAY
    VIDEO_NODES_AVAILABLE = True
except ImportError:
    VIDEO_PATCHLIFT_MAPPINGS = {}
    VIDEO_PATCHLIFT_DISPLAY = {}
    VIDEO_NODES_AVAILABLE = False
    print("⚠️ Studio42 Video Nodes not available. Please install dependencies (opencv-python, ffmpeg-python).")

# Import LCARS nodes with graceful fallback
try:
    from .lcars_qwen_nodes import NODE_CLASS_MAPPINGS as QWEN_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as QWEN_DISPLAY
    QWEN_NODES_AVAILABLE = True
except ImportError as e:
    QWEN_MAPPINGS = {}
    QWEN_DISPLAY = {}
    QWEN_NODES_AVAILABLE = False
    print(f"⚠️ LCARS Qwen Nodes not available: {str(e)}")

try:
    from .lcars_wan_nodes import NODE_CLASS_MAPPINGS as WAN_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as WAN_DISPLAY
    WAN_NODES_AVAILABLE = True
except ImportError as e:
    WAN_MAPPINGS = {}
    WAN_DISPLAY = {}
    WAN_NODES_AVAILABLE = False
    print(f"⚠️ LCARS Wan Nodes not available: {str(e)}")

try:
    from .lcars_procedural_nodes import NODE_CLASS_MAPPINGS as PROCEDURAL_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PROCEDURAL_DISPLAY
    PROCEDURAL_NODES_AVAILABLE = True
except ImportError as e:
    PROCEDURAL_MAPPINGS = {}
    PROCEDURAL_DISPLAY = {}
    PROCEDURAL_NODES_AVAILABLE = False
    print(f"⚠️ LCARS Procedural Nodes not available: {str(e)}")

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **BG_REMOVER_MAPPINGS,
    **LAYER_COMPOSER_MAPPINGS,
    **PATCHLIFT_MAPPINGS,
    **PATCHDROP_MAPPINGS,
    **AUDIO_LOADER_MAPPINGS,
    **AUDIO_MIXER_MAPPINGS,
    **VIDEO_PATCHLIFT_MAPPINGS,
    **QWEN_MAPPINGS,
    **WAN_MAPPINGS,
    **PROCEDURAL_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **BG_REMOVER_DISPLAY,
    **LAYER_COMPOSER_DISPLAY,
    **PATCHLIFT_DISPLAY,
    **PATCHDROP_DISPLAY,
    **AUDIO_LOADER_DISPLAY,
    **AUDIO_MIXER_DISPLAY,
    **VIDEO_PATCHLIFT_DISPLAY,
    **QWEN_DISPLAY,
    **WAN_DISPLAY,
    **PROCEDURAL_DISPLAY,
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Print installation info
print("=" * 70)
print("🚀 Studio42 LCARS Edition v3.0.0 - The Future is Now!")
print("=" * 70)
print("\n✨ LCARS + Hitchhiker's Guide UI Theme Enabled!")
print("   Futuristic nodes with orange glow and sleek design\n")

print("📋 Available node categories:")
print(f"   🎬 Classic Studio42: {len(BG_REMOVER_MAPPINGS) + len(LAYER_COMPOSER_MAPPINGS) + len(PATCHLIFT_MAPPINGS) + len(PATCHDROP_MAPPINGS)} nodes")
print(f"   🎵 Audio Processing: {len(AUDIO_LOADER_MAPPINGS) + len(AUDIO_MIXER_MAPPINGS)} nodes")

if VIDEO_NODES_AVAILABLE:
    print(f"   🎥 Video Processing: {len(VIDEO_PATCHLIFT_MAPPINGS)} nodes")
else:
    print("   🎥 Video Processing: Not available (install: opencv-python, ffmpeg-python)")

if QWEN_NODES_AVAILABLE:
    print(f"   🤖 Qwen AI Models: {len(QWEN_MAPPINGS)} nodes")
else:
    print("   🤖 Qwen AI Models: Not available (install: transformers)")

if WAN_NODES_AVAILABLE:
    print(f"   🎬 Wan Video Models: {len(WAN_MAPPINGS)} nodes")
else:
    print("   🎬 Wan Video Models: Not available (install: diffusers)")

if PROCEDURAL_NODES_AVAILABLE:
    print(f"   🌫️  Procedural Generators: {len(PROCEDURAL_MAPPINGS)} nodes")
else:
    print("   🌫️  Procedural Generators: Not available")

total_nodes = len(NODE_CLASS_MAPPINGS)
print(f"\n📊 Total nodes loaded: {total_nodes}")

print("\n🎨 Special Features:")
print("   • LCARS-themed UI with futuristic styling")
print("   • Easy custom node creation system")
print("   • Qwen language & vision models")
print("   • Wan video generation & enhancement")
print("   • Procedural noise, patterns, fractals & gradients")

print("\n💡 Quick Start:")
print("   • Check web/css/lcars-theme.css for styling")
print("   • Use lcars_base_node.py to create custom nodes")
print("   • Run lcars_node_template_generator.py for easy node creation")

print("\n" + "=" * 70)
print("🚀 DON'T PANIC - Your LCARS ComfyUI is ready!")
print("=" * 70)