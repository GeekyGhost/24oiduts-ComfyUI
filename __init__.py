"""
Studio42 Image, Audio, and Video Editing Suite for ComfyUI
Advanced nodes for background removal, layer composition, and patch manipulation for both images and videos.
Author: Studio42
Version: 2.0.1
"""

# Import all node mappings
from .studio42_bg_remover import NODE_CLASS_MAPPINGS as BG_REMOVER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BG_REMOVER_DISPLAY
from .studio42_layer_composer import NODE_CLASS_MAPPINGS as LAYER_COMPOSER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LAYER_COMPOSER_DISPLAY  
from .studio42_patchlift_loader import NODE_CLASS_MAPPINGS as PATCHLIFT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PATCHLIFT_DISPLAY
from .studio42_patchdrop import NODE_CLASS_MAPPINGS as PATCHDROP_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PATCHDROP_DISPLAY

# Gracefully import video nodes
try:
    from .studio42_video_patchlift_loader import NODE_CLASS_MAPPINGS as VIDEO_PATCHLIFT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VIDEO_PATCHLIFT_DISPLAY
    VIDEO_NODES_AVAILABLE = True
except ImportError:
    VIDEO_PATCHLIFT_MAPPINGS = {}
    VIDEO_PATCHLIFT_DISPLAY = {}
    VIDEO_NODES_AVAILABLE = False
    print("⚠️ Studio42 Video Nodes not available. Please install dependencies (opencv-python, ffmpeg-python).")

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **BG_REMOVER_MAPPINGS,
    **LAYER_COMPOSER_MAPPINGS,
    **PATCHLIFT_MAPPINGS,
    **PATCHDROP_MAPPINGS,
    **VIDEO_PATCHLIFT_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **BG_REMOVER_DISPLAY,
    **LAYER_COMPOSER_DISPLAY,
    **PATCHLIFT_DISPLAY,
    **PATCHDROP_DISPLAY,
    **VIDEO_PATCHLIFT_DISPLAY,
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Print installation info
print("🎬 Studio42 Image and Video Editing Suite v2.0.1 loaded!")
print("📋 Available nodes:")
for display_name in sorted(NODE_DISPLAY_NAME_MAPPINGS.values()):
    print(f"   • {display_name}")
if VIDEO_NODES_AVAILABLE:
    print("✅ Video capabilities enabled.")
else:
    print("💡 For video nodes, run: pip install opencv-python ffmpeg-python")
print("🚀 Ready for advanced editing workflows!")
