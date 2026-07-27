"""ComfyUI entry point for ComfyUI-Flux2Dev-Enhancer."""

if __package__:
    from .comfyui_flux2dev_enhancer import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        __version__,
    )
else:  # Direct import used by test collectors and diagnostic tooling.
    from comfyui_flux2dev_enhancer import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        __version__,
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
