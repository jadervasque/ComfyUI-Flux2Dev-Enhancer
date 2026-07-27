"""Standalone ComfyUI FLUX.2 enhancement package."""

from .registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .version import __version__

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
