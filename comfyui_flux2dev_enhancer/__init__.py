"""Standalone ComfyUI FLUX.2 enhancement package."""

from .runtime_compatibility import install_flux_forward_runtime_guard

# Install the guard during custom-node import. It checks the effective
# ``Flux.forward_orig`` again at model-call time, so later monkey patches are also
# covered without depending on any project node being active in the workflow.
install_flux_forward_runtime_guard()

from .registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .version import __version__

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
