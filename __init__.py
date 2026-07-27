"""ComfyUI FLUX.2 Enhancer node registration.

New model-neutral nodes are registered under ``conditioning/flux2``. Original
Klein node identifiers remain available for workflow compatibility and are marked
as legacy in their display names.
"""

from .flux2_conditioning import (
    NODE_CLASS_MAPPINGS as CONDITIONING_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as CONDITIONING_NAMES,
)
from .flux2_diagnostics import (
    NODE_CLASS_MAPPINGS as DIAGNOSTIC_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as DIAGNOSTIC_NAMES,
)
from .flux2_guidance import (
    NODE_CLASS_MAPPINGS as GUIDANCE_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as GUIDANCE_NAMES,
)
from .flux2_identity_nodes import (
    NODE_CLASS_MAPPINGS as IDENTITY_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as IDENTITY_NAMES,
)
from .flux2_reference_controls import (
    NODE_CLASS_MAPPINGS as REFERENCE_CONTROL_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as REFERENCE_CONTROL_NAMES,
)
from .flux2_reference_latent import (
    NODE_CLASS_MAPPINGS as REFERENCE_LATENT_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as REFERENCE_LATENT_NAMES,
)

# Earlier algorithms and Klein-only text tools are retained without being used by
# the new model-neutral node IDs. This prevents existing workflow JSON files from
# failing to load while making the compatibility boundary explicit.
from .identity_feature_transfer import (
    IdentityFeatureTransfer,
    IdentityFeatureTransferAdvanced,
    IdentityFeatureTransferV3,
)
from .flux2_klein_enhancer import Flux2KleinEnhancer, Flux2KleinDetailController
from .flux2_klein_text_enhancer import Flux2KleinTextEnhancer
from .flux2_sectioned_encoder import Flux2KleinSectionedEncoder
from .Flux2klein_Ksampler_exp import (
    NODE_CLASS_MAPPINGS as EXPERIMENTAL_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as EXPERIMENTAL_NAMES,
)


LEGACY_NODE_CLASS_MAPPINGS = {
    "Flux2KleinEnhancer": Flux2KleinEnhancer,
    "Flux2KleinDetailController": Flux2KleinDetailController,
    "Flux2KleinTextEnhancer": Flux2KleinTextEnhancer,
    "Flux2KleinSectionedEncoder": Flux2KleinSectionedEncoder,
    "IdentityFeatureTransfer": IdentityFeatureTransfer,
    "IdentityFeatureTransferAdvanced": IdentityFeatureTransferAdvanced,
    "IdentityFeatureTransferV3": IdentityFeatureTransferV3,
    **EXPERIMENTAL_NODES,
}

LEGACY_NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinEnhancer": "FLUX.2 Klein Enhancer (Legacy)",
    "Flux2KleinDetailController": "FLUX.2 Klein Detail Controller (Legacy)",
    "Flux2KleinTextEnhancer": "FLUX.2 Klein Text Enhancer (Legacy)",
    "Flux2KleinSectionedEncoder": "FLUX.2 Klein Sectioned Encoder (Legacy)",
    "IdentityFeatureTransfer": "FLUX.2 Klein Identity Feature Transfer (Legacy)",
    "IdentityFeatureTransferAdvanced": "FLUX.2 Klein Identity Feature Transfer Advanced (Legacy)",
    "IdentityFeatureTransferV3": "FLUX.2 Klein Identity Feature Transfer V3 (Legacy)",
    **{
        node_id: f"{display_name} (Legacy)"
        for node_id, display_name in EXPERIMENTAL_NAMES.items()
    },
}

NODE_CLASS_MAPPINGS = {
    **DIAGNOSTIC_NODES,
    **CONDITIONING_NODES,
    **REFERENCE_LATENT_NODES,
    **REFERENCE_CONTROL_NODES,
    **GUIDANCE_NODES,
    **IDENTITY_NODES,
    **LEGACY_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **DIAGNOSTIC_NAMES,
    **CONDITIONING_NAMES,
    **REFERENCE_LATENT_NAMES,
    **REFERENCE_CONTROL_NAMES,
    **GUIDANCE_NAMES,
    **IDENTITY_NAMES,
    **LEGACY_NODE_DISPLAY_NAME_MAPPINGS,
}

__version__ = "4.0.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
