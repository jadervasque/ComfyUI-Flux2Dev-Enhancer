"""Canonical ComfyUI node registry for the standalone extension."""

from __future__ import annotations

from .conditioning import (
    Flux2ConditioningEnhancer,
    Flux2DetailController,
    Flux2SectionedEncoder,
    Flux2TextConditioningEnhancer,
)
from .constants import CATEGORY_ROOT
from .diagnostics import Flux2ArchitectureInspector
from .guidance import Flux2ColorAnchor, Flux2IdentityGuidance
from .identity_transfer import Flux2IdentityFeatureTransfer
from .reference_controls import (
    Flux2ReferenceAttentionControl,
    Flux2ReferenceLatentMask,
    Flux2ReferenceWeight,
    Flux2TextReferenceBalance,
)
from .reference_latent import Flux2MultiReferenceLatent


def _categorize(category: str, *classes) -> None:
    full_category = f"{CATEGORY_ROOT}/{category}"
    for node_class in classes:
        node_class.CATEGORY = full_category


_categorize("Diagnostics", Flux2ArchitectureInspector)
_categorize(
    "Conditioning",
    Flux2ConditioningEnhancer,
    Flux2TextConditioningEnhancer,
    Flux2SectionedEncoder,
    Flux2DetailController,
)
_categorize(
    "References",
    Flux2MultiReferenceLatent,
    Flux2ReferenceAttentionControl,
    Flux2ReferenceWeight,
    Flux2TextReferenceBalance,
    Flux2ReferenceLatentMask,
)
_categorize("Guidance", Flux2ColorAnchor, Flux2IdentityGuidance)
_categorize("Identity", Flux2IdentityFeatureTransfer)

NODE_CLASS_MAPPINGS = {
    "Flux2ArchitectureInspector": Flux2ArchitectureInspector,
    "Flux2ConditioningEnhancer": Flux2ConditioningEnhancer,
    "Flux2TextConditioningEnhancer": Flux2TextConditioningEnhancer,
    "Flux2SectionedEncoder": Flux2SectionedEncoder,
    "Flux2DetailController": Flux2DetailController,
    "Flux2MultiReferenceLatent": Flux2MultiReferenceLatent,
    "Flux2ReferenceAttentionControl": Flux2ReferenceAttentionControl,
    "Flux2ReferenceWeight": Flux2ReferenceWeight,
    "Flux2TextReferenceBalance": Flux2TextReferenceBalance,
    "Flux2ReferenceLatentMask": Flux2ReferenceLatentMask,
    "Flux2ColorAnchor": Flux2ColorAnchor,
    "Flux2IdentityGuidance": Flux2IdentityGuidance,
    "Flux2IdentityFeatureTransfer": Flux2IdentityFeatureTransfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2ArchitectureInspector": "FLUX.2 Architecture Inspector",
    "Flux2ConditioningEnhancer": "FLUX.2 Conditioning Enhancer",
    "Flux2TextConditioningEnhancer": "FLUX.2 Text Conditioning Enhancer",
    "Flux2SectionedEncoder": "FLUX.2 Sectioned Encoder",
    "Flux2DetailController": "FLUX.2 Detail Controller",
    "Flux2MultiReferenceLatent": "FLUX.2 Multi Reference Latent",
    "Flux2ReferenceAttentionControl": "FLUX.2 Reference Attention Control",
    "Flux2ReferenceWeight": "FLUX.2 Reference Weight",
    "Flux2TextReferenceBalance": "FLUX.2 Text/Reference Balance",
    "Flux2ReferenceLatentMask": "FLUX.2 Reference Latent Mask",
    "Flux2ColorAnchor": "FLUX.2 Color Anchor",
    "Flux2IdentityGuidance": "FLUX.2 Identity Guidance",
    "Flux2IdentityFeatureTransfer": "FLUX.2 Identity Feature Transfer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
