from __future__ import annotations

from comfyui_flux2dev_enhancer.guidance import (
    Flux2ColorAnchor,
    Flux2IdentityGuidance,
)
from comfyui_flux2dev_enhancer.identity_transfer import Flux2IdentityFeatureTransfer
from comfyui_flux2dev_enhancer.reference_controls import (
    Flux2ReferenceWeight,
    Flux2TextReferenceBalance,
)


class CloneOnlyModel:
    def clone(self):
        return self


def test_disabled_identity_transfer_is_loader_independent():
    model = CloneOnlyModel()
    assert Flux2IdentityFeatureTransfer().apply(model, enabled=False)[0] is model


def test_zero_total_identity_transfer_is_loader_independent():
    model = CloneOnlyModel()
    assert (
        Flux2IdentityFeatureTransfer().apply(
            model, strength_mode="normalized_total", total_strength=0.0
        )[0]
        is model
    )


def test_neutral_reference_weight_is_loader_independent():
    model = CloneOnlyModel()
    assert Flux2ReferenceWeight().apply(model, weight=1.0)[0] is model


def test_neutral_text_reference_balance_is_loader_independent():
    model = CloneOnlyModel()
    output_model, conditioning = Flux2TextReferenceBalance().balance(
        model, [[None, {}]], balance=0.5
    )
    assert output_model is model
    assert conditioning == [[None, {}]]


def test_zero_identity_guidance_is_loader_independent():
    model = CloneOnlyModel()
    assert Flux2IdentityGuidance().apply(model, {}, strength=0.0)[0] is model


def test_zero_color_anchor_is_loader_independent():
    model = CloneOnlyModel()
    assert Flux2ColorAnchor().apply(model, [], strength=0.0)[0] is model
