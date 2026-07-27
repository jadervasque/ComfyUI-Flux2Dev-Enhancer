from __future__ import annotations

from flux2_enhancer_under_test.flux2_guidance import (
    Flux2ColorAnchor,
    Flux2IdentityGuidance,
)
from flux2_enhancer_under_test.flux2_identity_nodes import Flux2IdentityFeatureTransfer
from flux2_enhancer_under_test.flux2_reference_controls import (
    Flux2ReferenceWeight,
    Flux2TextReferenceBalance,
)


class CloneOnlyModel:
    def clone(self):
        return self


def test_disabled_identity_transfer_is_loader_independent():
    model = CloneOnlyModel()
    assert Flux2IdentityFeatureTransfer().apply(model, enabled=False)[0] is model


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
