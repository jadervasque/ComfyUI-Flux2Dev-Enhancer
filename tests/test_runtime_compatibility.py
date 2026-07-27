from __future__ import annotations

from types import SimpleNamespace

import pytest

from comfyui_flux2dev_enhancer.architecture import (
    Flux2CompatibilityError,
    ensure_forward_orig_compatibility,
)
from comfyui_flux2dev_enhancer.runtime_compatibility import (
    install_flux_forward_runtime_guard,
)


class FakeModel:
    def __init__(self, forward_orig):
        self.model = SimpleNamespace(
            diffusion_model=SimpleNamespace(forward_orig=forward_orig)
        )


def test_inactive_timestep_zero_keyword_is_removed_for_older_patch():
    calls = []

    def patched_forward_orig(*args, transformer_options=None, attn_mask=None):
        calls.append((args, transformer_options, attn_mask))
        return "ok"

    model = FakeModel(patched_forward_orig)

    assert ensure_forward_orig_compatibility(model) is True
    result = model.model.diffusion_model.forward_orig(
        "img",
        timestep_zero_index=None,
        transformer_options={"patches": {}},
        attn_mask=None,
    )

    assert result == "ok"
    assert calls == [(("img",), {"patches": {}}, None)]
    assert ensure_forward_orig_compatibility(model) is False


def test_active_timestep_zero_keyword_is_not_silently_discarded():
    def patched_forward_orig(*args, transformer_options=None, attn_mask=None):
        return args

    model = FakeModel(patched_forward_orig)
    ensure_forward_orig_compatibility(model)

    with pytest.raises(Flux2CompatibilityError, match="timestep_zero_index"):
        model.model.diffusion_model.forward_orig(
            "img",
            timestep_zero_index=[[10, 20]],
            transformer_options={},
            attn_mask=None,
        )


def test_active_attention_mask_is_not_silently_discarded():
    def patched_forward_orig(*args, transformer_options=None):
        return args

    model = FakeModel(patched_forward_orig)
    ensure_forward_orig_compatibility(model)

    with pytest.raises(Flux2CompatibilityError, match="attn_mask"):
        model.model.diffusion_model.forward_orig(
            "img",
            timestep_zero_index=None,
            transformer_options={},
            attn_mask=object(),
        )


def test_current_forward_signature_is_left_unchanged():
    def forward_orig(
        *args,
        timestep_zero_index=None,
        transformer_options=None,
        attn_mask=None,
    ):
        return args

    model = FakeModel(forward_orig)
    original = model.model.diffusion_model.forward_orig

    assert ensure_forward_orig_compatibility(model) is False
    assert model.model.diffusion_model.forward_orig is original


def test_runtime_guard_repairs_monkey_patch_installed_after_guard():
    class FakeFlux:
        def __init__(self):
            self.double_blocks = [object()]
            self.single_blocks = [object()]

        def forward_orig(
            self,
            *args,
            timestep_zero_index=None,
            transformer_options=None,
            attn_mask=None,
        ):
            return "native"

        def _forward(self):
            return self.forward_orig(
                "img",
                timestep_zero_index=None,
                transformer_options={"patches": {}},
                attn_mask=None,
            )

    assert install_flux_forward_runtime_guard(FakeFlux) is True

    # Reproduce comfyui-flux2fun-controlnet: replace the class method after our
    # package has already been imported, without the new ComfyUI keyword.
    def patched_forward_orig(
        self,
        *args,
        transformer_options=None,
        attn_mask=None,
    ):
        return "third-party"

    FakeFlux.forward_orig = patched_forward_orig

    assert FakeFlux()._forward() == "third-party"
    assert install_flux_forward_runtime_guard(FakeFlux) is False


def test_runtime_guard_preserves_active_timestep_protection():
    class FakeFlux:
        def __init__(self):
            self.double_blocks = [object()]
            self.single_blocks = [object()]

        def forward_orig(
            self,
            *args,
            transformer_options=None,
            attn_mask=None,
        ):
            return "outdated"

        def _forward(self):
            return self.forward_orig(
                "img",
                timestep_zero_index=[[4, 8]],
                transformer_options={},
                attn_mask=None,
            )

    install_flux_forward_runtime_guard(FakeFlux)

    with pytest.raises(Flux2CompatibilityError, match="timestep_zero_index"):
        FakeFlux()._forward()
