"""ComfyUI-facing guards for identity-transfer neutral and mask-only cases."""

from __future__ import annotations

from .flux2_identity_transfer import (
    Flux2IdentityFeatureTransfer as _Flux2IdentityFeatureTransfer,
    LegacyIdentityFeatureTransferFinal as _LegacyIdentityFeatureTransferFinal,
)


_MASK_NAMES = tuple(f"subject_mask_{index}" for index in range(1, 9))


class Flux2IdentityFeatureTransfer(_Flux2IdentityFeatureTransfer):
    """Preserve a true no-op before architecture capability validation."""

    def apply(self, model, preset="AUTO_BALANCED", enabled=True, **kwargs):
        if not bool(enabled):
            return (model.clone(),)
        if (
            kwargs.get("strength_mode", "normalized_total") == "normalized_total"
            and float(kwargs.get("total_strength", 0.65)) <= 0.0
        ):
            return (model.clone(),)
        if kwargs.get("mask_behavior", "focus_only") == "zero_unmasked_tokens":
            has_mask = any(kwargs.get(name) is not None for name in _MASK_NAMES)
            if not has_mask:
                kwargs["mask_behavior"] = "focus_only"
        return super().apply(model=model, preset=preset, enabled=enabled, **kwargs)


class LegacyIdentityFeatureTransferFinal(_LegacyIdentityFeatureTransferFinal):
    def apply(self, model, preset="HARD_LOCK", enabled=True, **kwargs):
        if not bool(enabled):
            return (model.clone(),)
        if kwargs.get("mask_behavior", "focus_only") == "zero_unmasked_tokens":
            has_mask = any(kwargs.get(name) is not None for name in _MASK_NAMES)
            if not has_mask:
                kwargs["mask_behavior"] = "focus_only"
        return super().apply(model=model, preset=preset, enabled=enabled, **kwargs)


NODE_CLASS_MAPPINGS = {
    "Flux2IdentityFeatureTransfer": Flux2IdentityFeatureTransfer,
    "IdentityFeatureTransferFinal": LegacyIdentityFeatureTransferFinal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2IdentityFeatureTransfer": "FLUX.2 Identity Feature Transfer",
    "IdentityFeatureTransferFinal": "Identity Feature Transfer Final (Legacy)",
}
