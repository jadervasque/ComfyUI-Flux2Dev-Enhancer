"""Latent-space guidance and color anchoring for compatible FLUX.2 models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .architecture import require_capabilities
from .constants import PROJECT_NAME


def _reference_from_conditioning(conditioning, index: int):
    for _, meta in conditioning or []:
        references = meta.get("reference_latents", []) or []
        if 0 <= index < len(references):
            return references[index]
        model_conds = meta.get("model_conds", {}) or {}
        ref_cond = model_conds.get("ref_latents")
        values = getattr(ref_cond, "cond", None)
        if values is not None and 0 <= index < len(values):
            return values[index]
    return None


def _prepare_sigmas(sigmas):
    if sigmas is None:
        return None
    if not torch.is_tensor(sigmas):
        raise ValueError("sigmas must be a SIGMAS tensor")
    values = sigmas.detach().flatten().double().cpu()
    if values.numel() < 2 or not torch.isfinite(values).all().item():
        raise ValueError("sigmas must contain at least two finite values")
    return values


def _sigma_index(values: torch.Tensor, sigma: float) -> int:
    differences = torch.abs(values[:-1] - float(sigma))
    return int(torch.argmin(differences).item())


def _progress(values, sigma: float, state: dict) -> tuple[float, int]:
    if values is not None:
        index = _sigma_index(values, sigma)
        return index / max(values.numel() - 2, 1), index
    previous = state.get("last_sigma")
    maximum = state.get("sigma_max")
    step = int(state.get("step", 0))
    if maximum is None or sigma > maximum or (
        previous is not None and sigma > previous + 1e-7
    ):
        maximum = sigma
        step = 0
    state["sigma_max"] = maximum
    state["last_sigma"] = sigma
    state["step"] = step + 1
    sigma_progress = (
        (maximum - sigma) / maximum if maximum and maximum > 1e-8 else 0.0
    )
    return min(max(sigma_progress, 0.0), 1.0), step


def _install_post_cfg_callback(model, callback):
    patched = model.clone()
    callbacks = list(
        patched.model_options.get("sampler_post_cfg_function", []) or []
    )
    callbacks.append(callback)
    patched.model_options["sampler_post_cfg_function"] = callbacks
    return patched


class Flux2ColorAnchor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
            "optional": {
                "ramp_curve": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1},
                ),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                "channel_weights": (
                    ["uniform", "by_variance"],
                    {"default": "uniform"},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2"

    def apply(
        self,
        model,
        conditioning,
        strength=0.5,
        ramp_curve=1.5,
        reference_index=0,
        channel_weights="uniform",
        start_percent=0.0,
        end_percent=1.0,
        sigmas=None,
        debug=False,
    ):
        if float(strength) <= 0.0:
            return (model.clone(),)
        require_capabilities(model, sampler_post_cfg=True)
        reference = _reference_from_conditioning(conditioning, int(reference_index))
        if not torch.is_tensor(reference) or reference.dim() != 4:
            raise ValueError(
                f"FLUX.2 Color Anchor: reference {reference_index} is missing or is not a 4D latent."
            )
        ref_float = reference.float()
        ref_mean = ref_float.mean(dim=(-2, -1), keepdim=True)
        trust = None
        if channel_weights == "by_variance":
            variance = ref_float.var(dim=(-2, -1), keepdim=True)
            trust = 1.0 / (1.0 + variance)
            trust /= trust.max().clamp(min=1e-8)
        sigma_values = _prepare_sigmas(sigmas)
        state = {"last_sigma": None, "sigma_max": None, "step": 0}
        start = min(max(float(start_percent), 0.0), 1.0)
        end = min(max(float(end_percent), start), 1.0)
        curve = max(float(ramp_curve), 1e-3)

        def callback(args):
            denoised = args["denoised"]
            sigma_value = float(args["sigma"].detach().flatten()[0].cpu().item())
            progress, step = _progress(sigma_values, sigma_value, state)
            if progress < start or progress > end:
                return denoised
            if denoised.shape[1] != ref_mean.shape[1]:
                raise ValueError(
                    "FLUX.2 Color Anchor: generated and reference latent channel counts differ."
                )
            color_progress = max(progress, 1.0 - 0.5 ** (step + 1))
            effective = float(strength) * color_progress ** (1.0 / curve)
            reference_mean = ref_mean.to(denoised.device, denoised.dtype)
            current_mean = denoised.mean(dim=(-2, -1), keepdim=True)
            correction = reference_mean - current_mean
            if trust is not None:
                correction *= trust.to(denoised.device, denoised.dtype)
            if debug:
                print(
                    f"[{PROJECT_NAME}:ColorAnchor] step={step} "
                    f"sigma={sigma_value:.6f} progress={progress:.3f} "
                    f"effective={effective:.3f}"
                )
            return denoised + correction * effective

        return (_install_post_cfg_callback(model, callback),)


class Flux2IdentityGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "identity_latent": ("LATENT",),
                "strength": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "mode": (
                    ["adaptive", "direct", "channel_match"],
                    {"default": "adaptive"},
                ),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2"

    def apply(
        self,
        model,
        identity_latent,
        strength=0.3,
        start_percent=0.0,
        end_percent=0.8,
        mode="adaptive",
        sigmas=None,
        debug=False,
    ):
        if float(strength) <= 0.0:
            return (model.clone(),)
        require_capabilities(model, sampler_post_cfg=True)
        reference = (
            identity_latent.get("samples")
            if isinstance(identity_latent, dict)
            else None
        )
        if not torch.is_tensor(reference) or reference.dim() != 4:
            raise ValueError(
                "FLUX.2 Identity Guidance requires a 4D latent samples tensor."
            )
        sigma_values = _prepare_sigmas(sigmas)
        state = {"last_sigma": None, "sigma_max": None, "step": 0}
        start = min(max(float(start_percent), 0.0), 1.0)
        end = min(max(float(end_percent), start), 1.0)

        def callback(args):
            denoised = args["denoised"]
            sigma_value = float(args["sigma"].detach().flatten()[0].cpu().item())
            progress, step = _progress(sigma_values, sigma_value, state)
            if progress < start or progress > end:
                return denoised
            ref = reference.to(denoised.device, denoised.dtype)
            if ref.shape[0] != denoised.shape[0]:
                ref = ref[:1].expand(denoised.shape[0], -1, -1, -1)
            if ref.shape[1] != denoised.shape[1]:
                raise ValueError(
                    "FLUX.2 Identity Guidance: generated and identity latent channel counts differ."
                )
            if ref.shape[-2:] != denoised.shape[-2:]:
                ref = F.interpolate(
                    ref,
                    size=denoised.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if mode == "direct":
                output = denoised + (ref - denoised) * float(strength)
            elif mode == "channel_match":
                ref_mean = ref.mean(dim=(-2, -1), keepdim=True)
                ref_std = ref.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
                current_mean = denoised.mean(dim=(-2, -1), keepdim=True)
                current_std = denoised.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
                matched = (denoised - current_mean) / current_std * ref_std + ref_mean
                output = denoised + (matched - denoised) * float(strength)
            else:
                similarity = F.cosine_similarity(
                    denoised.flatten(2), ref.flatten(2), dim=1
                ).clamp(0.0, 1.0)
                weight = similarity.view(
                    denoised.shape[0], 1, *denoised.shape[-2:]
                )
                output = denoised + (ref - denoised) * weight * float(strength)
            if debug:
                print(
                    f"[{PROJECT_NAME}:IdentityGuidance] step={step} "
                    f"sigma={sigma_value:.6f} progress={progress:.3f} "
                    f"mode={mode} strength={strength}"
                )
            return output

        return (_install_post_cfg_callback(model, callback),)


NODE_CLASS_MAPPINGS = {
    "Flux2ColorAnchor": Flux2ColorAnchor,
    "Flux2IdentityGuidance": Flux2IdentityGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2ColorAnchor": "FLUX.2 Color Anchor",
    "Flux2IdentityGuidance": "FLUX.2 Identity Guidance",
}

__all__ = [
    "Flux2ColorAnchor",
    "Flux2IdentityGuidance",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "_progress",
    "_reference_from_conditioning",
]
