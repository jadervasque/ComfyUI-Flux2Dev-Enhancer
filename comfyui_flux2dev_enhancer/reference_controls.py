"""Reference-attention and reference-latent controls for FLUX.2."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .architecture import require_capabilities
from .constants import PROJECT_NAME


def _reference_slice(ref_tokens, reference_index: int, sequence_length: int):
    if not ref_tokens or reference_index < 0 or reference_index >= len(ref_tokens):
        return None
    total = int(sum(ref_tokens))
    offset = int(sum(ref_tokens[:reference_index]))
    count = int(ref_tokens[reference_index])
    if count <= 0:
        return None
    start = sequence_length - total + offset
    return start, start + count


def _factor_grid(count: int, target_ratio: float) -> tuple[int, int]:
    count = max(1, int(count))
    best = (1, count)
    best_error = float("inf")
    for height in range(1, int(count**0.5) + 3):
        if count % height:
            continue
        width = count // height
        for candidate in ((height, width), (width, height)):
            error = abs(candidate[0] / max(candidate[1], 1) - target_ratio)
            if error < best_error:
                best, best_error = candidate, error
    return best


def _spatial_weights(count, latent, mode, fade_strength, device):
    if mode == "none" or latent is None or not torch.is_tensor(latent):
        return None
    height, width = latent.shape[-2:]
    grid_h, grid_w = _factor_grid(count, height / max(width, 1))
    y = torch.linspace(0.0, 1.0, grid_h, device=device)
    x = torch.linspace(0.0, 1.0, grid_w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    if mode == "center_out":
        distance = torch.sqrt((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
        distance = distance / distance.max().clamp(min=1e-8)
        weights = 1.0 - distance * fade_strength
    elif mode == "edges_out":
        distance = torch.sqrt((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
        distance = distance / distance.max().clamp(min=1e-8)
        weights = (1.0 - fade_strength) + distance * fade_strength
    elif mode == "top_down":
        weights = 1.0 - yy * fade_strength
    elif mode == "left_right":
        weights = 1.0 - xx * fade_strength
    else:
        return None
    return weights.clamp(0.0, 5.0).flatten()[:count]


def _find_reference_latent(conditioning, index: int):
    for _, meta in conditioning or []:
        references = meta.get("reference_latents", []) or []
        if 0 <= index < len(references):
            return references[index]
    return None


class Flux2ReferenceAttentionControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05},
                ),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 63}),
            },
            "optional": {
                "spatial_fade": (
                    ["none", "center_out", "edges_out", "top_down", "left_right"],
                    {"default": "none"},
                ),
                "spatial_fade_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "control"
    CATEGORY = "conditioning/flux2"

    def control(
        self,
        model,
        conditioning,
        strength=1.0,
        reference_index=0,
        spatial_fade="none",
        spatial_fade_strength=0.5,
        debug=False,
    ):
        patched = model.clone()
        if float(strength) == 1.0 and spatial_fade == "none":
            return patched, conditioning
        require_capabilities(model, attn_input=True)
        reference_latent = _find_reference_latent(conditioning, int(reference_index))

        def attention_patch(q, k, v, extra_options=None, **kwargs):
            options = extra_options or kwargs.get("extra_options") or {}
            ref_tokens = options.get("reference_image_num_tokens", []) or []
            selected = _reference_slice(ref_tokens, int(reference_index), int(k.shape[2]))
            if selected is None:
                return {"q": q, "k": k, "v": v}
            start, end = selected
            token_weights = _spatial_weights(
                end - start,
                reference_latent,
                spatial_fade,
                float(spatial_fade_strength),
                k.device,
            )
            scale = float(strength)
            if token_weights is not None:
                scale = (token_weights * scale).view(1, 1, -1, 1).to(k.dtype)
            new_k, new_v = k.clone(), v.clone()
            new_k[:, :, start:end, :] *= scale
            new_v[:, :, start:end, :] *= scale
            if debug:
                print(
                    f"[{PROJECT_NAME}:ReferenceAttentionControl] "
                    f"block={options.get('block_type')}:{options.get('block_index')} "
                    f"reference={reference_index} tokens={start}:{end} strength={strength}"
                )
            return {"q": q, "k": new_k, "v": new_v}

        patched.set_model_attn1_patch(attention_patch)
        return patched, conditioning


class Flux2ReferenceWeight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2"

    def apply(self, model, reference_index=0, weight=1.0):
        patched = model.clone()
        if float(weight) == 1.0:
            return (patched,)
        require_capabilities(model, attn_input=True)

        def attention_patch(q, k, v, extra_options=None, **kwargs):
            options = extra_options or kwargs.get("extra_options") or {}
            selected = _reference_slice(
                options.get("reference_image_num_tokens", []) or [],
                int(reference_index),
                int(k.shape[2]),
            )
            if selected is None:
                return {"q": q, "k": k, "v": v}
            start, end = selected
            new_k, new_v = k.clone(), v.clone()
            new_k[:, :, start:end, :] *= float(weight)
            new_v[:, :, start:end, :] *= float(weight)
            return {"q": q, "k": new_k, "v": new_v}

        patched.set_model_attn1_patch(attention_patch)
        return (patched,)


class Flux2TextReferenceBalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "balance": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            },
            "optional": {"debug": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "balance"
    CATEGORY = "conditioning/flux2"

    def balance(self, model, conditioning, balance=0.5, debug=False):
        patched = model.clone()
        if float(balance) == 0.5:
            return patched, conditioning
        require_capabilities(model, attn_input=True)
        if balance <= 0.5:
            text_scale, reference_scale = balance * 2.0, 1.0
        else:
            text_scale, reference_scale = 1.0, (1.0 - balance) * 2.0

        def attention_patch(q, k, v, extra_options=None, **kwargs):
            options = extra_options or kwargs.get("extra_options") or {}
            img_slice = options.get("img_slice")
            ref_tokens = options.get("reference_image_num_tokens", []) or []
            new_k, new_v = k.clone(), v.clone()
            if img_slice is not None and text_scale != 1.0:
                text_end = int(img_slice[0])
                new_k[:, :, :text_end, :] *= text_scale
                new_v[:, :, :text_end, :] *= text_scale
            if ref_tokens and reference_scale != 1.0:
                count = int(sum(ref_tokens))
                new_k[:, :, -count:, :] *= reference_scale
                new_v[:, :, -count:, :] *= reference_scale
            if debug:
                print(
                    f"[{PROJECT_NAME}:TextReferenceBalance] "
                    f"text={text_scale:.3f} reference={reference_scale:.3f}"
                )
            return {"q": q, "k": new_k, "v": new_v}

        patched.set_model_attn1_patch(attention_patch)
        return patched, conditioning


class Flux2ReferenceLatentMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"conditioning": ("CONDITIONING",), "mask": ("MASK",)},
            "optional": {
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 64}),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2"

    @staticmethod
    def _resize(mask, height, width):
        if mask.dim() == 2:
            value = mask[None, None].float()
        elif mask.dim() == 3:
            value = mask[:1, None].float()
        elif mask.dim() == 4:
            value = mask[:1, :1].float()
        else:
            raise ValueError(f"Unexpected mask shape {tuple(mask.shape)}")
        return F.interpolate(
            value, size=(height, width), mode="bilinear", align_corners=False
        )

    @staticmethod
    def _feather(mask, radius):
        if radius <= 0:
            return mask
        kernel_size = radius * 2 + 1
        sigma = max(radius / 3.0, 1e-6)
        axis = torch.arange(kernel_size, dtype=torch.float32, device=mask.device) - radius
        gaussian = torch.exp(-0.5 * (axis / sigma) ** 2)
        gaussian /= gaussian.sum()
        kernel = (gaussian[:, None] * gaussian[None, :])[None, None]
        return F.conv2d(mask, kernel, padding=radius).clamp(0.0, 1.0)

    def apply(
        self,
        conditioning,
        mask,
        strength=1.0,
        invert_mask=False,
        feather=0,
        reference_index=0,
        debug=False,
    ):
        if not conditioning or strength <= 0.0:
            return (conditioning,)
        output = []
        for cond, meta in conditioning:
            new_meta = dict(meta)
            references = list(meta.get("reference_latents", []) or [])
            if reference_index < 0 or reference_index >= len(references):
                output.append([cond, new_meta])
                continue
            original = references[reference_index]
            reference = original.float().clone()
            spatial = self._resize(mask, reference.shape[-2], reference.shape[-1])
            if invert_mask:
                spatial = 1.0 - spatial
            spatial = self._feather(spatial, int(feather))
            multiplier = 1.0 - float(strength) * (1.0 - spatial.to(reference.device))
            references[reference_index] = (reference * multiplier).to(original.dtype)
            new_meta["reference_latents"] = references
            output.append([cond, new_meta])
            if debug:
                print(
                    f"[{PROJECT_NAME}:ReferenceLatentMask] "
                    f"reference={reference_index} shape={tuple(reference.shape)} "
                    f"strength={strength} feather={feather}"
                )
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Flux2ReferenceAttentionControl": Flux2ReferenceAttentionControl,
    "Flux2ReferenceWeight": Flux2ReferenceWeight,
    "Flux2TextReferenceBalance": Flux2TextReferenceBalance,
    "Flux2ReferenceLatentMask": Flux2ReferenceLatentMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2ReferenceAttentionControl": "FLUX.2 Reference Attention Control",
    "Flux2ReferenceWeight": "FLUX.2 Reference Weight",
    "Flux2TextReferenceBalance": "FLUX.2 Text/Reference Balance",
    "Flux2ReferenceLatentMask": "FLUX.2 Reference Latent Mask",
}

__all__ = [
    "Flux2ReferenceAttentionControl",
    "Flux2ReferenceLatentMask",
    "Flux2ReferenceWeight",
    "Flux2TextReferenceBalance",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "_reference_slice",
]
