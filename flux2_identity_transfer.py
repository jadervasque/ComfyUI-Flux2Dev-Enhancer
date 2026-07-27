"""Architecture-aware identity feature transfer for the FLUX.2 model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .architecture import Flux2Architecture, require_capabilities
from .scheduling import (
    auto_identity_preset,
    normalized_per_application,
    parse_block_schedule,
    parse_reference_indices,
    project_schedule,
)


LEGACY_HARD_DOUBLE = "0-7:mid_img=0.55"
LEGACY_HARD_SINGLE = (
    "0:mid_img=0.22; 1:mid_img=0.24; 3:mid_img=0.28; "
    "4:mid_img=0.22; 6:mid_img=0.26; 7:mid_img=0.27; "
    "8:mid_img=0.25; 10:mid_img=0.27; 13:mid_img=0.27"
)

_LEGACY_PRESETS = {
    "KLEIN_LEGACY_HARD": (0.040, 0.0250, 1.0),
    "KLEIN_LEGACY_MID": (0.200, 0.0700, 1.0),
    "KLEIN_LEGACY_SOFT": (0.500, 0.0700, 1.0),
}


@dataclass(frozen=True)
class TransferConfig:
    similarity_floor: float
    temperature: float
    mask_threshold: float
    double_schedule: dict[int, float]
    single_schedule: dict[int, float]
    architecture: Flux2Architecture


def _prepare_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None or not torch.is_tensor(mask):
        return None
    value = mask.detach().float().cpu()
    if value.dim() == 4:
        if value.shape[-1] in (1, 3, 4):
            value = value[0].mean(dim=-1)
        else:
            value = value[0, 0]
    elif value.dim() == 3:
        if value.shape[-1] in (1, 3, 4) and value.shape[0] != 1:
            value = value.mean(dim=-1)
        else:
            value = value[0]
    elif value.dim() != 2:
        return None
    return value.contiguous().clamp(0.0, 1.0)


def _grid_for_tokens(count: int, mask: torch.Tensor) -> Tuple[int, int]:
    count = max(1, int(count))
    mask_h, mask_w = mask.shape[-2:]
    target_ratio = mask_h / max(mask_w, 1)
    best = (1, count)
    best_error = float("inf")
    for height in range(1, int(count**0.5) + 3):
        if count % height:
            continue
        width = count // height
        for candidate_h, candidate_w in ((height, width), (width, height)):
            error = abs(candidate_h / max(candidate_w, 1) - target_ratio)
            if error < best_error:
                best = (candidate_h, candidate_w)
                best_error = error
    return best


def _scalar_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().flatten()[0].double().cpu().item())
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _sigma_schedule(sigmas):
    if sigmas is None:
        return None
    if not torch.is_tensor(sigmas):
        raise ValueError("FLUX.2 Identity Feature Transfer: sigmas must be a SIGMAS tensor.")
    values = sigmas.detach().flatten().double().cpu()
    if values.numel() < 2 or not torch.isfinite(values).all().item():
        raise ValueError("FLUX.2 Identity Feature Transfer: invalid sigma schedule.")
    deltas = (values[:-1] - values[1:]).abs()
    if torch.any(deltas <= 0).item():
        raise ValueError("FLUX.2 Identity Feature Transfer: sigma intervals must be non-zero.")
    return values, deltas[0] / deltas


def _sigma_step(values: torch.Tensor, current_sigma: float) -> int:
    step_values = values[:-1]
    differences = torch.abs(step_values - float(current_sigma))
    tolerance = max(1e-7, abs(float(current_sigma)) * 1e-6)
    exact = torch.nonzero(differences <= tolerance, as_tuple=False).flatten()
    if exact.numel():
        return int(exact[0].item())
    for index in range(values.numel() - 1):
        start = float(values[index].item())
        end = float(values[index + 1].item())
        if min(start, end) <= current_sigma <= max(start, end):
            return index
    return int(torch.argmin(differences).item())


def _normalize_pair(
    double_schedule: dict[int, float],
    single_schedule: dict[int, float],
    total_strength: float,
) -> tuple[dict[int, float], dict[int, float]]:
    count = len(double_schedule) + len(single_schedule)
    if count <= 0:
        return {}, {}
    maximum = max([*double_schedule.values(), *single_schedule.values()]) or 1.0
    base = normalized_per_application(total_strength, count)
    return (
        {idx: base * (value / maximum) for idx, value in double_schedule.items()},
        {idx: base * (value / maximum) for idx, value in single_schedule.items()},
    )


def resolve_transfer_config(
    architecture: Flux2Architecture,
    preset: str,
    similarity_floor: float,
    softmax_temperature: float,
    mask_threshold: float,
    double_blocks: str,
    single_blocks: str,
    strength_mode: str,
    total_strength: float,
) -> TransferConfig:
    preset_key = str(preset).upper()
    if preset_key in {"AUTO_SOFT", "AUTO_BALANCED", "AUTO_STRONG"}:
        resolved = auto_identity_preset(
            preset_key, architecture.double_blocks, architecture.single_blocks
        )
        double_schedule = dict(resolved.double_schedule)
        single_schedule = dict(resolved.single_schedule)
        similarity_floor = resolved.similarity_floor
        softmax_temperature = resolved.temperature
        mask_threshold = resolved.mask_threshold
    elif preset_key in _LEGACY_PRESETS:
        similarity_floor, softmax_temperature, mask_threshold = _LEGACY_PRESETS[preset_key]
        legacy_double = parse_block_schedule(LEGACY_HARD_DOUBLE, 7)
        legacy_single = parse_block_schedule(LEGACY_HARD_SINGLE, 23)
        double_schedule = project_schedule(legacy_double, 8, architecture.double_blocks)
        single_schedule = project_schedule(legacy_single, 24, architecture.single_blocks)
    else:
        double_schedule = parse_block_schedule(
            double_blocks, architecture.max_double_block, strict=True
        )
        single_schedule = parse_block_schedule(
            single_blocks, architecture.max_single_block, strict=True
        )

    if str(strength_mode) == "normalized_total":
        double_schedule, single_schedule = _normalize_pair(
            double_schedule, single_schedule, total_strength
        )

    return TransferConfig(
        similarity_floor=float(min(max(similarity_floor, 0.0), 0.95)),
        temperature=float(max(softmax_temperature, 1e-6)),
        mask_threshold=float(min(max(mask_threshold, 0.0), 1.0)),
        double_schedule=double_schedule,
        single_schedule=single_schedule,
        architecture=architecture,
    )


class Flux2IdentityFeatureTransfer:
    """Pull generated image features toward matching reference-image features."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (
                    [
                        "AUTO_BALANCED",
                        "AUTO_SOFT",
                        "AUTO_STRONG",
                        "KLEIN_LEGACY_HARD",
                        "KLEIN_LEGACY_MID",
                        "KLEIN_LEGACY_SOFT",
                        "CUSTOM",
                    ],
                    {"default": "AUTO_BALANCED"},
                ),
                "enabled": ("BOOLEAN", {"default": True}),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                "reference_indices": ("STRING", {"default": "all", "multiline": False}),
                "strength_mode": (
                    ["normalized_total", "legacy_per_block"],
                    {"default": "normalized_total"},
                ),
                "total_strength": (
                    "FLOAT",
                    {
                        "default": 0.65,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Approximate aggregate transfer strength when normalized_total is selected.",
                    },
                ),
                "similarity_floor": (
                    "FLOAT",
                    {"default": 0.20, "min": 0.0, "max": 0.95, "step": 0.001},
                ),
                "softmax_temperature": (
                    "FLOAT",
                    {"default": 0.07, "min": 0.0001, "max": 0.25, "step": 0.0001},
                ),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "double_blocks": (
                    "STRING",
                    {"default": "0-7:mid_img=0.30", "multiline": False},
                ),
                "single_blocks": (
                    "STRING",
                    {"default": "0:mid_img=0.20", "multiline": False},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "sigma_scaling": (["none", "equal_energy"], {"default": "none"}),
                "mask_behavior": (
                    ["focus_only", "zero_unmasked_tokens"],
                    {"default": "focus_only"},
                ),
                "query_chunk_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 32,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Limits the similarity matrix size to reduce peak VRAM.",
                    },
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "subject_mask_1": ("MASK",),
                "subject_mask_2": ("MASK",),
                "subject_mask_3": ("MASK",),
                "subject_mask_4": ("MASK",),
                "subject_mask_5": ("MASK",),
                "subject_mask_6": ("MASK",),
                "subject_mask_7": ("MASK",),
                "subject_mask_8": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False

    def apply(
        self,
        model,
        preset="AUTO_BALANCED",
        enabled=True,
        reference_index=0,
        reference_indices="all",
        strength_mode="normalized_total",
        total_strength=0.65,
        similarity_floor=0.20,
        softmax_temperature=0.07,
        mask_threshold=0.95,
        double_blocks="0-7:mid_img=0.30",
        single_blocks="0:mid_img=0.20",
        start_percent=0.0,
        end_percent=1.0,
        sigma_scaling="none",
        mask_behavior="focus_only",
        query_chunk_size=256,
        debug=False,
        sigmas=None,
        subject_mask_1=None,
        subject_mask_2=None,
        subject_mask_3=None,
        subject_mask_4=None,
        subject_mask_5=None,
        subject_mask_6=None,
        subject_mask_7=None,
        subject_mask_8=None,
    ):
        architecture = require_capabilities(
            model,
            attn_output=True,
            attn_input=mask_behavior == "zero_unmasked_tokens",
        )
        patched = model.clone()
        if not bool(enabled):
            return (patched,)

        config = resolve_transfer_config(
            architecture,
            preset,
            similarity_floor,
            softmax_temperature,
            mask_threshold,
            double_blocks,
            single_blocks,
            strength_mode,
            total_strength,
        )
        if not config.double_schedule and not config.single_schedule:
            return (patched,)

        masks = [
            _prepare_mask(subject_mask_1),
            _prepare_mask(subject_mask_2),
            _prepare_mask(subject_mask_3),
            _prepare_mask(subject_mask_4),
            _prepare_mask(subject_mask_5),
            _prepare_mask(subject_mask_6),
            _prepare_mask(subject_mask_7),
            _prepare_mask(subject_mask_8),
        ]
        mask_cache: Dict[Tuple[int, int, float], torch.Tensor] = {}
        sigma_info = _sigma_schedule(sigmas)
        start = min(max(float(start_percent), 0.0), 1.0)
        end = min(max(float(end_percent), start), 1.0)
        chunk_size = max(32, int(query_chunk_size))
        debug_steps: set[tuple[str, int]] = set()

        def mask_indices(reference_id: int, count: int, device):
            if reference_id < 0 or reference_id >= len(masks):
                return None
            source = masks[reference_id]
            if source is None:
                return None
            key = (reference_id, int(count), config.mask_threshold)
            if key not in mask_cache:
                grid = _grid_for_tokens(count, source)
                pooled = F.adaptive_avg_pool2d(source[None, None], grid).view(-1)
                keep = pooled >= config.mask_threshold
                mask_cache[key] = torch.nonzero(keep, as_tuple=False).squeeze(-1).long().cpu()
            return mask_cache[key].to(device)

        def selected_slices(ref_tokens: Sequence[int], base: int):
            selected = parse_reference_indices(
                reference_indices, len(ref_tokens), fallback=reference_index
            )
            selected_set = set(selected)
            slices = []
            offset = 0
            for ref_id, count in enumerate(ref_tokens):
                count = int(count)
                ref_start = base + offset
                ref_end = ref_start + count
                if ref_id in selected_set and count > 0:
                    slices.append((ref_id, ref_start, ref_end))
                offset += count
            return slices

        def reference_bank(tokens: torch.Tensor, slices):
            parts = []
            for ref_id, ref_start, ref_end in slices:
                reference = tokens[:, ref_start:ref_end]
                indices = mask_indices(ref_id, ref_end - ref_start, tokens.device)
                if indices is not None:
                    if indices.numel() == 0:
                        continue
                    reference = reference.index_select(1, indices)
                if reference.shape[1]:
                    parts.append(reference)
            if not parts:
                return None
            bank = torch.cat(parts, dim=1)
            if bank.shape[0] == 1 and tokens.shape[0] > 1:
                bank = bank.expand(tokens.shape[0], -1, -1)
            return bank

        def step_controls(extra_options):
            current_sigma = _scalar_float(extra_options.get("sigmas"))
            if sigma_info is not None and current_sigma is not None:
                values, ratios = sigma_info
                step_index = _sigma_step(values, current_sigma)
                denominator = max(values.numel() - 2, 1)
                progress = step_index / denominator
                multiplier = (
                    float(ratios[step_index].item())
                    if sigma_scaling == "equal_energy"
                    else 1.0
                )
                return progress, multiplier, step_index, current_sigma
            progress = (
                min(max(1.0 - current_sigma, 0.0), 1.0)
                if current_sigma is not None
                else 0.5
            )
            return progress, 1.0, None, current_sigma

        def pull_features(generated: torch.Tensor, reference: torch.Tensor, strength: float):
            if strength <= 0.0 or reference is None or reference.shape[1] == 0:
                return None
            gen_float = generated.float()
            ref_float = reference.float()
            ref_centered = ref_float - ref_float.mean(dim=1, keepdim=True)
            ref_norm = F.normalize(ref_centered, dim=-1)
            gen_mean = gen_float.mean(dim=1, keepdim=True)
            output_chunks = []
            for chunk_start in range(0, gen_float.shape[1], chunk_size):
                chunk_end = min(chunk_start + chunk_size, gen_float.shape[1])
                gen_chunk = gen_float[:, chunk_start:chunk_end]
                gen_centered = gen_chunk - gen_mean
                gen_norm = F.normalize(gen_centered, dim=-1)
                similarity = torch.bmm(gen_norm, ref_norm.transpose(1, 2))
                negative = torch.finfo(similarity.dtype).min
                similarity = torch.where(
                    similarity >= config.similarity_floor,
                    similarity,
                    torch.full_like(similarity, negative),
                )
                weights = torch.softmax(similarity / config.temperature, dim=-1)
                weights = torch.nan_to_num(weights, nan=0.0)
                pooled = torch.bmm(weights, ref_float)
                best = similarity.max(dim=-1).values
                best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
                confidence = (
                    (best - config.similarity_floor)
                    / max(1.0 - config.similarity_floor, 1e-6)
                ).clamp(0.0, 1.0)
                blend = (confidence * float(strength)).unsqueeze(-1)
                output_chunks.append((pooled - gen_chunk) * blend)
            return torch.cat(output_chunks, dim=1).to(generated.dtype)

        def source_mask_patch(
            q,
            k,
            v,
            pe=None,
            attn_mask=None,
            extra_options=None,
            **kwargs,
        ):
            options = extra_options or kwargs.get("extra_options") or {}
            ref_tokens = options.get("reference_image_num_tokens", []) or []
            if not ref_tokens:
                return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
            total_sequence = int(k.shape[2])
            total_reference = int(sum(ref_tokens))
            allow = torch.ones(
                (k.shape[0], total_sequence), dtype=torch.bool, device=k.device
            )
            changed = False
            for ref_id, ref_start, ref_end in selected_slices(
                ref_tokens, total_sequence - total_reference
            ):
                indices = mask_indices(ref_id, ref_end - ref_start, k.device)
                if indices is None:
                    continue
                allow[:, ref_start:ref_end] = False
                if indices.numel():
                    allow[:, ref_start + indices] = True
                changed = True
            if not changed:
                return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
            key_allow = allow[:, None, None, :]
            if attn_mask is None:
                combined = key_allow
            elif attn_mask.dtype == torch.bool:
                existing = attn_mask
                if existing.ndim == 2:
                    existing = (
                        existing[:, None, None, :]
                        if existing.shape[0] == k.shape[0]
                        else existing[None, None, :, :]
                    )
                elif existing.ndim == 3:
                    existing = existing[:, None, :, :]
                combined = existing & key_allow
            else:
                existing = attn_mask
                if existing.ndim == 2:
                    existing = (
                        existing[:, None, None, :]
                        if existing.shape[0] == k.shape[0]
                        else existing[None, None, :, :]
                    )
                elif existing.ndim == 3:
                    existing = existing[:, None, :, :]
                bias = torch.zeros(
                    (k.shape[0], 1, 1, total_sequence),
                    dtype=existing.dtype,
                    device=k.device,
                )
                bias.masked_fill_(~key_allow, torch.finfo(existing.dtype).min)
                combined = existing + bias
            return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": combined}

        def output_patch(attention: torch.Tensor, extra_options):
            ref_tokens = extra_options.get("reference_image_num_tokens", []) or []
            img_slice = extra_options.get("img_slice")
            if not ref_tokens or img_slice is None:
                return attention
            block_type = str(extra_options.get("block_type", "double"))
            block_index = int(extra_options.get("block_index", 0))
            if block_type == "double":
                strength = config.double_schedule.get(block_index, 0.0)
            elif block_type == "single":
                strength = config.single_schedule.get(block_index, 0.0)
            else:
                return attention
            if strength <= 0.0:
                return attention

            progress, sigma_multiplier, sigma_step, current_sigma = step_controls(extra_options)
            if progress < start or progress > end:
                return attention
            strength *= sigma_multiplier
            if strength <= 0.0:
                return attention

            text_end, declared_total = int(img_slice[0]), int(img_slice[1])
            total_sequence = min(int(attention.shape[1]), declared_total)
            total_reference = int(sum(ref_tokens))
            generated_start = text_end
            generated_end = total_sequence - total_reference
            if generated_end <= generated_start or total_reference <= 0:
                return attention
            slices = selected_slices(ref_tokens, total_sequence - total_reference)
            reference = reference_bank(attention, slices)
            if reference is None:
                return attention
            generated = attention[:, generated_start:generated_end]
            delta = pull_features(generated, reference, strength)
            if delta is None:
                return attention
            output = attention.clone()
            output[:, generated_start:generated_end] = generated + delta

            debug_key = (block_type, block_index)
            if debug and debug_key not in debug_steps:
                debug_steps.add(debug_key)
                print(
                    "[Flux2IdentityFeatureTransfer] "
                    f"variant={architecture.variant} block={block_type}:{block_index} "
                    f"strength={strength:.6f} progress={progress:.3f} "
                    f"sigma_step={sigma_step} sigma={current_sigma} "
                    f"generated_tokens={generated.shape[1]} "
                    f"reference_tokens={reference.shape[1]}"
                )
            return output

        if debug:
            print(
                "[Flux2IdentityFeatureTransfer] "
                f"architecture={architecture.to_dict()} preset={preset} "
                f"sim={config.similarity_floor:.4f} temp={config.temperature:.4f} "
                f"double={config.double_schedule} single={config.single_schedule} "
                f"mask_behavior={mask_behavior} sigma_scaling={sigma_scaling}"
            )

        patched.set_model_attn1_output_patch(output_patch)
        if mask_behavior == "zero_unmasked_tokens" and any(
            mask is not None for mask in masks
        ):
            patched.set_model_attn1_patch(source_mask_patch)
        return (patched,)


class LegacyIdentityFeatureTransferFinal:
    """Compatibility adapter for workflows using the original node identifier."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (
                    ["HARD_LOCK", "MID_LOCK", "SOFT_LOCK", "custom"],
                    {"default": "HARD_LOCK"},
                ),
                "enabled": ("BOOLEAN", {"default": True}),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 15}),
                "reference_indices": ("STRING", {"default": "all", "multiline": False}),
                "similarity_floor": (
                    "FLOAT",
                    {"default": 0.040, "min": 0.0, "max": 0.95, "step": 0.001},
                ),
                "softmax_temperature": (
                    "FLOAT",
                    {"default": 0.025, "min": 0.0001, "max": 0.25, "step": 0.0001},
                ),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "double_blocks": (
                    "STRING",
                    {"default": LEGACY_HARD_DOUBLE, "multiline": False},
                ),
                "single_blocks": (
                    "STRING",
                    {"default": LEGACY_HARD_SINGLE, "multiline": False},
                ),
                "debug": ("BOOLEAN", {"default": False}),
                "mask_behavior": (
                    ["focus_only", "zero_unmasked_tokens"],
                    {"default": "focus_only"},
                ),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "subject_mask_1": ("MASK",),
                "subject_mask_2": ("MASK",),
                "subject_mask_3": ("MASK",),
                "subject_mask_4": ("MASK",),
                "subject_mask_5": ("MASK",),
                "subject_mask_6": ("MASK",),
                "subject_mask_7": ("MASK",),
                "subject_mask_8": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2/legacy"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False

    def apply(
        self,
        model,
        preset="HARD_LOCK",
        enabled=True,
        reference_index=0,
        reference_indices="all",
        similarity_floor=0.040,
        softmax_temperature=0.025,
        mask_threshold=1.0,
        double_blocks=LEGACY_HARD_DOUBLE,
        single_blocks=LEGACY_HARD_SINGLE,
        debug=False,
        mask_behavior="focus_only",
        sigmas=None,
        **masks,
    ):
        preset_map = {
            "HARD_LOCK": "KLEIN_LEGACY_HARD",
            "MID_LOCK": "KLEIN_LEGACY_MID",
            "SOFT_LOCK": "KLEIN_LEGACY_SOFT",
            "custom": "CUSTOM",
        }
        return Flux2IdentityFeatureTransfer().apply(
            model=model,
            preset=preset_map.get(preset, "KLEIN_LEGACY_HARD"),
            enabled=enabled,
            reference_index=reference_index,
            reference_indices=reference_indices,
            strength_mode="legacy_per_block",
            total_strength=1.0,
            similarity_floor=similarity_floor,
            softmax_temperature=softmax_temperature,
            mask_threshold=mask_threshold,
            double_blocks=double_blocks,
            single_blocks=single_blocks,
            start_percent=0.0,
            end_percent=1.0,
            sigma_scaling="equal_energy" if sigmas is not None else "none",
            mask_behavior=mask_behavior,
            query_chunk_size=256,
            debug=debug,
            sigmas=sigmas,
            **masks,
        )


NODE_CLASS_MAPPINGS = {
    "Flux2IdentityFeatureTransfer": Flux2IdentityFeatureTransfer,
    "IdentityFeatureTransferFinal": LegacyIdentityFeatureTransferFinal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2IdentityFeatureTransfer": "FLUX.2 Identity Feature Transfer",
    "IdentityFeatureTransferFinal": "Identity Feature Transfer Final (Legacy)",
}


__all__ = [
    "Flux2IdentityFeatureTransfer",
    "LegacyIdentityFeatureTransferFinal",
    "TransferConfig",
    "resolve_transfer_config",
]
