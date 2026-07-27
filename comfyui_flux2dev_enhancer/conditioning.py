"""Text-conditioning utilities for official FLUX.2 encoder wrappers."""

from __future__ import annotations

import math
import re
from collections import deque
from typing import Dict, Optional, Tuple

import torch

from .constants import PROJECT_NAME

_SEPARATORS = {"comma": ", ", "period": ". ", "space": " ", "newline": "\n"}


def _active_end(meta: dict, sequence_length: int, override: int = 0) -> int:
    if override > 0:
        return min(int(override), sequence_length)
    attention_mask = meta.get("attention_mask")
    if torch.is_tensor(attention_mask) and attention_mask.numel() > 0:
        mask = attention_mask
        while mask.dim() > 2:
            mask = mask[0]
        if mask.dim() == 2:
            active = torch.nonzero(mask[0] > 0, as_tuple=False).flatten()
        else:
            active = torch.nonzero(mask > 0, as_tuple=False).flatten()
        if active.numel() > 0:
            return min(int(active[-1].item()) + 1, sequence_length)
    return sequence_length


def _parse_marker_sections(text: str) -> Optional[Dict[str, str]]:
    if not text:
        return None
    pattern = r"\[(FRONT|MID|END)\](.*?)(?=\[(?:FRONT|MID|END)\]|$)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    sections = {"front": "", "mid": "", "end": ""}
    for name, content in matches:
        sections[name.lower()] = content.strip()
    return sections


def _find_tokenizer_adapter(clip):
    """Return the HF tokenizer, wrapper template, leading-token count, and backend."""

    root = getattr(clip, "tokenizer", None)
    if root is None:
        return None, None, 0, "unavailable"
    template = getattr(root, "llama_template", None)
    queue = deque([(root, 0)])
    visited = set()
    preferred_attrs = (
        "mistral3_24b",
        "qwen3_8b",
        "qwen3_4b",
        "tokenizer",
        "clip_l",
        "t5xxl",
    )
    while queue:
        current, depth = queue.popleft()
        if current is None or id(current) in visited or depth > 3:
            continue
        visited.add(id(current))
        candidate = getattr(current, "tokenizer", None)
        if candidate is not None and callable(candidate):
            start_token = getattr(current, "start_token", None)
            leading = 1 if start_token is not None else 0
            return candidate, template, leading, current.__class__.__name__
        for attr in preferred_attrs:
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                queue.append((child, depth + 1))
    return None, template, 0, root.__class__.__name__


def _count_tokens(tokenizer, text: str) -> int:
    result = tokenizer(text, add_special_tokens=False, return_tensors=None)
    input_ids = result["input_ids"] if isinstance(result, dict) else result.input_ids
    if torch.is_tensor(input_ids):
        if input_ids.dim() > 1:
            input_ids = input_ids[0]
        return int(input_ids.numel())
    if len(input_ids) > 0 and isinstance(input_ids[0], (list, tuple)):
        input_ids = input_ids[0]
    return len(input_ids)


def _compute_section_ranges(clip, sections: Dict[str, str], separator: str):
    tokenizer, template, leading, backend = _find_tokenizer_adapter(clip)
    if tokenizer is None or not template or "{}" not in template:
        return None, backend
    prefix, _suffix = template.split("{}", 1)
    ranges: Dict[str, Tuple[int, int]] = {}
    nonempty = [key for key in ("front", "mid", "end") if sections[key]]
    consumed = ""
    for key in ("front", "mid", "end"):
        text = sections[key]
        start = leading + _count_tokens(tokenizer, prefix + consumed)
        if text:
            consumed += text
        end = leading + _count_tokens(tokenizer, prefix + consumed)
        ranges[key] = (start, end)
        if text and key in nonempty and nonempty.index(key) < len(nonempty) - 1:
            consumed += separator
    return ranges, backend


def _three_slices(width: int):
    if width > 0 and width % 3 == 0:
        size = width // 3
        return (0, size), (size, size * 2), (size * 2, width)
    return None


class Flux2ConditioningEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "active_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05},
                ),
                "per_token_whiten": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 5.0, "step": 0.05},
                ),
                "norm_equalize": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
            "optional": {
                "early_layer_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05},
                ),
                "mid_layer_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05},
                ),
                "late_layer_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05},
                ),
                "preserve_original": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "active_end_override": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192},
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "enhance"
    CATEGORY = "conditioning/flux2"

    def enhance(
        self,
        conditioning,
        active_scale=1.0,
        per_token_whiten=0.0,
        norm_equalize=0.0,
        early_layer_scale=1.0,
        mid_layer_scale=1.0,
        late_layer_scale=1.0,
        preserve_original=0.0,
        active_end_override=0,
        debug=False,
    ):
        neutral = (
            active_scale == 1.0
            and per_token_whiten == 0.0
            and norm_equalize == 0.0
            and early_layer_scale == 1.0
            and mid_layer_scale == 1.0
            and late_layer_scale == 1.0
            and preserve_original == 0.0
        )
        if not conditioning or neutral:
            return (conditioning,)
        output = []
        for item_index, (tensor, meta) in enumerate(conditioning):
            if not torch.is_tensor(tensor) or tensor.dim() != 3:
                output.append([tensor, dict(meta)])
                continue
            original_dtype = tensor.dtype
            value = tensor.float().clone()
            end = _active_end(meta, value.shape[1], active_end_override)
            active = value[:, :end].clone()
            original = active.clone()
            if per_token_whiten != 0.0 and active.shape[1] > 0:
                mean = active.mean(dim=1, keepdim=True)
                active = mean + (active - mean) * (1.0 + float(per_token_whiten))
            if norm_equalize > 0.0 and active.shape[1] > 0:
                norms = active.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                target = norms.mean(dim=1, keepdim=True)
                equalized = active / norms * target
                active = active * (1.0 - norm_equalize) + equalized * norm_equalize
            if active_scale != 1.0:
                active *= float(active_scale)
            slices = _three_slices(active.shape[-1])
            scales = (early_layer_scale, mid_layer_scale, late_layer_scale)
            if slices is not None:
                for (start, stop), scale in zip(slices, scales):
                    if scale != 1.0:
                        active[:, :, start:stop] *= float(scale)
            elif any(scale != 1.0 for scale in scales):
                raise ValueError(
                    "FLUX.2 Conditioning Enhancer: per-layer scaling requires a conditioning width divisible by three."
                )
            if preserve_original > 0.0:
                active = active * (1.0 - preserve_original) + original * preserve_original
            value[:, :end] = active
            output.append([value.to(dtype=original_dtype), dict(meta)])
            if debug:
                print(
                    f"[{PROJECT_NAME}:ConditioningEnhancer] item={item_index} "
                    f"shape={tuple(tensor.shape)} active_end={end} "
                    f"three_slices={slices is not None}"
                )
        return (output,)


class Flux2TextConditioningEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "magnitude": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
                ),
            },
            "optional": {
                "contrast": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.05},
                ),
                "normalize_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "skip_first_token": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "enhance"
    CATEGORY = "conditioning/flux2"

    def enhance(
        self,
        conditioning,
        magnitude=1.0,
        contrast=0.0,
        normalize_strength=0.0,
        skip_first_token=True,
        debug=False,
    ):
        if not conditioning or (
            magnitude == 1.0 and contrast == 0.0 and normalize_strength == 0.0
        ):
            return (conditioning,)
        output = []
        for item_index, (tensor, meta) in enumerate(conditioning):
            if not torch.is_tensor(tensor) or tensor.dim() != 3:
                output.append([tensor, dict(meta)])
                continue
            value = tensor.float().clone()
            end = _active_end(meta, value.shape[1])
            start = 1 if skip_first_token and end > 1 else 0
            active = value[:, start:end]
            if active.shape[1] > 0:
                if normalize_strength > 0.0:
                    norms = active.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    target = norms.mean(dim=1, keepdim=True)
                    normalized = active / norms * target
                    active = (
                        active * (1.0 - normalize_strength)
                        + normalized * normalize_strength
                    )
                if contrast != 0.0:
                    mean = active.mean(dim=1, keepdim=True)
                    scale = 1.0 + contrast if contrast >= 0 else math.exp(contrast)
                    active = mean + (active - mean) * scale
                if magnitude != 1.0:
                    active *= float(magnitude)
                value[:, start:end] = active
            output.append([value.to(dtype=tensor.dtype), dict(meta)])
            if debug:
                print(
                    f"[{PROJECT_NAME}:TextConditioningEnhancer] "
                    f"item={item_index} active={start}:{end}"
                )
        return (output,)


class Flux2SectionedEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"clip": ("CLIP",)},
            "optional": {
                "front_text": ("STRING", {"multiline": True, "default": ""}),
                "mid_text": ("STRING", {"multiline": True, "default": ""}),
                "end_text": ("STRING", {"multiline": True, "default": ""}),
                "combined_prompt": ("STRING", {"multiline": True, "default": ""}),
                "separator": (list(_SEPARATORS), {"default": "comma"}),
                "show_preview": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "conditioning",
        "front_section",
        "mid_section",
        "end_section",
        "full_prompt",
    )
    FUNCTION = "encode_sectioned"
    CATEGORY = "conditioning/flux2"
    OUTPUT_NODE = True

    def encode_sectioned(
        self,
        clip,
        front_text="",
        mid_text="",
        end_text="",
        combined_prompt="",
        separator="comma",
        show_preview=True,
        debug=False,
    ):
        sections = _parse_marker_sections(combined_prompt) or {
            "front": front_text or "",
            "mid": mid_text or "",
            "end": end_text or "",
        }
        separator_text = _SEPARATORS.get(separator, ", ")
        full_prompt = separator_text.join(
            sections[key] for key in ("front", "mid", "end") if sections[key]
        )
        ranges, backend = _compute_section_ranges(clip, sections, separator_text)
        tokens = clip.tokenize(full_prompt)
        add_dict = {"flux2_section_backend": backend}
        if ranges is not None:
            add_dict["flux2_sections"] = ranges
        conditioning = clip.encode_from_tokens_scheduled(tokens, add_dict=add_dict)
        if show_preview or debug:
            print(
                "\n".join(
                    [
                        "=" * 70,
                        f"{PROJECT_NAME} — FLUX.2 Sectioned Encoding",
                        f"Tokenizer backend: {backend}",
                        f"Ranges: {ranges if ranges is not None else 'unavailable'}",
                        f"Prompt: {full_prompt!r}",
                        "=" * 70,
                    ]
                )
            )
        return (
            conditioning,
            sections["front"],
            sections["mid"],
            sections["end"],
            full_prompt,
        )


class Flux2DetailController:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"conditioning": ("CONDITIONING",)},
            "optional": {
                "front_mult": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05},
                ),
                "mid_mult": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05},
                ),
                "end_mult": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05},
                ),
                "emphasis_start": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "emphasis_end": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "emphasis_mult": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "preserve_original": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "fallback_mode": (
                    ["relative_sections", "no_op"],
                    {"default": "relative_sections"},
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "control"
    CATEGORY = "conditioning/flux2"

    def control(
        self,
        conditioning,
        front_mult=1.0,
        mid_mult=1.0,
        end_mult=1.0,
        emphasis_start=0,
        emphasis_end=0,
        emphasis_mult=1.0,
        preserve_original=0.0,
        fallback_mode="relative_sections",
        debug=False,
    ):
        neutral = (
            front_mult == mid_mult == end_mult == 1.0
            and (emphasis_end <= emphasis_start or emphasis_mult == 1.0)
            and preserve_original == 0.0
        )
        if not conditioning or neutral:
            return (conditioning,)
        output = []
        for item_index, (tensor, meta) in enumerate(conditioning):
            if not torch.is_tensor(tensor) or tensor.dim() != 3:
                output.append([tensor, dict(meta)])
                continue
            value = tensor.float().clone()
            original = value.clone()
            end = _active_end(meta, value.shape[1])
            ranges = meta.get("flux2_sections")
            source = "metadata"
            if not ranges:
                if fallback_mode == "no_op":
                    output.append([tensor, dict(meta)])
                    continue
                first = int(end * 0.25)
                second = int(end * 0.75)
                ranges = {
                    "front": (0, first),
                    "mid": (first, second),
                    "end": (second, end),
                }
                source = "relative fallback"
            for key, multiplier in (
                ("front", front_mult),
                ("mid", mid_mult),
                ("end", end_mult),
            ):
                if key not in ranges or multiplier == 1.0:
                    continue
                start_idx, stop_idx = ranges[key]
                start_idx = min(max(int(start_idx), 0), end)
                stop_idx = min(max(int(stop_idx), start_idx), end)
                value[:, start_idx:stop_idx] *= float(multiplier)
            if emphasis_end > emphasis_start and emphasis_mult != 1.0:
                start_idx = min(max(int(emphasis_start), 0), end)
                stop_idx = min(max(int(emphasis_end), start_idx), end)
                value[:, start_idx:stop_idx] *= float(emphasis_mult)
            if preserve_original > 0.0:
                value = value * (1.0 - preserve_original) + original * preserve_original
            output.append([value.to(dtype=tensor.dtype), dict(meta)])
            if debug:
                print(
                    f"[{PROJECT_NAME}:DetailController] item={item_index} "
                    f"source={source} ranges={ranges}"
                )
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Flux2ConditioningEnhancer": Flux2ConditioningEnhancer,
    "Flux2TextConditioningEnhancer": Flux2TextConditioningEnhancer,
    "Flux2SectionedEncoder": Flux2SectionedEncoder,
    "Flux2DetailController": Flux2DetailController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2ConditioningEnhancer": "FLUX.2 Conditioning Enhancer",
    "Flux2TextConditioningEnhancer": "FLUX.2 Text Conditioning Enhancer",
    "Flux2SectionedEncoder": "FLUX.2 Sectioned Encoder",
    "Flux2DetailController": "FLUX.2 Detail Controller",
}

__all__ = [
    "Flux2ConditioningEnhancer",
    "Flux2DetailController",
    "Flux2SectionedEncoder",
    "Flux2TextConditioningEnhancer",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "_active_end",
    "_compute_section_ranges",
]
