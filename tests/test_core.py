from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from comfyui_flux2dev_enhancer.architecture import inspect_flux2_architecture
from comfyui_flux2dev_enhancer.conditioning import (
    Flux2ConditioningEnhancer,
    Flux2DetailController,
    _active_end,
    _compute_section_ranges,
)
from comfyui_flux2dev_enhancer.guidance import _progress, _reference_from_conditioning
from comfyui_flux2dev_enhancer.identity_transfer import resolve_transfer_config
from comfyui_flux2dev_enhancer.reference_controls import _reference_slice
from comfyui_flux2dev_enhancer.reference_latent import (
    apply_reference_metadata,
    split_reference_batches,
)
from comfyui_flux2dev_enhancer.registry import NODE_CLASS_MAPPINGS
from comfyui_flux2dev_enhancer.scheduling import (
    ScheduleParseError,
    normalized_per_application,
    parse_block_schedule,
    parse_reference_indices,
)


class FakePatcher:
    def __init__(self, params):
        diffusion = SimpleNamespace(
            params=params,
            double_blocks=[object()] * params.depth,
            single_blocks=[object()] * params.depth_single_blocks,
            patch_size=1,
        )
        self.model = SimpleNamespace(diffusion_model=diffusion)
        self.model_options = {}

    def clone(self):
        return self

    def set_model_attn1_patch(self, function):
        self.input_patch = function

    def set_model_attn1_output_patch(self, function):
        self.output_patch = function

    def set_model_post_input_patch(self, function):
        self.post_patch = function


def make_params(hidden, heads, double, single, context, guidance):
    return SimpleNamespace(
        hidden_size=hidden,
        num_heads=heads,
        depth=double,
        depth_single_blocks=single,
        context_in_dim=context,
        in_channels=128,
        axes_dim=[32, 32, 32, 32],
        global_modulation=True,
        guidance_embed=guidance,
        default_ref_method="index",
        ref_index_scale=10.0,
    )


class FakeHFTokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": list(range(len(text.split())))}


class FakeTokenizerChild:
    tokenizer = FakeHFTokenizer()
    start_token = None


class FakeTokenizerWrapper:
    llama_template = "PREFIX {} SUFFIX"
    qwen3_8b = FakeTokenizerChild()


class FakeClip:
    tokenizer = FakeTokenizerWrapper()


def test_detects_flux2_dev():
    architecture = inspect_flux2_architecture(
        FakePatcher(make_params(6144, 48, 8, 48, 15360, True))
    )
    assert architecture.variant == "flux2_dev"
    assert architecture.max_single_block == 47
    assert architecture.guidance_embed is True


def test_detects_klein_4b():
    architecture = inspect_flux2_architecture(
        FakePatcher(make_params(3072, 24, 5, 20, 7680, False))
    )
    assert architecture.variant == "flux2_klein_4b"
    assert architecture.double_blocks == 5


def test_detects_klein_9b():
    architecture = inspect_flux2_architecture(
        FakePatcher(make_params(4096, 32, 8, 24, 12288, False))
    )
    assert architecture.variant == "flux2_klein_9b"
    assert architecture.single_blocks == 24


def test_parse_schedule_and_strict_range():
    assert parse_block_schedule("0-2:mid_img=0.5; 4:0.25", 5) == {
        0: 0.5,
        1: 0.5,
        2: 0.5,
        4: 0.25,
    }
    with pytest.raises(ScheduleParseError):
        parse_block_schedule("0-24:0.2", 23)


def test_normalized_strength_composes():
    per_application = normalized_per_application(0.8, 16)
    composed = 1.0 - (1.0 - per_application) ** 16
    assert math.isclose(composed, 0.8, rel_tol=1e-7)


def test_reference_index_parser():
    assert parse_reference_indices("0,2-3", 5) == [0, 2, 3]
    assert parse_reference_indices("invalid", 3, fallback=2) == [2]


def test_auto_normalized_schedule_is_architecture_aware():
    architecture = inspect_flux2_architecture(
        FakePatcher(make_params(6144, 48, 8, 48, 15360, True))
    )
    config = resolve_transfer_config(
        architecture,
        "AUTO_BALANCED",
        0.2,
        0.07,
        0.95,
        "",
        "",
        "normalized_total",
        0.65,
    )
    assert config.double_schedule and config.single_schedule
    assert max(config.single_schedule) <= architecture.max_single_block
    assert max(config.double_schedule.values()) < 0.2


def test_custom_per_block_schedule_is_preserved():
    architecture = inspect_flux2_architecture(
        FakePatcher(make_params(4096, 32, 8, 24, 12288, False))
    )
    config = resolve_transfer_config(
        architecture,
        "CUSTOM",
        0.3,
        0.08,
        0.9,
        "0-1:0.2",
        "3:0.25",
        "per_block",
        0.5,
    )
    assert config.double_schedule == {0: 0.2, 1: 0.2}
    assert config.single_schedule == {3: 0.25}


def test_unknown_preset_is_rejected():
    architecture = inspect_flux2_architecture(
        FakePatcher(make_params(4096, 32, 8, 24, 12288, False))
    )
    with pytest.raises(ValueError):
        resolve_transfer_config(
            architecture, "REMOVED_PRESET", 0.2, 0.07, 0.95, "", "", "per_block", 0.5
        )


def test_batch_reference_split_is_stable():
    latent = {"samples": torch.arange(16.0).reshape(2, 2, 2, 2)}
    references = split_reference_batches([latent])
    assert len(references) == 2
    assert references[0][0, 0, 0, 0].item() == 0
    assert references[1][0, 0, 0, 0].item() == 8


def test_reference_metadata_append_and_model_default():
    existing = torch.zeros(1, 2, 2, 2)
    added = torch.ones(1, 2, 2, 2)
    conditioning = [[
        torch.zeros(1, 1, 4),
        {"reference_latents": [existing], "reference_latents_method": "index"},
    ]]
    appended = apply_reference_metadata(
        conditioning, [added], mode="append", reference_method="index"
    )
    assert len(appended[0][1]["reference_latents"]) == 2
    assert appended[0][1] is not conditioning[0][1]
    defaulted = apply_reference_metadata(
        conditioning, [added], reference_method="model_default"
    )
    assert "reference_latents_method" not in defaulted[0][1]


def test_reference_slice_uses_runtime_token_counts():
    assert _reference_slice([10, 20, 5], 1, 100) == (75, 95)


def test_active_end_uses_attention_mask():
    meta = {"attention_mask": torch.tensor([[1, 1, 1, 0, 0]])}
    assert _active_end(meta, 5) == 3


def test_section_ranges_are_monotonic():
    ranges, backend = _compute_section_ranges(
        FakeClip(),
        {"front": "one two", "mid": "three", "end": "four five"},
        ", ",
    )
    assert backend == "FakeTokenizerChild"
    assert ranges["front"][0] <= ranges["front"][1]
    assert ranges["front"][1] <= ranges["mid"][0]
    assert ranges["mid"][1] <= ranges["end"][0]


def test_conditioning_neutral_is_exact_pass_through():
    conditioning = [[torch.randn(1, 4, 12), {}]]
    result = Flux2ConditioningEnhancer().enhance(conditioning)[0]
    assert result is conditioning


def test_detail_controller_uses_section_metadata():
    tensor = torch.ones(1, 6, 3)
    conditioning = [[
        tensor,
        {"flux2_sections": {"front": (0, 2), "mid": (2, 4), "end": (4, 6)}},
    ]]
    output = Flux2DetailController().control(conditioning, front_mult=2.0)[0]
    assert torch.all(output[0][0][:, :2] == 2)
    assert torch.all(output[0][0][:, 2:] == 1)


def test_guidance_reference_lookup_and_progress():
    reference = torch.zeros(1, 128, 2, 2)
    conditioning = [[torch.zeros(1, 1, 3), {"reference_latents": [reference]}]]
    assert _reference_from_conditioning(conditioning, 0) is reference
    progress, step = _progress(
        torch.tensor([1.0, 0.8, 0.4, 0.0], dtype=torch.float64), 0.4, {}
    )
    assert step == 2 and progress == 1.0


def test_registry_exposes_only_canonical_node_ids():
    assert set(NODE_CLASS_MAPPINGS) == {
        "Flux2ArchitectureInspector",
        "Flux2ConditioningEnhancer",
        "Flux2TextConditioningEnhancer",
        "Flux2SectionedEncoder",
        "Flux2DetailController",
        "Flux2MultiReferenceLatent",
        "Flux2ReferenceAttentionControl",
        "Flux2ReferenceWeight",
        "Flux2TextReferenceBalance",
        "Flux2ReferenceLatentMask",
        "Flux2ColorAnchor",
        "Flux2IdentityGuidance",
        "Flux2IdentityFeatureTransfer",
    }


def test_registry_categories_use_project_namespace():
    for node_class in NODE_CLASS_MAPPINGS.values():
        assert node_class.CATEGORY.startswith("ComfyUI-Flux2Dev-Enhancer/")
