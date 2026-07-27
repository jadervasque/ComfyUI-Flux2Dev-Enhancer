# PLAN0 — FLUX.2 Multi-Variant Compatibility

## Goal

Convert this fork from a Klein-specific ComfyUI extension into a professional FLUX.2 enhancement suite that supports the open-weight FLUX.2 family through runtime architecture detection rather than hard-coded model names or block counts.

Primary supported families:

- FLUX.2 [dev]
- FLUX.2 [klein] 4B distilled
- FLUX.2 [klein] 4B base
- FLUX.2 [klein] 9B distilled
- FLUX.2 [klein] 9B base
- FLUX.2 [klein] 9B KV, where the ComfyUI model wrapper exposes compatible attention hooks and reference metadata
- Quantized checkpoints of the same architectures, provided their loader preserves ComfyUI model patching APIs and transformer metadata

The implementation must preserve existing workflows through legacy node identifiers while exposing new model-neutral FLUX.2 node names.

## Project principles

1. Detect capabilities at runtime; do not infer compatibility from checkpoint filenames.
2. Prefer architecture and metadata inspection over fixed parameter-count assumptions.
3. Keep neutral settings exact pass-throughs.
4. Preserve legacy node IDs and inputs where practical.
5. Fail safely when a loader does not expose required hooks or metadata.
6. Separate model-family logic, schedules, node registration, and algorithms.
7. Document verified support separately from expected compatibility.
8. Preserve the original MIT copyright notice and prominently credit capitan01R as the original author.

## Current architecture assessment

The existing project contains four categories of behavior.

### Already mostly architecture-neutral

- Multi-reference latent injection through `reference_latents`
- Reference K/V weighting through `reference_image_num_tokens`
- Text/reference attention balance
- Reference latent masking
- Color anchoring in latent space
- Post-CFG identity guidance
- Basic text-conditioning magnitude and contrast transforms

These components primarily need naming, validation, defensive checks, metadata normalization, and documentation updates.

### Klein-specific but generalizable

- Identity Feature Transfer schedules assume 8 double blocks and 24 single blocks.
- Identity Feature Transfer presets are calibrated for Klein 9B.
- Text-conditioning layer slices are described as Qwen-specific even though the implementation can operate on any three-slice conditioning tensor.
- Detail Controller metadata uses `klein_sections`.
- Categories and display names contain `flux2klein` or `Klein`.

### Strongly Klein-specific

- Sectioned Encoder directly discovers Qwen3 tokenizer wrappers and hard-codes the Klein chat template.
- Experimental KSampler uses Klein-oriented defaults, labels, and schedule assumptions.

### Compatibility risk areas

- Third-party GGUF and quantized loaders may not expose `set_model_attn1_patch`, `set_model_attn1_output_patch`, model parameters, or standard `transformer_options`.
- KV-cached Klein variants may alter reference-token execution across denoising steps.
- FLUX.2 [dev] has more single-stream depth than Klein 9B, so per-block strengths cannot be copied directly.
- Text encoder structures differ between Klein and dev.
- Applying transfer over more active blocks can unintentionally multiply effective strength.

## Target package structure

```text
ComfyUI-Flux2Dev-Enhancer/
├── __init__.py
├── architecture.py
├── scheduling.py
├── validation.py
├── flux2_identity_transfer.py
├── flux2_reference_latent.py
├── flux2_reference_controls.py
├── flux2_conditioning.py
├── flux2_sampling.py
├── legacy.py
├── identity_feature_transfer.py          # retained for source compatibility
├── multi_reference_latent.py             # retained for source compatibility
├── existing legacy modules               # retained unless safely superseded
├── tests/
│   ├── test_architecture.py
│   ├── test_scheduling.py
│   ├── test_reference_metadata.py
│   ├── test_identity_slicing.py
│   └── test_legacy_registration.py
├── plans/
│   └── PLAN0.md
└── README.md
```

The final implementation may use fewer modules if consolidation improves maintainability, but architecture detection and schedule handling must remain isolated from node UI definitions.

## Phase 1 — Architecture and capability detection

Create `architecture.py` with:

- A frozen `Flux2Architecture` data model.
- Safe extraction of the diffusion model from a ComfyUI `MODEL` patcher.
- Runtime inspection of:
  - hidden size
  - attention head count
  - double-block count
  - single-block count
  - context width
  - patch size
  - guidance embedding support
  - default reference method
  - reference index scale
  - global modulation
- Capability detection for:
  - attention input patching
  - attention output patching
  - post-input patching
  - sampler post-CFG hooks
  - reference-latent support
- Variant classification using architecture fingerprints when known.
- A generic fallback profile for future FLUX.2 variants with compatible hooks.

Known profiles should include:

- `flux2_dev`
- `flux2_klein_4b`
- `flux2_klein_9b`
- `flux2_klein_9b_kv`
- `flux2_unknown_compatible`

Base and distilled checkpoints normally share architecture; distillation status should therefore remain a sampling/profile attribute rather than an architecture identity unless it can be reliably detected.

Acceptance criteria:

- No node depends on checkpoint filenames.
- Unsupported models return actionable errors.
- Unknown compatible FLUX.2 models can run in conservative mode.
- Detection functions are unit-testable without importing ComfyUI.

## Phase 2 — Dynamic schedules and strength normalization

Create `scheduling.py` with:

- A strict block-schedule parser.
- Dynamic validation against actual double/single block counts.
- Relative schedule projection between architecture profiles.
- Optional total-strength normalization.
- Sigma-aware scaling helpers.
- Denoising-window helpers.

Strength modes:

- `legacy_per_block`: preserves existing behavior.
- `normalized_total`: converts a requested aggregate strength into a per-application strength based on active block count.

Suggested normalization:

```python
per_application = 1.0 - (1.0 - total_strength) ** (1.0 / active_applications)
```

This is an approximation and must be documented as such.

Acceptance criteria:

- FLUX.2 [dev] schedules support its full detected single-block range.
- Klein presets remain backward-compatible.
- Invalid schedules fail with clear messages.
- Neutral or empty schedules are exact no-ops.

## Phase 3 — Generic Identity Feature Transfer

Implement a new model-neutral node:

- Node ID: `Flux2IdentityFeatureTransfer`
- Display name: `FLUX.2 Identity Feature Transfer`
- Category: `conditioning/flux2`

Core behavior:

- Read `reference_image_num_tokens`, `img_slice`, `block_type`, and `block_index` at runtime.
- Separate generated image tokens and reference tokens defensively.
- Support one or multiple selected references.
- Support per-reference masks.
- Preserve `focus_only` and `zero_unmasked_tokens` mask behaviors.
- Support dynamic double/single schedules.
- Support architecture-aware presets.
- Support optional sigmas and denoising windows.
- Support debug diagnostics without logging tensors or excessive per-token data.
- Validate batch and sequence dimensions.
- Avoid mutable state leaking across generations.

Presets:

- `AUTO_SOFT`
- `AUTO_BALANCED`
- `AUTO_STRONG`
- `KLEIN_LEGACY_HARD`
- `KLEIN_LEGACY_MID`
- `KLEIN_LEGACY_SOFT`
- `CUSTOM`

`AUTO_*` presets must resolve schedules and strengths from detected depth. Initial FLUX.2 [dev] presets must be conservative and explicitly described as starting points requiring empirical tuning.

Legacy compatibility:

- Keep `IdentityFeatureTransferFinal` registered.
- Route it to the new implementation with legacy defaults.
- Preserve earlier Identity Feature Transfer classes as deprecated aliases unless their behavior can be safely migrated.

Acceptance criteria:

- No hard-coded `0-23` limit in generic execution.
- `enabled=false` and zero strengths are no-ops.
- Reference slicing is verified for mixed reference resolutions.
- Hooks do nothing safely when no reference tokens are present.
- Existing Klein workflows continue to load.

## Phase 4 — Generic reference-latent nodes

Implement:

- `Flux2MultiReferenceLatent`
- `Flux2ReferenceLatentMask`
- `Flux2ReferenceAttentionControl`
- `Flux2ReferenceWeight`
- `Flux2TextReferenceBalance`

Enhance Multi Reference Latent with:

- Up to eight latent inputs.
- Batch splitting.
- `replace` and `append` modes.
- Configurable reference method:
  - `model_default`
  - `index`
  - `offset`
  - `uxo`
  - `index_timestep_zero`
- Duplicate-safe metadata copying.
- Input validation and optional debug summary.

Do not silently force `index` for every architecture when `model_default` is selected.

Legacy IDs must remain registered as aliases.

Acceptance criteria:

- Metadata is copied rather than mutated in place.
- Existing references can be appended or replaced intentionally.
- Batch items become individual references in stable order.
- Unsupported reference methods are rejected before sampling.

## Phase 5 — Generic conditioning tools

Rename and generalize:

- `FLUX.2 Conditioning Enhancer`
- `FLUX.2 Text Conditioning Enhancer`
- `FLUX.2 Sectioned Encoder`
- `FLUX.2 Detail Controller`

Changes:

- Replace `klein_sections` with `flux2_sections`.
- Read legacy `klein_sections` as a fallback.
- Store encoder metadata without discarding existing conditioning metadata.
- Make active token detection consistent and avoid the legacy fixed-77 fallback when a reliable mask or full sequence is available.
- Describe three-slice controls as conditional capabilities rather than guaranteed Qwen semantics.
- Detect tokenizer backends through a small adapter registry.
- Support Qwen3 Klein tokenizers.
- Add a safe generic fallback for token boundary estimation.
- Support Mistral-backed FLUX.2 [dev] only when exact section boundaries can be derived; otherwise encode normally and report that section metadata is unavailable.

Acceptance criteria:

- Standard prompt encoding remains valid even when exact section boundaries cannot be calculated.
- No Klein-specific metadata is required by generic nodes.
- Neutral settings preserve the input object.

## Phase 6 — Generic latent-space guidance and color anchor

Rename:

- `FLUX.2 Color Anchor`
- `FLUX.2 Identity Guidance`

Changes:

- Add model/capability validation where a model input is available.
- Normalize progress from the actual sigma schedule where possible.
- Validate latent channel counts and spatial resizing.
- Keep direct, adaptive, and channel-statistics modes.
- Document that these are latent corrections, not semantic identity extractors.

Acceptance criteria:

- Works with FLUX.2 latents using detected channel dimensions.
- No-op settings do not register unnecessary hooks.
- Callback state resets between generations.

## Phase 7 — Experimental sampler

Rename to `FLUX.2 KSampler Experimental` and generalize only if the direct-forward path can safely preserve ComfyUI conditioning metadata, model wrappers, controls, LoRAs, callbacks, and reference methods.

Preferred approach:

- Reuse ComfyUI scheduler and sampling APIs instead of directly calling the raw diffusion model.
- Detect guidance embedding support.
- Select model-family defaults:
  - distilled Klein: low-step profile
  - Klein base: full-step profile
  - dev: dev scheduler profile
- Preserve the standard sampler stack wherever possible.

If safe generalization cannot be completed, retain the old sampler under a clearly deprecated legacy name and exclude it from the primary compatibility claim.

Acceptance criteria:

- No claim of complete compatibility if ControlNet, wrappers, or conditioning metadata are bypassed.
- Standard ComfyUI samplers remain the documented recommendation.

## Phase 8 — Registration and migration

Update `__init__.py`:

- Register model-neutral node IDs first.
- Register old IDs as compatibility aliases.
- Use model-neutral display names for new nodes.
- Keep legacy display names marked `(Legacy)`.
- Move categories from `conditioning/flux2klein` to `conditioning/flux2` for new nodes.
- Increment package version consistently.

Add `legacy.py` to centralize alias mappings and reduce duplication.

Acceptance criteria:

- Old workflow JSON files load without missing-node errors.
- New workflows contain no Klein-specific IDs unless using explicit legacy nodes.
- Duplicate node IDs are prevented by tests.

## Phase 9 — Documentation and attribution

Rewrite `README.md` with:

- New project name and scope.
- Supported model matrix.
- Verified versus expected compatibility.
- Installation instructions using this fork URL.
- Recommended node order in ComfyUI.
- Architecture and loader requirements.
- Migration table from old node names to new names.
- Identity transfer theory and limitations.
- Preset guidance per family.
- Quantized-loader caveats.
- Troubleshooting and debug instructions.
- Explicit upstream attribution.
- License notes distinguishing extension code from model licenses.

Attribution requirements:

- Credit capitan01R as the original author of `ComfyUI-Flux2Klein-Enhancer`.
- Link the upstream repository.
- Preserve the original MIT notice in `LICENSE`.
- State that this repository is an independent fork and is not affiliated with Black Forest Labs or ComfyUI.

Update `pyproject.toml`:

- New project metadata.
- Fork repository URLs.
- New publisher/display metadata where appropriate.
- Consistent semantic version.

## Phase 10 — Tests and validation

Add pure-Python tests for:

- Architecture extraction and classification.
- Capability detection.
- Schedule parsing and projection.
- Aggregate strength normalization.
- Reference-index parsing.
- Token slicing for one and multiple references.
- Mask-to-token alignment.
- Metadata append/replace behavior.
- Legacy and new node registration.

Runtime validation matrix:

| Family | Distillation | Precision/loader | Minimum tests |
|---|---|---|---|
| Klein 4B | distilled | BF16/FP8 where available | T2I, one ref, multiple refs, mask |
| Klein 4B | base | BF16/FP8 where available | T2I, one ref, schedule |
| Klein 9B | distilled | BF16/FP8 | T2I, one ref, multiple refs, mask |
| Klein 9B KV | distilled | supported loader | KV cache plus identity transfer |
| Klein 9B | base | BF16/FP8 | T2I, one ref, schedule |
| FLUX.2 dev | guidance-distilled | BF16/FP8/GGUF as available | T2I, one ref, multiple refs, mask, LoRA |

Image-quality validation must use fixed prompts, seeds, resolutions, and references. Record:

- baseline without patch
- soft, balanced, and strong presets
- prompt adherence
- identity similarity
- unwanted pose/background copying
- runtime
- peak VRAM
- console diagnostics

## Planned commit sequence

1. `docs: add FLUX.2 multi-variant implementation plan`
2. `feat: add architecture detection and schedule utilities`
3. `feat: implement generic FLUX.2 identity transfer`
4. `feat: generalize reference latent and attention controls`
5. `feat: generalize conditioning and latent guidance nodes`
6. `refactor: register model-neutral nodes with legacy aliases`
7. `test: add architecture schedule and metadata coverage`
8. `docs: rewrite README for multi-variant support and attribution`
9. `chore: update package metadata and compatibility notes`

Each commit must keep the branch importable. Functional changes should not be committed directly to `main`.

## Definition of done

The implementation is complete when:

- New node names are model-neutral.
- Legacy workflows still load.
- FLUX.2 architecture depth is detected dynamically.
- FLUX.2 [dev] is not restricted to Klein's 24-single-block schedules.
- Reference metadata is handled safely across supported variants.
- Tests cover architecture, schedules, slicing, masks, metadata, and registration.
- Documentation clearly distinguishes verified, expected, and unsupported configurations.
- Upstream attribution and license preservation are complete.
- A draft pull request summarizes changes, risks, validation, and remaining empirical tuning work.
