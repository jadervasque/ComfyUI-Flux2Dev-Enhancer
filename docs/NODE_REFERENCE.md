# Node reference

This document defines the supported public node IDs for **ComfyUI-Flux2Dev-Enhancer 1.0**. Node IDs not listed here are not part of the standalone API.

## Diagnostics

### `Flux2ArchitectureInspector`

**Display name:** FLUX.2 Architecture Inspector  
**Input:** `MODEL`  
**Outputs:** model plus architecture report values  
**Purpose:** identifies structural FLUX.2 parameters and loader hook capabilities. Place immediately after the model loader when validating a checkpoint or quantized loader.

## Conditioning

### `Flux2ConditioningEnhancer`

Scales active conditioning, adjusts token contrast/whitening, equalizes norms, and optionally scales the three stacked encoder-layer slices. Neutral values leave conditioning unchanged.

### `Flux2TextConditioningEnhancer`

Provides simpler magnitude, contrast, and token-norm controls. Use when per-layer controls are unnecessary.

### `Flux2SectionedEncoder`

Encodes FRONT/MID/END prompt sections and records tokenizer-derived ranges when the official Qwen or Mistral wrapper exposes its tokenizer. Exact range metadata is omitted rather than fabricated when unavailable.

### `Flux2DetailController`

Scales section ranges or explicit token ranges in existing conditioning. It can use `flux2_sections` metadata or a documented fallback strategy.

## References

### `Flux2MultiReferenceLatent`

Adds one to eight FLUX.2 VAE-encoded latents to conditioning metadata.

Key controls:

- `mode`: `replace` or `append`;
- `reference_method`: `model_default`, `index`, `offset`, `uxo/uno`, or `index_timestep_zero`;
- stable ordering: `latent_1`, `latent_2`, …, `latent_8`.

Use `model_default` first unless a loader-specific method is required.

### `Flux2ReferenceAttentionControl`

Scales the selected reference's attention key/value tokens. `strength=1.0` and `spatial_fade=none` are neutral. Optional spatial fades derive a token grid from the selected reference latent.

### `Flux2ReferenceWeight`

A model-only reference K/V multiplier. `weight=1.0` is neutral.

### `Flux2TextReferenceBalance`

Trades off text and reference attention around `balance=0.5`:

- below `0.5`: attenuate text;
- `0.5`: neutral;
- above `0.5`: attenuate references.

### `Flux2ReferenceLatentMask`

Attenuates masked regions directly in one stored reference latent. Supports strength, inversion, feathering, and zero-based reference selection.

## Identity

### `Flux2IdentityFeatureTransfer`

Installs an attention-output patch that matches generated image tokens against selected reference tokens and transfers confidence-gated features.

Presets:

- `AUTO_SOFT`
- `AUTO_BALANCED`
- `AUTO_STRONG`
- `CUSTOM`

Strength modes:

- `normalized_total`: distributes aggregate strength across active blocks;
- `per_block`: uses schedule values directly.

Important inputs:

- `reference_indices`: `all`, a single index, lists, or ranges;
- `similarity_floor`: rejects weak matches;
- `softmax_temperature`: controls reference-token pooling sharpness;
- `double_blocks` / `single_blocks`: custom schedule strings;
- `start_percent` / `end_percent`: denoising window;
- `sigma_scaling`: optional equal-energy adjustment;
- `subject_mask_1` ↔ `latent_1`, through `subject_mask_8` ↔ `latent_8`;
- `query_chunk_size`: limits similarity-matrix VRAM.

The node transfers internal features. Strong or early transfer can also influence pose, framing, lighting, clothing, or background.

## Guidance

### `Flux2ColorAnchor`

Applies a post-CFG correction to per-channel spatial means using a selected reference. It does not copy spatial detail. `strength=0` is neutral.

### `Flux2IdentityGuidance`

Applies post-CFG latent correction from an identity latent.

Modes:

- `adaptive`: similarity-weighted spatial pull;
- `direct`: direct latent interpolation and highest layout-copy risk;
- `channel_match`: matches channel statistics without direct spatial copying.

Connect the actual `SIGMAS` schedule when precise denoising windows matter.

## Common compatibility requirements

Model-patching nodes may require one or more of:

- attention-input patch support;
- attention-output patch support;
- sampler post-CFG callback support;
- `reference_image_num_tokens`;
- `img_slice`, `block_type`, and `block_index`;
- reference latent forwarding.

Use Architecture Inspector and `docs/ARCHITECTURE.md` for loader diagnostics.
