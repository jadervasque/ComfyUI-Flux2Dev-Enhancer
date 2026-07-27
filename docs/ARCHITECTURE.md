# Architecture

## Repository entry point

ComfyUI loads the repository root as a Python package. The root `__init__.py` is intentionally small and re-exports:

```python
NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS
__version__
```

All runtime implementation lives in `comfyui_flux2dev_enhancer/`.

## Package modules

| Module | Responsibility |
|---|---|
| `constants.py` | Canonical project identity and repository URLs. |
| `version.py` | Single runtime version source. |
| `registry.py` | Canonical public node IDs, display names, and project categories. |
| `architecture.py` | FLUX.2 architecture fingerprinting and loader-capability validation. |
| `scheduling.py` | Block schedule parsing, reference selection, automatic presets, and strength normalization. |
| `conditioning.py` | Text-conditioning enhancement, section encoding, and section/token scaling. |
| `reference_latent.py` | Injection of one to eight VAE-encoded references into conditioning metadata. |
| `reference_controls.py` | Attention K/V weighting, text/reference balance, and latent masking. |
| `identity_transfer.py` | Attention-output feature matching and transfer. |
| `guidance.py` | Post-CFG latent color anchoring and identity guidance. |
| `diagnostics.py` | Architecture and loader-capability report node. |

## Public registry policy

`registry.py` is the only authoritative public registry. Node implementations may define local mappings for development convenience, but ComfyUI receives only the allowlisted canonical IDs from the package registry. Compatibility aliases and historical node IDs are intentionally excluded.

## Model path

```text
Diffusion model loader
  -> optional LoRA/model patches
  -> reference attention controls
  -> identity feature transfer
  -> guider / sampler
```

Every model-modifying node clones the incoming ComfyUI model patcher before installing hooks. Neutral settings return a clone without requiring hook capabilities.

## Conditioning path

```text
FLUX.2 text encoder
  -> text conditioning tools
  -> multi-reference latent metadata
  -> optional reference controls
  -> positive conditioning input
```

Reference images must be encoded with the FLUX.2 VAE. Batch entries are split into individual references in stable order.

## Runtime metadata contract

Reference and identity nodes rely on metadata exposed by the ComfyUI FLUX.2 implementation:

- `reference_latents`
- `reference_latents_method`
- `reference_image_num_tokens`
- `img_slice`
- `block_type`
- `block_index`
- current `sigmas`, when supplied by the sampler path

A quantized or third-party loader is compatible only when it preserves the required model patch methods and metadata.

## Hook contracts

Depending on the node, the loader/model patcher must preserve:

- `clone()`
- `set_model_attn1_patch()`
- `set_model_attn1_output_patch()`
- `set_model_post_input_patch()` when required by future extensions
- `model_options["sampler_post_cfg_function"]`
- access to the diffusion model parameters and block lists

`Flux2ArchitectureInspector` should be used to inspect an unfamiliar loader before debugging image quality.

## Architecture detection

Detection uses structural parameters rather than checkpoint filenames. Known profiles include:

| Profile | Hidden size | Heads | Double blocks | Single blocks | Conditioning width |
|---|---:|---:|---:|---:|---:|
| FLUX.2 dev | 6144 | 48 | 8 | 48 | 15360 |
| FLUX.2 Klein 9B | 4096 | 32 | 8 | 24 | 12288 |
| FLUX.2 Klein 4B | 3072 | 24 | 5 | 20 | 7680 |

Unknown profiles may be reported as structurally compatible, but automatic presets remain conservative.

## Identity feature transfer

The identity node patches attention output. During active blocks and denoising steps it:

1. separates text, generated-image, and reference-image tokens;
2. applies reference selection and optional masks;
3. centers and normalizes feature vectors;
4. computes chunked cosine similarity;
5. rejects weak matches;
6. pools reference features with temperature-controlled softmax;
7. gates the transfer by confidence and configured strength;
8. returns modified generated-image tokens to the remaining transformer blocks.

The algorithm transfers internal features, not pixels. Identity, pose, clothing, lighting, and background remain partially entangled.

## Release and compatibility boundary

This repository is a standalone project. Only node IDs documented in `docs/NODE_REFERENCE.md` are supported. Removed historical IDs are not loaded, migrated, or aliased. Public socket or node-ID changes require a major release.
