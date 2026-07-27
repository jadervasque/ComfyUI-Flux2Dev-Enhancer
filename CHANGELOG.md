# Changelog

All notable changes to this fork are documented here.

## 4.0.0 — FLUX.2 multi-variant beta

### Added

- Runtime architecture detection for FLUX.2 [dev], Klein 4B, and Klein 9B
  fingerprints, plus a conservative unknown-compatible profile.
- Loader capability detection for attention input/output patches, post-input
  patches, reference metadata, and sampler post-CFG callbacks.
- `FLUX.2 Architecture Inspector` diagnostic node.
- Dynamic block-schedule parsing and validation.
- Relative projection of Klein 9B legacy schedules to other FLUX.2 depths.
- Approximate aggregate-strength normalization across active blocks.
- Architecture-aware `FLUX.2 Identity Feature Transfer` with automatic presets,
  multi-reference masks, denoising windows, sigma scaling, and chunked matching.
- Generic multi-reference latent node with append/replace behavior and selectable
  reference placement methods.
- Generic reference attention, weight, text/reference balance, and latent-mask
  nodes.
- Generic conditioning and text-conditioning enhancers.
- Qwen/Mistral template-aware section encoding with safe metadata fallback.
- Generic color anchoring and latent identity guidance with explicit sigma input.
- Four model-neutral example workflows for FLUX.2 dev, Klein, masked
  multireference identity transfer, and ordered reference-attention controls.
- Native ComfyUI `MarkdownNote` documentation beside every Enhancer node used in
  the recommended workflows.
- English, Brazilian Portuguese, and Spanish READMEs with reciprocal language
  navigation.
- Unit tests for architecture detection, schedules, metadata, tokenizer sections,
  guidance progress, registration, neutral execution, workflow links, visual
  layout, Markdown documentation, and localized README navigation.
- `NOTICE.md`, migration documentation, compatibility matrix, troubleshooting,
  and implementation plan.

### Changed

- New nodes use model-neutral `FLUX.2` names and the `conditioning/flux2`
  category.
- Package metadata is now `ComfyUI-Flux2-Enhancer`, version 4.0.0.
- Existing node identifiers remain registered with `(Legacy)` display names.
- `IdentityFeatureTransferFinal` now routes through the architecture-aware
  compatibility implementation.
- `Flux2KleinMultiReferenceLatent` and original reference-control identifiers
  route through generic implementations while preserving their input surfaces.
- Neutral settings avoid installing hooks or requiring loader capabilities.
- Sampler callback lists are copied before adding latent-guidance callbacks.
- Recommended workflows now use a consistent four-zone left-to-right layout with
  non-overlapping nodes and labeled groups.

### Deprecated

- Basic, Advanced, and V3 Identity Feature Transfer nodes remain available for
  workflow compatibility but retain Klein-oriented schedules.
- `Flux2Klein KSampler Experimental` remains available but is excluded from the
  multi-variant compatibility claim. Standard ComfyUI samplers are recommended.

### Known limitations

- Automatic presets are conservative starting points and require visual tuning per
  model, resolution, sampler, quantization, and LoRA combination.
- GGUF, FP8, and KV-cache behavior depends on the third-party loader preserving
  ComfyUI patch interfaces and reference metadata.
- Automated tests validate code paths, graph structure, documentation, and
  metadata handling; they do not replace image-quality testing with real FLUX.2
  checkpoints on supported GPUs.
