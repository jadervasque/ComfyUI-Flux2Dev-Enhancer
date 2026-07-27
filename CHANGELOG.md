# Changelog

All notable changes to **ComfyUI-Flux2Dev-Enhancer** are documented in this file.

The project follows Semantic Versioning and the principles of Keep a Changelog.

## [Unreleased]

### Validation pending

- Broader fixed-seed GPU validation across FLUX.2 dev, Klein 4B, Klein 9B, and representative quantized loaders.

## [1.0.0-beta.1] — 2026-07-26

First standalone beta release.

### Added

- Dedicated runtime package: `comfyui_flux2dev_enhancer`.
- Thin ComfyUI entry point at repository root.
- Canonical registry with thirteen supported `Flux2...` node IDs.
- Project-specific node categories under `ComfyUI-Flux2Dev-Enhancer/...`.
- Runtime architecture detection for FLUX.2 dev, Klein 4B, and Klein 9B.
- Architecture Inspector for loader and hook diagnostics.
- Architecture-aware Identity Feature Transfer with automatic presets, custom schedules, masks, denoising windows, sigma scaling, and chunked similarity matching.
- Multi-reference latent conditioning and reference attention controls.
- Conditioning enhancement, sectioned encoding, and detail control.
- Post-CFG Color Anchor and Identity Guidance.
- Four maintained and visually organized example workflows with embedded Markdown Notes.
- English, Brazilian Portuguese, and Spanish README files.
- Architecture, development, node-reference, and release documentation.
- Contribution, security, support, conduct, authorship, citation, and attribution files.
- Structured bug and feature issue forms, pull-request template, CODEOWNERS, Dependabot, and GitHub Actions CI.
- Repository contract tests for exact node registry, metadata consistency, project naming, removed files, and maintained workflows.

### Changed

- Canonical project and package identity is now `ComfyUI-Flux2Dev-Enhancer`.
- Versioning restarts at `1.0.0b1` for the independent standalone release line.
- Runtime implementation moved from repository-root modules into a conventional package.
- Package metadata lists Jader Vasque as current author and maintainer; upstream attribution is maintained separately.
- Conditioning section metadata uses the standalone `flux2_sections` contract only.
- Identity transfer strength mode `per_block` replaces historical terminology.
- Debug prefixes and categories use the canonical project identity.

### Removed

- All historical node registrations and aliases.
- Inherited Identity Feature Transfer basic, Advanced, V3, and Final compatibility nodes.
- Former `Flux2Klein...` node aliases and wrappers.
- Historical identity presets and schedule projection behavior.
- Direct experimental sampler.
- Inherited root implementation modules.
- Historical example workflows and backward-compatibility claims.
- Silent fallback to inherited section metadata.

### Known limitations

- Automated tests validate runtime structure and mathematical code paths but do not prove image quality.
- FP8, GGUF, and KV-cache behavior depends on the loader preserving ComfyUI hook interfaces and reference metadata.
- Identity features remain entangled with pose, lighting, clothing, hair, and background.

[Unreleased]: https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer/compare/v1.0.0-beta.1...HEAD
[1.0.0-beta.1]: https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer/releases/tag/v1.0.0-beta.1
