# Release process

## Versioning

ComfyUI-Flux2Dev-Enhancer follows Semantic Versioning:

- **Major:** public node IDs, socket order/types, metadata contracts, or documented behavior change incompatibly.
- **Minor:** backward-compatible nodes, inputs, presets, or capabilities are added.
- **Patch:** compatible bug fixes, documentation corrections, and internal improvements.

Pre-release identifiers such as `b1` indicate that runtime and image-quality validation is still in progress.

## Sources of truth

The version must match in:

- `comfyui_flux2dev_enhancer/version.py`
- `pyproject.toml`
- `CITATION.cff`
- `CHANGELOG.md`

`tests/test_repository_contract.py` enforces runtime and package metadata consistency.

## Release checklist

1. Confirm the intended public node registry in `registry.py`.
2. Update all affected node documentation and maintained workflows.
3. Update English, Portuguese, and Spanish README files when user-facing behavior changes.
4. Add a dated changelog section.
5. Run local checks:

   ```bash
   python -m compileall -q __init__.py comfyui_flux2dev_enhancer tests
   ruff check comfyui_flux2dev_enhancer tests --select E9,F63,F7,F82
   pytest -q
   ```

6. Validate maintained workflows in a current ComfyUI installation.
7. Test representative FLUX.2 profiles and record checkpoints, loaders, precision, samplers, steps, seeds, references, masks, VRAM, and visual observations.
8. Confirm model and third-party loader license statements remain accurate.
9. Merge through a reviewed pull request with passing CI.
10. Create an annotated Git tag matching the version.
11. Publish GitHub release notes derived from `CHANGELOG.md`.

## Beta-to-stable criteria

A stable 1.0 release requires:

- passing repository CI;
- successful import in a current ComfyUI release;
- fixed-seed tests on FLUX.2 dev, Klein 4B, and Klein 9B;
- at least one native or standard loader validation per architecture;
- reference, mask, conditioning, identity-transfer, and guidance coverage;
- no known data-loss, execution, or severe artifact regressions;
- complete node reference and maintained example workflows.

FP8, GGUF, and KV-cache loaders remain loader-specific. Stable project status does not imply that every third-party wrapper is supported.

## Hotfixes

Security and severe runtime regressions may use a focused patch release. The change must still include a test that reproduces the defect and a changelog entry.

## Deprecation policy

The standalone project does not maintain aliases for removed historical nodes. Future deprecations should be announced for at least one minor release when practical, but aliases that materially complicate runtime architecture may be omitted in a major release.
