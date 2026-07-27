## Summary

<!-- Explain what changed and why. -->

## Public impact

<!-- List affected node IDs, sockets, metadata, workflows, performance, VRAM, or documentation. State "None" where applicable. -->

## Validation

- [ ] `python -m compileall -q __init__.py comfyui_flux2dev_enhancer tests`
- [ ] `ruff check comfyui_flux2dev_enhancer tests --select E9,F63,F7,F82`
- [ ] `pytest -q`
- [ ] Maintained workflow JSON files still load and are visually organized.
- [ ] GPU/runtime validation completed, or clearly documented as not applicable/pending.

### Runtime matrix

<!-- For image/runtime changes: ComfyUI commit, model, encoder, VAE, loader, precision, sampler, steps, seed, resolution, references, masks, VRAM, results. -->

## Documentation

- [ ] `docs/NODE_REFERENCE.md` updated for node behavior changes.
- [ ] README translations updated for user-facing changes.
- [ ] Example workflow and adjacent Markdown Note updated when applicable.
- [ ] `CHANGELOG.md` updated.

## Safety and compatibility

- [ ] No removed historical node IDs or compatibility aliases were reintroduced.
- [ ] Neutral values remain true no-ops.
- [ ] Model patchers are cloned before mutation.
- [ ] New loader requirements fail with actionable errors.
- [ ] No model files, private images, credentials, or local absolute paths are included.

## Checklist

- [ ] The change is focused and reviewable.
- [ ] Tests reproduce fixed defects or validate new behavior.
- [ ] Attribution and license notices remain intact.
