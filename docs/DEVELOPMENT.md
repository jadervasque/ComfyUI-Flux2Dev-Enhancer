# Development guide

## Requirements

- Python 3.10–3.12
- Git
- A development environment with PyTorch
- ComfyUI for runtime/GPU validation

## Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

## Local checks

```bash
python -m compileall -q __init__.py comfyui_flux2dev_enhancer tests
ruff check comfyui_flux2dev_enhancer tests --select E9,F63,F7,F82
pytest -q
```

## Package rules

- Runtime modules belong in `comfyui_flux2dev_enhancer/`.
- The root `__init__.py` must remain a thin ComfyUI entry point.
- Public node registration belongs only in `registry.py`.
- Project metadata belongs in `constants.py` and `version.py`.
- Do not add compatibility aliases for removed IDs.
- Do not add runtime dependencies unless ComfyUI does not already provide the capability and the dependency is justified.

## Adding a node

1. Choose the module by responsibility.
2. Implement `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, and `CATEGORY`.
3. Ensure neutral settings return without installing hooks.
4. Clone model patchers before mutation.
5. Raise actionable errors when a required loader capability is unavailable.
6. Add the class and display name to `registry.py`.
7. Add unit tests and update `docs/NODE_REFERENCE.md`.
8. Add or update a maintained example workflow and its adjacent Markdown Note when appropriate.

## Testing strategy

### Pure-Python tests

Use small fake model patchers and tensors to test:

- architecture fingerprints;
- schedules and strength normalization;
- metadata copy semantics;
- reference token slicing;
- neutral no-op behavior;
- registry allowlist;
- repository identity and removed-module audits;
- workflow graph integrity and visual layout.

### GPU validation

Record:

- ComfyUI commit;
- extension commit;
- model, text encoder, VAE, loader, and precision;
- sampler, scheduler, steps, guidance, resolution, and seed;
- reference count, masks, and node settings;
- VRAM peak and elapsed time;
- expected and observed visual behavior.

Automated tests do not prove likeness, prompt fidelity, or artifact quality.

## Workflow files

Maintained workflows are stored in `example_workflow/`. They must:

- use canonical node IDs only;
- contain valid links and socket types;
- use four ordered visual groups;
- avoid node overlap;
- include an English `MarkdownNote` for every project node demonstrated;
- use editable placeholder asset names rather than private local paths.

## Documentation

Public behavior changes require updates to the relevant README translations, node reference, architecture or development documentation, and changelog. English technical documentation is authoritative when translations temporarily differ.

## Commit and PR style

Use focused Conventional Commit-style messages. Keep PRs draft while validation is incomplete. Include exact test commands and distinguish code-level validation from GPU/image-quality validation.
