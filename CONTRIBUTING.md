# Contributing

Contributions to **ComfyUI-Flux2Dev-Enhancer** are welcome when they preserve the standalone architecture, public node contract, and documentation quality of the project.

## Before opening a change

1. Search existing issues and pull requests.
2. Use a feature request for substantial node or API proposals.
3. Keep changes focused. Do not combine unrelated refactors, new nodes, and visual workflow changes in one pull request.
4. Do not reintroduce removed compatibility aliases or inherited node IDs.

## Development setup

```bash
git clone https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer.git
cd ComfyUI-Flux2Dev-Enhancer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

## Required checks

```bash
python -m compileall -q __init__.py comfyui_flux2dev_enhancer tests
ruff check comfyui_flux2dev_enhancer tests --select E9,F63,F7,F82
pytest -q
```

Tests validate code structure and metadata. Changes affecting image quality must also include manual ComfyUI validation details: checkpoint, text encoder, VAE, loader, precision, sampler, steps, seed, resolution, references, masks, VRAM, and before/after observations.

## Repository conventions

- Python: 3.10 or newer.
- Formatting: four spaces, LF line endings, UTF-8.
- Branches: short descriptive names; automation branches use `agent/<description>`.
- Commits: Conventional Commit-style prefixes such as `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, and `ci:`.
- Public node IDs must begin with `Flux2` and be registered only in `comfyui_flux2dev_enhancer/registry.py`.
- Runtime code belongs in `comfyui_flux2dev_enhancer/`, not at repository root.
- Neutral settings must remain true no-ops and must not require loader hooks.
- Model mutations must operate on a cloned ComfyUI model patcher.

## Adding or changing a node

A node change must include:

1. Implementation and defensive input validation.
2. Registry entry and display name.
3. Category assignment under `ComfyUI-Flux2Dev-Enhancer/...`.
4. Unit tests for neutral, invalid, and expected behavior.
5. Updates to `docs/NODE_REFERENCE.md`.
6. Updated example workflow and adjacent Markdown Note when the node is demonstrated.
7. Changelog entry when the public contract changes.

Breaking node-ID or socket changes require a major version and explicit release notes. Compatibility aliases are not accepted in the standalone codebase.

## Pull requests

Complete the pull-request template, describe validation honestly, and keep the PR in draft while checks or GPU tests are incomplete. By contributing, you agree that your contribution is licensed under the repository's MIT license.
