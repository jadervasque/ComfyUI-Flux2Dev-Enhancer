# PLAN1 — Standalone repository professionalization

## Status

- **Target branch:** `agent/standalone-professionalization`
- **Base:** `agent/flux2-multivariant-compatibility` at `d9a5b0dd2f6a668d5f4c9d994b16a59050ee0b70`
- **Project identity:** `ComfyUI-Flux2Dev-Enhancer`
- **Compatibility policy:** breaking cleanup; legacy node identifiers, adapters, algorithms, examples, and compatibility promises will be removed.
- **Attribution policy:** preserve the upstream MIT copyright notice and credit `capitan01R/ComfyUI-Flux2Klein-Enhancer` in `NOTICE.md`, `AUTHORS.md`, documentation, and project metadata without presenting this repository as a fork or drop-in continuation.

## 1. Objectives

1. Convert the repository from an evolved fork layout into a standalone, maintainable ComfyUI extension.
2. Establish one canonical project name everywhere: `ComfyUI-Flux2Dev-Enhancer`.
3. Remove all runtime and documentation compatibility layers for the former Klein-specific project.
4. Keep support for the open-weight FLUX.2 architecture family through the new architecture-aware nodes.
5. Introduce a conventional Python package boundary while retaining ComfyUI's root `__init__.py` entry point.
6. Add the minimum governance, contribution, security, support, issue, pull-request, and CI documentation expected from a professional public GitHub repository.
7. Make repository structure and automated checks explicit enough that future changes can be reviewed and released consistently.

## 2. Final repository structure

```text
.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   ├── feature_request.yml
│   │   └── config.yml
│   ├── workflows/
│   │   └── ci.yml
│   ├── CODEOWNERS
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
├── comfyui_flux2dev_enhancer/
│   ├── __init__.py
│   ├── architecture.py
│   ├── conditioning.py
│   ├── constants.py
│   ├── diagnostics.py
│   ├── guidance.py
│   ├── identity_transfer.py
│   ├── reference_controls.py
│   ├── reference_latent.py
│   ├── registry.py
│   ├── scheduling.py
│   └── version.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   ├── NODE_REFERENCE.md
│   └── RELEASES.md
├── example_workflow/
│   ├── README.md
│   └── four maintained FLUX.2 example workflows
├── plans/
│   ├── PLAN0.md
│   └── PLAN1.md
├── tests/
├── .editorconfig
├── .gitattributes
├── .gitignore
├── AUTHORS.md
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── NOTICE.md
├── README.md
├── README.pt-BR.md
├── README.es.md
├── SECURITY.md
├── SUPPORT.md
├── __init__.py
└── pyproject.toml
```

The root `__init__.py` remains a small ComfyUI loader that re-exports the canonical mappings from the package. Runtime implementation will no longer live at repository root.

## 3. Legacy removal

### 3.1 Delete inherited implementation modules

Remove the former project modules and the experimental direct sampler:

- `identity_feature_transfer.py`
- `flux2_klein_enhancer.py`
- `flux2_klein_text_enhancer.py`
- `flux2_sectioned_encoder.py`
- `Flux2klein_Ksampler_exp.py`

### 3.2 Delete compatibility adapters and identifiers

Remove registrations and classes for identifiers such as:

- `IdentityFeatureTransfer`
- `IdentityFeatureTransferAdvanced`
- `IdentityFeatureTransferV3`
- `IdentityFeatureTransferFinal`
- `Flux2KleinEnhancer`
- `Flux2KleinDetailController`
- `Flux2KleinTextEnhancer`
- `Flux2KleinSectionedEncoder`
- `Flux2KleinMultiReferenceLatent`
- all former `Flux2Klein...` reference/guidance aliases
- `Flux2KleinKSamplerExperimental`

The final registry will expose only the standalone architecture-aware `Flux2...` identifiers.

### 3.3 Remove legacy presets and terminology

- Remove `KLEIN_LEGACY_HARD`, `KLEIN_LEGACY_MID`, and `KLEIN_LEGACY_SOFT`.
- Rename `legacy_per_block` to `per_block`.
- Remove relative projection of historical Klein schedules when it is no longer used.
- Remove documentation that promises old workflow compatibility.
- Delete inherited example workflows and retain only maintained standalone examples.

Supporting Klein 4B/9B as FLUX.2 architectures remains in scope; preserving old node contracts does not.

## 4. Package and registration design

1. Move implementation modules into `comfyui_flux2dev_enhancer/`.
2. Consolidate the neutral identity guards into the canonical identity-transfer node instead of maintaining an adapter module.
3. Add `constants.py` with:
   - canonical project name;
   - category root;
   - repository URL;
   - documentation URL.
4. Add `version.py` as the single version source.
5. Add `registry.py` as the only place that merges node class/display mappings.
6. Keep node IDs stable for the new standalone API, but use project-specific categories such as `ComfyUI-Flux2Dev-Enhancer/...`.
7. Keep user-facing node display names concise (`FLUX.2 ...`) while the extension identity remains unambiguous in categories and metadata.

## 5. Project identity and attribution

1. Set the Python/Comfy package name and display name to `ComfyUI-Flux2Dev-Enhancer`.
2. Replace former project names in runtime docstrings, debug prefixes, categories, metadata, examples, and tests.
3. Retain upstream names only in historical attribution contexts.
4. Treat Jader Vasque as the standalone project maintainer/author in package metadata.
5. Move the original author to explicit attribution rather than co-ownership of the new package metadata.
6. Preserve the MIT license notice required by the upstream license.

## 6. Professional repository documentation

### Required community files

- `CONTRIBUTING.md`: development setup, branch/commit conventions, tests, node/API changes, documentation expectations.
- `CODE_OF_CONDUCT.md`: Contributor Covenant-compatible conduct and enforcement contact.
- `SECURITY.md`: supported versions, private reporting process, disclosure expectations, scope limitations.
- `SUPPORT.md`: supported questions, required diagnostic data, unsupported scenarios, issue routing.
- `AUTHORS.md`: standalone maintainer and upstream attribution.
- `CITATION.cff`: citation metadata for the repository and upstream acknowledgement.

### Technical documentation

- `docs/ARCHITECTURE.md`: ComfyUI entry point, package modules, data paths, hooks, metadata contracts, compatibility boundaries.
- `docs/DEVELOPMENT.md`: environment, test strategy, formatting, adding nodes, workflow validation.
- `docs/NODE_REFERENCE.md`: canonical node IDs, inputs, outputs, neutral settings, hook requirements.
- `docs/RELEASES.md`: semantic versioning, changelog process, beta/stable criteria, release checklist.

### GitHub templates

- Structured bug and feature issue forms.
- Pull-request checklist.
- `CODEOWNERS` for repository-wide review ownership.
- Dependabot configuration for GitHub Actions.

## 7. Quality automation

1. Add a GitHub Actions CI workflow for supported Python versions.
2. Run:
   - Python compilation;
   - `pytest`;
   - package/import registration checks;
   - workflow JSON/layout tests;
   - documentation/project-name/legacy-string audit.
3. Configure pytest and optional static tooling in `pyproject.toml` without adding runtime dependencies to ComfyUI.
4. Add `.editorconfig`, `.gitattributes`, and `.gitignore` for deterministic repository behavior.

## 8. Test migration and additions

1. Update imports to the new package paths.
2. Remove assertions that require legacy registrations.
3. Add a registry allowlist test to ensure only canonical node IDs are exposed.
4. Add a repository audit test that rejects former project/runtime names outside approved attribution files.
5. Add metadata consistency tests for project name, version, package display name, and URLs.
6. Keep architecture, scheduling, guidance, workflow-link, visual-layout, Markdown Note, and multilingual README tests.
7. Validate that all maintained example workflows use canonical IDs only.

## 9. Documentation migration

1. Rewrite all three root READMEs as standalone-project documentation.
2. Remove fork/migration/backward-compatibility framing.
3. Keep an attribution section that clearly states which concepts originated upstream.
4. Update example documentation to remove historical workflow references.
5. Update `CHANGELOG.md` to describe the standalone breaking release.
6. Update `NOTICE.md` to distinguish current ownership from upstream attribution.
7. Keep `PLAN0.md` as historical engineering context and mark it superseded by `PLAN1.md` for repository structure and compatibility policy.

## 10. Commit strategy

Changes will be committed in reviewable stages:

1. `docs(plan): define standalone repository professionalization`
2. `refactor(package): establish standalone package layout`
3. `refactor(legacy): remove inherited nodes and compatibility adapters`
4. `refactor(identity): remove legacy presets and terminology`
5. `docs(community): add governance and contribution files`
6. `ci: add repository quality workflow and templates`
7. `docs: rewrite standalone project documentation`
8. `test: enforce canonical registry and repository structure`
9. `chore: finalize metadata and release documentation`

## 11. Definition of done

- [ ] `plans/PLAN1.md` exists before functional changes.
- [ ] Runtime implementation is contained in `comfyui_flux2dev_enhancer/`.
- [ ] Root `__init__.py` is a thin ComfyUI entry point.
- [ ] No legacy node IDs are registered.
- [ ] No inherited implementation modules remain.
- [ ] No legacy presets or compatibility terminology remain in the public API.
- [ ] Only maintained standalone workflows remain in `example_workflow/`.
- [ ] Canonical project name is consistent across code and metadata.
- [ ] Upstream credit and MIT notice are preserved in approved attribution locations.
- [ ] Community health files and GitHub templates exist.
- [ ] CI exists and runs compilation/tests/audits.
- [ ] English, Portuguese, and Spanish READMEs describe the standalone project.
- [ ] Tests validate registry, metadata, workflows, documentation, and absence of legacy runtime names.
- [ ] The branch remains isolated from `main` until reviewed.

## 12. Known risks

- Removing old IDs is intentionally breaking for workflows created before this standalone release.
- Moving modules can break relative imports if the root entry point and tests are not updated together.
- Quantized loaders remain compatible only when they preserve ComfyUI hook and reference metadata contracts.
- Automated tests cannot establish image quality; GPU validation remains required before declaring a stable release.
