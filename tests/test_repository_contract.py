from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from comfyui_flux2dev_enhancer.constants import PROJECT_NAME, REPOSITORY_URL
from comfyui_flux2dev_enhancer.registry import NODE_CLASS_MAPPINGS
from comfyui_flux2dev_enhancer.version import __version__


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "comfyui_flux2dev_enhancer"

FORBIDDEN_RUNTIME_TERMS = (
    "Flux2Klein",
    "IdentityFeatureTransferFinal",
    "IdentityFeatureTransferAdvanced",
    "IdentityFeatureTransferV3",
    "KLEIN_LEGACY",
    "legacy_per_block",
    "klein_sections",
    "ComfyUI-Flux2Klein-Enhancer",
)

REMOVED_MODULES = (
    "identity_feature_transfer.py",
    "flux2_klein_enhancer.py",
    "flux2_klein_text_enhancer.py",
    "flux2_sectioned_encoder.py",
    "Flux2klein_Ksampler_exp.py",
    "flux2_identity_nodes.py",
)

EXPECTED_EXAMPLES = {
    "FLUX2_dev_single_reference_identity.json",
    "FLUX2_klein_single_reference_identity.json",
    "FLUX2_multi_reference_masked_identity.json",
    "FLUX2_reference_attention_controls.json",
}


def test_runtime_package_contains_no_compatibility_code():
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_RUNTIME_TERMS:
            assert term not in text, f"{term!r} found in {path.relative_to(ROOT)}"


def test_inherited_modules_are_removed():
    for filename in REMOVED_MODULES:
        assert not (ROOT / filename).exists(), filename


def test_only_maintained_example_workflows_remain():
    actual = {path.name for path in (ROOT / "example_workflow").glob("*.json")}
    assert actual == EXPECTED_EXAMPLES


def test_project_metadata_matches_runtime_identity():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    comfy = metadata["tool"]["comfy"]
    assert PROJECT_NAME == "ComfyUI-Flux2Dev-Enhancer"
    assert project["name"] == PROJECT_NAME
    assert project["version"] == __version__
    assert project["urls"]["Repository"] == REPOSITORY_URL
    assert comfy["DisplayName"] == PROJECT_NAME


def test_node_ids_are_project_owned_and_unique():
    assert len(NODE_CLASS_MAPPINGS) == len(set(NODE_CLASS_MAPPINGS))
    assert all(node_id.startswith("Flux2") for node_id in NODE_CLASS_MAPPINGS)
