from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "example_workflow"

RECOMMENDED = (
    "FLUX2_dev_single_reference_identity.json",
    "FLUX2_klein_single_reference_identity.json",
    "FLUX2_multi_reference_masked_identity.json",
    "FLUX2_reference_attention_controls.json",
)

CUSTOM_NODE_TYPES = {
    "Flux2MultiReferenceLatent",
    "Flux2IdentityFeatureTransfer",
    "Flux2ReferenceAttentionControl",
    "Flux2TextReferenceBalance",
    "Flux2ReferenceWeight",
    "Flux2ReferenceLatentMask",
    "Flux2ConditioningEnhancer",
    "Flux2TextConditioningEnhancer",
    "Flux2SectionedEncoder",
    "Flux2DetailController",
    "Flux2ColorAnchor",
    "Flux2IdentityGuidance",
    "Flux2ArchitectureInspector",
}


def _load(name: str) -> dict:
    return json.loads((WORKFLOW_DIR / name).read_text(encoding="utf-8"))


def _rectangles_overlap(a: dict, b: dict, padding: float = 5.0) -> bool:
    ax, ay = a["pos"]
    aw, ah = a["size"]
    bx, by = b["pos"]
    bw, bh = b["size"]
    return not (
        ax + aw + padding <= bx
        or bx + bw + padding <= ax
        or ay + ah + padding <= by
        or by + bh + padding <= ay
    )


def test_recommended_workflows_have_documented_enhancer_nodes():
    for name in RECOMMENDED:
        workflow = _load(name)
        custom_nodes = [
            node for node in workflow["nodes"] if node["type"] in CUSTOM_NODE_TYPES
        ]
        notes = [
            node for node in workflow["nodes"] if node["type"] == "MarkdownNote"
        ]

        assert custom_nodes
        assert len(notes) >= len(custom_nodes)

        note_text = "\n".join(note["widgets_values"][0] for note in notes)
        for node in custom_nodes:
            display_term = {
                "Flux2MultiReferenceLatent": "Multi Reference Latent",
                "Flux2IdentityFeatureTransfer": "Identity Feature Transfer",
                "Flux2ReferenceAttentionControl": "Reference Attention Control",
                "Flux2TextReferenceBalance": "Text / Reference Balance",
            }.get(node["type"], node["type"])
            assert display_term in note_text

        for note in notes:
            assert note["inputs"] == []
            assert note["outputs"] == []
            assert note["properties"] == {}
            assert note["title"]
            assert note["widgets_values"]
            assert note.get("color") == "#222"
            assert note.get("bgcolor") == "#000"


def test_recommended_workflow_nodes_do_not_overlap():
    for name in RECOMMENDED:
        workflow = _load(name)
        overlaps = [
            (left["id"], right["id"])
            for left, right in combinations(workflow["nodes"], 2)
            if _rectangles_overlap(left, right)
        ]
        assert overlaps == [], f"{name} has overlapping nodes: {overlaps}"


def test_recommended_workflows_use_four_ordered_visual_groups():
    expected_prefixes = (
        "1 — Model and prompt",
        "2 — Reference preparation",
        "3 — ",
        "4 — Sampling and output",
    )
    for name in RECOMMENDED:
        workflow = _load(name)
        titles = [group["title"] for group in workflow["groups"]]
        assert len(titles) == 4
        for title, prefix in zip(titles, expected_prefixes):
            assert title.startswith(prefix)


def test_localized_readmes_have_reciprocal_language_navigation():
    files = {
        "README.md": "# ComfyUI-Flux2Dev-Enhancer",
        "README.pt-BR.md": "# ComfyUI-Flux2Dev-Enhancer",
        "README.es.md": "# ComfyUI-Flux2Dev-Enhancer",
    }
    required_links = (
        "[English](README.md)",
        "[Português (Brasil)](README.pt-BR.md)",
        "[Español](README.es.md)",
    )
    for filename, heading in files.items():
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert heading in text
        for link in required_links:
            assert link in text


def test_localized_readmes_are_not_identical_copies():
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    portuguese = (ROOT / "README.pt-BR.md").read_text(encoding="utf-8")
    spanish = (ROOT / "README.es.md").read_text(encoding="utf-8")

    assert "## Installation" in english
    assert "## Instalação" in portuguese
    assert "## Instalación" in spanish
    assert len({english, portuguese, spanish}) == 3
