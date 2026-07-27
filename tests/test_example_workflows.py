from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "example_workflow"
RECOMMENDED = {
    "FLUX2_dev_single_reference_identity.json",
    "FLUX2_klein_single_reference_identity.json",
    "FLUX2_multi_reference_masked_identity.json",
    "FLUX2_reference_attention_controls.json",
}
LEGACY_NODE_IDS = {
    "IdentityFeatureTransferFinal",
    "IdentityFeatureTransferV3",
    "Flux2KleinMultiReferenceLatent",
    "Flux2KleinRefLatentController",
    "Flux2KleinTextRefBalance",
    "Flux2KleinColorAnchor",
    "Flux2KleinKSamplerExperimental",
}


def load_workflow(name: str) -> dict:
    return json.loads((WORKFLOW_DIR / name).read_text(encoding="utf-8"))


def node_types(workflow: dict) -> set[str]:
    return {node["type"] for node in workflow["nodes"]}


def validate_links(workflow: dict) -> None:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links, default=0)

    for link_id, from_id, from_slot, to_id, to_slot, link_type in workflow["links"]:
        assert from_id in nodes
        assert to_id in nodes
        source = nodes[from_id]
        target = nodes[to_id]
        assert 0 <= from_slot < len(source["outputs"])
        assert 0 <= to_slot < len(target["inputs"])
        assert source["outputs"][from_slot]["type"] == link_type
        assert target["inputs"][to_slot]["type"] == link_type
        assert link_id in (source["outputs"][from_slot].get("links") or [])
        assert target["inputs"][to_slot].get("link") == link_id

    for node in nodes.values():
        for socket in node.get("inputs", []):
            link_id = socket.get("link")
            if link_id is not None:
                assert link_id in links


def test_recommended_workflows_exist_and_parse():
    assert RECOMMENDED.issubset({path.name for path in WORKFLOW_DIR.glob("*.json")})
    for name in RECOMMENDED:
        workflow = load_workflow(name)
        assert workflow["version"] == 0.4
        assert workflow["nodes"]
        validate_links(workflow)


def test_recommended_workflows_use_model_neutral_nodes():
    for name in RECOMMENDED:
        types = node_types(load_workflow(name))
        assert "Flux2MultiReferenceLatent" in types
        assert "Flux2IdentityFeatureTransfer" in types
        assert types.isdisjoint(LEGACY_NODE_IDS)


def test_dev_example_uses_dev_model_and_mistral_encoder():
    workflow = load_workflow("FLUX2_dev_single_reference_identity.json")
    loader_values = [node.get("widgets_values", []) for node in workflow["nodes"]]
    flattened = {value for values in loader_values for value in values if isinstance(value, str)}
    assert "flux2-dev.safetensors" in flattened
    assert "mistral_3_small_flux2_bf16.safetensors" in flattened


def test_klein_example_uses_klein_model_and_qwen_encoder():
    workflow = load_workflow("FLUX2_klein_single_reference_identity.json")
    loader_values = [node.get("widgets_values", []) for node in workflow["nodes"]]
    flattened = {value for values in loader_values for value in values if isinstance(value, str)}
    assert "flux-2-klein-9b.safetensors" in flattened
    assert "qwen_3_8b_fp8mixed.safetensors" in flattened


def test_multi_reference_example_connects_two_latents_and_masks():
    workflow = load_workflow("FLUX2_multi_reference_masked_identity.json")
    multi = next(node for node in workflow["nodes"] if node["type"] == "Flux2MultiReferenceLatent")
    identity = next(node for node in workflow["nodes"] if node["type"] == "Flux2IdentityFeatureTransfer")
    assert multi["inputs"][1]["link"] is not None
    assert multi["inputs"][2]["link"] is not None
    assert identity["inputs"][2]["link"] is not None
    assert identity["inputs"][3]["link"] is not None
    assert "zero_unmasked_tokens" in identity["widgets_values"]


def test_reference_controls_example_contains_ordered_control_chain():
    workflow = load_workflow("FLUX2_reference_attention_controls.json")
    types = node_types(workflow)
    assert "Flux2ReferenceAttentionControl" in types
    assert "Flux2TextReferenceBalance" in types
    assert "Flux2IdentityFeatureTransfer" in types
