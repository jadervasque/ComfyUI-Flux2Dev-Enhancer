# Example workflows

This directory contains two generations of workflows.

## Recommended FLUX.2 workflows

These workflows use the model-neutral node IDs under `conditioning/flux2` and
require only ComfyUI core nodes plus this extension.

| File | Model/profile | Demonstrates |
|---|---|---|
| `FLUX2_dev_single_reference_identity.json` | FLUX.2 dev | One identity reference, VAE encoding, `Flux2MultiReferenceLatent`, sigma-aware `Flux2IdentityFeatureTransfer`, and standard ComfyUI sampling. |
| `FLUX2_klein_single_reference_identity.json` | FLUX.2 Klein 9B distilled | The same architecture-aware path with a four-step Klein profile. Change the checkpoint and Qwen encoder for Klein 4B or base variants. |
| `FLUX2_multi_reference_masked_identity.json` | FLUX.2 dev by default | Two references in stable order, one mask per reference, late-start transfer, `zero_unmasked_tokens`, and a smaller query chunk for VRAM control. |
| `FLUX2_reference_attention_controls.json` | FLUX.2 dev by default | `Flux2ReferenceAttentionControl`, `Flux2TextReferenceBalance`, and conservative identity transfer in one ordered chain. |

All model and image filenames are editable placeholders. After importing a
workflow, select the files installed in your local ComfyUI directories.

## Visual organization

Every recommended workflow follows the same left-to-right structure:

1. **Model and prompt**
2. **Reference preparation**
3. **ComfyUI FLUX.2 Enhancer nodes**
4. **Sampling and output**

The sections are represented by labeled ComfyUI groups. Nodes use a consistent
grid with clear horizontal flow, and the graphs are saved at a scale that exposes
the complete pipeline on first import.

No executable nodes overlap.

## Embedded Markdown documentation

Every `ComfyUI-Flux2Dev-Enhancer` node used in a recommended workflow has an
adjacent native ComfyUI `MarkdownNote`.

The notes are written in English and document:

- what the node changes internally;
- where it belongs in the model or conditioning path;
- neutral values and important controls;
- reference/mask ordering;
- denoising and sigma behavior;
- VRAM implications;
- limitations such as pose or background transfer.

The Markdown Notes are documentation only. They have no inputs, outputs, or
execution effect.

## Suggested model substitutions

- FLUX.2 dev: `flux2-dev.safetensors` plus a compatible Mistral FLUX.2 text encoder.
- Klein 9B: `flux-2-klein-9b.safetensors` plus a Qwen3 8B FLUX.2 text encoder.
- Klein 4B: use the Klein 4B checkpoint and matching Qwen3 4B encoder; the generic
  nodes detect five double blocks and twenty single blocks automatically.
- Base Klein: increase scheduler steps according to the checkpoint guidance rather
  than using the distilled four-step default.
- FP8 or GGUF: change the loader only when it preserves ComfyUI attention-patch
  APIs and reference metadata.

## Mask inputs

The examples connect the `MASK` output of `LoadImage` for convenience. Use an
image with alpha or replace it with your preferred mask-generation node.

Reference and mask order must match:

```text
latent_1 <-> subject_mask_1
latent_2 <-> subject_mask_2
...
```

## Node order

```text
MODEL:        loader -> optional LoRAs -> reference controls -> identity transfer -> guider/sampler
CONDITIONING: text encode -> multi reference latent -> optional reference controls -> guider/sampler
REFERENCES:   image -> scale/crop -> FLUX.2 VAE encode -> latent_N
```

The standard ComfyUI sampler stack remains the recommended path. The historical
direct Klein sampler is not used by the new examples.

## Historical legacy workflows

The remaining JSON files were inherited from the upstream Klein-specific project.
They are retained to verify backward compatibility and may contain IDs displayed
with `(Legacy)`, including:

- `IdentityFeatureTransferFinal`
- `Flux2KleinMultiReferenceLatent`
- `Flux2KleinColorAnchor`
- `IdentityFeatureTransferV3`
- `Flux2KleinKSamplerExperimental`

Use the four `FLUX2_*.json` workflows for new projects. Legacy files are not the
reference implementation for FLUX.2 dev or Klein 4B.

## Validation

`tests/test_example_workflows.py` validates graph schema, links, model profiles,
multireference wiring, and control order.

`tests/test_workflow_layout_and_readmes.py` additionally verifies:

- one Markdown Note for every Enhancer node used;
- note content and native `MarkdownNote` schema;
- non-overlapping node rectangles;
- the four visual workflow groups;
- localized README files and reciprocal language links.

These tests validate structure and documentation, not image quality. Visual
validation still requires actual checkpoints, local images, and a GPU runtime.
