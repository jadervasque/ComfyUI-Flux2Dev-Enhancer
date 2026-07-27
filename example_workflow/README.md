# Example workflows

This directory contains two generations of workflows.

## Recommended FLUX.2 workflows

These workflows use the new model-neutral node IDs under `conditioning/flux2` and only require ComfyUI core nodes plus this extension.

| File | Model/profile | Demonstrates |
|---|---|---|
| `FLUX2_dev_single_reference_identity.json` | FLUX.2 dev | One identity reference, VAE encoding, `Flux2MultiReferenceLatent`, sigma-aware `Flux2IdentityFeatureTransfer`, and standard ComfyUI sampling. |
| `FLUX2_klein_single_reference_identity.json` | FLUX.2 Klein 9B distilled | The same architecture-aware identity path with a four-step Klein profile. Change the checkpoint and Qwen encoder for Klein 4B or base variants. |
| `FLUX2_multi_reference_masked_identity.json` | FLUX.2 dev by default | Two references in stable order, one mask per reference, late-start transfer, `zero_unmasked_tokens`, and a smaller query chunk for VRAM control. |
| `FLUX2_reference_attention_controls.json` | FLUX.2 dev by default | `Flux2ReferenceAttentionControl`, `Flux2TextReferenceBalance`, and a conservative identity-transfer stage in one model/conditioning chain. |

All model and image filenames are editable placeholders. After importing a workflow, select the files installed in your local ComfyUI directories.

### Suggested model substitutions

- FLUX.2 dev: `flux2-dev.safetensors` plus a compatible Mistral FLUX.2 text encoder.
- Klein 9B: `flux-2-klein-9b.safetensors` plus a Qwen3 8B FLUX.2 text encoder.
- Klein 4B: select the Klein 4B checkpoint and matching Qwen3 4B text encoder; the generic nodes detect its five double blocks and twenty single blocks automatically.
- Base Klein: increase the scheduler steps according to the checkpoint guidance rather than using the distilled four-step default.
- FP8 or GGUF: change only the loader when it preserves ComfyUI's attention-patch APIs and reference metadata.

### Mask inputs

The examples connect the `MASK` output of `LoadImage` for convenience. Use an image with alpha or replace that connection with your preferred mask-generation node. Reference and mask order must match:

```text
latent_1 <-> subject_mask_1
latent_2 <-> subject_mask_2
...
```

### Node order

```text
MODEL:        loader -> optional LoRAs -> reference controls -> identity transfer -> guider/sampler
CONDITIONING: text encode -> multi reference latent -> optional reference controls -> guider/sampler
REFERENCES:   image -> scale/crop -> FLUX.2 VAE encode -> latent_N
```

The standard ComfyUI sampler stack remains the recommended path. The historical direct Klein sampler is not used by the new examples.

## Historical legacy workflows

The remaining JSON files were inherited from the upstream Klein-specific project. They are intentionally retained to verify backward compatibility and may contain IDs now displayed with `(Legacy)`, including:

- `IdentityFeatureTransferFinal`
- `Flux2KleinMultiReferenceLatent`
- `Flux2KleinColorAnchor`
- `IdentityFeatureTransferV3`
- `Flux2KleinKSamplerExperimental`

Use the four `FLUX2_*.json` workflows above for new projects. The legacy files are not the reference implementation for FLUX.2 dev or Klein 4B.

## Validation

`tests/test_example_workflows.py` parses every recommended JSON workflow and checks:

- workflow schema essentials;
- link endpoint and socket consistency;
- absence of legacy custom-node IDs;
- presence of model-neutral reference and identity nodes;
- the intended dev, Klein, multireference, and reference-control profiles.

These tests validate graph structure, not image quality. Visual validation still requires the actual checkpoints, local images, and a GPU runtime.
