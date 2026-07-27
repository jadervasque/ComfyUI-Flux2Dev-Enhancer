# Maintained example workflows

This directory contains the supported workflow examples for **ComfyUI-Flux2Dev-Enhancer**.

| File | Default profile | Demonstrates |
|---|---|---|
| `FLUX2_dev_single_reference_identity.json` | FLUX.2 dev | One identity reference, VAE encoding, sigma-aware Identity Feature Transfer, and standard ComfyUI sampling. |
| `FLUX2_klein_single_reference_identity.json` | FLUX.2 Klein 9B distilled | The same architecture-aware identity path with a four-step Klein profile. |
| `FLUX2_multi_reference_masked_identity.json` | FLUX.2 dev | Two ordered references, one mask per reference, late-start transfer, source masking, and lower query chunk size. |
| `FLUX2_reference_attention_controls.json` | FLUX.2 dev | Multi Reference Latent, Reference Attention Control, Text/Reference Balance, and Identity Feature Transfer in one chain. |

All model and image filenames are editable placeholders. Select the files available in your local ComfyUI installation after importing a workflow.

## Visual structure

Every maintained workflow uses four left-to-right groups:

1. **Model and prompt**
2. **Reference preparation**
3. **ComfyUI-Flux2Dev-Enhancer nodes**
4. **Sampling and output**

Executable nodes are positioned on a consistent grid without overlap. Each project node demonstrated has an adjacent native ComfyUI `MarkdownNote` in English.

The Markdown Notes document:

- internal implementation role;
- correct position in the graph;
- neutral values and important controls;
- reference and mask order;
- denoising and sigma behavior;
- VRAM implications;
- limitations such as pose, composition, or background transfer.

Markdown Notes have no inputs or outputs and do not affect execution.

## Model substitutions

- **FLUX.2 dev:** use a compatible dev checkpoint, Mistral FLUX.2 text encoder, and FLUX.2 VAE.
- **Klein 9B distilled:** the maintained example uses a Qwen3 8B encoder and four-step scheduler profile.
- **Klein 4B:** replace the checkpoint and use the matching Qwen3 4B encoder. Architecture-aware nodes detect its block counts automatically.
- **Base Klein variants:** increase scheduler steps according to the model guidance instead of retaining the distilled four-step profile.
- **BF16, FP8, GGUF, or KV loaders:** the loader must preserve ComfyUI attention-patch interfaces and reference metadata. Use Architecture Inspector before debugging node behavior.

## Reference and mask order

```text
latent_1 <-> subject_mask_1
latent_2 <-> subject_mask_2
...
latent_8 <-> subject_mask_8
```

The examples connect the `MASK` output of `LoadImage` for convenience. Use alpha-aware images or replace those links with dedicated mask-generation nodes.

## Recommended node order

```text
MODEL:
loader -> optional LoRAs -> reference controls -> identity transfer -> guider/sampler

CONDITIONING:
text encode -> optional conditioning tools -> multi reference latent
-> optional reference controls -> guider/sampler

REFERENCES:
image -> scale/crop -> FLUX.2 VAE encode -> latent_N
```

Use the standard ComfyUI sampler stack. The standalone project does not include a direct experimental sampler.

## Validation

The repository tests verify:

- JSON parsing;
- link and socket consistency;
- canonical project node IDs;
- dev, Klein, multi-reference, and attention-control profiles;
- one Markdown Note per demonstrated project node;
- four ordered visual groups;
- absence of overlapping nodes.

These checks validate graph structure, not visual quality. Runtime validation still requires the referenced checkpoints, local images, and a compatible GPU environment.
