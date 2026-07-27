<div align="center">

[English](README.md) | [Português (Brasil)](README.pt-BR.md) | [Español](README.es.md)

</div>

# ComfyUI FLUX.2 Enhancer

Architecture-aware conditioning, reference-latent, identity-transfer, color-control,
and guidance nodes for the open-weight FLUX.2 family in ComfyUI.

This project is an independent fork of
[`capitan01R/ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer).
The original project and algorithms were created by **capitan01R**. This fork
preserves the upstream MIT notice and generalizes the extension beyond a fixed
Klein 9B architecture.

> [!IMPORTANT]
> FLUX.2 model licenses are separate from this extension's MIT license. “Open
> weight” does not imply that every checkpoint permits the same commercial use.
> Review the license attached to the exact model you load.

## Status

Version: **4.0.0 beta**

The extension performs runtime architecture and loader-capability detection. It
does not decide compatibility from checkpoint filenames.

| Model family | Architecture detection | Generic nodes | Preset state | Runtime image validation |
|---|---:|---:|---|---|
| FLUX.2 [dev] | Implemented | Implemented | Conservative automatic presets | Pending broader GPU validation |
| FLUX.2 [klein] 4B distilled | Implemented | Implemented | Automatic presets | Pending broader GPU validation |
| FLUX.2 [klein] 4B base | Implemented | Implemented | Automatic presets | Pending broader GPU validation |
| FLUX.2 [klein] 9B distilled | Implemented | Implemented | Automatic and legacy presets | Existing upstream behavior plus new code paths |
| FLUX.2 [klein] 9B base | Implemented | Implemented | Automatic and legacy presets | Pending broader GPU validation |
| FLUX.2 [klein] 9B KV | Architecture-compatible | Loader-dependent | Conservative | Requires KV-loader testing |
| BF16 / FP8 repacks | Architecture-compatible | Loader-dependent | Same architecture profile | Test the specific loader |
| GGUF repacks | Architecture-compatible | Loader-dependent | Same architecture profile | Test the specific loader |

“Loader-dependent” means the loader must preserve ComfyUI's model-patching APIs
and runtime transformer metadata. Use **FLUX.2 Architecture Inspector** to check
the loaded model.

## Why architecture detection matters

The official FLUX.2 variants do not have the same transformer depth:

| Profile | Hidden width | Attention heads | Double blocks | Single blocks | Text-conditioning width |
|---|---:|---:|---:|---:|---:|
| FLUX.2 [dev] | 6144 | 48 | 8 | 48 | 15360 |
| FLUX.2 [klein] 9B | 4096 | 32 | 8 | 24 | 12288 |
| FLUX.2 [klein] 4B | 3072 | 24 | 5 | 20 | 7680 |

A schedule calibrated for Klein 9B cannot be copied unchanged to dev or Klein 4B.
The extension reads the actual block lists, validates custom schedules against
them, and projects legacy schedules by relative depth.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer.git
```

Restart ComfyUI after installing or updating.

No additional runtime packages are required beyond dependencies already used by
ComfyUI. Development tests require `pytest`.

## Example workflows

Recommended workflows are stored in [`example_workflow`](example_workflow):

- `FLUX2_dev_single_reference_identity.json`
- `FLUX2_klein_single_reference_identity.json`
- `FLUX2_multi_reference_masked_identity.json`
- `FLUX2_reference_attention_controls.json`

The recommended graphs are visually separated into model/prompt, reference
preparation, Enhancer nodes, and sampling/output sections. Every Enhancer node
used in a graph has an adjacent native **Markdown Note** explaining its role,
placement, runtime behavior, and important controls in English.

Inherited Klein-specific workflows remain available as legacy compatibility
fixtures. See [`example_workflow/README.md`](example_workflow/README.md).

## Recommended node order

### Model path

```text
Load Diffusion Model
        ↓
Apply LoRA(s), when used
        ↓
FLUX.2 reference/model controls, when used
        ↓
FLUX.2 Identity Feature Transfer
        ↓
Guider / KSampler / SamplerCustom
```

### Conditioning and reference path

```text
FLUX.2 text encoder
        ↓
Prompt conditioning
        ↓
FLUX.2 Multi Reference Latent
        ↓
Optional reference conditioning controls
        ↓
Sampler positive conditioning

Reference image 1 → FLUX.2 VAE Encode → latent_1
Reference image 2 → FLUX.2 VAE Encode → latent_2
...
```

### Output path

```text
Empty FLUX.2 Latent or encoded source image
        ↓
Sampler
        ↓
FLUX.2 VAE Decode
        ↓
Image
```

The output canvas is controlled by the latent supplied to the sampler. Reference
image dimensions do not automatically set output dimensions.

## Model-neutral nodes

| Node | Purpose |
|---|---|
| **FLUX.2 Architecture Inspector** | Reports detected variant, block counts, conditioning width, guidance support, reference method, and loader hook capabilities. |
| **FLUX.2 Identity Feature Transfer** | Pulls generated attention features toward matching reference features using architecture-aware schedules and masks. |
| **FLUX.2 Multi Reference Latent** | Adds up to eight VAE-encoded references with explicit placement and append/replace behavior. |
| **FLUX.2 Reference Attention Control** | Scales one reference's attention keys and values, optionally with a spatial fade. |
| **FLUX.2 Reference Weight** | Lightweight model-only multiplier for one reference's keys and values. |
| **FLUX.2 Text/Reference Balance** | Attenuates text or reference attention around a neutral midpoint. |
| **FLUX.2 Reference Latent Mask** | Attenuates black regions directly in one encoded reference latent. |
| **FLUX.2 Conditioning Enhancer** | Scales, whitens, equalizes, and optionally adjusts the three stacked encoder-layer slices. |
| **FLUX.2 Text Conditioning Enhancer** | Simpler magnitude, contrast, and token-norm controls. |
| **FLUX.2 Sectioned Encoder** | Encodes FRONT/MID/END prompt sections and records tokenizer-derived ranges when available. |
| **FLUX.2 Detail Controller** | Scales section ranges or explicit token ranges. |
| **FLUX.2 Color Anchor** | Corrects per-channel latent color statistics toward a selected reference. |
| **FLUX.2 Identity Guidance** | Applies a post-CFG latent correction toward an identity latent. |

## FLUX.2 Identity Feature Transfer

The node patches attention output during denoising. For each active transformer
block it:

1. Reads `reference_image_num_tokens`, `img_slice`, `block_type`, and
   `block_index` from ComfyUI.
2. Separates text, generated-image, and reference-image tokens.
3. Builds a reference bank from selected references and masks.
4. Centers and normalizes generated and reference features.
5. Computes cosine similarity between generated and reference tokens.
6. Rejects matches below `similarity_floor`.
7. Pools reference features using temperature-controlled softmax weights.
8. Applies confidence-gated feature transfer to generated-image tokens.
9. Returns the modified attention output to the remaining transformer layers.

It does not copy pixels and it is not a face-swap postprocessor. Identity, pose,
lighting, hair, clothing, and background remain entangled in model features.

### Presets

| Preset | Intended use |
|---|---|
| `AUTO_SOFT` | Preserve some likeness while allowing substantial prompt and composition freedom. |
| `AUTO_BALANCED` | General starting point for identity-preserving edits. |
| `AUTO_STRONG` | Strong lock; use masks and inspect unwanted pose/background transfer. |
| `KLEIN_LEGACY_HARD` | Original hard schedule projected to the loaded architecture. |
| `KLEIN_LEGACY_MID` | Original medium similarity settings with projected depth. |
| `KLEIN_LEGACY_SOFT` | Original selective settings with projected depth. |
| `CUSTOM` | Use the supplied double and single schedule strings directly. |

Automatic presets are architecture-aware starting points, not guarantees of equal
visual intensity across models, quantizations, resolutions, samplers, or LoRAs.

### Strength modes

`normalized_total` treats `total_strength` as an approximate aggregate blend and
distributes it across active blocks:

```python
per_application = 1 - (1 - total_strength) ** (1 / active_applications)
```

`legacy_per_block` applies schedule values directly to every active block. It more
closely reproduces legacy behavior, but can become very strong as block count or
step count increases.

### Denoising and VRAM controls

- `start_percent` and `end_percent` gate transfer over denoising progress.
- Connect the real `SIGMAS` output for reliable progress and equal-energy scaling.
- Later windows generally preserve composition more freely.
- `query_chunk_size` limits the similarity matrix. Lower it to reduce peak VRAM.
- Reduce reference resolution or count if the reference bank itself is too large.

### Reference masks

- `focus_only`: masks limit the explicit transfer bank while native attention can
  still see the complete reference.
- `zero_unmasked_tokens`: masked-out tokens are also blocked as native attention
  sources and therefore require an attention-input-patch-capable loader.

Reference order is shared with masks:

```text
latent_1 ↔ subject_mask_1
latent_2 ↔ subject_mask_2
...
latent_8 ↔ subject_mask_8
```

## Multi Reference Latent

Inputs must be FLUX.2 VAE-encoded `LATENT` values. Batch items are split into
individual references in stable order.

- `replace`: replace an existing reference list.
- `append`: preserve existing references and add the connected references.
- `model_default`: do not force a reference method; recommended first choice.
- `index`, `offset`, `uxo/uno`, and `index_timestep_zero`: explicit methods for
  compatible models and loaders.

A method being listed in the UI does not guarantee that a third-party loader
implements it correctly.

## Reference attention controls

**Reference Attention Control** multiplies the selected reference's key/value
tokens. A strength of `1.0` is neutral. Spatial fades derive token-space weights
from the corresponding reference latent.

**Text/Reference Balance** is neutral at `0.5`. Below that value text is
attenuated; above it references are attenuated. It is a tradeoff control rather
than an independent gain for both streams.

## Conditioning tools

Official FLUX.2 text encoders expose three selected hidden-state slices stacked in
the final conditioning width. The generic Conditioning Enhancer can scale those
slices when the width is divisible by three.

The Sectioned Encoder supports separate fields or a combined prompt:

```text
[FRONT] subject and primary action
[MID] clothing, props, and scene details
[END] lighting, camera, and rendering style
```

With official Qwen and Mistral wrappers, the node reads the actual template and
underlying tokenizer. If a third-party loader hides the tokenizer, encoding still
works, but exact section metadata is omitted rather than fabricated.

## Color Anchor and Identity Guidance

These nodes act on denoised latent predictions after CFG. They are not semantic
identity extractors.

- **Color Anchor** adjusts per-channel spatial means toward a reference while
  preserving spatial deviations.
- **Identity Guidance — adaptive** weights latent correction by local similarity.
- **Identity Guidance — direct** pulls every latent position toward the reference.
- **Identity Guidance — channel_match** matches channel statistics without direct
  spatial copying.

Connect `SIGMAS` when precise start/end windows matter.

## Architecture Inspector

Run **FLUX.2 Architecture Inspector** after the model loader when checking a new
checkpoint or quantized loader.

Example report:

```json
{
  "variant": "flux2_dev",
  "hidden_size": 6144,
  "num_heads": 48,
  "double_blocks": 8,
  "single_blocks": 48,
  "context_in_dim": 15360,
  "supports_attn_input_patch": true,
  "supports_attn_output_patch": true
}
```

If a loader removes a required hook, compatible nodes stop with an actionable
error instead of silently claiming to work.

## Legacy workflow compatibility

Existing IDs remain registered so old workflow JSON files continue to load.

| Legacy ID | Recommended new ID |
|---|---|
| `IdentityFeatureTransferFinal` | `Flux2IdentityFeatureTransfer` |
| `Flux2KleinMultiReferenceLatent` | `Flux2MultiReferenceLatent` |
| `Flux2KleinRefLatentController` | `Flux2ReferenceAttentionControl` |
| `Flux2KleinRefLatentWeight` | `Flux2ReferenceWeight` |
| `Flux2KleinTextRefBalance` | `Flux2TextReferenceBalance` |
| `Flux2KleinMaskRefController` | `Flux2ReferenceLatentMask` |
| `Flux2KleinColorAnchor` | `Flux2ColorAnchor` |
| `IdentityGuidance` | `Flux2IdentityGuidance` |
| `Flux2KleinEnhancer` | `Flux2ConditioningEnhancer` |
| `Flux2KleinTextEnhancer` | `Flux2TextConditioningEnhancer` |
| `Flux2KleinSectionedEncoder` | `Flux2SectionedEncoder` |
| `Flux2KleinDetailController` | `Flux2DetailController` |

The basic, Advanced, and V3 identity nodes remain Klein-oriented legacy
algorithms. The direct experimental Klein sampler is also retained only for
workflow compatibility and is excluded from the multi-variant claim.

## Loader and quantization requirements

A compatible loader must preserve the capabilities required by each node:

- access to the FLUX.2 diffusion model and real double/single block lists;
- `set_model_attn1_patch`;
- `set_model_attn1_output_patch`;
- `model_options` and sampler post-CFG callbacks;
- `reference_image_num_tokens`;
- `img_slice`, `block_type`, and `block_index`;
- reference-latent forwarding.

Weight quantization alone does not prevent compatibility. Loader wrappers that
replace or bypass ComfyUI's patch interfaces can.

## Troubleshooting

### Identity Transfer has no effect

- Confirm reference latents reach positive conditioning.
- Enable `debug` and inspect reference token counts.
- Confirm at least one block has nonzero strength.
- Check `start_percent` and `end_percent`.
- Verify masks contain white pixels after pooling.

### Identity Transfer causes OOM

- Lower `query_chunk_size`.
- Reduce reference resolution.
- Use fewer references.
- Use a face crop when only identity is needed.
- Shorten the active block schedule.

### Result copies pose, framing, or background

- Use a tighter subject mask.
- Start with `AUTO_SOFT`.
- Move the active denoising window later.
- Raise `similarity_floor`.
- Try `focus_only` before `zero_unmasked_tokens`.

### Section controls do not correspond to text

- Inspect `flux2_section_backend`.
- Use the official ComfyUI Mistral or Qwen tokenizer loader.
- Use a no-op fallback when approximate sections are unwanted.

## Development and tests

```bash
python -m py_compile architecture.py scheduling.py flux2_*.py
pytest -q
```

Tests cover architecture fingerprints, schedules, legacy projection, strength
normalization, reference metadata, token slicing, section ranges, registration,
neutral execution, workflow graph integrity, Markdown Note documentation, and
localized README navigation.

Code-level tests do not prove image quality. Visual validation still requires real
checkpoints, fixed seeds, reference images, and GPU execution.

## Implementation plan

See [`plans/PLAN0.md`](plans/PLAN0.md).

## Attribution

Original project and algorithms:

- **capitan01R**
- [`ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer)

Multi-variant fork and architecture generalization:

- **Jader Vasque**
- [`ComfyUI-Flux2Dev-Enhancer`](https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer)

See [`NOTICE.md`](NOTICE.md) and [`LICENSE`](LICENSE).

This repository is not affiliated with or endorsed by Black Forest Labs, ComfyUI,
or the upstream author.
