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

The extension now performs runtime architecture and loader-capability detection.
It does not decide compatibility from checkpoint filenames.

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

“Loader-dependent” means the loader must preserve ComfyUI's model patch APIs and
runtime transformer metadata. Use **FLUX.2 Architecture Inspector** to check the
loaded model.

## Why architecture detection matters

The official FLUX.2 variants do not have the same transformer depth:

| Profile | Hidden width | Attention heads | Double blocks | Single blocks | Text-conditioning width |
|---|---:|---:|---:|---:|---:|
| FLUX.2 [dev] | 6144 | 48 | 8 | 48 | 15360 |
| FLUX.2 [klein] 9B | 4096 | 32 | 8 | 24 | 12288 |
| FLUX.2 [klein] 4B | 3072 | 24 | 5 | 20 | 7680 |

A schedule calibrated for Klein 9B cannot be copied unchanged to dev or Klein 4B.
This fork reads the actual block lists from the loaded model, validates custom
schedules against them, and projects legacy schedules by relative depth when a
legacy preset is selected.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer.git
```

Restart ComfyUI after installing or updating.

No additional runtime packages are required beyond the dependencies already used
by ComfyUI. Development tests require `pytest`.

## Recommended workflow

### Model path

```text
Load Diffusion Model
        ↓
Apply LoRA(s), when used
        ↓
FLUX.2 Identity Feature Transfer
        ↓
KSampler / SamplerCustom
```

### Conditioning and reference path

```text
FLUX.2 text encoder
        ↓
Prompt conditioning
        ↓
FLUX.2 Multi Reference Latent
        ↓
KSampler positive conditioning

Reference image 1 → FLUX.2 VAE Encode → latent_1
Reference image 2 → FLUX.2 VAE Encode → latent_2
...
```

### Output path

```text
Empty FLUX.2 Latent or encoded source image
        ↓
KSampler
        ↓
FLUX.2 VAE Decode
        ↓
Image
```

The output canvas is controlled by the latent supplied to the sampler. Reference
image dimensions do not automatically set output dimensions.

## New model-neutral nodes

| Node | Purpose |
|---|---|
| **FLUX.2 Architecture Inspector** | Reports detected variant, block counts, conditioning width, guidance support, reference method, and loader hook capabilities. |
| **FLUX.2 Identity Feature Transfer** | Pulls generated attention features toward matching reference features with architecture-aware schedules and masks. |
| **FLUX.2 Multi Reference Latent** | Adds up to eight VAE-encoded references using an explicit placement method and append/replace behavior. |
| **FLUX.2 Reference Attention Control** | Scales one reference's attention keys and values, optionally with a spatial fade. |
| **FLUX.2 Reference Weight** | Lightweight model-only multiplier for one reference's keys and values. |
| **FLUX.2 Text/Reference Balance** | Attenuates text or reference attention around a neutral midpoint. |
| **FLUX.2 Reference Latent Mask** | Attenuates black regions directly in one encoded reference latent. |
| **FLUX.2 Conditioning Enhancer** | Scales, whitens, equalizes, and optionally adjusts the three stacked encoder-layer slices. |
| **FLUX.2 Text Conditioning Enhancer** | Simpler magnitude, contrast, and token-norm controls. |
| **FLUX.2 Sectioned Encoder** | Encodes FRONT/MID/END prompt sections and records tokenizer-derived ranges when the loader exposes the tokenizer. |
| **FLUX.2 Detail Controller** | Scales section ranges or explicit token ranges. |
| **FLUX.2 Color Anchor** | Corrects per-channel latent color statistics toward a selected reference. |
| **FLUX.2 Identity Guidance** | Applies a post-CFG latent correction toward an identity latent. |

## FLUX.2 Identity Feature Transfer

### What the node does

The node patches attention output during denoising. For each active transformer
block it:

1. Reads `reference_image_num_tokens`, `img_slice`, `block_type`, and
   `block_index` from ComfyUI.
2. Separates text, generated-image, and reference-image tokens.
3. Builds a reference bank from the selected references and masks.
4. Centers and normalizes generated and reference features.
5. Computes cosine similarity between generated tokens and reference tokens.
6. Rejects matches below `similarity_floor`.
7. Pools reference features using temperature-controlled softmax weights.
8. Applies confidence-gated feature transfer to generated-image tokens.
9. Returns the modified attention output to the remaining transformer layers.

It does not copy pixels and it is not a face-swap postprocessor. Identity, pose,
lighting, hair, clothing, and background are entangled in model features. Masks,
schedules, and conservative strength are therefore important.

### Presets

| Preset | Intended use |
|---|---|
| `AUTO_SOFT` | Preserve some likeness while allowing substantial prompt and composition freedom. |
| `AUTO_BALANCED` | General starting point for identity-preserving edits. |
| `AUTO_STRONG` | Strong lock; use masks and inspect unwanted pose/background transfer. |
| `KLEIN_LEGACY_HARD` | Original hard schedule projected to the loaded architecture. |
| `KLEIN_LEGACY_MID` | Original medium similarity settings with projected depth. |
| `KLEIN_LEGACY_SOFT` | Original selective settings with projected depth. |
| `CUSTOM` | Use the provided double and single schedule strings directly. |

Automatic presets are architecture-aware starting points, not guarantees of the
same visual intensity on every model, quantization, resolution, sampler, or LoRA.

### Strength modes

#### `normalized_total`

Treats `total_strength` as an approximate aggregate blend and distributes it over
the active blocks. This prevents a 48-single-block model from automatically
receiving roughly twice the repeated transfer of a 24-single-block model.

The approximation is:

```python
per_application = 1 - (1 - total_strength) ** (1 / active_applications)
```

#### `legacy_per_block`

Uses schedule values directly on every active block. This reproduces legacy
behavior more closely but can become very strong as block count or step count
increases.

### Denoising controls

- `start_percent` and `end_percent` gate the transfer over denoising progress.
- Connect the actual `SIGMAS` output for the most reliable window and
  equal-energy behavior.
- `sigma_scaling=equal_energy` adjusts block strengths according to the current
  sigma interval.
- Later windows generally preserve composition more freely; early windows can
  transfer pose and global structure more strongly.

### VRAM control

`query_chunk_size` limits how many generated tokens are compared with the complete
reference bank at once. Lower it when similarity matching causes a VRAM spike.
It changes memory use, not the intended mathematical result.

Suggested sequence:

1. Start at `256`.
2. Lower to `128` or `64` if memory is insufficient.
3. Reduce reference resolution or reference count if the bank itself is too large.

### Reference selection

`reference_indices` accepts:

```text
all
0
0,2,3
0-3
0,2-4
```

Indices are zero-based. `reference_index` is the fallback when the selection text
does not resolve to a valid reference.

### Mask behavior

- `focus_only`: masks limit the explicit transfer bank, while native attention can
  still see the full reference.
- `zero_unmasked_tokens`: masked-out reference tokens are also blocked as native
  attention sources. This is more restrictive and requires an attention-input
  patch-capable loader.

Reference order is shared with masks:

```text
latent_1 ↔ subject_mask_1
latent_2 ↔ subject_mask_2
...
latent_8 ↔ subject_mask_8
```

## FLUX.2 Multi Reference Latent

Inputs must be FLUX.2 VAE-encoded `LATENT` values. Batch items are split into
individual references in stable order.

### Mode

- `replace`: replace any reference list already stored in conditioning.
- `append`: preserve existing references and add the new references afterward.

### Reference method

| Method | Behavior |
|---|---|
| `model_default` | Do not force a method; use the loaded model's default. Recommended first choice. |
| `index` | Give every reference its own indexed position. |
| `offset` | Use offset placement supported by the ComfyUI FLUX implementation. |
| `uxo/uno` | Use the UXO spatial offset method. |
| `index_timestep_zero` | Indexed references with reference timestep-zero behavior, used by compatible KV-cache paths. |

A method being available in the UI does not guarantee that a third-party loader
implements it correctly.

## Conditioning tools

Official FLUX.2 text encoders expose three selected hidden-state slices stacked in
the final conditioning width. The generic Conditioning Enhancer can scale those
three slices when the width is divisible by three.

The labels `early`, `mid`, and `late` refer to the selected encoder-layer slices,
not to denoising time.

### Sectioned Encoder

The node supports either separate fields or a combined prompt:

```text
[FRONT] subject and primary action
[MID] clothing, props, and scene details
[END] lighting, camera, and rendering style
```

For official ComfyUI Qwen and Mistral FLUX.2 tokenizer wrappers, the node reads the
actual wrapper template and underlying tokenizer to estimate section ranges. If a
third-party loader hides the tokenizer, the prompt is still encoded normally, but
exact section metadata is omitted rather than fabricated.

The Detail Controller can then:

- use `flux2_sections` metadata;
- read legacy `klein_sections` metadata;
- fall back to relative 25% / 50% / 25% sections;
- or perform no section scaling when metadata is unavailable.

## Color Anchor and Identity Guidance

These nodes act on denoised latent predictions after CFG. They are not semantic
identity extractors.

### Color Anchor

Adjusts only per-channel spatial means toward a selected reference. Spatial
variations remain unchanged. `by_variance` trusts stable reference channels more
than highly varying channels.

### Identity Guidance

- `adaptive`: spatially weights correction by latent similarity.
- `direct`: pulls every latent position toward the reference and can copy layout.
- `channel_match`: matches channel statistics without direct spatial copying.

Connect `SIGMAS` when precise start/end windows matter.

## Architecture Inspector

Run **FLUX.2 Architecture Inspector** immediately after the model loader when
checking a new checkpoint or quantized loader.

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
Legacy display names include `(Legacy)`.

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

The original basic, Advanced, and V3 Identity Feature Transfer nodes remain
Klein-oriented legacy algorithms. Use the new **FLUX.2 Identity Feature Transfer**
for architecture-aware execution.

## Experimental sampler

`Flux2Klein KSampler Experimental` is retained only for workflow compatibility.
It directly invokes the diffusion model and does not expose every feature of the
standard ComfyUI sampler stack. It is not part of the multi-variant compatibility
claim.

Use standard `KSampler`, `SamplerCustom`, and the official FLUX.2 scheduler nodes
for new workflows.

## Loader and quantization requirements

A compatible loader must preserve, as required by each node:

- `model.diffusion_model` or an equivalent double/single-stream model object;
- actual double and single block lists;
- `set_model_attn1_patch`;
- `set_model_attn1_output_patch`;
- `model_options` and sampler post-CFG callbacks;
- `reference_image_num_tokens`;
- `img_slice`, `block_type`, and `block_index`;
- reference latent forwarding.

Quantization of weights does not by itself prevent these nodes from working.
Loader wrappers that bypass or replace ComfyUI's patch interfaces can.

## Troubleshooting

### Node reports an unknown or incompatible architecture

- Run Architecture Inspector.
- Update ComfyUI.
- Test with the native ComfyUI diffusion-model loader.
- Confirm the checkpoint is FLUX.2 rather than FLUX.1 or another architecture.

### Identity Transfer has no effect

- Confirm reference latents reach positive conditioning.
- Enable `debug` and look for reference token counts.
- Confirm at least one block has nonzero strength.
- Check `start_percent` / `end_percent`.
- Verify that masks contain white pixels after pooling.

### Identity Transfer causes OOM

- Lower `query_chunk_size`.
- Reduce reference resolution.
- Use fewer references.
- Avoid full-resolution references when only a face crop is needed.
- Test a shorter active block schedule.

### Result copies pose, framing, or background

- Use a tighter subject mask.
- Start with `AUTO_SOFT`.
- Move the active denoising window later.
- Raise `similarity_floor`.
- Use `focus_only` before trying `zero_unmasked_tokens`.

### Section controls do not correspond to text

- Inspect `flux2_section_backend` metadata in debug output.
- Use the official ComfyUI Mistral or Qwen tokenizer loader.
- Set Detail Controller fallback to `no_op` if approximate sections are unwanted.

## Development and tests

```bash
python -m py_compile architecture.py scheduling.py flux2_*.py
pytest -q
```

The pure-Python tests cover:

- official architecture fingerprints;
- dynamic block limits;
- legacy schedule projection;
- cumulative-strength normalization;
- reference-index parsing;
- batch reference splitting;
- append/replace metadata behavior;
- token slicing;
- tokenizer-derived section ranges;
- neutral pass-through behavior;
- generic node registration.

Image-quality validation still requires real FLUX.2 checkpoints, fixed seeds,
reference images, and GPU execution. Code-level tests do not prove visual quality.

## Implementation plan

The implementation and validation plan is maintained in
[`plans/PLAN0.md`](plans/PLAN0.md).

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
