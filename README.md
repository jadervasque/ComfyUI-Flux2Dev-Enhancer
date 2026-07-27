<div align="center">

[English](README.md) | [Português (Brasil)](README.pt-BR.md) | [Español](README.es.md)

</div>

# ComfyUI-Flux2Dev-Enhancer

Architecture-aware conditioning, reference, identity-transfer, and latent-guidance nodes for the open-weight FLUX.2 family in ComfyUI.

> **Status:** `1.0.0b1` standalone beta. The public API contains only the canonical nodes documented in [`docs/NODE_REFERENCE.md`](docs/NODE_REFERENCE.md). Historical node IDs and compatibility aliases are intentionally not included.

## Highlights

- Runtime detection for FLUX.2 dev, Klein 4B, and Klein 9B transformer profiles.
- Identity Feature Transfer with architecture-aware schedules, masks, denoising windows, sigma scaling, and chunked matching.
- Single- and multi-reference latent conditioning.
- Reference attention weighting and text/reference balance.
- Conditioning enhancement and tokenizer-aware prompt sections.
- Post-CFG color anchoring and latent identity guidance.
- Architecture Inspector for loader and quantization diagnostics.
- Maintained, visually organized workflows with embedded English Markdown Notes.
- Automated tests and GitHub Actions CI for Python 3.10–3.12.

## Supported profiles

| Profile | Architecture detection | Canonical nodes | Validation status |
|---|---:|---:|---|
| FLUX.2 dev | Implemented | Implemented | Broader GPU validation in progress |
| FLUX.2 Klein 4B | Implemented | Implemented | Broader GPU validation in progress |
| FLUX.2 Klein 9B | Implemented | Implemented | Architecture and code paths covered |
| KV-cache variants | Architecture-compatible | Loader-dependent | Requires a compatible KV loader |
| BF16 / FP8 repacks | Architecture-compatible | Loader-dependent | Test the exact loader |
| GGUF repacks | Architecture-compatible | Loader-dependent | Test the exact loader |

Loader-dependent means the loader must preserve the ComfyUI model-patch interfaces and runtime metadata described in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer.git
```

Restart ComfyUI after installing or updating.

No additional runtime package is required beyond the dependencies normally provided by ComfyUI. Development dependencies are optional:

```bash
python -m pip install -e ".[dev]"
```

## Recommended workflow order

### Model path

```text
Load Diffusion Model
        ↓
Apply LoRA(s), when used
        ↓
Reference controls, when used
        ↓
FLUX.2 Identity Feature Transfer
        ↓
Guider / KSampler / SamplerCustom
```

### Conditioning path

```text
FLUX.2 text encoder
        ↓
Prompt conditioning
        ↓
Optional conditioning tools
        ↓
FLUX.2 Multi Reference Latent
        ↓
Optional reference controls
        ↓
Positive conditioning input
```

### Reference path

```text
Reference image
        ↓
Scale or crop
        ↓
FLUX.2 VAE Encode
        ↓
latent_1 ... latent_8
```

Reference order and mask order must match:

```text
latent_1 ↔ subject_mask_1
latent_2 ↔ subject_mask_2
...
latent_8 ↔ subject_mask_8
```

## Canonical nodes

| Node | Purpose |
|---|---|
| **FLUX.2 Architecture Inspector** | Reports architecture, block counts, conditioning width, and loader hook capabilities. |
| **FLUX.2 Conditioning Enhancer** | Scales and normalizes active conditioning and encoder-layer slices. |
| **FLUX.2 Text Conditioning Enhancer** | Provides simpler magnitude, contrast, and norm controls. |
| **FLUX.2 Sectioned Encoder** | Encodes FRONT/MID/END prompt sections and records tokenizer-derived ranges. |
| **FLUX.2 Detail Controller** | Scales prompt sections or explicit token ranges. |
| **FLUX.2 Multi Reference Latent** | Adds up to eight VAE-encoded references to conditioning. |
| **FLUX.2 Reference Attention Control** | Scales one reference's attention keys and values, optionally with spatial fade. |
| **FLUX.2 Reference Weight** | Applies a lightweight model-only reference multiplier. |
| **FLUX.2 Text/Reference Balance** | Attenuates text or references around a neutral midpoint. |
| **FLUX.2 Reference Latent Mask** | Attenuates masked regions inside one stored reference latent. |
| **FLUX.2 Identity Feature Transfer** | Matches generated features to reference features during denoising. |
| **FLUX.2 Color Anchor** | Corrects latent channel means toward a selected reference. |
| **FLUX.2 Identity Guidance** | Applies adaptive, direct, or channel-statistic latent correction. |

See the complete input, output, neutral-value, and compatibility contract in [`docs/NODE_REFERENCE.md`](docs/NODE_REFERENCE.md).

## Identity Feature Transfer

The identity node clones the model and installs an attention-output patch. At selected blocks and denoising steps it:

1. separates text, generated-image, and reference-image tokens;
2. selects references and applies optional masks;
3. centers and normalizes feature vectors;
4. computes chunked cosine similarity;
5. rejects weak matches below `similarity_floor`;
6. pools reference features with a temperature-controlled softmax;
7. applies confidence-gated transfer to generated-image tokens.

It transfers internal features, not pixels. Identity, pose, lighting, clothing, hair, and background remain partially entangled. Conservative presets, masks, later denoising windows, and fixed-seed comparisons are recommended.

### Presets

- `AUTO_SOFT`: more prompt and composition freedom.
- `AUTO_BALANCED`: general starting point.
- `AUTO_STRONG`: stronger reference lock and higher copying risk.
- `CUSTOM`: explicit double- and single-block schedules.

### Strength modes

- `normalized_total`: distributes an approximate aggregate strength across active blocks.
- `per_block`: applies schedule strengths directly at every active block.

Lower `query_chunk_size` when similarity matching causes a VRAM spike.

## Example workflows

Maintained workflows are stored in [`example_workflow/`](example_workflow/):

- FLUX.2 dev, single-reference identity transfer;
- Klein 9B, single-reference identity transfer;
- two masked references;
- reference attention controls plus identity transfer.

Each workflow uses four visual zones and includes an English `MarkdownNote` beside every project node demonstrated.

## Standalone API policy

This repository is an independent standalone extension. Only the canonical `Flux2...` IDs in the node reference are supported. Removed historical IDs, direct experimental samplers, inherited presets, and compatibility adapters are not registered.

A public node-ID or socket-breaking change requires a major release. Quantized loader compatibility is based on preserved ComfyUI hooks and metadata, not checkpoint filenames.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Node reference](docs/NODE_REFERENCE.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Release process](docs/RELEASES.md)
- [Example workflows](example_workflow/README.md)
- [Implementation plan 1](plans/PLAN1.md)

## Development

```bash
python -m compileall -q __init__.py comfyui_flux2dev_enhancer tests
ruff check comfyui_flux2dev_enhancer tests --select E9,F63,F7,F82
pytest -q
```

Automated tests validate code, metadata, registry, workflow structure, layout, and documentation. They do not replace image-quality testing with real checkpoints and GPUs.

See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), [`SUPPORT.md`](SUPPORT.md), and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Attribution

ComfyUI-Flux2Dev-Enhancer is maintained as an independent project by **Jader Vasque**.

Early identity-transfer concepts and portions of the initial implementation were derived from **capitan01R's** [`ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer). The upstream copyright and MIT notice are preserved in [`LICENSE`](LICENSE), [`NOTICE.md`](NOTICE.md), and [`AUTHORS.md`](AUTHORS.md).

This repository is not affiliated with or endorsed by Black Forest Labs, ComfyUI, or the upstream author. FLUX.2 model licenses are separate from this extension's MIT license; review the license of every checkpoint you use.
