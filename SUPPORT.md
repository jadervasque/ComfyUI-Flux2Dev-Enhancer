# Support

## Where to ask

- **Bug reports:** use the GitHub bug-report form.
- **Feature proposals:** use the feature-request form.
- **Security issues:** follow `SECURITY.md`; do not use public issues.
- **General ComfyUI questions:** use ComfyUI community support unless the problem is specific to this extension.

## Information required for technical support

Include all relevant details:

- extension version or commit SHA;
- ComfyUI version/commit;
- operating system, Python version, GPU, VRAM, and driver;
- checkpoint, text encoder, VAE, loader, and precision/quantization;
- sampler, scheduler, steps, guidance, resolution, and batch size;
- complete error traceback;
- exported workflow JSON with secrets and private paths removed;
- whether the problem reproduces without other custom nodes;
- Architecture Inspector output when loader compatibility is involved.

Reports without enough reproduction information may be closed until the missing details are supplied.

## Supported scope

The project supports the canonical nodes registered by the current release and the maintained example workflows. Compatibility depends on loaders preserving the ComfyUI patch interfaces and FLUX.2 reference metadata documented in `docs/ARCHITECTURE.md`.

## Out of scope

- removed node IDs or historical workflows;
- installation or licensing support for third-party checkpoints;
- model files distributed by other organizations;
- guaranteed behavior from loaders that replace ComfyUI attention hooks;
- image-quality guarantees across every checkpoint, LoRA, quantization, seed, or workflow;
- private workflow design or model-training services.

Support is provided on a best-effort basis without service-level guarantees.
