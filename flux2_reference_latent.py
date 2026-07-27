"""Reference-latent conditioning nodes shared by all compatible FLUX.2 variants."""

from __future__ import annotations

from typing import Iterable

import torch


_REFERENCE_METHODS = {
    "model_default",
    "index",
    "offset",
    "uxo",
    "uxo/uno",
    "index_timestep_zero",
}


def latent_samples(latent):
    if latent is None:
        return None
    value = latent.get("samples") if isinstance(latent, dict) else latent
    if torch.is_tensor(value) and value.ndim == 4:
        return value
    return None


def split_reference_batches(latents: Iterable) -> list[torch.Tensor]:
    references: list[torch.Tensor] = []
    for latent in latents:
        samples = latent_samples(latent)
        if samples is None:
            continue
        for batch_index in range(samples.shape[0]):
            references.append(samples[batch_index : batch_index + 1].detach())
    return references


def apply_reference_metadata(
    conditioning,
    references: list[torch.Tensor],
    *,
    mode: str = "replace",
    reference_method: str = "model_default",
):
    if not references:
        return conditioning
    if mode not in {"replace", "append"}:
        raise ValueError(f"Unsupported reference mode {mode!r}; use replace or append.")
    method = str(reference_method)
    if method not in _REFERENCE_METHODS:
        raise ValueError(f"Unsupported FLUX.2 reference method {method!r}.")
    if method == "uxo/uno":
        method = "uxo"

    output = []
    for cond, meta in conditioning:
        new_meta = dict(meta)
        existing = list(meta.get("reference_latents", []) or [])
        new_meta["reference_latents"] = (
            existing + list(references) if mode == "append" else list(references)
        )
        if method == "model_default":
            new_meta.pop("reference_latents_method", None)
        else:
            new_meta["reference_latents_method"] = method
        output.append([cond, new_meta])
    return output


class Flux2MultiReferenceLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent_1": ("LATENT",),
                "mode": (["replace", "append"], {"default": "replace"}),
                "reference_method": (
                    [
                        "model_default",
                        "index",
                        "offset",
                        "uxo/uno",
                        "index_timestep_zero",
                    ],
                    {"default": "model_default"},
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "latent_2": ("LATENT",),
                "latent_3": ("LATENT",),
                "latent_4": ("LATENT",),
                "latent_5": ("LATENT",),
                "latent_6": ("LATENT",),
                "latent_7": ("LATENT",),
                "latent_8": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False

    def apply(
        self,
        conditioning,
        latent_1,
        mode="replace",
        reference_method="model_default",
        debug=False,
        **optional,
    ):
        ordered = [latent_1] + [optional[key] for key in sorted(optional)]
        references = split_reference_batches(ordered)
        output = apply_reference_metadata(
            conditioning,
            references,
            mode=mode,
            reference_method=reference_method,
        )
        if debug:
            shapes = [tuple(reference.shape) for reference in references]
            print(
                "[Flux2MultiReferenceLatent] "
                f"mode={mode} method={reference_method} "
                f"references={len(references)} shapes={shapes}"
            )
        return (output,)


class LegacyMultiReferenceLatent:
    """Original input surface: replace references and force indexed placement."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent_1": ("LATENT",),
            },
            "optional": {
                "latent_2": ("LATENT",),
                "latent_3": ("LATENT",),
                "latent_4": ("LATENT",),
                "latent_5": ("LATENT",),
                "latent_6": ("LATENT",),
                "latent_7": ("LATENT",),
                "latent_8": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2/legacy"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False

    def apply(self, conditioning, latent_1, **optional):
        return Flux2MultiReferenceLatent().apply(
            conditioning,
            latent_1,
            mode="replace",
            reference_method="index",
            debug=False,
            **optional,
        )


NODE_CLASS_MAPPINGS = {
    "Flux2MultiReferenceLatent": Flux2MultiReferenceLatent,
    "Flux2KleinMultiReferenceLatent": LegacyMultiReferenceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2MultiReferenceLatent": "FLUX.2 Multi Reference Latent",
    "Flux2KleinMultiReferenceLatent": "Multi ReferenceLatent (Legacy)",
}
