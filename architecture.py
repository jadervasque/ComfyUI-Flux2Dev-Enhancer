"""Runtime architecture and capability detection for ComfyUI FLUX.2 models.

The module intentionally avoids importing ComfyUI so its pure inspection logic can
be unit-tested with light-weight fakes. Nodes should inspect model objects instead
of checkpoint filenames: quantized and repackaged checkpoints frequently rename the
same underlying architecture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class Flux2CompatibilityError(ValueError):
    """Raised when a model does not expose the FLUX.2 interfaces a node needs."""


@dataclass(frozen=True)
class Flux2Architecture:
    variant: str
    hidden_size: int
    num_heads: int
    double_blocks: int
    single_blocks: int
    context_in_dim: int
    patch_size: int
    guidance_embed: bool
    global_modulation: bool
    default_ref_method: str
    ref_index_scale: float
    supports_attn_input_patch: bool
    supports_attn_output_patch: bool
    supports_post_input_patch: bool
    supports_sampler_post_cfg: bool
    likely_kv_cached: bool = False

    @property
    def is_flux2(self) -> bool:
        return self.variant.startswith("flux2_")

    @property
    def is_klein(self) -> bool:
        return "klein" in self.variant

    @property
    def max_double_block(self) -> int:
        return max(0, self.double_blocks - 1)

    @property
    def max_single_block(self) -> int:
        return max(0, self.single_blocks - 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Official architecture fingerprints from the BFL FLUX.2 reference implementation.
_KNOWN_FINGERPRINTS = {
    (6144, 48, 8, 48, 15360): "flux2_dev",
    (4096, 32, 8, 24, 12288): "flux2_klein_9b",
    (3072, 24, 5, 20, 7680): "flux2_klein_4b",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def unwrap_diffusion_model(model: Any) -> Any:
    """Return the underlying diffusion model from common ComfyUI wrappers."""

    if model is None:
        raise Flux2CompatibilityError("A MODEL input is required.")

    inner = getattr(model, "model", None)
    if inner is not None:
        diffusion = getattr(inner, "diffusion_model", None)
        if diffusion is not None:
            return diffusion

    diffusion = getattr(model, "diffusion_model", None)
    if diffusion is not None:
        return diffusion

    for candidate in (model, inner):
        if candidate is not None and hasattr(candidate, "double_blocks") and hasattr(candidate, "single_blocks"):
            return candidate

    raise Flux2CompatibilityError(
        "Unable to locate the diffusion model. The loader must expose "
        "model.diffusion_model or a compatible FLUX transformer object."
    )


def _read_param(diffusion_model: Any, name: str, fallback: Any = None) -> Any:
    params = getattr(diffusion_model, "params", None)
    if params is not None and hasattr(params, name):
        return getattr(params, name)
    if hasattr(diffusion_model, name):
        return getattr(diffusion_model, name)
    return fallback


def _detect_kv_mode(model: Any, diffusion_model: Any, default_ref_method: str) -> bool:
    if default_ref_method == "index_timestep_zero":
        return True
    names = " ".join(
        str(getattr(obj, "__class__", type(obj)).__name__).lower()
        for obj in (model, getattr(model, "model", None), diffusion_model)
        if obj is not None
    )
    options = getattr(model, "model_options", {}) or {}
    return "kv" in names or bool(options.get("flux_kv_cache"))


def inspect_flux2_architecture(model: Any, *, require_known: bool = False) -> Flux2Architecture:
    """Inspect a ComfyUI model patcher and return a FLUX.2 architecture profile.

    Unknown models are accepted only when they expose FLUX.2 structural signals:
    image channels used by FLUX.2, double/single streams and four-axis positional
    encoding. Callers can set ``require_known`` for stricter validation.
    """

    diffusion = unwrap_diffusion_model(model)
    double_blocks = len(getattr(diffusion, "double_blocks", []) or [])
    single_blocks = len(getattr(diffusion, "single_blocks", []) or [])
    hidden_size = _safe_int(_read_param(diffusion, "hidden_size", 0))
    num_heads = _safe_int(_read_param(diffusion, "num_heads", 0))
    context_in_dim = _safe_int(_read_param(diffusion, "context_in_dim", 0))
    patch_size = _safe_int(_read_param(diffusion, "patch_size", 1), 1)
    global_modulation = bool(_read_param(diffusion, "global_modulation", False))
    guidance_embed = bool(
        _read_param(
            diffusion,
            "guidance_embed",
            _read_param(diffusion, "use_guidance_embed", False),
        )
    )
    default_ref_method = str(_read_param(diffusion, "default_ref_method", "index"))
    ref_index_scale = _safe_float(_read_param(diffusion, "ref_index_scale", 1.0), 1.0)

    fingerprint = (hidden_size, num_heads, double_blocks, single_blocks, context_in_dim)
    variant = _KNOWN_FINGERPRINTS.get(fingerprint)

    axes_dim = _read_param(diffusion, "axes_dim", None)
    in_channels = _safe_int(_read_param(diffusion, "in_channels", 0))
    looks_like_flux2 = (
        double_blocks > 0
        and single_blocks > 0
        and hidden_size > 0
        and num_heads > 0
        and (
            in_channels == 128
            or global_modulation
            or (isinstance(axes_dim, (list, tuple)) and len(axes_dim) == 4)
        )
    )

    if variant is None:
        if require_known:
            raise Flux2CompatibilityError(
                "Unknown FLUX.2 architecture fingerprint: "
                f"hidden={hidden_size}, heads={num_heads}, double={double_blocks}, "
                f"single={single_blocks}, context={context_in_dim}."
            )
        if not looks_like_flux2:
            raise Flux2CompatibilityError(
                "The loaded model does not expose a compatible FLUX.2 double/single-stream architecture."
            )
        variant = "flux2_unknown_compatible"

    likely_kv = _detect_kv_mode(model, diffusion, default_ref_method)
    if variant == "flux2_klein_9b" and likely_kv:
        variant = "flux2_klein_9b_kv"

    return Flux2Architecture(
        variant=variant,
        hidden_size=hidden_size,
        num_heads=num_heads,
        double_blocks=double_blocks,
        single_blocks=single_blocks,
        context_in_dim=context_in_dim,
        patch_size=max(1, patch_size),
        guidance_embed=guidance_embed,
        global_modulation=global_modulation,
        default_ref_method=default_ref_method,
        ref_index_scale=ref_index_scale,
        supports_attn_input_patch=callable(getattr(model, "set_model_attn1_patch", None)),
        supports_attn_output_patch=callable(getattr(model, "set_model_attn1_output_patch", None)),
        supports_post_input_patch=callable(getattr(model, "set_model_post_input_patch", None)),
        supports_sampler_post_cfg=hasattr(model, "model_options"),
        likely_kv_cached=likely_kv,
    )


def require_capabilities(
    model: Any,
    *,
    attn_input: bool = False,
    attn_output: bool = False,
    sampler_post_cfg: bool = False,
) -> Flux2Architecture:
    architecture = inspect_flux2_architecture(model)
    missing: list[str] = []
    if attn_input and not architecture.supports_attn_input_patch:
        missing.append("set_model_attn1_patch")
    if attn_output and not architecture.supports_attn_output_patch:
        missing.append("set_model_attn1_output_patch")
    if sampler_post_cfg and not architecture.supports_sampler_post_cfg:
        missing.append("model_options/sampler_post_cfg_function")
    if missing:
        raise Flux2CompatibilityError(
            "The current model loader does not preserve required ComfyUI patch interfaces: "
            + ", ".join(missing)
            + ". Try a native ComfyUI loader or a loader that preserves model patch hooks."
        )
    return architecture


__all__ = [
    "Flux2Architecture",
    "Flux2CompatibilityError",
    "inspect_flux2_architecture",
    "require_capabilities",
    "unwrap_diffusion_model",
]
