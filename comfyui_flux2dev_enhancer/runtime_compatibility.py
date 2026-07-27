"""Runtime guards for third-party FLUX monkey patches.

Some ComfyUI extensions replace ``Flux.forward_orig`` globally during import. Those
replacements can lag behind the current ComfyUI call signature and fail before the
first denoising step. This module installs a lightweight guard around ``Flux._forward``
so the effective replacement is checked immediately before every model call, including
patches installed after this package was imported.
"""

from __future__ import annotations

import functools
from typing import Any

from .architecture import ensure_forward_orig_compatibility

_RUNTIME_GUARD_MARKER = "_flux2dev_runtime_forward_guard"


def install_flux_forward_runtime_guard(flux_class: Any = None) -> bool:
    """Install an idempotent guard around ComfyUI's ``Flux._forward``.

    ``flux_class`` is injectable for unit tests. In ComfyUI, the class is imported
    lazily so this package remains importable in documentation and test environments
    where ComfyUI itself is unavailable.
    """

    if flux_class is None:
        try:
            from comfy.ldm.flux.model import Flux as flux_class
        except (ImportError, ModuleNotFoundError):
            return False

    current_forward = getattr(flux_class, "_forward", None)
    if not callable(current_forward):
        return False
    if getattr(current_forward, _RUNTIME_GUARD_MARKER, False):
        return False

    @functools.wraps(current_forward)
    def guarded_forward(self, *args, **kwargs):
        # Inspect the effective method at call time. This intentionally happens on
        # every call because another custom node may replace ``forward_orig`` after
        # this package has already been imported.
        if ensure_forward_orig_compatibility(self):
            replacement = getattr(self, "forward_orig", None)
            wrapped = getattr(replacement, "__wrapped__", None)
            replacement_name = getattr(
                wrapped or replacement,
                "__name__",
                type(wrapped or replacement).__name__,
            )
            print(
                "[ComfyUI-Flux2Dev-Enhancer] Adapted outdated external "
                f"Flux.forward_orig replacement: {replacement_name}"
            )
        return current_forward(self, *args, **kwargs)

    setattr(guarded_forward, _RUNTIME_GUARD_MARKER, True)
    setattr(flux_class, "_forward", guarded_forward)
    return True


__all__ = ["install_flux_forward_runtime_guard"]
