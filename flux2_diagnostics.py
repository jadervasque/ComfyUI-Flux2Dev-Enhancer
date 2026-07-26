"""Diagnostic nodes for validating FLUX.2 loader compatibility."""

from __future__ import annotations

import json

from .architecture import inspect_flux2_architecture


class Flux2ArchitectureInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {"print_to_console": ("BOOLEAN", {"default": True})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("architecture_report",)
    FUNCTION = "inspect"
    CATEGORY = "conditioning/flux2/diagnostics"
    OUTPUT_NODE = True

    def inspect(self, model, print_to_console=True):
        architecture = inspect_flux2_architecture(model)
        report = json.dumps(architecture.to_dict(), indent=2, sort_keys=True)
        if print_to_console:
            print(f"[Flux2ArchitectureInspector]\n{report}")
        return (report,)


NODE_CLASS_MAPPINGS = {"Flux2ArchitectureInspector": Flux2ArchitectureInspector}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2ArchitectureInspector": "FLUX.2 Architecture Inspector"
}
