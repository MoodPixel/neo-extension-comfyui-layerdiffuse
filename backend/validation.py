from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

EXTENSION_ID = "image.layerdiffuse"
VALIDATION_VERSION = "layerdiffuse-workflow-validation-v1"

SUPPORTED_WORKFLOWS = {"txt2img", "img2img"}
SUPPORTED_MODEL_FAMILIES = {"sdxl", "sd", "sd15", "sd1.5", "sd_1_5", "sd1_5", "sdxl_sd", "sdxl/sd", "sdxl_sd_family"}
WARN_MODEL_FAMILIES = {"flux", "qwen", "qwen-image", "zimage", "unknown", ""}

MODE_TEMPLATES = {
    "transparent_asset": "transparent_asset_sdxl.json",
    "rgb_alpha_split": "transparent_asset_split_sdxl.json",
    "foreground_on_background": "foreground_on_background_sdxl.json",
    "background_aware_blend": "background_aware_blend_sdxl.json",
    "extract_foreground": "extract_foreground_from_composite_sdxl.json",
    "overlay_fx": "overlay_fx_transparent_sdxl.json",
}

MODE_REQUIREMENTS = {
    "transparent_asset": {
        "requires_prompt": True,
        "required_images": [],
        "recommended_decode": "rgba",
        "outputs_expected": ["rgba_image", "alpha_mask", "preview_image"],
        "patch_strategy": "replace_workflow",
    },
    "rgb_alpha_split": {
        "requires_prompt": True,
        "required_images": [],
        "recommended_decode": "split",
        "outputs_expected": ["rgba_image", "rgb_image", "alpha_mask", "preview_image"],
        "patch_strategy": "replace_workflow",
    },
    "foreground_on_background": {
        "requires_prompt": True,
        "required_images": ["background_image_id"],
        "recommended_decode": "rgba",
        "outputs_expected": ["rgba_image", "alpha_mask", "preview_image"],
        "patch_strategy": "replace_workflow",
    },
    "background_aware_blend": {
        "requires_prompt": True,
        "required_images": ["foreground_image_id", "background_image_id"],
        "recommended_decode": "preview_only",
        "outputs_expected": ["preview_image", "composited_image"],
        "patch_strategy": "sidecar_run",
    },
    "extract_foreground": {
        "requires_prompt": False,
        "required_images": ["source_image_id", "background_image_id"],
        "recommended_decode": "split",
        "outputs_expected": ["rgba_image", "rgb_image", "alpha_mask", "preview_image"],
        "patch_strategy": "sidecar_run",
    },
    "overlay_fx": {
        "requires_prompt": True,
        "required_images": [],
        "recommended_decode": "split",
        "outputs_expected": ["rgba_image", "alpha_mask", "preview_image"],
        "patch_strategy": "replace_workflow",
    },
}

REQUIRED_NODES = [
    "LayeredDiffusionApply",
    "LayeredDiffusionJointApply",
    "LayeredDiffusionCondApply",
    "LayeredDiffusionCondJointApply",
    "LayeredDiffusionDiffApply",
    "LayeredDiffusionDecode",
    "LayeredDiffusionDecodeRGBA",
    "LayeredDiffusionDecodeSplit",
]


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _family(value: Any) -> str:
    return str(value or "unknown").strip().lower()


@dataclass
class ValidationResult:
    valid: bool = True
    status: str = "valid"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    autofixes: list[dict[str, Any]] = field(default_factory=list)
    effective_state: dict[str, Any] = field(default_factory=dict)
    disabled_reason: str | None = None
    contract_version: str = VALIDATION_VERSION

    def block(self, message: str) -> None:
        if message not in self.errors:
            self.errors.append(message)
        self.valid = False
        self.status = "blocked"
        self.disabled_reason = self.disabled_reason or message

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
        if self.status == "valid":
            self.status = "warning"

    def autofix(self, key: str, before: Any, after: Any, reason: str) -> None:
        self.autofixes.append({"key": key, "before": before, "after": after, "reason": reason})
        self.warn(f"Auto-fix: {reason}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_raw_state(raw_state: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(raw_state or {})
    return {
        "enabled": _truthy(raw.get("enabled", False)),
        "mode": _clean(raw.get("mode")) or "transparent_asset",
        "source_type": _clean(raw.get("source_type")) or "prompt",
        "source_image_id": _clean(raw.get("source_image_id")),
        "background_image_id": _clean(raw.get("background_image_id")),
        "foreground_image_id": _clean(raw.get("foreground_image_id")),
        "decode_mode": _clean(raw.get("decode_mode")) or "rgba",
        "output_policy": _clean(raw.get("output_policy")) or "new_run",
        "replace_target_id": _clean(raw.get("replace_target_id")),
        "replace_confirmed": _truthy(raw.get("replace_confirmed", False)),
        "save_rgba": _truthy(raw.get("save_rgba", True)),
        "save_rgb": _truthy(raw.get("save_rgb", False)),
        "save_alpha": _truthy(raw.get("save_alpha", True)),
        "save_metadata": _truthy(raw.get("save_metadata", True)),
        "compatibility_mode": _clean(raw.get("compatibility_mode")) or "auto",
    }


def validate_layerdiffuse_workflow(
    raw_state: Mapping[str, Any] | None,
    *,
    workflow: str | None = None,
    model_family: str | None = None,
    batch_size: int | None = None,
    available_nodes: list[str] | None = None,
    available_templates: list[str] | None = None,
    execution_enabled: bool = True,
) -> dict[str, Any]:
    """Validate LayerDiffuse extension state before workflow execution.

    This function is extension-local. It does not mutate Neo's core workflow graph.
    It returns a transparent result with blocked states, warnings, visible auto-fixes,
    and the effective state Neo may use in later execution phases.
    """
    raw = normalize_raw_state(raw_state)
    result = ValidationResult()

    if not raw["enabled"]:
        result.status = "disabled"
        result.effective_state = {"extension_id": EXTENSION_ID, "active": False, "raw_state": raw}
        return result.to_dict()

    mode = raw["mode"]
    if mode not in MODE_REQUIREMENTS:
        result.block(f"Unsupported LayerDiffuse mode: {mode}")
        requirements = None
    else:
        requirements = MODE_REQUIREMENTS[mode]

    workflow_name = (workflow or "txt2img").strip().lower()
    if workflow_name not in SUPPORTED_WORKFLOWS:
        result.block(f"LayerDiffuse does not support workflow '{workflow_name}'. Supported: txt2img, img2img.")

    family = _family(model_family)
    if family not in SUPPORTED_MODEL_FAMILIES:
        if family in WARN_MODEL_FAMILIES or family.startswith("flux") or family.startswith("qwen"):
            result.warn(f"Model family '{family}' is not confirmed for LayerDiffuse. Use SDXL/SD1.5-compatible templates first.")
        else:
            result.warn(f"Unknown/unsupported model family '{family}'. Validation will allow UI state but runtime may block execution.")

    requested_batch = int(batch_size or 1)
    effective_batch = requested_batch
    if requested_batch > 1:
        effective_batch = 1
        result.autofix("batch_size", requested_batch, 1, "LayerDiffuse enforces batch size 1 for transparent RGBA output mapping.")

    if requirements:
        for key in requirements["required_images"]:
            if not raw.get(key):
                result.block(f"{key} is required for mode '{mode}'.")
        recommended_decode = requirements["recommended_decode"]
        if raw["decode_mode"] != recommended_decode:
            result.autofix("decode_mode", raw["decode_mode"], recommended_decode, f"Mode '{mode}' should use decode mode '{recommended_decode}'.")
            raw["decode_mode"] = recommended_decode

    if raw["output_policy"] == "replace":
        if not raw["replace_target_id"]:
            result.block("Replace output policy requires replace_target_id.")
        if not raw["replace_confirmed"]:
            result.block("Replace output policy requires visible replace_confirmed=true.")

    if not raw["save_metadata"]:
        result.warn("Metadata save is disabled; traceability will be reduced.")

    template = MODE_TEMPLATES.get(mode)
    if not template:
        result.block(f"No workflow template mapping exists for mode '{mode}'.")
    elif available_templates is not None and template not in set(available_templates):
        result.block(f"Missing workflow template: {template}")

    if available_nodes is not None:
        missing = [node for node in REQUIRED_NODES if node not in set(available_nodes)]
        if missing:
            result.block("Missing LayerDiffuse ComfyUI nodes: " + ", ".join(missing))

    if not execution_enabled:
        result.block("LayerDiffuse execution is disabled by config; executable Phase 11 graph wiring requires execution_enabled=true.")

    result.effective_state = {
        "extension_id": EXTENSION_ID,
        "active": bool(raw["enabled"] and result.valid),
        "mode": mode,
        "workflow": workflow_name,
        "model_family": family,
        "workflow_template": template,
        "patch_strategy": requirements["patch_strategy"] if requirements else None,
        "batch_size": effective_batch,
        "batch_policy": "force_1",
        "source_resolved": {
            "type": raw["source_type"],
            "source_image_id": raw["source_image_id"],
            "background_image_id": raw["background_image_id"],
            "foreground_image_id": raw["foreground_image_id"],
        },
        "decode_mode": raw["decode_mode"],
        "output_policy": raw["output_policy"],
        "replace_target_id": raw["replace_target_id"],
        "outputs_expected": requirements["outputs_expected"] if requirements else [],
        "raw_state": raw,
        "validation_contract": VALIDATION_VERSION,
        "hidden_mutations_allowed": False,
    }
    return result.to_dict()
