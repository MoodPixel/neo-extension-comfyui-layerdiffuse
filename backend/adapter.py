"""LayerDiffuse external extension adapter for Neo Studio.

Phase 4 scope:
- normalize raw LayerDiffuse UI state
- resolve mode -> workflow template + patch strategy
- validate visible guardrails before workflow mutation
- produce a transparent effective_state + workflow_patch contract

This adapter is intentionally extension-local. It does not import Neo core modules,
mutate base workflows, or execute ComfyUI directly. Neo's global extension runtime
should call `build_execution_plan(...)` and then decide whether/how to apply the
returned workflow_patch.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from .output_metadata import build_metadata_block, build_output_bundle, OUTPUT_CONTRACT_VERSION
    from .asset_library import ASSET_CONTRACT_VERSION
    from .editor_export import EDITOR_EXPORT_CONTRACT_VERSION
    from .graph_wiring import build_comfyui_graph, GRAPH_WIRING_VERSION
except Exception:  # pragma: no cover - allows direct script execution in tests/tools
    import sys
    _backend_dir = Path(__file__).resolve().parent
    if str(_backend_dir) not in sys.path:
        sys.path.insert(0, str(_backend_dir))
    from output_metadata import build_metadata_block, build_output_bundle, OUTPUT_CONTRACT_VERSION
    from asset_library import ASSET_CONTRACT_VERSION
    from editor_export import EDITOR_EXPORT_CONTRACT_VERSION
    from graph_wiring import build_comfyui_graph, GRAPH_WIRING_VERSION
import json

EXTENSION_ID = "image.layerdiffuse"
ADAPTER_VERSION = "layerdiffuse-adapter-v1"
PAYLOAD_CONTRACT_VERSION = "layerdiffuse-payload-contract-v1"
METADATA_CONTRACT_VERSION = "layerdiffuse-metadata-contract-v1"
TEMPLATE_DIR_NAME = "workflow_templates"

SUPPORTED_MODEL_FAMILIES = {"sdxl", "sdxl_sd", "sdxl/sd", "sdxl_sd_family", "sd", "sd15", "sd1.5", "sd_1_5", "sd1_5"}
SD15_ALIASES = {"sd", "sd15", "sd1.5", "sd_1_5", "sd1_5"}
SDXL_ALIASES = {"sdxl", "sdxl_sd", "sdxl/sd", "sdxl_sd_family", "sdxl-base", "sdxl_base"}
SUPPORTED_WORKFLOWS = {"txt2img", "img2img"}
SUPPORTED_OUTPUT_POLICIES = {"preview", "new_run", "append", "replace"}
SUPPORTED_DECODE_MODES = {"rgba", "split", "preview_only"}

DEFAULT_STATE: dict[str, Any] = {
    "enabled": False,
    "mode": "transparent_asset",
    "source_type": "prompt",
    "source_image_id": None,
    "background_image_id": None,
    "foreground_image_id": None,
    "decode_mode": "rgba",
    "output_policy": "new_run",
    "replace_target_id": None,
    "save_rgba": True,
    "save_rgb": False,
    "save_alpha": True,
    "save_metadata": True,
    "compatibility_mode": "auto",
}

MODE_CONFIG: dict[str, dict[str, Any]] = {
    "transparent_asset": {
        "template": {"sdxl": "transparent_asset_sdxl.json", "sd15": "transparent_asset_sd15.json"},
        "fallback_template": "transparent_asset_sdxl.json",
        "strategy": "replace_workflow",
        "decode_mode": "rgba",
        "requires_prompt": True,
        "requires": [],
        "outputs": ["rgba_image", "preview_image"],
    },
    "rgb_alpha_split": {
        "template": {"sdxl": "transparent_asset_split_sdxl.json"},
        "fallback_template": "transparent_asset_split_sdxl.json",
        "strategy": "replace_workflow",
        "decode_mode": "split",
        "requires_prompt": True,
        "requires": [],
        "outputs": ["rgba_image", "rgb_image", "alpha_mask", "preview_image"],
    },
    "foreground_on_background": {
        "template": {"sdxl": "foreground_on_background_sdxl.json"},
        "fallback_template": "foreground_on_background_sdxl.json",
        "strategy": "replace_workflow",
        "decode_mode": "rgba",
        "requires_prompt": True,
        "requires": ["background_image_id"],
        "outputs": ["rgba_image", "preview_image"],
    },
    "background_aware_blend": {
        "template": {"sdxl": "background_aware_blend_sdxl.json"},
        "fallback_template": "background_aware_blend_sdxl.json",
        "strategy": "sidecar_run",
        "decode_mode": "preview_only",
        "requires_prompt": True,
        "requires": ["foreground_image_id", "background_image_id"],
        "outputs": ["blended_image", "preview_image"],
    },
    "extract_foreground": {
        "template": {"sdxl": "extract_foreground_from_composite_sdxl.json"},
        "fallback_template": "extract_foreground_from_composite_sdxl.json",
        "strategy": "sidecar_run",
        "decode_mode": "split",
        "requires_prompt": False,
        "requires": ["source_image_id", "background_image_id"],
        "outputs": ["rgba_image", "rgb_image", "alpha_mask", "preview_image"],
    },
    "overlay_fx": {
        "template": {"sdxl": "overlay_fx_transparent_sdxl.json"},
        "fallback_template": "overlay_fx_transparent_sdxl.json",
        "strategy": "replace_workflow",
        "decode_mode": "split",
        "requires_prompt": True,
        "requires": [],
        "outputs": ["rgba_image", "rgb_image", "alpha_mask", "preview_image"],
    },
}

REQUIRED_NODE_NAMES = [
    "LayeredDiffusionApply",
    "LayeredDiffusionJointApply",
    "LayeredDiffusionCondApply",
    "LayeredDiffusionCondJointApply",
    "LayeredDiffusionDiffApply",
    "LayeredDiffusionDecode",
    "LayeredDiffusionDecodeRGBA",
    "LayeredDiffusionDecodeSplit",
]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _key(value: Any) -> str:
    return _clean(value).lower().replace(" ", "_")


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _none_if_blank(value: Any) -> str | None:
    text = _clean(value)
    return text or None


def extension_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {"_invalid_json": str(path)}


def normalize_model_family(context: Mapping[str, Any] | None, compatibility_mode: str = "auto") -> str:
    context = _as_dict(context)
    requested = _key(compatibility_mode)
    if requested == "sd15":
        return "sd15"
    if requested == "sdxl":
        return "sdxl"
    family = _key(context.get("model_family") or context.get("family") or context.get("generation_family") or "sdxl")
    if family in SD15_ALIASES:
        return "sd15"
    if family in SDXL_ALIASES or family == "":
        return "sdxl"
    return family


def normalize_raw_state(raw_state: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = deepcopy(DEFAULT_STATE)
    incoming = _as_dict(raw_state)
    raw.update({k: v for k, v in incoming.items() if k in DEFAULT_STATE or k in {"replace_confirmed", "identity_context_confirmed"}})

    raw["enabled"] = _bool(raw.get("enabled"), False)
    raw["mode"] = _key(raw.get("mode")) or DEFAULT_STATE["mode"]
    if raw["mode"] not in MODE_CONFIG:
        raw["mode"] = DEFAULT_STATE["mode"]
        raw.setdefault("_normalization_warnings", []).append("unknown_mode_reset_to_transparent_asset")

    raw["source_type"] = _key(raw.get("source_type")) or DEFAULT_STATE["source_type"]
    if raw["source_type"] == "output":
        raw["source_type"] = "previous_output"
    if raw["source_type"] not in {"prompt", "selected_image", "upload", "previous_output"}:
        raw["source_type"] = DEFAULT_STATE["source_type"]
        raw.setdefault("_normalization_warnings", []).append("unknown_source_type_reset_to_prompt")

    raw["decode_mode"] = _key(raw.get("decode_mode")) or MODE_CONFIG[raw["mode"]]["decode_mode"]
    if raw["decode_mode"] not in SUPPORTED_DECODE_MODES:
        raw["decode_mode"] = MODE_CONFIG[raw["mode"]]["decode_mode"]
        raw.setdefault("_normalization_warnings", []).append("unknown_decode_mode_reset_to_mode_default")

    raw["output_policy"] = _key(raw.get("output_policy")) or DEFAULT_STATE["output_policy"]
    if raw["output_policy"] not in SUPPORTED_OUTPUT_POLICIES:
        raw["output_policy"] = DEFAULT_STATE["output_policy"]
        raw.setdefault("_normalization_warnings", []).append("unknown_output_policy_reset_to_new_run")

    raw["compatibility_mode"] = _key(raw.get("compatibility_mode")) or DEFAULT_STATE["compatibility_mode"]
    if raw["compatibility_mode"] not in {"auto", "sdxl", "sd15"}:
        raw["compatibility_mode"] = "auto"

    for key_name in ("source_image_id", "background_image_id", "foreground_image_id", "replace_target_id"):
        raw[key_name] = _none_if_blank(raw.get(key_name))
    for key_name in ("save_rgba", "save_rgb", "save_alpha", "save_metadata"):
        raw[key_name] = _bool(raw.get(key_name), bool(DEFAULT_STATE[key_name]))
    if "replace_confirmed" in raw:
        raw["replace_confirmed"] = _bool(raw.get("replace_confirmed"), False)
    return raw


def resolve_template_id(mode: str, model_family: str) -> str:
    mode_cfg = MODE_CONFIG.get(mode) or MODE_CONFIG[DEFAULT_STATE["mode"]]
    template_map = mode_cfg.get("template") or {}
    return template_map.get(model_family) or mode_cfg.get("fallback_template")


def template_path_for(template_id: str, extension_root: Path | None = None) -> Path:
    root = extension_root or extension_root_from_file()
    return root / TEMPLATE_DIR_NAME / template_id


def load_template(template_id: str, extension_root: Path | None = None) -> dict[str, Any]:
    return load_json(template_path_for(template_id, extension_root))


def _context_has_prompt(context: Mapping[str, Any]) -> bool:
    return bool(_clean(context.get("prompt") or context.get("positive") or context.get("positive_prompt")))


def _context_workflow(context: Mapping[str, Any]) -> str:
    return _key(context.get("workflow") or context.get("workflow_type") or context.get("mode") or "txt2img") or "txt2img"


def validate_raw_state(raw_state: Mapping[str, Any] | None, context: Mapping[str, Any] | None = None, *, extension_root: Path | None = None) -> dict[str, Any]:
    raw = normalize_raw_state(raw_state)
    context = _as_dict(context)
    errors: list[str] = []
    warnings: list[str] = list(raw.pop("_normalization_warnings", []))
    auto_fixes: list[str] = []

    if not raw["enabled"]:
        return {
            "ok": True,
            "blocked": False,
            "disabled_reason": "extension_disabled",
            "warnings": warnings,
            "errors": [],
            "auto_fixes": auto_fixes,
            "raw_state": raw,
        }

    workflow = _context_workflow(context)
    if workflow not in SUPPORTED_WORKFLOWS:
        errors.append(f"unsupported_workflow:{workflow}")

    model_family = normalize_model_family(context, raw["compatibility_mode"])
    if model_family not in {"sdxl", "sd15"}:
        errors.append(f"unsupported_model_family:{model_family}")

    mode_cfg = MODE_CONFIG[raw["mode"]]
    template_id = resolve_template_id(raw["mode"], model_family)
    template = load_template(template_id, extension_root)
    if not template or template.get("_invalid_json"):
        errors.append(f"workflow_template_missing_or_invalid:{template_id}")
    elif template.get("status") == "mapping_only":
        warnings.append("workflow_template_mapping_only:not_executable_until_socket_mapping_phase")
    elif template.get("status") in {"executable", "verified_executable"} and not (template.get("graph") or {}).get("nodes"):
        errors.append(f"workflow_template_executable_declared_but_graph_nodes_missing:{template_id}")

    if mode_cfg.get("requires_prompt") and not _context_has_prompt(context):
        errors.append("prompt_required")

    for required_key in mode_cfg.get("requires", []):
        if not raw.get(required_key):
            errors.append(f"required_source_missing:{required_key}")

    if template and template.get("status") == "blocked_until_exported_workflow_mapping":
        errors.append(f"workflow_template_not_executable_until_exported_mapping:{template_id}")

    if raw["output_policy"] == "replace" and not raw.get("replace_target_id"):
        errors.append("replace_target_required")
    if raw["output_policy"] == "replace" and not raw.get("replace_confirmed", False):
        errors.append("replace_requires_visible_confirmation")

    batch_size = int(context.get("batch_size") or context.get("batch") or 1)
    if batch_size > 1:
        warnings.append("batch_force_1_requires_visible_clamp")
        auto_fixes.append("effective_batch_size=1")

    if raw["decode_mode"] != mode_cfg.get("decode_mode"):
        warnings.append(f"decode_mode_auto_fixed:{raw['decode_mode']}->{mode_cfg.get('decode_mode')}")
        raw["decode_mode"] = mode_cfg.get("decode_mode")
        auto_fixes.append("decode_mode=mode_default")

    return {
        "ok": not errors,
        "blocked": bool(errors),
        "disabled_reason": errors[0] if errors else None,
        "warnings": warnings,
        "errors": errors,
        "auto_fixes": auto_fixes,
        "raw_state": raw,
    }


def build_effective_state(raw_state: Mapping[str, Any] | None, context: Mapping[str, Any] | None = None, *, extension_root: Path | None = None) -> dict[str, Any]:
    context = _as_dict(context)
    validation = validate_raw_state(raw_state, context, extension_root=extension_root)
    raw = validation["raw_state"]
    model_family = normalize_model_family(context, raw.get("compatibility_mode", "auto"))
    mode_cfg = MODE_CONFIG[raw["mode"]]
    template_id = resolve_template_id(raw["mode"], model_family)
    template = load_template(template_id, extension_root)
    output_types = [item.get("type") for item in template.get("outputs", []) if isinstance(item, dict)] if template else []
    if not output_types:
        output_types = list(mode_cfg.get("outputs") or [])

    effective_enabled = bool(raw.get("enabled")) and validation["ok"]
    return {
        "extension_id": EXTENSION_ID,
        "adapter_version": ADAPTER_VERSION,
        "enabled": bool(raw.get("enabled")),
        "effective_enabled": effective_enabled,
        "active": effective_enabled,
        "mode": raw["mode"],
        "workflow": _context_workflow(context),
        "model_family": model_family,
        "batch_size": 1,
        "workflow_template": template_id,
        "template_status": template.get("status") or "missing",
        "patch_strategy": mode_cfg["strategy"],
        "decode_mode": raw["decode_mode"],
        "output_policy": raw["output_policy"],
        "source_resolved": {
            "type": raw.get("source_type"),
            "source_image_id": raw.get("source_image_id"),
            "background_image_id": raw.get("background_image_id"),
            "foreground_image_id": raw.get("foreground_image_id"),
            "replace_target_id": raw.get("replace_target_id"),
        },
        "save": {
            "rgba": bool(raw.get("save_rgba")),
            "rgb": bool(raw.get("save_rgb")),
            "alpha": bool(raw.get("save_alpha")),
            "metadata": bool(raw.get("save_metadata")),
        },
        "outputs_expected": output_types,
        "required_nodes": template.get("required_nodes") or REQUIRED_NODE_NAMES,
        "validation": {
            "ok": validation["ok"],
            "blocked": validation["blocked"],
            "disabled_reason": validation["disabled_reason"],
            "warnings": validation["warnings"],
            "errors": validation["errors"],
            "auto_fixes": validation["auto_fixes"],
        },
    }


def build_workflow_patch(raw_state: Mapping[str, Any] | None, context: Mapping[str, Any] | None = None, *, extension_root: Path | None = None) -> dict[str, Any]:
    raw = normalize_raw_state(raw_state)
    effective = build_effective_state(raw, context, extension_root=extension_root)
    template = load_template(effective["workflow_template"], extension_root)
    graph_package = build_comfyui_graph(raw, context, effective) if effective.get("effective_enabled") else {}
    return {
        "extension_id": EXTENSION_ID,
        "adapter_version": ADAPTER_VERSION,
        "enabled": effective["effective_enabled"],
        "blocked": not effective["effective_enabled"] and bool(raw.get("enabled")),
        "strategy": effective["patch_strategy"],
        "template": effective["workflow_template"],
        "template_status": effective["template_status"],
        "mapping_only": effective["template_status"] not in {"executable", "verified_executable"},
        "engine": "comfyui",
        "graph_wiring_version": GRAPH_WIRING_VERSION,
        "comfyui_graph": graph_package,
        "graph": graph_package.get("graph") if graph_package.get("executable") else {},
        "primary_output_type": graph_package.get("primary_output_type"),
        "output_bindings": graph_package.get("output_bindings") or {},
        "bindings": template.get("bindings") or {},
        "required_nodes": effective["required_nodes"],
        "inputs": {
            "prompt": "$context.prompt",
            "negative_prompt": "$context.negative_prompt",
            "model": "$context.model",
            "source_image_id": raw.get("source_image_id"),
            "background_image_id": raw.get("background_image_id"),
            "foreground_image_id": raw.get("foreground_image_id"),
            "decode_mode": effective["decode_mode"],
            "output_policy": effective["output_policy"],
        },
        "outputs_expected": effective["outputs_expected"],
        "output_policy": effective["output_policy"],
        "validation": effective["validation"],
        "notes": [
            "Phase 12 syncs prompt-driven ComfyUI graph wiring to the verified standalone LayerDiffuse RGBA workflow.",
            "Primary output must be the LayeredDiffusionDecodeRGBA SaveImage node, not the plain VAEDecode preview.",
            "Background/composite modes remain blocked until a working exported local ComfyUI workflow/object_info is mapped.",
        ],
    }


def build_payload_entry(raw_state: Mapping[str, Any] | None, context: Mapping[str, Any] | None = None, *, extension_root: Path | None = None) -> dict[str, Any]:
    """Build the canonical Neo external_extensions entry for LayerDiffuse.

    The entry is transparent by design:
    - raw_state preserves user/UI selections after type normalization only
    - effective_state records the backend-resolved executable intent
    - workflow_patch is visible but not applied by this adapter
    - invalid enabled states remain present and resolve effective_enabled=false
    """
    raw = normalize_raw_state(raw_state)
    effective = build_effective_state(raw, context, extension_root=extension_root)
    patch = build_workflow_patch(raw, context, extension_root=extension_root)
    validation = deepcopy(effective.get("validation") or {})
    return {
        "extension_id": EXTENSION_ID,
        "enabled": bool(raw.get("enabled")),
        "effective_enabled": bool(effective.get("effective_enabled")),
        "payload_contract_version": PAYLOAD_CONTRACT_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "source": raw.get("source_type"),
        "source_policy": raw.get("source_type"),
        "output_policy": raw.get("output_policy"),
        "batch_policy": "force_1",
        "context_policy": ["prompt", "model", "image", "metadata"],
        "target_sections": ["extensions", "output", "assets"],
        "raw_state": raw,
        "effective_state": effective,
        "workflow_patch": patch,
        "workflow_validation": validation,
        "warnings": list(validation.get("warnings") or []),
        "disabled_reason": validation.get("disabled_reason"),
        "transparency": {
            "raw_state_mutated": False,
            "effective_state_separate": True,
            "hidden_mutations_allowed": False,
            "extension_executes_workflow_directly": False,
        },
    }


def build_payload_block(raw_state: Mapping[str, Any] | None, context: Mapping[str, Any] | None = None, *, extension_root: str | Path | None = None) -> dict[str, Any]:
    root_path = Path(extension_root) if extension_root else extension_root_from_file()
    return {EXTENSION_ID: build_payload_entry(raw_state, context, extension_root=root_path)}


def build_execution_plan(raw_state: Mapping[str, Any] | None, context: Mapping[str, Any] | None = None, *, extension_root: str | Path | None = None) -> dict[str, Any]:
    root_path = Path(extension_root) if extension_root else extension_root_from_file()
    raw = normalize_raw_state(raw_state)
    effective = build_effective_state(raw, context, extension_root=root_path)
    patch = build_workflow_patch(raw, context, extension_root=root_path)
    payload_entry = build_payload_entry(raw, context, extension_root=root_path)
    payload_block = {EXTENSION_ID: payload_entry}
    return {
        "extension_id": EXTENSION_ID,
        "adapter_version": ADAPTER_VERSION,
        "payload_contract_version": PAYLOAD_CONTRACT_VERSION,
        "raw_state": raw,
        "effective_state": effective,
        "workflow_patch": patch,
        "payload_entry": payload_entry,
        "payload_block": payload_block,
        "metadata": build_run_metadata(raw, effective, patch),
    }


def build_run_metadata(
    raw_state: Mapping[str, Any],
    effective_state: Mapping[str, Any],
    workflow_patch: Mapping[str, Any],
    *,
    run_id: str | None = None,
    base_dir: str = "layerdiffuse_outputs",
    produced_files: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build Neo run metadata with a Phase 8 output bundle.

    The metadata keeps raw_state and effective_state separate, records the workflow
    patch, and declares saveable RGBA/RGB/alpha/preview outputs without touching
    files directly. Actual file writes remain owned by Neo's output collector.
    """
    block = build_metadata_block(
        raw_state,
        effective_state,
        workflow_patch,
        run_id=run_id,
        base_dir=base_dir,
        produced_files=produced_files,
    )
    entry = block.setdefault("_neo_external_extensions", {}).setdefault(EXTENSION_ID, {})
    entry.update({
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "payload_contract_version": PAYLOAD_CONTRACT_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "asset_contract_version": ASSET_CONTRACT_VERSION,
        "editor_export_contract_version": EDITOR_EXPORT_CONTRACT_VERSION,
        "enabled": bool(raw_state.get("enabled")),
        "effective_enabled": bool(effective_state.get("effective_enabled", effective_state.get("active"))),
        "template_status": effective_state.get("template_status"),
        "outputs_expected": list(effective_state.get("outputs_expected") or []),
        "validation": deepcopy(effective_state.get("validation") or {}),
        "mapping_only": bool(workflow_patch.get("mapping_only")),
        "policy": "transparent_raw_effective_no_hidden_mutation",
        "phase11_policy": "transparent_raw_effective_output_bundle_asset_manifest_real_rgba_graph_no_hidden_mutation",
    })
    return block


def get_capabilities() -> dict[str, Any]:
    return {
        "extension_id": EXTENSION_ID,
        "adapter_version": ADAPTER_VERSION,
        "modes": sorted(MODE_CONFIG.keys()),
        "supported_workflows": sorted(SUPPORTED_WORKFLOWS),
        "supported_model_families": sorted(SUPPORTED_MODEL_FAMILIES),
        "output_policies": sorted(SUPPORTED_OUTPUT_POLICIES),
        "decode_modes": sorted(SUPPORTED_DECODE_MODES),
        "payload_contract_version": PAYLOAD_CONTRACT_VERSION,
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "asset_contract_version": ASSET_CONTRACT_VERSION,
        "editor_export_contract_version": EDITOR_EXPORT_CONTRACT_VERSION,
        "editor_export_presets": ["after_effects_png_bundle", "transparent_overlay_pack", "thumbnail_poster_asset_pack"],
        "required_nodes": REQUIRED_NODE_NAMES,
        "template_dir": TEMPLATE_DIR_NAME,
        "graph_wiring_version": GRAPH_WIRING_VERSION,
        "prompt_driven_executable_modes": ["transparent_asset", "rgb_alpha_split", "overlay_fx"],
    }


@dataclass(frozen=True)
class AdapterResult:
    """Optional typed wrapper for tests/tools that prefer object-style results."""

    raw_state: dict[str, Any]
    effective_state: dict[str, Any]
    workflow_patch: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_state": deepcopy(self.raw_state),
            "effective_state": deepcopy(self.effective_state),
            "workflow_patch": deepcopy(self.workflow_patch),
            "metadata": deepcopy(self.metadata),
        }
