"""Executable ComfyUI graph wiring for the LayerDiffuse external extension.

Phase 12 scope:
- Sync Neo prompt-driven templates to the verified standalone ComfyUI SDXL RGBA workflow.
- Use the official LayeredDiffusionApply -> KSampler -> VAEDecode -> LayeredDiffusionDecodeRGBA chain.
- Save the RGBA decode as the primary output instead of the plain RGB preview.
- Keep RGB preview/split outputs as explicit sidecars.

This module is extension-local. It does not patch Neo core. Neo's external
workflow runtime may call `build_comfyui_graph(...)` from the workflow patch.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

GRAPH_WIRING_VERSION = "layerdiffuse-verified-comfy-graph-v2"
EXTENSION_ID = "image.layerdiffuse"
VERIFIED_REFERENCE_WORKFLOW = "layer_diffusion_fg_example_rgba.json"
VERIFIED_REFERENCE_CHAIN = [
    "CheckpointLoaderSimple",
    "LayeredDiffusionApply",
    "KSampler",
    "VAEDecode",
    "LayeredDiffusionDecodeRGBA",
    "SaveImage",
]

PROMPT_DRIVEN_EXECUTABLE_MODES = {"transparent_asset", "rgb_alpha_split", "overlay_fx"}
BACKGROUND_MODES_NEED_OBJECT_INFO = {"foreground_on_background", "background_aware_blend", "extract_foreground"}

DEFAULTS: dict[str, Any] = {
    "ckpt_name": "sd_xl_base_1.0.safetensors",
    "width": 1024,
    "height": 1024,
    "batch_size": 1,
    "seed": 0,
    "steps": 28,
    "cfg": 5.0,
    "sampler_name": "dpmpp_2m_sde",
    "scheduler": "karras",
    "denoise": 1.0,
    "negative_prompt": "solid background, gray background, white background, black background, scenery, watermark, text, blurry, low quality",
    "layerdiffuse_weight": 1.0,
    "sub_batch_size": 16,
}


def _clean(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text or fallback


def _num(value: Any, fallback: int | float) -> int | float:
    try:
        if isinstance(fallback, int) and not isinstance(fallback, bool):
            return int(value)
        return float(value)
    except Exception:
        return fallback


def _multiple_of_64(value: Any, fallback: int) -> int:
    try:
        ivalue = int(value)
    except Exception:
        ivalue = fallback
    ivalue = max(64, ivalue)
    return int(round(ivalue / 64.0) * 64)


def sd_version_for(model_family: str | None) -> str:
    family = _clean(model_family, "sdxl").lower().replace(" ", "_")
    if family in {"sd", "sd15", "sd1.5", "sd_1_5", "sd1_5"}:
        return "SD1x"
    return "SDXL"


def layerdiffuse_config_for(mode: str, model_family: str | None) -> str:
    sd_version = sd_version_for(model_family)
    if sd_version == "SD1x":
        return "SD15, Attention Injection, attn_sharing"
    # Phase 12 sync: the verified standalone ComfyUI workflow from ComfyUI-layerdiffuse
    # uses SDXL Conv Injection for foreground RGBA generation. Keep this default so Neo
    # matches the known-good graph before adding advanced per-mode tuning.
    return "SDXL, Conv Injection"


def context_value(context: Mapping[str, Any] | None, *keys: str, fallback: Any = None) -> Any:
    context = context or {}
    for key in keys:
        value = context.get(key)
        if value not in (None, ""):
            return value
    return fallback


def normalized_context(context: Mapping[str, Any] | None, effective_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    context = dict(context or {})
    effective_state = dict(effective_state or {})
    model_family = _clean(effective_state.get("model_family") or context_value(context, "model_family", "family", fallback="sdxl"), "sdxl")
    width = _multiple_of_64(context_value(context, "width", "W", fallback=DEFAULTS["width"]), DEFAULTS["width"])
    height = _multiple_of_64(context_value(context, "height", "H", fallback=DEFAULTS["height"]), DEFAULTS["height"])
    return {
        "ckpt_name": _clean(context_value(context, "ckpt_name", "checkpoint", "model", fallback=DEFAULTS["ckpt_name"]), DEFAULTS["ckpt_name"]),
        "positive_prompt": _clean(context_value(context, "prompt", "positive_prompt", "positive", fallback="")),
        "negative_prompt": _clean(context_value(context, "negative_prompt", "negative", fallback=DEFAULTS["negative_prompt"]), DEFAULTS["negative_prompt"]),
        "width": width,
        "height": height,
        "batch_size": 1,
        "seed": int(_num(context_value(context, "seed", fallback=DEFAULTS["seed"]), DEFAULTS["seed"])),
        "steps": int(_num(context_value(context, "steps", fallback=DEFAULTS["steps"]), DEFAULTS["steps"])),
        "cfg": float(_num(context_value(context, "cfg", "cfg_scale", fallback=DEFAULTS["cfg"]), DEFAULTS["cfg"])),
        "sampler_name": _clean(context_value(context, "sampler_name", "sampler", fallback=DEFAULTS["sampler_name"]), DEFAULTS["sampler_name"]),
        "scheduler": _clean(context_value(context, "scheduler", fallback=DEFAULTS["scheduler"]), DEFAULTS["scheduler"]),
        "denoise": float(_num(context_value(context, "denoise", "denoising_strength", fallback=DEFAULTS["denoise"]), DEFAULTS["denoise"])),
        "sd_version": sd_version_for(model_family),
        "model_family": model_family,
        "sub_batch_size": int(_num(context_value(context, "sub_batch_size", fallback=DEFAULTS["sub_batch_size"]), DEFAULTS["sub_batch_size"])),
        "layerdiffuse_weight": float(_num(context_value(context, "layerdiffuse_weight", fallback=DEFAULTS["layerdiffuse_weight"]), DEFAULTS["layerdiffuse_weight"])),
    }


def _save_node(prefix: str, source_node: str = "10") -> dict[str, Any]:
    return {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": prefix,
            "images": [source_node, 0],
        },
    }


def build_prompt_driven_graph(
    *,
    mode: str,
    context: Mapping[str, Any] | None = None,
    effective_state: Mapping[str, Any] | None = None,
    filename_prefix: str = "Neo_LayerDiffuse",
) -> dict[str, Any]:
    """Build an executable ComfyUI API prompt for prompt-driven LayerDiffuse modes.

    Primary output is node 11, which saves the image produced by
    LayeredDiffusionDecodeRGBA. Node 12 saves the plain RGB VAEDecode preview only.
    For split modes, node 13/14 save split outputs from LayeredDiffusionDecode and
    keep RGBA as the primary output for editor-safe transparency.
    """
    mode = _clean(mode, "transparent_asset")
    if mode not in PROMPT_DRIVEN_EXECUTABLE_MODES:
        raise ValueError(f"Mode '{mode}' is not prompt-driven executable in Phase 11.")
    ctx = normalized_context(context, effective_state)
    ld_config = layerdiffuse_config_for(mode, ctx["model_family"])
    prefix = _clean(filename_prefix, "Neo_LayerDiffuse")

    graph: dict[str, dict[str, Any]] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ctx["ckpt_name"]},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ctx["positive_prompt"], "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ctx["negative_prompt"], "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": ctx["width"], "height": ctx["height"], "batch_size": 1},
        },
        "5": {
            "class_type": "LayeredDiffusionApply",
            "inputs": {"model": ["1", 0], "config": ld_config, "weight": ctx["layerdiffuse_weight"]},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": ctx["seed"],
                "steps": ctx["steps"],
                "cfg": ctx["cfg"],
                "sampler_name": ctx["sampler_name"],
                "scheduler": ctx["scheduler"],
                "denoise": ctx["denoise"],
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
        },
        "10": {
            "class_type": "LayeredDiffusionDecodeRGBA",
            "inputs": {
                "samples": ["6", 0],
                "images": ["7", 0],
                "sd_version": ctx["sd_version"],
                "sub_batch_size": ctx["sub_batch_size"],
            },
        },
        "11": _save_node(f"{prefix}_rgba", "10"),
        "12": _save_node(f"{prefix}_preview_rgb", "7"),
    }

    if mode in {"rgb_alpha_split", "overlay_fx"}:
        graph["13"] = {
            "class_type": "LayeredDiffusionDecode",
            "inputs": {
                "samples": ["6", 0],
                "images": ["7", 0],
                "sd_version": ctx["sd_version"],
                "sub_batch_size": ctx["sub_batch_size"],
            },
        }
        graph["14"] = _save_node(f"{prefix}_rgb", "13")
        # The LayeredDiffusionDecode MASK output is declared for the Neo output collector.
        # If the local runtime supports mask-saving nodes, Neo can bind ["13", 1] as alpha_mask.
        # We do not guess a custom mask-to-image saver here to avoid invalid node dependency.

    return graph


def build_output_bindings(mode: str) -> dict[str, Any]:
    base = {
        "rgba_image": {"node_id": "11", "source_node_id": "10", "source_output_index": 0, "role": "primary", "required": True},
        "preview_image": {"node_id": "12", "source_node_id": "7", "source_output_index": 0, "role": "preview", "required": True},
    }
    if mode in {"rgb_alpha_split", "overlay_fx"}:
        base["rgb_image"] = {"node_id": "14", "source_node_id": "13", "source_output_index": 0, "role": "sidecar", "required": False}
        base["alpha_mask"] = {"node_id": None, "source_node_id": "13", "source_output_index": 1, "role": "mask", "required": False, "collector_required": True}
    else:
        base["alpha_mask"] = {"node_id": None, "source_node_id": "10", "source_output_index": 0, "role": "mask", "required": False, "collector_required": True}
    return base


def build_comfyui_graph(
    raw_state: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    effective_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_state = dict(raw_state or {})
    effective_state = dict(effective_state or {})
    mode = _clean(effective_state.get("mode") or raw_state.get("mode"), "transparent_asset")
    if mode in BACKGROUND_MODES_NEED_OBJECT_INFO:
        return {
            "graph_wiring_version": GRAPH_WIRING_VERSION,
            "extension_id": EXTENSION_ID,
            "mode": mode,
            "executable": False,
            "blocked_reason": "phase11_background_modes_require_exported_working_comfyui_object_info",
            "notes": [
                "Prompt-driven transparent RGBA modes are wired in Phase 11.",
                "Background/composite modes need exact local object_info and a known working exported workflow before enabling execution.",
            ],
        }
    graph = build_prompt_driven_graph(mode=mode, context=context, effective_state=effective_state)
    return {
        "graph_wiring_version": GRAPH_WIRING_VERSION,
        "verified_reference_workflow": VERIFIED_REFERENCE_WORKFLOW,
        "verified_reference_chain": VERIFIED_REFERENCE_CHAIN,
        "extension_id": EXTENSION_ID,
        "mode": mode,
        "format": "comfyui_api_prompt",
        "executable": True,
        "primary_output_type": "rgba_image",
        "primary_output_node": "11",
        "preview_output_node": "12",
        "output_bindings": build_output_bindings(mode),
        "graph": graph,
        "guardrails": {
            "batch_size": 1,
            "dimensions_multiple_of": 64,
            "primary_output_must_use": "LayeredDiffusionDecodeRGBA -> SaveImage",
            "layerdiffuse_apply_must_feed_sampler": True,
            "plain_vaedecode_is_preview_only": True,
        },
    }


def assert_graph_routes_rgba_primary(graph_package: Mapping[str, Any]) -> None:
    """Raise AssertionError if the graph saves RGB preview as the primary output."""
    graph = dict(graph_package.get("graph") or {})
    rgba_node = graph.get("10") or {}
    save_node = graph.get("11") or {}
    preview_node = graph.get("12") or {}
    assert graph.get("5", {}).get("class_type") == "LayeredDiffusionApply", "node 5 must apply LayerDiffuse before sampling"
    assert graph.get("6", {}).get("inputs", {}).get("model") == ["5", 0], "KSampler must receive the LayerDiffuse-patched model"
    assert rgba_node.get("class_type") == "LayeredDiffusionDecodeRGBA", "node 10 must decode RGBA"
    assert save_node.get("class_type") == "SaveImage", "node 11 must save primary RGBA output"
    assert save_node.get("inputs", {}).get("images") == ["10", 0], "primary save must use RGBA decode output"
    assert preview_node.get("inputs", {}).get("images") == ["7", 0], "preview save must use plain VAE decode only"


def clone_graph_package(package: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(package))
