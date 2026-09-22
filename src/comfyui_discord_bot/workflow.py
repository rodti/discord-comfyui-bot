from __future__ import annotations

from typing import Any

SD_BASES = {"sd-1", "sd-2", "sd1", "sd2", "sdxl", "sdxl-refiner"}
FLUX1_BASES = {"flux", "flux-1", "flux1"}
FLUX2_BASES = {"flux2", "flux-2", "flux.2", "flux2-klein"}
SUPPORTED_BASES = SD_BASES | FLUX1_BASES | FLUX2_BASES

OUTPUT_NODE = "save"
# Streams the finished image over the websocket without writing it to disk.
# Ships with ComfyUI as custom_nodes/websocket_image_save.py.
WEBSOCKET_OUTPUT = "SaveImageWebsocket"
# Fallback: writes to ComfyUI's temp folder, which ComfyUI empties on startup.
TEMP_OUTPUT = "PreviewImage"


class WorkflowError(ValueError):
    pass


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def _filename(value: Any, setting: str) -> str:
    """Accept a resolved model dict (from the ComfyUI client) or a raw file name."""
    if isinstance(value, dict):
        value = value.get("name")
    if value in (None, ""):
        raise WorkflowError(f"ComfyUI model reference for {setting} is missing a file name")
    return str(value)


def build_workflow(
    values: dict[str, Any], model: dict[str, Any], output: str = WEBSOCKET_OUTPUT
) -> dict[str, Any]:
    """Build a ComfyUI API-format text-to-image workflow for SD 1/2, SDXL, FLUX.1, or FLUX.2."""
    base = str(model.get("base", model.get("base_model", ""))).lower()
    if base not in SUPPORTED_BASES:
        raise WorkflowError(
            f"Built-in generation does not support model base '{base or 'unknown'}'. "
            "Choose an installed SD 1.x, SD 2.x, SDXL, FLUX.1, or FLUX.2 model, or set its base "
            "in comfyui.model_bases."
        )
    is_flux1, is_flux2 = base in FLUX1_BASES, base in FLUX2_BASES
    width, height, seed = int(values["width"]), int(values["height"]), int(values["seed"])
    steps, cfg = int(values["steps"]), float(values["cfg_scale"])
    sampler = values.get("sampler") or "euler"
    nodes: dict[str, dict[str, Any]] = {}

    # --- Loaders ---------------------------------------------------------------
    clip_out: list[Any] | None = None
    vae_out: list[Any] | None = None
    if model.get("folder", "checkpoints") == "checkpoints":
        nodes["loader"] = _node("CheckpointLoaderSimple", ckpt_name=_filename(model, "model"))
        model_out, clip_out, vae_out = ["loader", 0], ["loader", 1], ["loader", 2]
    else:
        nodes["loader"] = _node("UNETLoader", unet_name=_filename(model, "model"), weight_dtype="default")
        model_out = ["loader", 0]

    if is_flux1:
        t5, clip_l = values.get("t5_encoder"), values.get("clip_encoder")
        if t5 or clip_l:
            if not (t5 and clip_l):
                raise WorkflowError("FLUX.1 needs both generation.t5_encoder and generation.clip_encoder")
            nodes["text_encoder"] = _node(
                "DualCLIPLoader", clip_name1=_filename(t5, "t5_encoder"),
                clip_name2=_filename(clip_l, "clip_encoder"), type="flux",
            )
            clip_out = ["text_encoder", 0]
        elif clip_out is None:
            raise WorkflowError("FLUX.1 diffusion models need generation.t5_encoder and generation.clip_encoder")
    elif is_flux2:
        if values.get("text_encoder"):
            nodes["text_encoder"] = _node(
                "CLIPLoader", clip_name=_filename(values["text_encoder"], "text_encoder"),
                type="flux2", device="default",
            )
            clip_out = ["text_encoder", 0]
        elif clip_out is None:
            raise WorkflowError("FLUX.2 diffusion models need generation.text_encoder")

    if (is_flux1 or is_flux2) and values.get("vae"):
        nodes["vae_loader"] = _node("VAELoader", vae_name=_filename(values["vae"], "vae"))
        vae_out = ["vae_loader", 0]
    if vae_out is None:
        raise WorkflowError("FLUX diffusion models need generation.vae")

    # --- LoRA ------------------------------------------------------------------
    if values.get("lora"):
        lora_name = _filename(values["lora"], "lora")
        if is_flux1 or is_flux2:
            nodes["lora"] = _node("LoraLoaderModelOnly", model=model_out, lora_name=lora_name, strength_model=1.0)
            model_out = ["lora", 0]
        else:
            nodes["lora"] = _node(
                "LoraLoader", model=model_out, clip=clip_out, lora_name=lora_name,
                strength_model=1.0, strength_clip=1.0,
            )
            model_out, clip_out = ["lora", 0], ["lora", 1]

    # --- Conditioning + sampling -----------------------------------------------
    nodes["positive"] = _node("CLIPTextEncode", text=values["prompt"], clip=clip_out)
    if is_flux2:
        nodes["negative"] = _node("CLIPTextEncode", text=values.get("negative_prompt", ""), clip=clip_out)
        nodes.update({
            "guider": _node("CFGGuider", model=model_out, positive=["positive", 0], negative=["negative", 0], cfg=cfg),
            "sampler_select": _node("KSamplerSelect", sampler_name=sampler),
            "scheduler": _node("Flux2Scheduler", steps=steps, width=width, height=height),
            "noise": _node("RandomNoise", noise_seed=seed),
            "latent": _node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1),
            "sample": _node(
                "SamplerCustomAdvanced", noise=["noise", 0], guider=["guider", 0],
                sampler=["sampler_select", 0], sigmas=["scheduler", 0], latent_image=["latent", 0],
            ),
        })
    else:
        if is_flux1:
            # FLUX.1 is guidance-distilled: CFG value becomes FluxGuidance, sampler CFG stays at 1.
            nodes["guidance"] = _node("FluxGuidance", conditioning=["positive", 0], guidance=cfg)
            nodes["negative"] = _node("ConditioningZeroOut", conditioning=["positive", 0])
            positive, sampler_cfg = ["guidance", 0], 1.0
            nodes["latent"] = _node("EmptySD3LatentImage", width=width, height=height, batch_size=1)
        else:
            nodes["negative"] = _node("CLIPTextEncode", text=values.get("negative_prompt", ""), clip=clip_out)
            positive, sampler_cfg = ["positive", 0], cfg
            nodes["latent"] = _node("EmptyLatentImage", width=width, height=height, batch_size=1)
        nodes["sample"] = _node(
            "KSampler", model=model_out, seed=seed, steps=steps, cfg=sampler_cfg,
            sampler_name=sampler, scheduler=values.get("scheduler") or ("simple" if is_flux1 else "normal"),
            positive=positive, negative=["negative", 0], latent_image=["latent", 0], denoise=1.0,
        )

    nodes["decode"] = _node("VAEDecode", samples=["sample", 0], vae=vae_out)
    if output not in (WEBSOCKET_OUTPUT, TEMP_OUTPUT):
        raise WorkflowError(f"Unsupported output node: {output}")
    nodes[OUTPUT_NODE] = _node(output, images=["decode", 0])
    return nodes
