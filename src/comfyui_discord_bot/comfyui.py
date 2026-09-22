from __future__ import annotations

import asyncio
import json
import struct
import uuid
from contextlib import suppress
from pathlib import PurePosixPath
from typing import Any

import aiohttp

from .workflow import OUTPUT_NODE, SUPPORTED_BASES, TEMP_OUTPUT, WEBSOCKET_OUTPUT

# folder -> (model type, fallback loader node, fallback input name)
MODEL_FOLDERS: dict[str, tuple[str, str, str]] = {
    "checkpoints": ("main", "CheckpointLoaderSimple", "ckpt_name"),
    "diffusion_models": ("main", "UNETLoader", "unet_name"),
    "loras": ("lora", "LoraLoader", "lora_name"),
    "vae": ("vae", "VAELoader", "vae_name"),
    "text_encoders": ("text_encoder", "CLIPLoader", "clip_name"),
}

PREVIEW_IMAGE_EVENT = 1
IMAGE_FORMATS = {1: "jpg", 2: "png"}

FALLBACK_SAMPLERS = [
    "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms",
    "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m",
    "dpmpp_2m_sde", "dpmpp_3m_sde", "ddpm", "lcm", "ipndm", "deis", "ddim", "uni_pc",
]


class ComfyUIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def guess_base(name: str, folder: str) -> str:
    """ComfyUI has no model metadata API, so infer the family from the file name."""
    text = name.casefold().replace("\\", "/")
    if any(tag in text for tag in ("flux2", "flux-2", "flux.2", "flux_2", "klein")):
        return "flux2"
    if "flux" in text:
        return "flux"
    if folder != "checkpoints":
        return ""
    if any(tag in text for tag in ("xl", "pony", "illustrious", "noob")):
        return "sdxl"
    if any(tag in text for tag in ("sd2", "sd_2", "v2-", "768-v", "2.1", "2-1")):
        return "sd-2"
    return "sd-1"


def _combo_options(spec: Any) -> list[str]:
    """Read a COMBO input from /object_info in both the legacy and the newer format."""
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):
            return [str(v) for v in spec[0]]
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return [str(v) for v in spec[1].get("options", [])]
    return []


def _aliases(model: dict[str, Any]) -> set[str]:
    name = str(model.get("name", ""))
    leaf = PurePosixPath(name.replace("\\", "/")).name
    return {s.casefold() for s in (str(model.get("key", "")), name, leaf, PurePosixPath(leaf).stem) if s}


class ComfyUIClient:
    def __init__(self, base_url: str, token: str | None, model_bases: dict[str, str] | None = None) -> None:
        self.base_url = base_url
        self.token = token
        self._websocket_output: bool | None = None
        self.model_bases = {str(k).casefold(): str(v).lower() for k, v in (model_bases or {}).items()}
        self.session: aiohttp.ClientSession | None = None

    def _session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.token}"} if self.token else None,
                timeout=aiohttp.ClientTimeout(total=60),
            )
        return self.session

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            async with self._session().request(method, self.base_url + path, **kwargs) as response:
                if response.status >= 400:
                    body = (await response.text())[:1000]
                    raise ComfyUIError(f"ComfyUI returned HTTP {response.status}: {body}", response.status)
                return await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ComfyUIError(f"Cannot communicate with ComfyUI: {exc}") from exc

    async def _list_folder(self, folder: str, node_class: str, input_name: str) -> list[str]:
        try:
            files = await self._json("GET", f"/models/{folder}")
            if isinstance(files, list):
                return [str(f) for f in files]
        except ComfyUIError as exc:
            if exc.status is None:
                raise
        # Older ComfyUI builds: read the loader node's dropdown instead.
        try:
            info = await self._json("GET", f"/object_info/{node_class}")
        except ComfyUIError as exc:
            if exc.status is None:
                raise
            return []
        inputs = info.get(node_class, {}).get("input", {})
        return _combo_options({**inputs.get("optional", {}), **inputs.get("required", {})}.get(input_name))

    def base_for(self, name: str, folder: str) -> str:
        probe = {"name": name, "key": f"{folder}/{name}"}
        for alias in _aliases(probe):
            if alias in self.model_bases:
                return self.model_bases[alias]
        return guess_base(name, folder)

    async def get_models(self) -> list[dict[str, Any]]:
        folders = list(MODEL_FOLDERS.items())
        listings = await asyncio.gather(*(self._list_folder(f, node, field) for f, (_, node, field) in folders))
        models: list[dict[str, Any]] = []
        for (folder, (kind, _, _)), names in zip(folders, listings):
            models.extend(
                {"key": f"{folder}/{name}", "name": name, "type": kind, "folder": folder, "base": self.base_for(name, folder)}
                for name in names
            )
        return models

    async def get_samplers(self) -> list[str]:
        try:
            info = await self._json("GET", "/object_info/KSampler")
            found = _combo_options(info.get("KSampler", {}).get("input", {}).get("required", {}).get("sampler_name"))
            return sorted(set(found)) or FALLBACK_SAMPLERS
        except ComfyUIError:
            return FALLBACK_SAMPLERS

    async def resolve_model(
        self, requested: Any = None, main_only: bool = True, models: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        all_models = models if models is not None else await self.get_models()
        main_models = [m for m in all_models if m.get("type") == "main"]
        candidates = main_models if main_only else all_models
        if isinstance(requested, dict):
            key = requested.get("key")
            match = next((m for m in candidates if m.get("key") == key), None)
            return match or requested
        if requested not in (None, ""):
            needle = str(requested).casefold()
            match = next((m for m in candidates if needle in _aliases(m)), None)
            if not match:
                raise ComfyUIError(f"Installed {'main ' if main_only else ''}model not found: {requested}")
            return match
        match = next((m for m in main_models if m.get("base") in SUPPORTED_BASES), None)
        if not match:
            raise ComfyUIError("No supported installed main generation model was found")
        return match

    async def output_node(self) -> str:
        """Prefer an output node that never writes the image to the ComfyUI machine's disk."""
        if self._websocket_output is None:
            try:
                info = await self._json("GET", f"/object_info/{WEBSOCKET_OUTPUT}")
                self._websocket_output = isinstance(info, dict) and WEBSOCKET_OUTPUT in info
            except ComfyUIError as exc:
                if exc.status is None:
                    raise
                self._websocket_output = False
        return WEBSOCKET_OUTPUT if self._websocket_output else TEMP_OUTPUT

    async def _queue(self, workflow: dict[str, Any], client_id: str) -> str:
        queued = await self._json("POST", "/prompt", json={"prompt": workflow, "client_id": client_id})
        prompt_id = queued.get("prompt_id") if isinstance(queued, dict) else None
        if not prompt_id:
            raise ComfyUIError(f"ComfyUI did not return a prompt ID: {queued}")
        if queued.get("node_errors"):
            raise ComfyUIError(f"ComfyUI rejected the workflow: {queued['node_errors']}")
        return prompt_id

    async def _cancel(self, prompt_id: str) -> None:
        with suppress(ComfyUIError):
            await self._json("POST", "/queue", json={"delete": [prompt_id]})

    async def generate(self, workflow: dict[str, Any], poll_interval: float, timeout: float) -> tuple[bytes, str]:
        if workflow.get(OUTPUT_NODE, {}).get("class_type") == WEBSOCKET_OUTPUT:
            return await self._generate_websocket(workflow, timeout)
        return await self._generate_polling(workflow, poll_interval, timeout)

    async def _generate_websocket(self, workflow: dict[str, Any], timeout: float) -> tuple[bytes, str]:
        """Receive the image straight from the websocket; nothing is written on the ComfyUI machine."""
        client_id = str(uuid.uuid4())  # one connection per job, so concurrent jobs never mix
        prompt_id: str | None = None
        try:
            async with self._session().ws_connect(
                f"{self.base_url}/ws", params={"clientId": client_id}, heartbeat=30, max_msg_size=0
            ) as ws:
                prompt_id = await self._queue(workflow, client_id)

                async def receive() -> tuple[bytes, str]:
                    current_node: str | None = None
                    image: tuple[bytes, str] | None = None
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            event = json.loads(msg.data)
                            data = event.get("data", {})
                            if data.get("prompt_id") != prompt_id:
                                continue
                            if event.get("type") == "execution_error":
                                raise ComfyUIError(
                                    f"{data.get('node_type', 'unknown node')}: "
                                    f"{data.get('exception_message', 'execution failed')}".strip()
                                )
                            if event.get("type") == "execution_interrupted":
                                raise ComfyUIError("generation was interrupted")
                            if event.get("type") == "executing":
                                current_node = data.get("node")
                                if current_node is None:
                                    break
                            if event.get("type") == "execution_success":
                                break
                        elif msg.type == aiohttp.WSMsgType.BINARY and current_node == OUTPUT_NODE:
                            if len(msg.data) > 8:
                                kind, fmt = struct.unpack(">II", msg.data[:8])
                                if kind == PREVIEW_IMAGE_EVENT:
                                    image = (msg.data[8:], f"discord-bot-{prompt_id[:8]}.{IMAGE_FORMATS.get(fmt, 'png')}")
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                    if image is None:
                        raise ComfyUIError("Generation completed but returned no image")
                    return image

                return await asyncio.wait_for(receive(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            if prompt_id:
                await self._cancel(prompt_id)
            raise ComfyUIError(f"generation timed out after {timeout:g} seconds") from exc
        except aiohttp.ClientError as exc:
            raise ComfyUIError(f"Cannot communicate with ComfyUI websocket: {exc}") from exc

    async def _generate_polling(self, workflow: dict[str, Any], poll_interval: float, timeout: float) -> tuple[bytes, str]:
        """Fallback when SaveImageWebsocket is unavailable: the image lands in ComfyUI's temp folder."""
        prompt_id = await self._queue(workflow, str(uuid.uuid4()))

        async def wait_for_result() -> dict[str, str]:
            while True:
                history = await self._json("GET", f"/history/{prompt_id}")
                entry = history.get(prompt_id) if isinstance(history, dict) else None
                if entry:
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        raise ComfyUIError(execution_error(status))
                    image = find_image(entry.get("outputs", {}))
                    if image:
                        return image
                    if status.get("completed"):
                        raise ComfyUIError("Generation completed but returned no image")
                await asyncio.sleep(poll_interval)

        try:
            image = await asyncio.wait_for(wait_for_result(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            await self._cancel(prompt_id)
            raise ComfyUIError(f"generation timed out after {timeout:g} seconds") from exc

        params = {"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image.get("type", "temp")}
        try:
            async with self._session().get(self.base_url + "/view", params=params) as response:
                if response.status >= 400:
                    raise ComfyUIError(f"image download returned HTTP {response.status}", response.status)
                return await response.read(), PurePosixPath(image["filename"]).name
        except aiohttp.ClientError as exc:
            raise ComfyUIError(f"cannot download generated image: {exc}") from exc


def execution_error(status: dict[str, Any]) -> str:
    for message in status.get("messages", []):
        if isinstance(message, list) and len(message) > 1 and message[0] == "execution_error":
            details = message[1] if isinstance(message[1], dict) else {}
            node = details.get("node_type", "unknown node")
            return f"{node}: {details.get('exception_message', 'execution failed')}".strip()
    return "generation failed"


def find_image(outputs: Any) -> dict[str, str] | None:
    """Find the saved image in a /history entry, preferring the workflow's output node."""
    if not isinstance(outputs, dict):
        return None
    ordered = [outputs.get(OUTPUT_NODE)] + [v for k, v in outputs.items() if k != OUTPUT_NODE]
    for output in ordered:
        if isinstance(output, dict):
            for image in output.get("images", []):
                if isinstance(image, dict) and image.get("filename"):
                    return image
    return None
