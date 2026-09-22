# Discord ComfyUI bot

A Discord `/dream` bot that generates images with a ComfyUI server, local or on another machine. Users do not create or export workflows: the bot builds the required ComfyUI API workflow internally from the selected installed model.

## Setup

1. Create a Discord application in the Developer Portal, add a bot, and invite it with the `bot` and `applications.commands` scopes.
2. Make sure ComfyUI is running and has a supported model installed. On a remote machine, start it listening on the network: `python main.py --listen 0.0.0.0 --port 8188`.
3. Copy `config.example.json` to `config.json` and set the Discord token and `comfyui.url` (for example `http://192.168.1.50:8188`).

## Run as native Python

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run.py
```

On Windows, activate with `.venv\Scripts\activate`. No Docker installation or `workflow.json` is needed.

## Run with Docker

```
cp config.example.json config.json
docker compose up --build -d
```

If ComfyUI runs on the Docker host itself, use `http://host.docker.internal:8188` as the URL.

## Models and configuration

Set `generation.model` to an installed file name (with or without extension, e.g. `juggernautXL` or `sdxl/juggernautXL.safetensors`). Models are listed from ComfyUI's `checkpoints` and `diffusion_models` folders. When it is `null`, the bot chooses the first supported installed model. Built-in text-to-image generation supports:

- Stable Diffusion 1.x and 2.x
- SDXL
- FLUX.1
- FLUX.2 Klein

ComfyUI has no model-metadata API, so the model family is inferred from the file name (`flux-2`/`klein` → FLUX.2, `flux` → FLUX.1, `xl`/`pony`/`illustrious` → SDXL, otherwise SD 1.x). If a file name doesn't reveal its family, map it in `comfyui.model_bases`, e.g. `{"my_merge.safetensors": "flux"}`. Valid bases: `sd-1`, `sd-2`, `sdxl`, `flux`, `flux2`.

Full checkpoints (in `models/checkpoints`) carry their own text encoders and VAE. FLUX.1 diffusion-only models (in `models/diffusion_models`) need `generation.t5_encoder`, `generation.clip_encoder`, and `generation.vae`. FLUX.2 needs `generation.text_encoder` (e.g. `qwen_3_4b`) and `generation.vae` (e.g. `flux2-vae`). References are resolved against ComfyUI's installed model lists.

The core generation settings control negative prompt, dimensions, seed, sampler, steps, and CFG/guidance. `generation.scheduler` optionally sets the ComfyUI noise schedule (defaults: `normal` for SD, `simple` for FLUX.1; FLUX.2 uses `Flux2Scheduler`). Slash-command values override configured defaults. A seed of `-1` chooses a random seed.

Each result includes Refresh, Edit prompt, Random, Tweak, and Delete controls. The Tweak panel loads model and LoRA choices from the connected ComfyUI installation and sampler choices from ComfyUI's `/object_info`; dropdown pages expose installations with more than 25 options. Refreshing or changing a result creates a new message and preserves the original image; only Delete removes it. Only the requester may use the controls. A yellow face alternates between closed and wide-open mouth frames as it munches through the “Working…” progress bar. Result metadata uses icons for model, sampler, seed, dimensions, steps, and CFG/guidance. Discord receives only a generic failure message; full errors are logged to the bot console.

Global slash-command registration can take up to an hour. Set `discord.guild_id` to a development server ID for immediate guild-scoped registration.

## Image storage

Generated images are never written to disk: the bot receives them over ComfyUI's websocket using the `SaveImageWebsocket` node, which ships with ComfyUI (`custom_nodes/websocket_image_save.py`), and holds them only in memory until they are posted to Discord. If that node is missing, the bot falls back to `PreviewImage`, which writes to ComfyUI's `temp` folder; ComfyUI empties that folder each time it starts.

## Environment overrides

Connection and runtime settings have optional environment overrides in `.env.example`. Environment values take precedence over `config.json`; `.env` is not required.

## Security

Do not commit `.env` or `config.json`, because either may contain tokens. Both are ignored by Git. ComfyUI has no built-in authentication, so anything that can reach its port can run jobs on it. Prefer a private network (LAN, VPN, Tailscale) or an authenticated reverse proxy; `comfyui.token` is sent as a `Bearer` header for proxies that require one.
