from __future__ import annotations

import asyncio
import copy
import io
import json
import logging
import random
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from .config import Settings
from .comfyui import ComfyUIClient
from .workflow import WorkflowError, build_workflow

log = logging.getLogger(__name__)

SUBJECTS = ("a forgotten moon temple", "a clockwork fox", "an underwater library", "a floating city", "a spectral forest", "a tiny dragon barista")
STYLES = ("cinematic concept art", "dreamlike oil painting", "intricate ink illustration", "retro-futurist poster", "editorial photography", "luminous fantasy art")
MOODS = ("at blue hour", "in warm volumetric light", "during a thunderstorm", "with an ethereal atmosphere", "under neon reflections", "surrounded by drifting petals")
PACMAN_DOTS = 12


def random_prompt() -> str:
    return f"{random.choice(SUBJECTS)}, {random.choice(STYLES)}, {random.choice(MOODS)}, highly detailed"


def pacman_frame(position: int) -> str:
    position %= PACMAN_DOTS + 1
    pacman = "😐" if position % 2 == 0 else "😮"
    return f"**Working…**\n`[{' ' * position}{pacman}{'·' * (PACMAN_DOTS - position)}]`"


async def animate_progress(message: discord.Message) -> None:
    position = 1
    try:
        while True:
            await asyncio.sleep(2)
            await message.edit(content=pacman_frame(position))
            position = (position + 1) % (PACMAN_DOTS + 1)
    except (discord.HTTPException, discord.NotFound):
        return


async def stop_animation(task: asyncio.Task[None]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def parse_value(value: str) -> Any:
    """Allow model/LoRA fields to be strings or JSON objects."""
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@dataclass
class GenerationState:
    owner_id: int
    prompt: str
    negative_prompt: str
    width: int
    height: int
    seed: int
    steps: int
    cfg_scale: float
    extras: dict[str, Any] = field(default_factory=dict)

    def graph_values(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            **self.extras,
        }


class ComfyBot(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        super().__init__(command_prefix=commands.when_mentioned, intents=discord.Intents.default())
        self.settings = settings
        self.comfy = ComfyUIClient(settings.comfyui_url, settings.comfyui_token, settings.model_bases)
        self.jobs = asyncio.Semaphore(settings.max_concurrent_jobs)

    async def setup_hook(self) -> None:
        if self.settings.guild_id:
            guild = discord.Object(id=self.settings.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

    async def close(self) -> None:
        await self.comfy.close()
        await super().close()

    async def render(self, state: GenerationState) -> tuple[discord.Embed, discord.File]:
        models = await self.comfy.get_models()
        model = await self.comfy.resolve_model(state.extras.get("model"), models=models)
        values = state.graph_values()
        for name in ("lora", "t5_encoder", "clip_encoder", "text_encoder", "vae"):
            if values.get(name) not in (None, ""):
                values[name] = await self.comfy.resolve_model(values[name], main_only=False, models=models)
        workflow = build_workflow(values, model, await self.comfy.output_node())
        async with self.jobs:
            image, image_name = await self.comfy.generate(
                workflow, self.settings.poll_interval, self.settings.timeout
            )
        embed = discord.Embed(description=state.prompt[:4096])
        base = str(model.get("base", model.get("base_model", ""))).lower()
        sampler = state.extras.get("sampler") or ("model default" if "flux" in base else "euler")
        model_name = model.get("name", model.get("key", "Unknown model"))
        embed.set_footer(
            text=f"🤖 {model_name} • 🎛️ {sampler} • 🌱 {state.seed} • "
            f"📐 {state.width}×{state.height} • 🔢 {state.steps} • 🧭 {state.cfg_scale:g}"
        )
        embed.set_image(url=f"attachment://{image_name}")
        # The image only ever lives in memory here; nothing is written to the bot's disk.
        return embed, discord.File(io.BytesIO(image), filename=image_name)


class PromptModal(discord.ui.Modal, title="Edit prompt"):
    prompt = discord.ui.TextInput(label="Prompt", style=discord.TextStyle.paragraph, max_length=1500)
    negative = discord.ui.TextInput(label="Negative prompt", style=discord.TextStyle.paragraph, max_length=1000, required=False)

    def __init__(self, view: "ResultView") -> None:
        super().__init__()
        self.result_view = view
        self.prompt.default = view.state.prompt
        self.negative.default = view.state.negative_prompt

    async def on_submit(self, interaction: discord.Interaction) -> None:
        state = copy.deepcopy(self.result_view.state)
        state.prompt = str(self.prompt)
        state.negative_prompt = str(self.negative)
        state.seed = random.randint(0, 2_147_483_647)
        await self.result_view.regenerate(interaction, state)


class GenerationSettingsModal(discord.ui.Modal, title="Generation settings"):
    generation = discord.ui.TextInput(label="Scale, steps, strength", placeholder="7, 30, 0.75", required=False, max_length=100)
    upscaling = discord.ui.TextInput(label="Upscaling", placeholder="true, false, or scale value", required=False, max_length=100)

    def __init__(self, view: "TweakView") -> None:
        super().__init__()
        self.tweak_view = view
        state = view.draft
        strength = state.extras.get("strength", "")
        self.generation.default = f"{state.cfg_scale:g}, {state.steps}, {strength}"
        self.upscaling.default = _display(state.extras.get("upscaling"))

    async def on_submit(self, interaction: discord.Interaction) -> None:
        view, state = self.tweak_view, self.tweak_view.draft
        parts = [part.strip() for part in str(self.generation).split(",")]
        try:
            if parts and parts[0]:
                state.cfg_scale = float(parts[0])
            if len(parts) > 1 and parts[1]:
                state.steps = int(parts[1])
            if len(parts) > 2 and parts[2]:
                state.extras["strength"] = float(parts[2])
            if not (0 <= state.cfg_scale <= 30 and 1 <= state.steps <= 100):
                raise ValueError
        except ValueError:
            await interaction.response.send_message("Use `scale, steps, strength`, for example `7, 30, 0.75`.", ephemeral=True)
            return
        upscaling = parse_value(str(self.upscaling))
        if upscaling is None:
            state.extras.pop("upscaling", None)
        else:
            state.extras["upscaling"] = upscaling
        state.seed = random.randint(0, 2_147_483_647)
        await view.source.regenerate(interaction, copy.deepcopy(state))


def _display(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)


class InstalledChoiceSelect(discord.ui.Select):
    def __init__(self, owner: "TweakView", kind: str) -> None:
        self.owner = owner
        self.kind = kind
        super().__init__(placeholder=kind.title(), min_values=1, max_values=1, options=owner.options_for(kind))

    async def callback(self, interaction: discord.Interaction) -> None:
        value = self.values[0]
        if self.kind == "model" and value != "__keep__":
            self.owner.draft.extras["model"] = value
        elif self.kind == "lora":
            if value == "__none__":
                self.owner.draft.extras.pop("lora", None)
            elif value != "__keep__":
                self.owner.draft.extras["lora"] = value
        elif self.kind == "sampler":
            if value == "__default__":
                self.owner.draft.extras.pop("sampler", None)
            elif value != "__keep__":
                self.owner.draft.extras["sampler"] = value
        await interaction.response.defer()


class TweakView(discord.ui.View):
    PAGE_SIZE = 24

    def __init__(self, source: "ResultView", models: list[dict[str, Any]], samplers: list[str]) -> None:
        super().__init__(timeout=300)
        self.source = source
        self.draft = copy.deepcopy(source.state)
        self.page = 0
        self.choices = {
            "model": sorted((m for m in models if str(m.get("type", "main")).lower() == "main"), key=lambda m: str(m.get("name", "")).casefold()),
            "lora": sorted((m for m in models if str(m.get("type", "")).lower() == "lora"), key=lambda m: str(m.get("name", "")).casefold()),
            "sampler": sorted(set(samplers)),
        }
        self.selects = {kind: InstalledChoiceSelect(self, kind) for kind in ("model", "sampler", "lora")}
        for select in self.selects.values():
            self.add_item(select)
        self._update_pager()

    def options_for(self, kind: str) -> list[discord.SelectOption]:
        first = self.page * self.PAGE_SIZE
        values = self.choices[kind][first:first + self.PAGE_SIZE]
        if kind in {"model", "lora"}:
            options = [
                discord.SelectOption(label=str(item.get("name") or item.get("key"))[:100], value=str(item.get("key"))[:100])
                for item in values
            ]
        else:
            options = [discord.SelectOption(label=value[:100], value=value[:100]) for value in values]
        if kind == "model":
            options.insert(0, discord.SelectOption(label="Keep current model", value="__keep__"))
        elif kind == "lora":
            options.insert(0, discord.SelectOption(label="No LoRA", value="__none__"))
        else:
            options.insert(0, discord.SelectOption(label="Model default", value="__default__"))
        return options

    def _update_pager(self) -> None:
        max_pages = max(1, max((len(items) + self.PAGE_SIZE - 1) // self.PAGE_SIZE for items in self.choices.values()))
        self.previous_page.disabled = self.page == 0
        self.next_page.disabled = self.page >= max_pages - 1

    async def _change_page(self, interaction: discord.Interaction, delta: int) -> None:
        self.page += delta
        for kind, select in self.selects.items():
            select.options = self.options_for(kind)
        self._update_pager()
        await interaction.response.edit_message(content=f"Choose installed options — page {self.page + 1}", view=self)

    @discord.ui.button(label="Previous", style=discord.ButtonStyle.secondary, row=3)
    async def previous_page(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await self._change_page(interaction, -1)

    @discord.ui.button(label="Next", style=discord.ButtonStyle.secondary, row=3)
    async def next_page(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await self._change_page(interaction, 1)

    @discord.ui.button(label="Other settings", style=discord.ButtonStyle.secondary, row=3)
    async def other_settings(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await interaction.response.send_modal(GenerationSettingsModal(self))

    @discord.ui.button(label="Generate", emoji="✨", style=discord.ButtonStyle.primary, row=3)
    async def generate(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        self.draft.seed = random.randint(0, 2_147_483_647)
        await self.source.regenerate(interaction, copy.deepcopy(self.draft))


class ResultView(discord.ui.View):
    def __init__(self, bot: ComfyBot, state: GenerationState) -> None:
        super().__init__(timeout=3600)
        self.bot = bot
        self.state = state

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.state.owner_id:
            return True
        await interaction.response.send_message("Only the person who created this image can change it.", ephemeral=True)
        return False

    async def regenerate(self, interaction: discord.Interaction, state: GenerationState) -> None:
        await interaction.response.defer()
        message = await interaction.followup.send(content=pacman_frame(0), wait=True)
        animation = asyncio.create_task(animate_progress(message))
        try:
            embed, file = await self.bot.render(state)
            await stop_animation(animation)
            await message.edit(content=None, embed=embed, attachments=[file], view=ResultView(self.bot, state))
        except Exception:
            await stop_animation(animation)
            log.exception("Regeneration failed")
            await message.edit(content="Oops, something's wrong.", embed=None, attachments=[], view=None)
        finally:
            if not animation.done():
                await stop_animation(animation)

    @discord.ui.button(label="Refresh", emoji="🔄", style=discord.ButtonStyle.primary)
    async def refresh(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        state = copy.deepcopy(self.state)
        state.seed = random.randint(0, 2_147_483_647)
        await self.regenerate(interaction, state)

    @discord.ui.button(label="Edit prompt", emoji="✏️", style=discord.ButtonStyle.secondary)
    async def edit_prompt(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await interaction.response.send_modal(PromptModal(self))

    @discord.ui.button(label="Random", emoji="🎲", style=discord.ButtonStyle.secondary)
    async def randomise(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        state = copy.deepcopy(self.state)
        state.prompt = random_prompt()
        state.seed = random.randint(0, 2_147_483_647)
        await self.regenerate(interaction, state)

    @discord.ui.button(label="Tweak", emoji="🎛️", style=discord.ButtonStyle.secondary)
    async def tweak(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            models, samplers = await asyncio.gather(
                self.bot.comfy.get_models(), self.bot.comfy.get_samplers()
            )
            view = TweakView(self, models, samplers)
            await interaction.edit_original_response(
                content="Choose installed options — page 1", view=view
            )
        except Exception:
            log.exception("Could not load ComfyUI tweak options")
            await interaction.edit_original_response(content="Oops, something's wrong.", view=None)

    @discord.ui.button(label="Delete", emoji="🗑️", style=discord.ButtonStyle.danger)
    async def delete(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await interaction.response.defer()
        await interaction.delete_original_response()


def create_bot(settings: Settings) -> ComfyBot:
    bot = ComfyBot(settings)

    @bot.tree.command(name="dream", description="Generate an image with ComfyUI")
    @app_commands.describe(prompt="What to generate", negative_prompt="What to avoid", width="Image width", height="Image height", seed="Random seed (-1 for random)", steps="Denoising steps", cfg_scale="Prompt guidance")
    async def dream(
        interaction: discord.Interaction,
        prompt: app_commands.Range[str, 1, 1500],
        negative_prompt: app_commands.Range[str, 0, 1000] | None = None,
        width: app_commands.Range[int, 256, 2048] | None = None,
        height: app_commands.Range[int, 256, 2048] | None = None,
        seed: app_commands.Range[int, -1, 2147483647] | None = None,
        steps: app_commands.Range[int, 1, 100] | None = None,
        cfg_scale: app_commands.Range[float, 0, 30] | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        defaults = settings.generation_defaults
        selected_seed = int(defaults["seed"]) if seed is None else seed
        extras = {
            name: defaults.get(name)
            for name in ("model", "lora", "sampler", "scheduler", "strength", "upscaling", "t5_encoder", "clip_encoder", "text_encoder", "vae")
            if defaults.get(name) is not None
        }
        state = GenerationState(
            interaction.user.id, str(prompt),
            str(defaults["negative_prompt"] if negative_prompt is None else negative_prompt),
            int(defaults["width"] if width is None else width),
            int(defaults["height"] if height is None else height),
            random.randint(0, 2_147_483_647) if selected_seed == -1 else selected_seed,
            int(defaults["steps"] if steps is None else steps),
            float(defaults["cfg_scale"] if cfg_scale is None else cfg_scale), extras,
        )
        message = await interaction.edit_original_response(content=pacman_frame(0))
        animation = asyncio.create_task(animate_progress(message))
        try:
            embed, file = await bot.render(state)
            await stop_animation(animation)
            await message.edit(content=None, embed=embed, attachments=[file], view=ResultView(bot, state))
        except Exception:
            await stop_animation(animation)
            log.exception("Generation failed")
            await message.edit(content="Oops, something's wrong.", embed=None, attachments=[], view=None)
        finally:
            if not animation.done():
                await stop_animation(animation)

    return bot


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = Settings.from_env()
        bot = create_bot(settings)
    except (ValueError, WorkflowError) as exc:
        raise SystemExit(f"Startup configuration error:\n{exc}") from None
    bot.run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
