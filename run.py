"""Run the bot directly from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from comfyui_discord_bot.bot import main  # noqa: E402


if __name__ == "__main__":
    main()
