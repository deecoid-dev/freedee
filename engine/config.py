"""Persistent engine settings saved to a JSON file next to the entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_FILE = "freedee_settings.json"

# All settings that survive a restart.  Missing keys fall back to the values
# set in Engine.__init__, so old saves keep working after new options appear.
DEFAULTS: dict = {
    "render_mode": 0,
    "resolution": 0,
    "windowed": True,
    "vsync": False,
    "brightness": 1.0,
    "gamma": 1.0,
    "fov": 50.0,
    "sensitivity": 4.5,
    "invert_y": False,
    "player_speed": 7.0,
    "auto_bunnyhop": False,
    "menu_theme": 0,
    "accent": 0,
    "menu_opacity": 0.96,
    "show_hud": True,
    "debug_overlay": False,
    "crosshair_enabled": True,
    "crosshair_color": 0,
    "crosshair_custom_r": 255,
    "crosshair_custom_g": 255,
    "crosshair_custom_b": 255,
    "crosshair_size": 10,
    "crosshair_gap": 6,
    "crosshair_thickness": 2,
    "crosshair_dot": True,
    "crosshair_dot_size": 3,
    "crosshair_outline": True,
    "crosshair_outline_width": 1,
    "crosshair_opacity": 100,
}


def sys_main_script() -> str | None:
    """Best effort path to the script that started the interpreter."""
    import sys

    main = getattr(sys, "argv", [None])[0]
    if not main:
        return None
    return str(Path(main).resolve())


def _settings_path(filename: str = DEFAULT_FILE) -> Path:
    """Place the save file next to the launched script or in the cwd."""
    main = Path(sys_main_script()) if sys_main_script() else Path.cwd()
    return (main.parent if main.is_file() else main) / filename


class Settings:
    """Tiny wrapper around a JSON file with defaults and atomic writes."""

    def __init__(self, filename: str = DEFAULT_FILE) -> None:
        self.path = _settings_path(filename)
        self.data: dict = dict(DEFAULTS)
        self.load()

    def load(self) -> dict:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError):
            return dict(self.data)
        if isinstance(loaded, dict):
            for key, default in DEFAULTS.items():
                value = loaded.get(key, default)
                if isinstance(value, type(default)) or default is None:
                    self.data[key] = value
        return dict(self.data)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(self.data, handle, indent=2, ensure_ascii=False)
            tmp.replace(self.path)
        except OSError:
            pass

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value) -> None:
        self.data[key] = value

    def get(self, key: str, default=None):
        return self.data.get(key, default)
