"""Explicit engine states shared by the frame loop and the menu flow."""

from __future__ import annotations

from enum import Enum, auto


class GameState(Enum):
    """Where the engine currently is.

    ``MAIN_MENU`` has no world loaded and never renders the 3D scene.
    ``PLAYING`` owns a loaded world and captures the mouse. ``PAUSED`` keeps
    the world loaded but freezes it behind the pause menu.
    """

    MAIN_MENU = auto()
    PLAYING = auto()
    PAUSED = auto()
