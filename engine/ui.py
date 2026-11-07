"""Immediate-mode overlay UI drawn into a pygame surface.

The surface is uploaded as a texture and blended over the OpenGL frame, so
the debugging panel, settings screen and HUD keep the same look as before
while the 3D scene is rendered on the GPU.
"""

from __future__ import annotations

import pygame

PANEL_FILL = (11, 17, 25)
PANEL_BORDER = (38, 55, 75)
ACCENT = (111, 227, 255)
TEXT = (219, 232, 245)
MUTED = (125, 145, 168)
HIGHLIGHT = (16, 30, 44)
DIM = (2, 5, 10)
HUD_TEXT = (84, 104, 127)


def _font(names: str, size: int, bold: bool = False) -> pygame.font.Font:
    path = pygame.font.match_font(names, bold=bold)
    if path:
        font = pygame.font.Font(path, size)
    else:
        font = pygame.font.Font(None, size)
    font.set_bold(bold)
    return font


class UI:
    """Collects the overlay drawing primitives and caches text surfaces."""

    def __init__(self) -> None:
        self.mono = _font("consolas,couriernew,dejavusansmono", 14)
        self.mono_small = _font("consolas,couriernew,dejavusansmono", 12)
        self.mono_bold = _font("consolas,couriernew,dejavusansmono", 13, True)
        self.ui = _font("segoeui,arial,dejavusans", 18)
        self.ui_bold = _font("segoeui,arial,dejavusans", 18, True)
        self.title = _font("segoeui,arial,dejavusans", 26, True)
        self.toast = _font("segoeui,arial,dejavusans", 20, True)
        self._cache: dict[tuple, pygame.Surface] = {}

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    def _text_surface(
        self, font: pygame.font.Font, text: str, color: tuple[int, int, int]
    ) -> pygame.Surface:
        key = (id(font), text, color)
        surface = self._cache.get(key)
        if surface is None:
            if len(self._cache) > 600:
                self._cache.clear()
            surface = font.render(text, True, color)
            self._cache[key] = surface
        return surface

    def text(
        self,
        target: pygame.Surface,
        font: pygame.font.Font,
        text: str,
        position: tuple[float, float],
        color: tuple[int, int, int] = TEXT,
        anchor: str = "topleft",
    ) -> pygame.Rect:
        surface = self._text_surface(font, text, color)
        rect = surface.get_rect()
        setattr(rect, anchor, (round(position[0]), round(position[1])))
        target.blit(surface, rect)
        return rect

    # ------------------------------------------------------------------
    # Debug panel (F3)
    # ------------------------------------------------------------------
    def draw_debug(
        self, target: pygame.Surface, rows: list[tuple[str, str]]
    ) -> None:
        x, y, pad = 16, 16, 14
        title_height = 28
        width = 372
        line_height = self.mono.get_linesize()
        height = pad * 2 + title_height + line_height * len(rows)

        pygame.draw.rect(target, PANEL_FILL, (x, y, width, height))
        pygame.draw.rect(
            target, PANEL_BORDER, (x, y, width, height), 1
        )
        pygame.draw.rect(target, ACCENT, (x, y, 3, height))

        self.text(
            target, self.mono_bold, "debug screen",
            (x + pad, y + pad - 3), ACCENT,
        )
        self.text(
            target, self.mono_small, "F3",
            (x + width - pad, y + pad - 2), MUTED, anchor="topright",
        )

        body_y = y + pad + title_height
        for index, (label, value) in enumerate(rows):
            line_y = body_y + index * line_height
            self.text(
                target, self.mono, label.upper(),
                (x + pad, line_y), MUTED,
            )
            self.text(
                target, self.mono, value,
                (x + 148, line_y), TEXT,
            )

    # ------------------------------------------------------------------
    # Settings screen (Esc)
    # ------------------------------------------------------------------
    ROWS = (
        "resolution",
        "fullscreen",
        "player_speed",
        "auto_bunnyhop",
        "back",
    )

    def draw_settings(
        self,
        target: pygame.Surface,
        width: int,
        height: int,
        state: dict,
    ) -> None:
        pygame.draw.rect(target, (*DIM, 190), (0, 0, width, height))

        panel_w = min(560, width - 40)
        panel_h = min(408, height - 40)
        x0 = (width - panel_w) // 2
        y0 = (height - panel_h) // 2
        x1 = x0 + panel_w
        y1 = y0 + panel_h

        pygame.draw.rect(target, PANEL_FILL, (x0, y0, panel_w, panel_h))
        pygame.draw.rect(target, PANEL_BORDER, (x0, y0, panel_w, panel_h), 1)
        pygame.draw.rect(target, ACCENT, (x0, y0, panel_w, 3))

        self.text(
            target, self.title, "SETTINGS", (x0 + 30, y0 + 18), TEXT
        )
        self.text(
            target, self.mono_small, "ESC  CLOSE",
            (x1 - 30, y0 + 27), MUTED, anchor="topright",
        )
        self.text(
            target, self.mono_small,
            f"WINDOW {state['window'][0]}×{state['window'][1]}     "
            f"DISPLAY {state['screen'][0]}×{state['screen'][1]}",
            (x0 + 30, y0 + 54), MUTED,
        )
        pygame.draw.line(
            target, (27, 42, 59), (x0 + 30, y0 + 76), (x1 - 30, y0 + 76)
        )

        cursor = state["cursor"]
        rows_y = y0 + 92
        row_step = 46
        for index, row in enumerate(self.ROWS):
            selected = index == cursor
            row_top = rows_y + index * row_step
            if selected:
                pygame.draw.rect(
                    target, HIGHLIGHT,
                    (x0 + 18, row_top - 6, panel_w - 36, 38),
                )
                pygame.draw.rect(
                    target, ACCENT,
                    (x0 + 18, row_top - 6, 3, 38),
                )

            label, value = self._row_text(row, state, selected)
            self.text(
                target, self.ui, label,
                (x0 + 34, row_top),
                TEXT if selected else (159, 178, 199),
            )
            self.text(
                target, self.ui_bold, value,
                (x1 - 36, row_top + 1),
                ACCENT if selected else TEXT, anchor="topright",
            )

        pygame.draw.line(
            target, (27, 42, 59),
            (x0 + 30, y1 - 58), (x1 - 30, y1 - 58),
        )
        self.text(
            target, self.mono_small,
            "↑ ↓  navigate     ← →  change     Enter  select     Esc  close",
            ((x0 + x1) / 2, y1 - 36), MUTED, anchor="center",
        )

    @staticmethod
    def _row_text(row: str, state: dict, selected: bool) -> tuple[str, str]:
        if row == "resolution":
            resolution = state["resolution"]
            if selected:
                return "Resolution", f"←   {resolution}   →"
            return "Resolution", resolution
        if row == "fullscreen":
            return "Fullscreen", "On" if state["fullscreen"] else "Off"
        if row == "player_speed":
            speed = f"{state['player_speed']:.1f} m/s"
            if selected:
                return "Player speed", f"←   {speed}   →"
            return "Player speed", speed
        if row == "auto_bunnyhop":
            return (
                "Auto bunnyhop",
                "On" if state["auto_bunnyhop"] else "Off",
            )
        return "Back", "Esc"

    # ------------------------------------------------------------------
    # HUD and notifications
    # ------------------------------------------------------------------
    def draw_hud(
        self,
        target: pygame.Surface,
        width: int,
        height: int,
        walking: bool = True,
    ) -> None:
        if walking:
            move_line = (
                "WASD move    Space jump (hold = bunnyhop)    Shift sprint    "
                "Mouse look    R respawn    Tab free camera"
            )
        else:
            move_line = (
                "WASD move    Q/E up-down    Mouse look    Wheel move    "
                "Shift boost    C reset camera    Tab walk mode"
            )
        self.text(
            target, self.mono_small, move_line,
            (16, height - 36), HUD_TEXT, anchor="bottomleft",
        )
        self.text(
            target, self.mono_small,
            "F3  debug     F7  render mode     F11  fullscreen     "
            "M  cat     Esc  settings / release mouse",
            (16, height - 16), HUD_TEXT, anchor="bottomleft",
        )

    def draw_mouse_hint(
        self, target: pygame.Surface, width: int, height: int
    ) -> None:
        text = "CLICK TO CAPTURE MOUSE"
        panel_width = 250
        x0 = (width - panel_width) / 2
        y0 = height - 96
        pygame.draw.rect(
            target, PANEL_FILL, (x0, y0, panel_width, 32)
        )
        pygame.draw.rect(
            target, PANEL_BORDER, (x0, y0, panel_width, 32), 1
        )
        self.text(
            target, self.mono_bold, text,
            (width / 2, y0 + 16), ACCENT, anchor="center",
        )

    def draw_toast(
        self, target: pygame.Surface, width: int, text: str
    ) -> None:
        rendered = self._text_surface(self.toast, text, ACCENT)
        panel_width = max(300, rendered.get_width() + 80)
        panel_height = 58
        x0 = (width - panel_width) / 2
        y0 = 24.0
        pygame.draw.rect(
            target, PANEL_FILL, (x0, y0, panel_width, panel_height)
        )
        pygame.draw.rect(
            target, PANEL_BORDER,
            (x0, y0, panel_width, panel_height), 1,
        )
        pygame.draw.rect(
            target, ACCENT, (x0, y0 + panel_height - 3, panel_width, 3)
        )
        self.text(
            target, self.toast, text,
            (width / 2, y0 + panel_height / 2 - 1), ACCENT,
            anchor="center",
        )
