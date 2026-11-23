"""GoldSource/Source-inspired overlay UI drawn into a pygame surface.

The surface is uploaded as a texture and blended over the OpenGL frame, so
the HUD, debug panel and the options window share one consistent, slightly
translucent "old engine" look.  The options window is a real little window:
it can be dragged by its title bar, has category tabs, dropdowns, checkboxes,
sliders and OK / Cancel / Apply buttons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pygame

from .crosshair import Crosshair

PANEL_ALPHA = 244
DIM_ALPHA = 150
TITLE_H = 24
TABS_H = 24
FOOTER_H = 46
WINDOW_W = 660
WINDOW_H = 500
MAP_WINDOW_W = 640
MAP_WINDOW_H = 470
MAP_ROW_H = 28
MAP_FOOTER_H = 48

#: Full-screen backdrop shown behind the main menu.
MENU_BACKGROUND_PATH = Path(__file__).resolve().parent / "menu_background.png"

TEXT = (219, 232, 245)
MUTED = (125, 145, 168)


@dataclass(frozen=True)
class MenuTheme:
    """Colours of one selectable menu skin."""

    name: str
    panel: tuple[int, int, int]
    dark: tuple[int, int, int]
    light: tuple[int, int, int]
    border: tuple[int, int, int]
    border_dark: tuple[int, int, int]
    text: tuple[int, int, int]
    muted: tuple[int, int, int]


#: Menu colours offered in Options > Interface.  The first one is the classic
#: olive GoldSource look from the reference screenshot.
MENU_THEMES: tuple[MenuTheme, ...] = (
    MenuTheme(
        "GoldSource",
        (59, 64, 45),
        (37, 41, 29),
        (78, 83, 59),
        (126, 130, 99),
        (22, 24, 17),
        (222, 222, 205),
        (154, 158, 132),
    ),
    MenuTheme(
        "Source",
        (42, 48, 56),
        (27, 31, 37),
        (58, 65, 75),
        (96, 105, 117),
        (14, 17, 21),
        (218, 224, 230),
        (138, 148, 160),
    ),
    MenuTheme(
        "Quake",
        (64, 48, 34),
        (42, 31, 22),
        (86, 66, 47),
        (138, 108, 76),
        (26, 18, 12),
        (230, 216, 198),
        (168, 148, 124),
    ),
    MenuTheme(
        "Midnight",
        (34, 42, 66),
        (22, 28, 45),
        (48, 58, 88),
        (86, 102, 140),
        (12, 16, 26),
        (214, 222, 242),
        (140, 152, 182),
    ),
    MenuTheme(
        "Crimson",
        (62, 38, 42),
        (42, 25, 28),
        (84, 52, 58),
        (136, 84, 92),
        (26, 14, 16),
        (236, 214, 216),
        (178, 144, 148),
    ),
    MenuTheme(
        "Forest",
        (38, 56, 42),
        (25, 38, 29),
        (52, 74, 56),
        (88, 118, 92),
        (14, 22, 17),
        (214, 232, 216),
        (144, 168, 148),
    ),
    MenuTheme(
        "Mono",
        (52, 52, 52),
        (34, 34, 34),
        (70, 70, 70),
        (110, 110, 110),
        (18, 18, 18),
        (224, 224, 224),
        (160, 160, 160),
    ),
)

#: Highlight colours offered in Options > Interface.
ACCENT_PRESETS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("Gold", (214, 196, 96)),
    ("Amber", (255, 196, 92)),
    ("Orange", (255, 138, 92)),
    ("Red", (255, 107, 107)),
    ("Magenta", (255, 114, 214)),
    ("Violet", (181, 140, 255)),
    ("Blue", (122, 162, 255)),
    ("Cyan", (111, 227, 255)),
    ("Mint", (121, 242, 178)),
    ("Lime", (184, 242, 107)),
    ("White", (232, 232, 220)),
)
DEFAULT_ACCENT = ACCENT_PRESETS[0][1]

#: Category tabs, in display order.
TABS: tuple[tuple[str, str], ...] = (
    ("video", "Video"),
    ("mouse", "Mouse"),
    ("game", "Game"),
    ("crosshair", "Crosshair"),
    ("interface", "Interface"),
    ("controls", "Controls"),
)


def _font(names: str, size: int, bold: bool = False) -> pygame.font.Font:
    path = pygame.font.match_font(names, bold=bold)
    if path:
        font = pygame.font.Font(path, size)
    else:
        font = pygame.font.Font(None, size)
    font.set_bold(bold)
    return font


def _mix(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    return tuple(
        round(first[channel] + (second[channel] - first[channel]) * amount)
        for channel in range(3)
    )


def _lighten(color: tuple[int, int, int], amount: float = 0.08) -> tuple[int, int, int]:
    return _mix(color, (255, 255, 255), amount)


def _darken(color: tuple[int, int, int], amount: float = 0.08) -> tuple[int, int, int]:
    return _mix(color, (0, 0, 0), amount)


class UI:
    """Collects the overlay drawing primitives and caches text surfaces."""

    TABS = TABS
    THEMES = MENU_THEMES
    ACCENTS = ACCENT_PRESETS

    def __init__(self) -> None:
        # Tahoma was the small UI font of the GoldSource era; the fallbacks
        # keep the same compact proportions on Linux.
        self.font = _font("tahoma,verdana,segoeui,dejavusans", 12)
        self.font_bold = _font("tahoma,verdana,segoeui,dejavusans", 12, True)
        self.font_small = _font("tahoma,verdana,segoeui,dejavusans", 11)
        self.font_title = _font("tahoma,verdana,segoeui,dejavusans", 12, True)
        self.mono = _font("consolas,couriernew,dejavusansmono", 13)
        self.mono_small = _font("consolas,couriernew,dejavusansmono", 11)
        self.mono_bold = _font("consolas,couriernew,dejavusansmono", 12, True)
        self.toast = _font("tahoma,verdana,segoeui,dejavusans", 14, True)
        # The front-end menu is a little larger than the in-game HUD.
        self.menu_font = _font("tahoma,verdana,segoeui,dejavusans", 15)
        self.menu_small = _font(
            "tahoma,verdana,segoeui,dejavusans", 10, True
        )
        self.menu_logo = _font(
            "tahoma,verdana,segoeui,dejavusans", 18, True
        )

        # Public aliases kept for the rest of the engine.
        self.ui = self.font
        self.ui_bold = self.font_bold
        self.title = self.font_title

        self._cache: dict[tuple, pygame.Surface] = {}
        self.theme_index = 0
        self.accent_index = 0
        self.menu_opacity = 0.96

        # Options window state.
        self.settings_offset = [0, 0]
        self.open_dropdown: str | None = None
        self._drag_start: tuple[int, int] | None = None
        self._drag_start_offset = (0, 0)

        # Front-end menu caches.
        self._menu_bg_source: pygame.Surface | None = None
        self._menu_bg_surface: pygame.Surface | None = None
        self._menu_bg_size: tuple[int, int] | None = None
        self._dim_cache: dict[tuple[int, int, int], pygame.Surface] = {}
        self._settings_layer: pygame.Surface | None = None
        self._settings_layer_size: tuple[int, int] | None = None
        self._map_layer: pygame.Surface | None = None
        self._map_layer_size: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Theme and accent
    # ------------------------------------------------------------------
    @property
    def theme(self) -> MenuTheme:
        return MENU_THEMES[self.theme_index]

    @property
    def accent(self) -> tuple[int, int, int]:
        return ACCENT_PRESETS[self.accent_index][1]

    @property
    def accent_name(self) -> str:
        return ACCENT_PRESETS[self.accent_index][0]

    @property
    def accent_hex(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.accent)

    @property
    def theme_names(self) -> tuple[str, ...]:
        return tuple(theme.name for theme in MENU_THEMES)

    @property
    def accent_names(self) -> tuple[str, ...]:
        return tuple(name for name, _color in ACCENT_PRESETS)

    def set_theme(self, index: int) -> None:
        self.theme_index = index % len(MENU_THEMES)

    def set_accent(self, index: int) -> None:
        self.accent_index = index % len(ACCENT_PRESETS)

    def cycle_accent(self, direction: int) -> None:
        self.set_accent(self.accent_index + direction)

    # ------------------------------------------------------------------
    # Window dragging
    # ------------------------------------------------------------------
    @property
    def dragging(self) -> bool:
        return self._drag_start is not None

    def start_drag(self, position: tuple[int, int]) -> None:
        self._drag_start = position
        self._drag_start_offset = tuple(self.settings_offset)

    def drag_to(self, position: tuple[int, int]) -> None:
        if self._drag_start is None:
            return
        self.settings_offset[0] = (
            self._drag_start_offset[0] + position[0] - self._drag_start[0]
        )
        self.settings_offset[1] = (
            self._drag_start_offset[1] + position[1] - self._drag_start[1]
        )

    def end_drag(self) -> None:
        self._drag_start = None

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    def _text_surface(
        self, font: pygame.font.Font, text: str, color: tuple[int, int, int]
    ) -> pygame.Surface:
        key = (id(font), text, color)
        surface = self._cache.get(key)
        if surface is None:
            if len(self._cache) > 800:
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
        shadow: bool = False,
    ) -> pygame.Rect:
        surface = self._text_surface(font, text, color)
        rect = surface.get_rect()
        setattr(rect, anchor, (round(position[0]), round(position[1])))
        if shadow:
            shadow_surface = self._text_surface(font, text, (0, 0, 0))
            target.blit(shadow_surface, rect.move(1, 1))
        target.blit(surface, rect)
        return rect

    def _ellipsize(
        self, font: pygame.font.Font, text: str, max_width: int
    ) -> str:
        """Trim long labels so a list row never spills into the next column."""
        if max_width <= 0:
            return ""
        if font.size(text)[0] <= max_width:
            return text
        ellipsis = "..."
        if font.size(ellipsis)[0] > max_width:
            return ""
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = text[:middle] + ellipsis
            if font.size(candidate)[0] <= max_width:
                low = middle
            else:
                high = middle - 1
        return text[:low] + ellipsis

    # ------------------------------------------------------------------
    # Options window layout
    # ------------------------------------------------------------------
    def settings_layout(
        self, width: int, height: int, state: dict
    ) -> dict:
        """Build every rectangle used by drawing and hit testing.

        The same layout is used by :meth:`draw_settings` and by the engine's
        mouse/keyboard handlers, so clicks always land on what is visible.
        """
        values: dict = state.get("values", {})
        options: dict = state.get("options", {})

        window_w = min(WINDOW_W, max(420, width - 32))
        window_h = min(WINDOW_H, max(320, height - 32))
        window_w = min(window_w, max(320, width - 8))
        window_h = min(window_h, max(240, height - 8))

        base_x = (width - window_w) // 2
        base_y = (height - window_h) // 2
        x = base_x + self.settings_offset[0]
        y = base_y + self.settings_offset[1]
        x = max(8, min(width - window_w - 8, x))
        y = max(8, min(height - window_h - 8, y))
        # Keep the stored offset in sync after clamping so dragging feels
        # continuous even when the window hits a screen edge.
        self.settings_offset[0] = x - base_x
        self.settings_offset[1] = y - base_y

        window = pygame.Rect(x, y, window_w, window_h)
        title = pygame.Rect(x, y, window_w, TITLE_H)
        close = pygame.Rect(x + window_w - 22, y + 4, 15, 15)

        tabs: list[tuple[str, str, pygame.Rect]] = []
        tab_x = x + 7
        for tab_id, label in TABS:
            tab_w = max(60, self.font.size(label)[0] + 22)
            if tab_x + tab_w > x + window_w - 7:
                tab_w = max(30, x + window_w - 7 - tab_x)
            tabs.append(
                (tab_id, label, pygame.Rect(tab_x, y + TITLE_H, tab_w, TABS_H))
            )
            tab_x += tab_w + 2

        content = pygame.Rect(
            x + 7,
            y + TITLE_H + TABS_H,
            window_w - 14,
            window_h - TITLE_H - TABS_H - FOOTER_H,
        )
        footer = pygame.Rect(
            x + 7, y + window_h - FOOTER_H, window_w - 14, FOOTER_H
        )

        button_w = 72
        button_h = 23
        button_gap = 7
        button_x = x + window_w - 10 - (button_w * 3 + button_gap * 2)
        button_y = y + window_h - 35
        buttons: dict[str, pygame.Rect] = {}
        for button_id, _label in (
            ("ok", "OK"),
            ("cancel", "Cancel"),
            ("apply", "Apply"),
        ):
            buttons[button_id] = pygame.Rect(
                button_x, button_y, button_w, button_h
            )
            button_x += button_w + button_gap

        widgets: dict[str, dict] = {}
        order: list[str] = []
        groups: list[tuple[str, int, int]] = []
        notes: list[tuple[str, int, int]] = []
        extras: dict = {}

        def option_list(key: str) -> list[str]:
            return list(options.get(key, ()))

        def dropdown(
            widget_id: str,
            label: str,
            wx: int,
            wy: int,
            ww: int,
            option_key: str | None = None,
        ) -> None:
            choices = option_list(option_key or widget_id)
            value = int(values.get(widget_id, 0))
            value = max(0, min(len(choices) - 1, value)) if choices else 0
            field = pygame.Rect(wx, wy + 16, ww, 24)
            swatch = None
            if widget_id == "menu_theme" and MENU_THEMES:
                swatch = MENU_THEMES[value % len(MENU_THEMES)].panel
            elif widget_id == "accent" and ACCENT_PRESETS:
                swatch = ACCENT_PRESETS[value % len(ACCENT_PRESETS)][1]
            elif widget_id == "crosshair_color":
                if value >= len(Crosshair.COLORS):
                    swatch = (
                        int(values.get("crosshair_custom_r", 255)),
                        int(values.get("crosshair_custom_g", 255)),
                        int(values.get("crosshair_custom_b", 255)),
                    )
                elif Crosshair.COLORS:
                    swatch = Crosshair.COLORS[value % len(Crosshair.COLORS)][1]
                    if swatch is None:
                        accent_index = int(values.get("accent", 0))
                        swatch = ACCENT_PRESETS[
                            accent_index % len(ACCENT_PRESETS)
                        ][1]
            widgets[widget_id] = {
                "kind": "dropdown",
                "label": label,
                "rect": pygame.Rect(wx, wy, ww, 42),
                "field": field,
                "options": choices,
                "value": value,
                "swatch": swatch,
            }
            order.append(widget_id)

        def checkbox(
            widget_id: str, label: str, wx: int, wy: int, ww: int
        ) -> None:
            box = pygame.Rect(wx, wy + 1, 14, 14)
            widgets[widget_id] = {
                "kind": "checkbox",
                "label": label,
                "rect": pygame.Rect(wx, wy, ww, 20),
                "box": box,
                "value": bool(values.get(widget_id, False)),
            }
            order.append(widget_id)

        def slider(
            widget_id: str,
            label: str,
            wx: int,
            wy: int,
            ww: int,
            minimum: float,
            maximum: float,
            step: float,
            integer: bool = False,
        ) -> None:
            value = values.get(widget_id, minimum)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = minimum
            value = max(minimum, min(maximum, value))
            track = pygame.Rect(wx + 5, wy + 24, max(20, ww - 10), 12)
            widgets[widget_id] = {
                "kind": "slider",
                "label": label,
                "rect": pygame.Rect(wx, wy, ww, 50),
                "track": track,
                "value": value,
                "min": minimum,
                "max": maximum,
                "step": step,
                "integer": integer,
            }
            order.append(widget_id)

        def button(
            widget_id: str,
            label: str,
            wx: int,
            wy: int,
            ww: int,
            wh: int = 23,
        ) -> None:
            widgets[widget_id] = {
                "kind": "button",
                "label": label,
                "rect": pygame.Rect(wx, wy, ww, wh),
            }
            order.append(widget_id)

        tab_index = int(state.get("tab", 0)) % len(TABS)
        tab_id = TABS[tab_index][0]
        content_x = content.left + 10
        content_w = content.width - 20

        if tab_id == "video":
            gap = 24
            col_w = max(160, (content_w - gap) // 2)
            left = content_x
            right = content_x + col_w + gap
            row_y = content.top + 8
            groups.append(("Renderer", left, row_y))
            row_y += 22
            dropdown("render_mode", "Render mode", left, row_y, col_w)
            dropdown("resolution", "Resolution", left, row_y + 52, col_w)
            checkbox("windowed", "Run in a window", right, row_y + 4, col_w)
            checkbox(
                "vsync", "Wait for vertical sync", right, row_y + 34, col_w
            )
            row_y += 116
            slider(
                "brightness", "Brightness", left, row_y, col_w,
                0.5, 1.5, 0.05,
            )
            slider(
                "gamma", "Gamma", right, row_y, col_w,
                0.5, 2.0, 0.05,
            )
            groups.append(("View", left, row_y + 58))
            slider(
                "fov", "Field of view", left, row_y + 80, col_w,
                60, 110, 1,
            )
            notes.append(
                (
                    "Note: changing video options will recreate the game "
                    "window.",
                    content_x,
                    content.bottom - 24,
                )
            )

        elif tab_id == "mouse":
            row_y = content.top + 8
            groups.append(("Mouse", content_x, row_y))
            row_y += 22
            slider(
                "sensitivity", "Sensitivity", content_x, row_y,
                min(340, content_w), 1.0, 10.0, 0.1,
            )
            row_y += 58
            checkbox(
                "invert_y", "Invert vertical axis", content_x, row_y, 320
            )
            row_y += 38
            groups.append(("Crosshair", content_x, row_y))
            row_y += 22
            checkbox(
                "crosshair_enabled", "Enable crosshair",
                content_x, row_y, 320,
            )
            button(
                "open_crosshair", "Customize crosshair...",
                content_x, row_y + 28, 160,
            )
            notes.append(
                (
                    "Sensitivity and inversion are applied when you return "
                    "to the game.",
                    content_x,
                    content.bottom - 24,
                )
            )

        elif tab_id == "game":
            row_y = content.top + 8
            groups.append(("Movement", content_x, row_y))
            row_y += 22
            slider(
                "player_speed", "Player speed", content_x, row_y,
                min(340, content_w), 3.0, 15.0, 0.5,
            )
            row_y += 58
            checkbox(
                "auto_bunnyhop", "Auto bunnyhop", content_x, row_y, 320
            )
            row_y += 40
            groups.append(("Defaults", content_x, row_y))
            button(
                "defaults", "Reset all options",
                content_x, row_y + 24, 160,
            )
            notes.append(
                (
                    "Auto bunnyhop keeps the jump key held for continuous "
                    "hops.",
                    content_x,
                    content.bottom - 24,
                )
            )

        elif tab_id == "crosshair":
            preview_w = min(176, max(132, content.width // 4))
            left_area = content.width - preview_w - 36
            gap = 16
            col_w = max(120, (left_area - gap) // 2)
            left = content_x
            right = content_x + col_w + gap
            groups.append(("Crosshair", content_x, content.top + 8))
            row_y = content.top + 30
            checkbox("crosshair_enabled", "Enable crosshair", left, row_y, col_w)
            row_y += 28
            dropdown("crosshair_color", "Color", left, row_y, col_w)
            row_y += 52
            slider(
                "crosshair_size", "Size", left, row_y, col_w,
                2, 48, 2, integer=True,
            )
            row_y += 52
            slider(
                "crosshair_gap", "Gap", left, row_y, col_w,
                0, 40, 1, integer=True,
            )
            row_y += 52
            slider(
                "crosshair_thickness", "Thickness", left, row_y, col_w,
                1, 10, 1, integer=True,
            )

            # Independent RGB colour, used when the Color dropdown is set to
            # "Custom".  Touching any of these switches the dropdown over.
            rgb_y = content.top + 272
            groups.append(("Custom color", left, rgb_y))
            rgb_gap = 6
            rgb_w = max(40, (left_area - rgb_gap * 2) // 3)
            for index, (widget_id, label) in enumerate(
                (
                    ("crosshair_custom_r", "Red"),
                    ("crosshair_custom_g", "Green"),
                    ("crosshair_custom_b", "Blue"),
                )
            ):
                slider(
                    widget_id, label,
                    left + index * (rgb_w + rgb_gap),
                    rgb_y + 22, rgb_w,
                    0, 255, 1, integer=True,
                )

            row_y = content.top + 30
            checkbox("crosshair_dot", "Dot", right, row_y, col_w)
            row_y += 28
            slider(
                "crosshair_dot_size", "Dot size", right, row_y, col_w,
                1, 16, 1, integer=True,
            )
            row_y += 52
            checkbox("crosshair_outline", "Outline", right, row_y, col_w)
            row_y += 28
            slider(
                "crosshair_outline_width", "Outline width", right, row_y,
                col_w, 1, 4, 1, integer=True,
            )
            row_y += 52
            slider(
                "crosshair_opacity", "Opacity", right, row_y, col_w,
                20, 100, 5, integer=True,
            )

            extras["preview"] = pygame.Rect(
                content.right - preview_w - 10,
                content.top + 46,
                preview_w,
                preview_w,
            )
            notes.append(
                (
                    "Moving an RGB slider switches Color to Custom.",
                    content_x,
                    content.bottom - 24,
                )
            )

        elif tab_id == "interface":
            row_y = content.top + 8
            groups.append(("Menu", content_x, row_y))
            row_y += 22
            dropdown("menu_theme", "Menu color", content_x, row_y, 300)
            dropdown(
                "accent", "Highlight color", content_x, row_y + 52, 300
            )
            slider(
                "menu_opacity", "Menu opacity",
                content_x, row_y + 104, 300, 0.55, 1.0, 0.05,
            )
            row_y += 184
            groups.append(("HUD", content_x, row_y))
            row_y += 22
            checkbox("show_hud", "Show HUD hints", content_x, row_y, 320)
            checkbox(
                "debug_overlay", "Show debug overlay (F3)",
                content_x, row_y + 28, 340,
            )
            notes.append(
                (
                    "Menu color applies to every HUD window, including this "
                    "one.",
                    content_x,
                    content.bottom - 24,
                )
            )

        elif tab_id == "controls":
            extras["controls"] = True

        for button_id, label in (
            ("ok", "OK"),
            ("cancel", "Cancel"),
            ("apply", "Apply"),
        ):
            widgets[button_id] = {
                "kind": "button",
                "label": label,
                "rect": buttons[button_id],
                "footer": True,
            }
            order.append(button_id)

        layout: dict = {
            "window": window,
            "title": title,
            "close": close,
            "tabs": tabs,
            "content": content,
            "footer": footer,
            "buttons": buttons,
            "widgets": widgets,
            "order": order,
            "groups": groups,
            "notes": notes,
            "dropdown": None,
            "tab": tab_index,
            "tab_id": tab_id,
        }
        layout.update(extras)

        open_id = self.open_dropdown
        if open_id is not None:
            widget = widgets.get(open_id)
            if widget is None or widget["kind"] != "dropdown":
                self.open_dropdown = None
            else:
                field = widget["field"]
                choices = widget["options"]
                row_h = 20
                list_h = row_h * len(choices) + 2
                list_y = field.bottom + 1
                if list_y + list_h > window.bottom - 6:
                    list_y = field.top - list_h - 1
                list_y = max(
                    y + TITLE_H + TABS_H + 1,
                    min(y + window_h - list_h - 1, list_y),
                )
                list_rect = pygame.Rect(
                    field.left, list_y, field.width, list_h
                )
                option_rects = [
                    pygame.Rect(
                        field.left + 1,
                        list_y + 1 + index * row_h,
                        field.width - 2,
                        row_h,
                    )
                    for index in range(len(choices))
                ]
                layout["dropdown"] = {
                    "id": open_id,
                    "list": list_rect,
                    "rects": option_rects,
                    "options": choices,
                    "value": widget["value"],
                }

        return layout

    # ------------------------------------------------------------------
    # Bevel helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bevel(
        target: pygame.Surface,
        rect: pygame.Rect,
        theme: MenuTheme,
        fill: tuple[int, int, int],
        raised: bool = True,
    ) -> None:
        pygame.draw.rect(target, fill, rect)
        pygame.draw.rect(target, theme.border_dark, rect, 1)
        top_left = theme.light if raised else theme.border_dark
        bottom_right = theme.border_dark if raised else theme.light
        pygame.draw.line(
            target, top_left,
            (rect.left + 1, rect.top + 1),
            (rect.right - 2, rect.top + 1),
        )
        pygame.draw.line(
            target, top_left,
            (rect.left + 1, rect.top + 1),
            (rect.left + 1, rect.bottom - 2),
        )
        pygame.draw.line(
            target, bottom_right,
            (rect.left + 1, rect.bottom - 2),
            (rect.right - 2, rect.bottom - 2),
        )
        pygame.draw.line(
            target, bottom_right,
            (rect.right - 2, rect.top + 1),
            (rect.right - 2, rect.bottom - 2),
        )

    @staticmethod
    def _draw_gear(
        target: pygame.Surface,
        rect: pygame.Rect,
        color: tuple[int, int, int],
        hole: tuple[int, int, int],
    ) -> None:
        """Small cog icon used in the options title bar."""
        cx, cy = rect.center
        points: list[tuple[float, float]] = []
        for index in range(16):
            angle = (index / 16.0) * math.tau
            radius = 6.2 if index % 2 == 0 else 4.2
            points.append(
                (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            )
        pygame.draw.polygon(target, color, points)
        pygame.draw.circle(target, hole, (cx, cy), 2)

    # ------------------------------------------------------------------
    # Options window drawing
    # ------------------------------------------------------------------
    def draw_settings(
        self,
        target: pygame.Surface,
        width: int,
        height: int,
        state: dict,
        dim: bool = True,
    ) -> None:
        if dim:
            self.draw_dim(target, width, height, DIM_ALPHA)

        values: dict = state.get("values", {})
        theme = MENU_THEMES[
            int(values.get("menu_theme", 0)) % len(MENU_THEMES)
        ]
        accent = ACCENT_PRESETS[
            int(values.get("accent", 0)) % len(ACCENT_PRESETS)
        ][1]

        opacity = float(values.get("menu_opacity", self.menu_opacity))
        opacity = max(0.0, min(1.0, opacity))
        panel_alpha = max(80, round(opacity * 255))

        layout = self.settings_layout(width, height, state)
        mouse = pygame.mouse.get_pos()

        # Compose the whole window on a transparent layer and blit it in one
        # go. pygame.draw.* replaces destination alpha instead of blending, so
        # a translucent panel drawn directly onto the overlay would punch a
        # hole through the menu backdrop and reveal whatever is behind it.
        layer = self._settings_layer_surface(width, height)

        self._draw_window_chrome(
            layer, layout, theme, accent, mouse, panel_alpha
        )

        content = layout["content"]
        pygame.draw.rect(layer, theme.border_dark, content, 1)
        pygame.draw.line(
            layer, theme.border,
            (content.left + 1, content.top + 1),
            (content.right - 2, content.top + 1),
        )

        self._draw_tabs(layer, layout, theme, accent, mouse)

        for label, gx, gy in layout["groups"]:
            self.text(
                layer, self.font_bold, label, (gx, gy), accent,
                shadow=True,
            )

        order = layout["order"]
        cursor = int(state.get("cursor", 0))
        focused_id = order[cursor % len(order)] if order else None
        for widget_id in order:
            widget = layout["widgets"][widget_id]
            if widget.get("footer"):
                continue
            self._draw_widget(
                layer,
                widget_id,
                widget,
                theme,
                accent,
                focused=widget_id == focused_id,
                hovered=widget["rect"].collidepoint(mouse),
                open_dropdown=self.open_dropdown == widget_id,
            )

        if layout["tab_id"] == "crosshair":
            self._draw_crosshair_preview(
                layer, layout, theme, accent, state.get("crosshair")
            )
        elif layout["tab_id"] == "controls":
            self._draw_controls(layer, layout, theme, accent)

        for note, nx, ny in layout["notes"]:
            self.text(
                layer, self.font_small, note, (nx, ny), theme.muted,
                shadow=True,
            )

        self._draw_footer(
            layer, layout, theme, accent, focused_id, mouse
        )

        dropdown = layout.get("dropdown")
        if dropdown is not None:
            self._draw_dropdown_list(
                layer, dropdown, theme, accent, mouse
            )

        target.blit(layer, (0, 0))

    def _settings_layer_surface(
        self, width: int, height: int
    ) -> pygame.Surface:
        """Return a cleared transparent layer for the options window."""
        if (
            self._settings_layer is None
            or self._settings_layer_size != (width, height)
        ):
            self._settings_layer = pygame.Surface(
                (width, height), pygame.SRCALPHA
            )
            self._settings_layer_size = (width, height)
        else:
            self._settings_layer.fill((0, 0, 0, 0))
        return self._settings_layer

    def _draw_window_chrome(
        self,
        target: pygame.Surface,
        layout: dict,
        theme: MenuTheme,
        accent: tuple[int, int, int],
        mouse: tuple[int, int],
        panel_alpha: int = PANEL_ALPHA,
        title_text: str = "Options",
        show_gear: bool = True,
    ) -> None:
        window: pygame.Rect = layout["window"]
        title: pygame.Rect = layout["title"]
        close: pygame.Rect = layout["close"]

        pygame.draw.rect(target, (0, 0, 0, 115), window.move(5, 5))
        pygame.draw.rect(target, (*theme.panel, panel_alpha), window)

        pygame.draw.rect(target, theme.border_dark, window, 1)
        pygame.draw.line(
            target, theme.border,
            (window.left + 1, window.top + 1),
            (window.right - 2, window.top + 1),
        )
        pygame.draw.line(
            target, theme.border,
            (window.left + 1, window.top + 1),
            (window.left + 1, window.bottom - 2),
        )
        pygame.draw.line(
            target, theme.border_dark,
            (window.left + 1, window.bottom - 2),
            (window.right - 2, window.bottom - 2),
        )
        pygame.draw.line(
            target, theme.border_dark,
            (window.right - 2, window.top + 1),
            (window.right - 2, window.bottom - 2),
        )

        pygame.draw.rect(target, theme.light, title)
        pygame.draw.line(
            target, theme.border,
            (title.left, title.top), (title.right, title.top),
        )
        pygame.draw.line(
            target, theme.border_dark,
            (title.left, title.bottom - 1),
            (title.right, title.bottom - 1),
        )

        title_x = title.left + 8
        if show_gear:
            self._draw_gear(
                target,
                pygame.Rect(title.left + 6, title.top + 5, 14, 14),
                accent,
                theme.light,
            )
            title_x = title.left + 27
        self.text(
            target, self.font_title, title_text,
            (title_x, title.top + 5), theme.text,
            shadow=True,
        )

        close_hover = close.collidepoint(mouse)
        close_fill = _lighten(theme.light, 0.16) if close_hover else theme.light
        self._bevel(target, close, theme, close_fill, raised=True)
        pygame.draw.line(
            target, theme.text,
            (close.left + 4, close.top + 4),
            (close.right - 4, close.bottom - 4),
        )
        pygame.draw.line(
            target, theme.text,
            (close.right - 4, close.top + 4),
            (close.left + 4, close.bottom - 4),
        )

    def _draw_tabs(
        self,
        target: pygame.Surface,
        layout: dict,
        theme: MenuTheme,
        accent: tuple[int, int, int],
        mouse: tuple[int, int],
    ) -> None:
        tabs: list[tuple[str, str, pygame.Rect]] = layout["tabs"]
        active_index = layout["tab"]

        for index, (_tab_id, label, rect) in enumerate(tabs):
            if index == active_index:
                continue
            hovered = rect.collidepoint(mouse)
            fill = _lighten(theme.dark, 0.10) if hovered else theme.dark
            pygame.draw.rect(target, fill, rect)
            pygame.draw.line(
                target, theme.light,
                (rect.left, rect.top), (rect.right - 1, rect.top),
            )
            pygame.draw.line(
                target, theme.border_dark,
                (rect.left, rect.bottom - 1),
                (rect.right - 1, rect.bottom - 1),
            )
            pygame.draw.line(
                target, theme.border_dark,
                (rect.left, rect.top), (rect.left, rect.bottom - 1),
            )
            pygame.draw.line(
                target, theme.border_dark,
                (rect.right - 1, rect.top),
                (rect.right - 1, rect.bottom - 1),
            )
            self.text(
                target, self.font, label, rect.center,
                theme.muted, anchor="center", shadow=True,
            )

        for index, (_tab_id, label, rect) in enumerate(tabs):
            if index != active_index:
                continue
            active = pygame.Rect(rect.left, rect.top, rect.width, rect.height + 1)
            pygame.draw.rect(target, theme.panel, active)
            pygame.draw.line(
                target, theme.border,
                (active.left, active.top), (active.right - 1, active.top),
            )
            pygame.draw.line(
                target, theme.border,
                (active.left, active.top), (active.left, active.bottom - 1),
            )
            pygame.draw.line(
                target, theme.border,
                (active.right - 1, active.top),
                (active.right - 1, active.bottom - 1),
            )
            self.text(
                target, self.font_bold, label, rect.center,
                accent, anchor="center", shadow=True,
            )

    def _draw_widget(
        self,
        target: pygame.Surface,
        widget_id: str,
        widget: dict,
        theme: MenuTheme,
        accent: tuple[int, int, int],
        focused: bool,
        hovered: bool,
        open_dropdown: bool,
    ) -> None:
        kind = widget["kind"]
        if kind == "checkbox":
            self._draw_checkbox(
                target, widget, theme, accent, focused, hovered
            )
        elif kind == "dropdown":
            self._draw_dropdown(
                target, widget_id, widget, theme, accent,
                focused, hovered, open_dropdown,
            )
        elif kind == "slider":
            self._draw_slider(
                target, widget_id, widget, theme, accent,
                focused, hovered,
            )
        elif kind == "button":
            self._draw_button(
                target, widget, theme, accent, focused, hovered
            )

    def _draw_checkbox(
        self,
        target: pygame.Surface,
        widget: dict,
        theme: MenuTheme,
        accent: tuple[int, int, int],
        focused: bool,
        hovered: bool,
    ) -> None:
        box: pygame.Rect = widget["box"]
        fill = _lighten(theme.dark, 0.10) if hovered else theme.dark
        self._bevel(target, box, theme, fill, raised=False)
        if widget["value"]:
            pygame.draw.lines(
                target, accent, False,
                (
                    (box.left + 3, box.centery),
                    (box.centerx - 1, box.bottom - 4),
                    (box.right - 3, box.top + 3),
                ),
                2,
            )
        self.text(
            target, self.font, widget["label"],
            (box.right + 7, box.top - 2), theme.text,
            shadow=True,
        )
        if focused:
            pygame.draw.rect(
                target, accent, box.inflate(4, 4), 1
            )

    def _draw_dropdown(
        self,
        target: pygame.Surface,
        widget_id: str,
        widget: dict,
        theme: MenuTheme,
        accent: tuple[int, int, int],
        focused: bool,
        hovered: bool,
        open_dropdown: bool,
    ) -> None:
        field: pygame.Rect = widget["field"]
        self.text(
            target, self.font, widget["label"],
            (field.left, field.top - 16), theme.text,
            shadow=True,
        )
        fill = theme.dark
        if open_dropdown:
            fill = _mix(theme.dark, accent, 0.16)
        elif hovered:
            fill = _lighten(theme.dark, 0.12)
        self._bevel(target, field, theme, fill, raised=False)

        text_x = field.left + 6
        swatch = widget.get("swatch")
        if swatch is not None:
            swatch_rect = pygame.Rect(
                field.left + 6, field.centery - 5, 10, 10
            )
            pygame.draw.rect(target, swatch, swatch_rect)
            pygame.draw.rect(target, theme.border_dark, swatch_rect, 1)
            text_x = swatch_rect.right + 6

        choices = widget["options"]
        value = widget["value"]
        label = choices[value] if choices else "—"
        self.text(
            target, self.font, label,
            (text_x, field.centery), theme.text,
            anchor="midleft", shadow=True,
        )

        arrow = pygame.Rect(
            field.right - 18, field.top + 2, 16, field.height - 4
        )
        self._bevel(target, arrow, theme, theme.light, raised=True)
        arrow_center = arrow.center
        pygame.draw.polygon(
            target, theme.text,
            (
                (arrow_center[0] - 4, arrow_center[1] - 2),
                (arrow_center[0] + 4, arrow_center[1] - 2),
                (arrow_center[0], arrow_center[1] + 3),
            ),
        )
        if focused:
            pygame.draw.rect(target, accent, field.inflate(4, 4), 1)

    def _slider_fraction(self, widget: dict) -> float:
        minimum = float(widget["min"])
        maximum = float(widget["max"])
        value = float(widget["value"])
        if maximum <= minimum:
            return 0.0
        return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))

    @staticmethod
    def _slider_value_text(widget_id: str, value: float) -> str:
        if widget_id in ("brightness", "gamma"):
            return f"{value:.2f}"
        if widget_id == "sensitivity":
            return f"{value:.1f}"
        if widget_id == "player_speed":
            return f"{value:.1f} m/s"
        if widget_id == "fov":
            return f"{int(round(value))}°"
        if widget_id == "menu_opacity":
            return f"{int(round(value * 100))} %"
        if widget_id == "crosshair_size":
            return f"{int(round(value))} px"
        if widget_id == "crosshair_gap":
            return f"{int(round(value))} px"
        if widget_id == "crosshair_thickness":
            return f"{int(round(value))} px"
        if widget_id == "crosshair_dot_size":
            return f"{int(round(value))} px"
        if widget_id == "crosshair_outline_width":
            return f"{int(round(value))} px"
        if widget_id == "crosshair_opacity":
            return f"{int(round(value))} %"
        if widget_id.startswith("crosshair_custom_"):
            return str(int(round(value)))
        return f"{value:.1f}"

    def _draw_slider(
        self,
        target: pygame.Surface,
        widget_id: str,
        widget: dict,
        theme: MenuTheme,
        accent: tuple[int, int, int],
        focused: bool,
        hovered: bool,
    ) -> None:
        track: pygame.Rect = widget["track"]
        self.text(
            target, self.font, widget["label"],
            (track.left - 5, track.top - 24), theme.text,
            shadow=True,
        )
        self.text(
            target, self.font, self._slider_value_text(
                widget_id, float(widget["value"])
            ),
            (track.right + 5, track.top - 24), accent,
            anchor="topright", shadow=True,
        )

        pygame.draw.line(
            target, theme.border_dark,
            (track.left, track.centery),
            (track.right, track.centery),
            2,
        )
        for index in range(9):
            tick_x = track.left + round(track.width * index / 8)
            pygame.draw.line(
                target, theme.border_dark,
                (tick_x, track.centery - 3),
                (tick_x, track.centery + 3),
            )

        fraction = self._slider_fraction(widget)
        handle_x = track.left + round(track.width * fraction)
        pygame.draw.line(
            target, accent,
            (track.left, track.centery),
            (handle_x, track.centery),
            1,
        )
        handle = pygame.Rect(handle_x - 4, track.centery - 7, 9, 14)
        fill = theme.light
        if hovered or focused:
            fill = _lighten(theme.light, 0.10)
        self._bevel(target, handle, theme, fill, raised=True)

        left_hint, right_hint = self._slider_hints(widget_id)
        if left_hint:
            self.text(
                target, self.font_small, left_hint,
                (track.left, track.bottom + 1), theme.muted,
                shadow=True,
            )
        if right_hint:
            self.text(
                target, self.font_small, right_hint,
                (track.right, track.bottom + 1), theme.muted,
                anchor="topright", shadow=True,
            )
        if focused:
            pygame.draw.rect(
                target, accent, track.inflate(8, 12), 1
            )

    @staticmethod
    def _slider_hints(widget_id: str) -> tuple[str, str]:
        if widget_id in ("brightness", "gamma"):
            return "Dark", "Light"
        if widget_id == "sensitivity":
            return "Low", "High"
        if widget_id == "player_speed":
            return "Slow", "Fast"
        if widget_id == "fov":
            return "Narrow", "Wide"
        if widget_id == "menu_opacity":
            return "Clear", "Solid"
        if widget_id == "crosshair_opacity":
            return "Faint", "Solid"
        return "", ""

    def _draw_button(
        self,
        target: pygame.Surface,
        widget: dict,
        theme: MenuTheme,
        accent: tuple[int, int, int],
        focused: bool,
        hovered: bool,
    ) -> None:
        rect: pygame.Rect = widget["rect"]
        disabled = bool(widget.get("disabled", False))
        fill = theme.dark if disabled else theme.panel
        if not disabled and (hovered or focused):
            fill = _lighten(theme.panel, 0.14)
        self._bevel(target, rect, theme, fill, raised=True)
        self.text(
            target, self.font, widget["label"], rect.center,
            theme.muted if disabled else theme.text,
            anchor="center", shadow=True,
        )
        if focused and not disabled:
            pygame.draw.rect(target, accent, rect.inflate(4, 4), 1)

    def _draw_footer(
        self,
        target: pygame.Surface,
        layout: dict,
        theme: MenuTheme,
        accent: tuple[int, int, int],
        focused_id: str | None,
        mouse: tuple[int, int],
    ) -> None:
        footer: pygame.Rect = layout["footer"]
        pygame.draw.line(
            target, theme.border_dark,
            (footer.left, footer.top), (footer.right, footer.top),
        )
        self.text(
            target, self.font_small,
            "Tab page   ↑↓ option   ←→ change   Enter select   Esc cancel",
            (footer.left + 5, footer.top + 13), theme.muted,
            shadow=True,
        )
        for button_id in ("ok", "cancel", "apply"):
            widget = layout["widgets"][button_id]
            self._draw_button(
                target,
                widget,
                theme,
                accent,
                focused=button_id == focused_id,
                hovered=widget["rect"].collidepoint(mouse),
            )

    def _draw_dropdown_list(
        self,
        target: pygame.Surface,
        dropdown: dict,
        theme: MenuTheme,
        accent: tuple[int, int, int],
        mouse: tuple[int, int],
    ) -> None:
        list_rect: pygame.Rect = dropdown["list"]
        pygame.draw.rect(target, (0, 0, 0, 120), list_rect.move(3, 3))
        pygame.draw.rect(target, theme.dark, list_rect)
        pygame.draw.rect(target, theme.border_dark, list_rect, 1)
        pygame.draw.line(
            target, theme.light,
            (list_rect.left + 1, list_rect.top + 1),
            (list_rect.right - 2, list_rect.top + 1),
        )

        for index, option_rect in enumerate(dropdown["rects"]):
            selected = index == dropdown["value"]
            hovered = option_rect.collidepoint(mouse)
            if selected:
                fill = _mix(theme.dark, accent, 0.26)
            elif hovered:
                fill = _lighten(theme.dark, 0.14)
            else:
                fill = theme.dark
            pygame.draw.rect(target, fill, option_rect)
            color = accent if selected else theme.text
            self.text(
                target, self.font, dropdown["options"][index],
                (option_rect.left + 6, option_rect.centery),
                color, anchor="midleft", shadow=True,
            )

    # ------------------------------------------------------------------
    # New Game map browser
    # ------------------------------------------------------------------
    def map_browser_layout(
        self, width: int, height: int, state: dict
    ) -> dict:
        """Rects for the map list window, shared by drawing and hit testing."""
        entries: list[dict] = list(state.get("entries", ()))
        error = str(state.get("error", "")).strip()

        window_w = min(MAP_WINDOW_W, max(420, width - 32))
        window_h = min(MAP_WINDOW_H, max(320, height - 32))
        window_w = min(window_w, max(320, width - 8))
        window_h = min(window_h, max(240, height - 8))

        x = (width - window_w) // 2
        y = (height - window_h) // 2
        window = pygame.Rect(x, y, window_w, window_h)
        title = pygame.Rect(x, y, window_w, TITLE_H)
        close = pygame.Rect(x + window_w - 22, y + 4, 15, 15)

        footer = pygame.Rect(
            x + 10, window.bottom - MAP_FOOTER_H - 8, window_w - 20,
            MAP_FOOTER_H,
        )
        content = pygame.Rect(
            x + 10,
            y + TITLE_H + 8,
            window_w - 20,
            window_h - TITLE_H - MAP_FOOTER_H - 24,
        )
        header = pygame.Rect(content.left, content.top, content.width, 20)
        list_top = content.top + 24
        list_height = footer.top - list_top - 8
        if error:
            list_height -= 22
        list_rect = pygame.Rect(
            content.left, list_top, content.width, max(24, list_height)
        )
        error_rect = None
        if error:
            error_rect = pygame.Rect(
                content.left,
                list_rect.bottom + 3,
                content.width,
                18,
            )

        visible_rows = max(1, list_rect.height // MAP_ROW_H)
        max_scroll = max(0, len(entries) - visible_rows)
        scroll = max(0, min(max_scroll, int(state.get("scroll", 0))))
        cursor = int(state.get("cursor", 0))
        if entries:
            cursor %= len(entries)

        rows: list[tuple[int, pygame.Rect]] = []
        list_width = list_rect.width - (12 if max_scroll > 0 else 0)
        for index in range(scroll, min(len(entries), scroll + visible_rows)):
            rows.append(
                (
                    index,
                    pygame.Rect(
                        list_rect.left,
                        list_rect.top + (index - scroll) * MAP_ROW_H,
                        list_width,
                        MAP_ROW_H - 1,
                    ),
                )
            )

        scrollbar = None
        if max_scroll > 0:
            track = pygame.Rect(
                list_rect.right - 9, list_rect.top + 2, 7,
                list_rect.height - 4,
            )
            thumb_height = max(
                22,
                round(track.height * visible_rows / max(1, len(entries))),
            )
            travel = max(1, track.height - thumb_height)
            thumb_y = track.top + round(
                travel * scroll / max(1, max_scroll)
            )
            scrollbar = {
                "track": track,
                "thumb": pygame.Rect(track.left, thumb_y, track.width, thumb_height),
            }

        button_w = 82
        button_h = 23
        gap = 7
        button_y = footer.bottom - button_h
        buttons: dict[str, pygame.Rect] = {}
        button_x = footer.right - (button_w * 3 + gap * 2)
        for button_id in ("cancel", "refresh", "launch"):
            buttons[button_id] = pygame.Rect(
                button_x, button_y, button_w, button_h
            )
            button_x += button_w + gap

        return {
            "window": window,
            "title": title,
            "close": close,
            "content": content,
            "header": header,
            "list": list_rect,
            "error": error_rect,
            "rows": rows,
            "visible_rows": visible_rows,
            "scroll": scroll,
            "max_scroll": max_scroll,
            "cursor": cursor,
            "scrollbar": scrollbar,
            "footer": footer,
            "buttons": buttons,
            "entries": entries,
        }

    def _map_layer_surface(
        self, width: int, height: int
    ) -> pygame.Surface:
        if self._map_layer is None or self._map_layer_size != (width, height):
            self._map_layer = pygame.Surface(
                (width, height), pygame.SRCALPHA
            )
            self._map_layer_size = (width, height)
        else:
            self._map_layer.fill((0, 0, 0, 0))
        return self._map_layer

    def draw_map_browser(
        self, target: pygame.Surface, width: int, height: int, state: dict
    ) -> None:
        """Draw the New Game window with the list of ``engine/maps`` files."""
        self.draw_dim(target, width, height)

        theme = self.theme
        accent = self.accent
        opacity = max(0.0, min(1.0, float(self.menu_opacity)))
        panel_alpha = max(80, round(opacity * 255))
        layout = self.map_browser_layout(width, height, state)
        mouse = pygame.mouse.get_pos()
        layer = self._map_layer_surface(width, height)

        self._draw_window_chrome(
            layer,
            layout,
            theme,
            accent,
            mouse,
            panel_alpha,
            title_text=str(state.get("title", "New game")),
            show_gear=False,
        )

        content: pygame.Rect = layout["content"]
        pygame.draw.rect(layer, theme.border_dark, content, 1)
        pygame.draw.line(
            layer, theme.border,
            (content.left + 1, content.top + 1),
            (content.right - 2, content.top + 1),
        )

        subtitle = str(state.get("subtitle", "Maps in engine/maps"))
        self.text(
            layer, self.font_bold, subtitle,
            (layout["header"].left, layout["header"].top),
            accent, shadow=True,
        )
        count_text = str(len(layout["entries"]))
        self.text(
            layer, self.font_small, f"{count_text} map(s)",
            (layout["header"].right, layout["header"].top + 3),
            theme.muted, anchor="topright", shadow=True,
        )

        list_rect: pygame.Rect = layout["list"]
        pygame.draw.rect(layer, theme.dark, list_rect)
        pygame.draw.rect(layer, theme.border_dark, list_rect, 1)
        pygame.draw.line(
            layer, theme.light,
            (list_rect.left + 1, list_rect.bottom - 2),
            (list_rect.right - 2, list_rect.bottom - 2),
        )

        entries = layout["entries"]
        if not entries:
            center_x = list_rect.centerx
            center_y = list_rect.centery - 14
            self.text(
                layer, self.font_bold, "No maps found",
                (center_x, center_y), theme.text,
                anchor="center", shadow=True,
            )
            self.text(
                layer, self.font_small,
                "Drop .map files into engine/maps, then press R to refresh.",
                (center_x, center_y + 20), theme.muted,
                anchor="center", shadow=True,
            )
        else:
            cursor = layout["cursor"]
            for index, row in layout["rows"]:
                selected = index == cursor
                hovered = row.collidepoint(mouse)
                if selected:
                    fill = _mix(theme.dark, accent, 0.26)
                elif hovered:
                    fill = _lighten(theme.dark, 0.12)
                else:
                    fill = theme.dark
                pygame.draw.rect(layer, fill, row)
                if selected:
                    pygame.draw.rect(
                        layer, accent,
                        (row.left + 1, row.top + 3, 3, row.height - 6),
                    )
                entry = entries[index]
                detail_width = self.font_small.size(
                    str(entry.get("detail", ""))
                )[0]
                name_width = max(
                    40, row.width - detail_width - 34
                )
                name = self._ellipsize(
                    self.font,
                    str(entry.get("name", "")),
                    name_width,
                )
                self.text(
                    layer, self.font, name,
                    (row.left + 13, row.centery),
                    accent if selected else theme.text,
                    anchor="midleft", shadow=True,
                )
                self.text(
                    layer, self.font_small,
                    str(entry.get("detail", "")),
                    (row.right - 8, row.centery),
                    theme.muted, anchor="midright", shadow=True,
                )
                if index + 1 < len(entries):
                    pygame.draw.line(
                        layer, theme.border_dark,
                        (row.left + 8, row.bottom),
                        (row.right - 8, row.bottom),
                    )

        scrollbar = layout.get("scrollbar")
        if scrollbar is not None:
            pygame.draw.rect(layer, theme.border_dark, scrollbar["track"])
            pygame.draw.rect(layer, theme.light, scrollbar["thumb"])
            pygame.draw.rect(layer, theme.border_dark, scrollbar["thumb"], 1)

        if layout.get("error") is not None:
            error_rect: pygame.Rect = layout["error"]
            error_text = self._ellipsize(
                self.font_small,
                str(state.get("error", "")),
                error_rect.width,
            )
            self.text(
                layer, self.font_small, error_text,
                (error_rect.left, error_rect.top),
                (255, 116, 116), shadow=True,
            )

        footer: pygame.Rect = layout["footer"]
        pygame.draw.line(
            layer, theme.border_dark,
            (footer.left, footer.top), (footer.right, footer.top),
        )
        self.text(
            layer, self.font_small,
            "Enter launch   R refresh   Esc cancel",
            (footer.left + 5, footer.top + 10), theme.muted,
            shadow=True,
        )

        buttons = layout["buttons"]
        for button_id, label in (
            ("cancel", "Cancel"),
            ("refresh", "Refresh"),
            ("launch", "Launch"),
        ):
            rect = buttons[button_id]
            widget = {
                "kind": "button",
                "label": label,
                "rect": rect,
                "disabled": button_id == "launch" and not entries,
            }
            self._draw_button(
                layer,
                widget,
                theme,
                accent,
                focused=False,
                hovered=not widget["disabled"] and rect.collidepoint(mouse),
            )

        target.blit(layer, (0, 0))

    def _draw_crosshair_preview(
        self,
        target: pygame.Surface,
        layout: dict,
        theme: MenuTheme,
        accent: tuple[int, int, int],
        crosshair: Crosshair | None,
    ) -> None:
        preview = layout.get("preview")
        if preview is None or crosshair is None:
            return

        self.text(
            target, self.font_bold, "Preview",
            (preview.left, preview.top - 18), accent, shadow=True,
        )
        pygame.draw.rect(target, theme.dark, preview)
        pygame.draw.rect(target, theme.border_dark, preview, 1)
        pygame.draw.line(
            target, theme.light,
            (preview.left + 1, preview.bottom - 2),
            (preview.right - 2, preview.bottom - 2),
        )
        pygame.draw.line(
            target, theme.light,
            (preview.right - 2, preview.top + 1),
            (preview.right - 2, preview.bottom - 2),
        )
        center = preview.center
        pygame.draw.line(
            target, _darken(theme.border, 0.25),
            (center[0], preview.top + 8),
            (center[0], preview.bottom - 8),
        )
        pygame.draw.line(
            target, _darken(theme.border, 0.25),
            (preview.left + 8, center[1]),
            (preview.right - 8, center[1]),
        )

        if crosshair.enabled:
            self._draw_crosshair_shape(
                target, center, crosshair, accent
            )
        else:
            self.text(
                target, self.font_bold, "OFF", center,
                theme.muted, anchor="center", shadow=True,
            )

        color = crosshair.color or accent
        info_y = preview.bottom + 9
        self.text(
            target, self.font_small, crosshair.color_name.upper(),
            (preview.left, info_y), accent, shadow=True,
        )
        self.text(
            target, self.font_small,
            "#{:02X}{:02X}{:02X}".format(*color),
            (preview.right, info_y), theme.muted,
            anchor="topright", shadow=True,
        )
        self.text(
            target, self.font_small,
            f"{crosshair.size}px arms   {crosshair.gap}px gap",
            (preview.left, info_y + 18), theme.muted, shadow=True,
        )
        self.text(
            target, self.font_small,
            f"{crosshair.thickness}px thick   {crosshair.opacity}% alpha",
            (preview.left, info_y + 36), theme.muted, shadow=True,
        )

    def _draw_controls(
        self,
        target: pygame.Surface,
        layout: dict,
        theme: MenuTheme,
        accent: tuple[int, int, int],
    ) -> None:
        content: pygame.Rect = layout["content"]
        x0 = content.left + 12
        column_w = (content.width - 24) // 2
        left = x0
        right = x0 + column_w + 16
        top = content.top + 8

        movement = (
            ("WASD", "move"),
            ("Space", "jump / hold to bunnyhop"),
            ("Shift", "sprint"),
            ("Mouse", "look around"),
            ("R", "respawn"),
            ("Tab", "walk / free camera"),
        )
        camera = (
            ("WASD", "move"),
            ("Q / E", "down / up"),
            ("Wheel", "move forward / back"),
            ("Shift", "boost"),
            ("C", "reset camera"),
        )
        interface = (
            ("F3", "debug overlay"),
            ("F7", "render mode"),
            ("F11", "fullscreen"),
            ("M", "cat overlay"),
            ("Esc", "options / release mouse"),
        )
        mouse = (
            ("LMB", "capture mouse"),
            ("Wheel", "camera move in free mode"),
            ("Drag", "move this window by its title bar"),
        )

        self._draw_control_block(
            target, "Movement", movement, left, top, theme, accent
        )
        self._draw_control_block(
            target, "Free camera", camera, right, top, theme, accent
        )
        interface_y = top + 22 + len(movement) * 19 + 20
        self._draw_control_block(
            target, "Interface", interface, left, interface_y, theme, accent
        )
        mouse_y = top + 22 + len(camera) * 19 + 20
        self._draw_control_block(
            target, "Mouse", mouse, right, mouse_y, theme, accent
        )

    def _draw_control_block(
        self,
        target: pygame.Surface,
        title: str,
        rows: tuple[tuple[str, str], ...],
        x: int,
        y: int,
        theme: MenuTheme,
        accent: tuple[int, int, int],
    ) -> None:
        self.text(
            target, self.font_bold, title, (x, y), accent, shadow=True
        )
        row_y = y + 22
        for key, description in rows:
            self.text(
                target, self.font_small, key, (x, row_y), accent,
                shadow=True,
            )
            self.text(
                target, self.font_small, description,
                (x + 76, row_y), theme.text, shadow=True,
            )
            row_y += 19

    # ------------------------------------------------------------------
    # Debug panel (F3)
    # ------------------------------------------------------------------
    def draw_debug(
        self, target: pygame.Surface, rows: list[tuple[str, str]]
    ) -> None:
        theme = self.theme
        accent = self.accent
        x, y, pad = 16, 16, 12
        title_height = 24
        width = 400
        line_height = self.mono.get_linesize()
        height = pad * 2 + title_height + line_height * len(rows)

        pygame.draw.rect(target, (0, 0, 0, 110), (x + 4, y + 4, width, height))
        pygame.draw.rect(target, (*theme.dark, 238), (x, y, width, height))
        pygame.draw.rect(target, theme.border_dark, (x, y, width, height), 1)
        pygame.draw.line(
            target, theme.light,
            (x + 1, y + 1), (x + width - 2, y + 1),
        )
        pygame.draw.rect(target, accent, (x, y, width, 2))

        self.text(
            target, self.mono_bold, "DEBUG",
            (x + pad, y + pad - 2), accent, shadow=True,
        )
        self.text(
            target, self.mono_small, "F3",
            (x + width - pad, y + pad - 1), theme.muted,
            anchor="topright", shadow=True,
        )

        body_y = y + pad + title_height
        for index, (label, value) in enumerate(rows):
            line_y = body_y + index * line_height
            self.text(
                target, self.mono_small, label.upper(),
                (x + pad, line_y), theme.muted, shadow=True,
            )
            self.text(
                target, self.mono, value,
                (x + 160, line_y), theme.text, shadow=True,
            )

    # ------------------------------------------------------------------
    # Crosshair editor and in-game crosshair
    # ------------------------------------------------------------------
    def crosshair_color(
        self, crosshair: Crosshair, accent: tuple[int, int, int] | None = None
    ) -> tuple[int, int, int]:
        """Resolve the crosshair colour, following the accent when selected."""
        color = crosshair.color
        return color if color is not None else (accent or self.accent)

    def draw_crosshair(
        self,
        target: pygame.Surface,
        width: int,
        height: int,
        crosshair: Crosshair,
    ) -> None:
        if not crosshair.enabled:
            return
        self._draw_crosshair_shape(
            target, (width / 2.0, height / 2.0), crosshair
        )

    def _draw_crosshair_shape(
        self,
        target: pygame.Surface,
        center: tuple[float, float],
        crosshair: Crosshair,
        accent: tuple[int, int, int] | None = None,
    ) -> None:
        alpha = max(0, min(255, round(crosshair.opacity * 2.55)))
        rects = crosshair.rects(center)
        if crosshair.outline and crosshair.outline_thickness > 0:
            outline = max(1, int(crosshair.outline_thickness))
            for x, y, w, h in rects:
                pygame.draw.rect(
                    target, (0, 0, 0, alpha),
                    (
                        x - outline, y - outline,
                        w + outline * 2, h + outline * 2,
                    ),
                )
        color = (*self.crosshair_color(crosshair, accent), alpha)
        for rect in rects:
            pygame.draw.rect(target, color, rect)

    # ------------------------------------------------------------------
    # Front-end menu
    # ------------------------------------------------------------------
    def _menu_background(
        self, width: int, height: int
    ) -> pygame.Surface | None:
        """Return the menu backdrop scaled to cover the whole window."""
        if width <= 0 or height <= 0:
            return None
        if self._menu_bg_source is None:
            try:
                self._menu_bg_source = pygame.image.load(
                    str(MENU_BACKGROUND_PATH)
                )
            except (pygame.error, OSError):
                return None

        if (
            self._menu_bg_surface is not None
            and self._menu_bg_size == (width, height)
        ):
            return self._menu_bg_surface

        source = self._menu_bg_source
        source_w, source_h = source.get_size()
        scale = max(width / source_w, height / source_h)
        scaled_w = max(width, round(source_w * scale))
        scaled_h = max(height, round(source_h * scale))
        if (scaled_w, scaled_h) == (source_w, source_h):
            scaled = source
        else:
            scaled = pygame.transform.smoothscale(
                source, (scaled_w, scaled_h)
            )
        if (scaled_w, scaled_h) == (width, height):
            cropped = scaled
        else:
            x = max(0, (scaled_w - width) // 2)
            y = max(0, (scaled_h - height) // 2)
            cropped = scaled.subsurface((x, y, width, height)).copy()

        # A front-end backdrop must fully cover the 3D frame, so drop any
        # residual alpha introduced by scaling the source image.
        opaque = pygame.Surface((width, height))
        opaque.blit(cropped, (0, 0))
        self._menu_bg_surface = opaque
        self._menu_bg_size = (width, height)
        return opaque

    def draw_dim(
        self,
        target: pygame.Surface,
        width: int,
        height: int,
        alpha: int = DIM_ALPHA,
    ) -> None:
        """Darken whatever is already on the overlay surface."""
        alpha = max(0, min(255, int(alpha)))
        key = (width, height, alpha)
        dim = self._dim_cache.get(key)
        if dim is None:
            dim = pygame.Surface((width, height), pygame.SRCALPHA)
            dim.fill((0, 0, 0, alpha))
            self._dim_cache[key] = dim
        target.blit(dim, (0, 0))

    def menu_layout(
        self,
        width: int,
        height: int,
        items: list[tuple[str, str]],
    ) -> dict:
        """Rects for the left-aligned front-end menu, used for hit testing."""
        margin_x = max(24, round(width * 0.035))
        margin_y = max(10, round(height * 0.02))
        row_h = max(28, min(36, round(height * 0.042)))
        start_y = round(height * 0.62)
        buttons: list[tuple[str, pygame.Rect]] = []
        for index, (item_id, label) in enumerate(items):
            text_w = self.menu_font.size(label)[0]
            buttons.append(
                (
                    item_id,
                    pygame.Rect(
                        margin_x - 10,
                        start_y + index * row_h,
                        max(150, text_w + 34),
                        row_h,
                    ),
                )
            )
        return {
            "buttons": buttons,
            "margin_x": margin_x,
            "margin_y": margin_y,
            "row_h": row_h,
            "start_y": start_y,
        }

    def draw_main_menu(
        self,
        target: pygame.Surface,
        width: int,
        height: int,
        state: dict,
    ) -> None:
        """Draw the HL2-2003 style front end or the in-game pause menu."""
        items = list(state.get("items", ()))
        if not items:
            return
        pause = bool(state.get("pause", False))
        cursor = int(state.get("cursor", 0)) % len(items)
        layout = self.menu_layout(width, height, items)
        theme = self.theme
        accent = self.accent

        if pause:
            if state.get("dim", True):
                self.draw_dim(target, width, height, 90)
        else:
            background = self._menu_background(width, height)
            if background is not None:
                target.blit(background, (0, 0))
            else:
                target.fill((*theme.dark, 255))

        margin_x = layout["margin_x"]
        margin_y = layout["margin_y"]

        if pause:
            self.text(
                target, self.menu_small, "PAUSED",
                (margin_x, layout["start_y"] - 24), accent,
                shadow=True,
            )

        for index, (_item_id, rect) in enumerate(layout["buttons"]):
            active = index == cursor
            color = accent if active else (224, 225, 213)
            if active:
                pygame.draw.rect(
                    target, (*accent, 255),
                    (rect.left + 2, rect.centery - 8, 3, 16),
                )
            self.text(
                target, self.menu_font, items[index][1],
                (rect.left + 10, rect.centery), color,
                anchor="midleft", shadow=True,
            )

        if pause:
            self.text(
                target, self.menu_small, "Esc  resume",
                (width - margin_x, height - margin_y - 2),
                (180, 184, 170), anchor="bottomright", shadow=True,
            )
        else:
            logo_y = height - max(34, round(height * 0.055))
            self.text(
                target, self.menu_logo, "F R E E D E E",
                (margin_x, logo_y), (218, 220, 208),
                anchor="bottomleft", shadow=True,
            )

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
        theme = self.theme
        accent = self.accent
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
        pygame.draw.rect(
            target, accent, (16, height - 54, 46, 2)
        )
        self.text(
            target, self.font_small, move_line,
            (16, height - 36), theme.text, anchor="bottomleft", shadow=True,
        )
        self.text(
            target, self.font_small,
            "F3  debug     F7  render mode     F11  fullscreen     "
            "M  cat     Esc  options",
            (16, height - 16), theme.muted, anchor="bottomleft", shadow=True,
        )

    def draw_mouse_hint(
        self, target: pygame.Surface, width: int, height: int
    ) -> None:
        theme = self.theme
        accent = self.accent
        text = "CLICK TO CAPTURE MOUSE"
        panel_width = 250
        x0 = (width - panel_width) / 2
        y0 = height - 96
        pygame.draw.rect(target, (0, 0, 0, 100), (x0 + 3, y0 + 3, panel_width, 32))
        pygame.draw.rect(target, (*theme.dark, 238), (x0, y0, panel_width, 32))
        pygame.draw.rect(target, theme.border_dark, (x0, y0, panel_width, 32), 1)
        pygame.draw.rect(target, accent, (x0, y0, panel_width, 2))
        self.text(
            target, self.mono_bold, text,
            (width / 2, y0 + 17), accent, anchor="center", shadow=True,
        )

    def draw_toast(
        self, target: pygame.Surface, width: int, text: str
    ) -> None:
        theme = self.theme
        accent = self.accent
        rendered = self._text_surface(self.toast, text, accent)
        panel_width = max(300, rendered.get_width() + 80)
        panel_height = 58
        x0 = (width - panel_width) / 2
        y0 = 24.0
        pygame.draw.rect(target, (0, 0, 0, 100), (x0 + 4, y0 + 4, panel_width, panel_height))
        pygame.draw.rect(target, (*theme.dark, 240), (x0, y0, panel_width, panel_height))
        pygame.draw.rect(target, theme.border_dark, (x0, y0, panel_width, panel_height), 1)
        pygame.draw.rect(
            target, accent,
            (x0, y0 + panel_height - 3, panel_width, 3),
        )
        self.text(
            target, self.toast, text,
            (width / 2, y0 + panel_height / 2 - 1), accent,
            anchor="center", shadow=True,
        )
