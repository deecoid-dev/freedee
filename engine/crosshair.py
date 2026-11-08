"""Crosshair settings and screen-space geometry.

The crosshair is drawn by :class:`engine.ui.UI` as a handful of rectangles
over the 3D frame, so the in-game overlay and the settings preview always use
exactly the same numbers.
"""

from __future__ import annotations

#: Colour choices offered in the crosshair editor. ``None`` means "follow the
#: UI accent colour", so the crosshair stays in sync with the rest of the HUD.
CROSSHAIR_COLORS: tuple[
    tuple[str, tuple[int, int, int] | None], ...
] = (
    ("Accent", None),
    ("White", (255, 255, 255)),
    ("Cyan", (111, 227, 255)),
    ("Mint", (121, 242, 178)),
    ("Lime", (184, 242, 107)),
    ("Amber", (255, 196, 92)),
    ("Orange", (255, 138, 92)),
    ("Red", (255, 107, 107)),
    ("Magenta", (255, 114, 214)),
    ("Violet", (181, 140, 255)),
    ("Blue", (122, 162, 255)),
)

#: Rows of the crosshair editor, in display order.
CROSSHAIR_ROWS = (
    "enabled",
    "color",
    "size",
    "gap",
    "thickness",
    "dot",
    "dot_size",
    "outline",
    "outline_thickness",
    "opacity",
    "back",
)

#: row name -> (step, minimum, maximum)
_NUMERIC_ROWS = {
    "size": (2, 2, 48),
    "gap": (1, 0, 40),
    "thickness": (1, 1, 10),
    "dot_size": (1, 1, 16),
    "outline_thickness": (1, 1, 4),
    "opacity": (5, 20, 100),
}


class Crosshair:
    """Mutable crosshair configuration shared by gameplay and settings."""

    ROWS = CROSSHAIR_ROWS
    COLORS = CROSSHAIR_COLORS

    def __init__(self) -> None:
        self.enabled = True
        self.color_index = 0  # Accent
        self.size = 10
        self.gap = 6
        self.thickness = 2
        self.dot = True
        self.dot_size = 3
        self.outline = True
        self.outline_thickness = 1
        self.opacity = 100

    # ------------------------------------------------------------------
    # Colour
    # ------------------------------------------------------------------
    @property
    def color_name(self) -> str:
        return CROSSHAIR_COLORS[self.color_index][0]

    @property
    def color(self) -> tuple[int, int, int] | None:
        """Static RGB, or ``None`` when the crosshair follows the UI accent."""
        return CROSSHAIR_COLORS[self.color_index][1]

    def cycle_color(self, direction: int) -> None:
        self.color_index = (
            self.color_index + direction
        ) % len(CROSSHAIR_COLORS)

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def step(self, row: str, direction: int) -> bool:
        """Apply one editor step. Returns ``True`` when the value changed."""
        if row == "enabled":
            self.enabled = not self.enabled
            return True
        if row == "color":
            self.cycle_color(direction)
            return True
        if row == "dot":
            self.dot = not self.dot
            return True
        if row == "outline":
            self.outline = not self.outline
            return True
        if row in _NUMERIC_ROWS:
            step, minimum, maximum = _NUMERIC_ROWS[row]
            value = getattr(self, row) + direction * step
            setattr(self, row, max(minimum, min(maximum, value)))
            return True
        return False

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def rects(
        self, center: tuple[float, float]
    ) -> list[tuple[int, int, int, int]]:
        """Return the filled rectangles (four arms, then the dot)."""
        cx = int(round(center[0]))
        cy = int(round(center[1]))
        thickness = max(1, int(self.thickness))
        gap = max(0, int(self.gap))
        arm = max(1, int(self.size))
        half = thickness // 2

        rects = [
            (cx - gap - arm, cy - half, arm, thickness),
            (cx + gap, cy - half, arm, thickness),
            (cx - half, cy - gap - arm, thickness, arm),
            (cx - half, cy + gap, thickness, arm),
        ]
        if self.dot:
            dot = max(1, int(self.dot_size))
            rects.append((cx - dot // 2, cy - dot // 2, dot, dot))
        return rects
