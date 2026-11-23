"""Texture sources and materials for world objects.

A material is a small, immutable bundle of a texture surface plus the UV scale
used by the renderer. Textures are looked up by material name in
``engine/assets`` first, which lets artists replace any procedural placeholder
without touching code:

    engine/assets/crate.png
    engine/assets/stone_wall.jpg

Supported extensions are ``.png``, ``.jpg``, ``.jpeg``, ``.bmp`` and ``.tga``.
If no matching file exists, a deterministic placeholder is generated in code
so the demo always runs out of the box.

The renderer maps textures onto geometry with a box projection, so no mesh UV
coordinates are required: every face samples the texture using the two object
space axes that are most perpendicular to its normal. ``uv_scale`` is measured
in texture repetitions per metre.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pygame

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
"""Folder scanned for user supplied object textures."""

MAP_TEXTURES_DIR = Path(__file__).resolve().parent / "maps" / "textures"
"""Folder scanned for textures referenced by ``.map`` files."""

TEXTURE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tga")
TEXTURE_SIZE = 256
"""Size of the generated placeholder textures, in pixels."""

#: Texture repetitions per metre, keyed by material name.
UV_SCALES = {
    "crate": 0.85,
    "wood_wall": 0.55,
    "stone_wall": 0.50,
    "brick_wall": 0.75,
    "metal_panel": 0.90,
    "concrete": 0.55,
    "floor_tiles": 0.50,
}


@dataclass(frozen=True)
class Material:
    """A texture surface and how densely it tiles across geometry."""

    name: str
    surface: pygame.Surface
    uv_scale: float = 1.0


def _shade(color: tuple[int, int, int], delta: int) -> tuple[int, int, int]:
    return tuple(max(0, min(255, channel + delta)) for channel in color)


def _speckle(
    surface: pygame.Surface,
    rng: random.Random,
    count: int,
    base: tuple[int, int, int],
    spread: int = 18,
) -> None:
    width, height = surface.get_size()
    for _ in range(count):
        color = _shade(base, rng.randint(-spread, spread))
        x = rng.randrange(width)
        y = rng.randrange(height)
        surface.set_at((x, y), color)


def _make_crate() -> pygame.Surface:
    """Wooden crate with plank seams, a diagonal brace and corner bolts."""
    rng = random.Random(101)
    surface = pygame.Surface((TEXTURE_SIZE, TEXTURE_SIZE))
    base = (146, 98, 52)
    surface.fill(base)

    plank_height = TEXTURE_SIZE // 4
    for row in range(4):
        top = row * plank_height
        color = _shade(base, rng.randint(-16, 16))
        pygame.draw.rect(
            surface, color, (0, top + 1, TEXTURE_SIZE, plank_height - 2)
        )
        for _ in range(11):
            y = top + rng.randint(4, plank_height - 4)
            x0 = rng.randint(0, TEXTURE_SIZE // 2)
            x1 = rng.randint(x0 + 24, TEXTURE_SIZE)
            pygame.draw.line(
                surface, _shade(color, -18), (x0, y), (x1, y), 1
            )
        pygame.draw.line(
            surface, _shade(base, -34),
            (0, top), (TEXTURE_SIZE, top), 2,
        )

    frame = (98, 63, 34)
    pygame.draw.rect(
        surface, frame, (0, 0, TEXTURE_SIZE, TEXTURE_SIZE), 12
    )
    pygame.draw.rect(
        surface, _shade(frame, 20),
        (12, 12, TEXTURE_SIZE - 24, TEXTURE_SIZE - 24), 2,
    )
    pygame.draw.line(
        surface, frame,
        (16, TEXTURE_SIZE - 16), (TEXTURE_SIZE - 16, 16), 14,
    )
    pygame.draw.line(
        surface, _shade(frame, 18),
        (20, TEXTURE_SIZE - 20), (TEXTURE_SIZE - 20, 20), 4,
    )

    for x in (18, TEXTURE_SIZE - 18):
        for y in (18, TEXTURE_SIZE - 18):
            pygame.draw.circle(surface, (74, 74, 70), (x, y), 5)
            pygame.draw.circle(surface, (154, 150, 138), (x - 1, y - 1), 2)
    return surface


def _make_wood_wall() -> pygame.Surface:
    """Vertical timber planks with cross beams and iron nails."""
    rng = random.Random(202)
    surface = pygame.Surface((TEXTURE_SIZE, TEXTURE_SIZE))
    base = (126, 84, 47)
    surface.fill(base)

    planks = 6
    plank_width = TEXTURE_SIZE // planks
    for index in range(planks):
        left = index * plank_width
        color = _shade(base, rng.randint(-20, 20))
        pygame.draw.rect(
            surface, color, (left + 1, 0, plank_width - 2, TEXTURE_SIZE)
        )
        for _ in range(9):
            x = left + rng.randint(3, plank_width - 3)
            y0 = rng.randint(0, TEXTURE_SIZE // 2)
            y1 = rng.randint(y0 + 24, TEXTURE_SIZE)
            pygame.draw.line(
                surface, _shade(color, -20), (x, y0), (x, y1), 1
            )
        for _ in range(2):
            kx = left + rng.randint(5, plank_width - 5)
            ky = rng.randint(12, TEXTURE_SIZE - 12)
            pygame.draw.circle(
                surface, _shade(color, -32), (kx, ky),
                rng.randint(2, 5), 1,
            )
        pygame.draw.line(
            surface, _shade(base, -46),
            (left, 0), (left, TEXTURE_SIZE), 2,
        )

    beam = (88, 58, 32)
    pygame.draw.rect(surface, beam, (0, 0, TEXTURE_SIZE, 15))
    pygame.draw.rect(
        surface, beam, (0, TEXTURE_SIZE - 15, TEXTURE_SIZE, 15)
    )
    pygame.draw.line(
        surface, _shade(beam, 20), (0, 15), (TEXTURE_SIZE, 15), 2
    )
    for x in (22, TEXTURE_SIZE // 2, TEXTURE_SIZE - 22):
        for y in (7, TEXTURE_SIZE - 7):
            pygame.draw.circle(surface, (72, 72, 68), (x, y), 3)
            pygame.draw.circle(surface, (160, 158, 146), (x - 1, y - 1), 1)
    return surface


def _make_stone_wall() -> pygame.Surface:
    """Irregular grey stone blocks set in dark mortar."""
    rng = random.Random(303)
    surface = pygame.Surface((TEXTURE_SIZE, TEXTURE_SIZE))
    surface.fill((72, 71, 67))

    y = 0
    while y < TEXTURE_SIZE:
        row_height = rng.randint(27, 43)
        x = -rng.randint(0, 28)
        while x < TEXTURE_SIZE:
            width = rng.randint(36, 74)
            rect = pygame.Rect(x + 3, y + 3, width - 6, row_height - 6)
            tone = rng.randint(104, 142)
            color = (
                max(0, min(255, tone + rng.randint(-4, 4))),
                max(0, min(255, tone - 2 + rng.randint(-4, 4))),
                max(0, min(255, tone - 7 + rng.randint(-4, 4))),
            )
            pygame.draw.rect(surface, color, rect, border_radius=4)
            pygame.draw.line(
                surface, _shade(color, 24),
                (rect.left + 3, rect.top + 3),
                (rect.right - 4, rect.top + 3), 2,
            )
            pygame.draw.line(
                surface, _shade(color, -28),
                (rect.left + 3, rect.bottom - 4),
                (rect.right - 4, rect.bottom - 4), 2,
            )
            for _ in range(3):
                px = rng.randint(
                    rect.left + 4, max(rect.left + 4, rect.right - 5)
                )
                py = rng.randint(
                    rect.top + 4, max(rect.top + 4, rect.bottom - 5)
                )
                pygame.draw.circle(
                    surface, _shade(color, rng.randint(-22, 18)), (px, py), 1
                )
            x += width
        y += row_height

    for _ in range(5):
        x = rng.randrange(TEXTURE_SIZE)
        y = rng.randrange(TEXTURE_SIZE)
        points = [(x, y)]
        for _ in range(rng.randint(3, 6)):
            x += rng.randint(-20, 20)
            y += rng.randint(8, 24)
            points.append((x, y))
        pygame.draw.lines(surface, (54, 53, 50), False, points, 2)
    return surface


def _make_brick_wall() -> pygame.Surface:
    """Offset red-brown bricks with pale mortar."""
    rng = random.Random(404)
    surface = pygame.Surface((TEXTURE_SIZE, TEXTURE_SIZE))
    surface.fill((122, 114, 102))

    brick_width = 48
    brick_height = 23
    gap = 4
    row = 0
    y = 0
    while y < TEXTURE_SIZE:
        offset = -(brick_width // 2) if row % 2 else 0
        x = offset
        while x < TEXTURE_SIZE:
            rect = pygame.Rect(
                x + gap // 2, y + gap // 2,
                brick_width - gap, brick_height - gap,
            )
            color = (
                rng.randint(138, 168),
                rng.randint(62, 84),
                rng.randint(48, 66),
            )
            pygame.draw.rect(surface, color, rect, border_radius=2)
            pygame.draw.line(
                surface, _shade(color, 20),
                (rect.left + 1, rect.top + 1),
                (rect.right - 2, rect.top + 1), 1,
            )
            screen_rect = rect.clip(surface.get_rect())
            if screen_rect.width and screen_rect.height:
                for _ in range(2):
                    pygame.draw.circle(
                        surface, _shade(color, rng.randint(-18, 14)),
                        (
                            rng.randint(
                                screen_rect.left, screen_rect.right - 1
                            ),
                            rng.randint(
                                screen_rect.top, screen_rect.bottom - 1
                            ),
                        ),
                        1,
                    )
            x += brick_width
        y += brick_height
        row += 1
    return surface


def _make_metal_panel() -> pygame.Surface:
    """Brushed steel panels with seams and rivets."""
    rng = random.Random(505)
    surface = pygame.Surface((TEXTURE_SIZE, TEXTURE_SIZE))
    base = (102, 112, 122)
    surface.fill(base)

    for x in range(TEXTURE_SIZE):
        wave = int(7.0 * math.sin(x * 0.09) + rng.randint(-3, 3))
        pygame.draw.line(
            surface, _shade(base, wave),
            (x, 0), (x, TEXTURE_SIZE), 1,
        )

    panel_height = 64
    for y in range(0, TEXTURE_SIZE, panel_height):
        pygame.draw.line(
            surface, (58, 64, 70),
            (0, y), (TEXTURE_SIZE, y), 3,
        )
        pygame.draw.line(
            surface, (142, 150, 158),
            (0, y + 3), (TEXTURE_SIZE, y + 3), 1,
        )

    for y in range(panel_height // 2, TEXTURE_SIZE, panel_height):
        for x in (20, TEXTURE_SIZE - 20):
            pygame.draw.circle(surface, (58, 62, 66), (x, y), 5)
            pygame.draw.circle(surface, (172, 178, 182), (x - 1, y - 1), 2)

    for _ in range(18):
        x0 = rng.randrange(TEXTURE_SIZE)
        y0 = rng.randrange(TEXTURE_SIZE)
        length = rng.randint(8, 42)
        pygame.draw.line(
            surface, _shade(base, rng.choice((-26, -18, 20))),
            (x0, y0), (x0 + length, y0 + rng.randint(-2, 2)), 1,
        )
    return surface


def _make_concrete() -> pygame.Surface:
    """Mottled concrete with a few hairline cracks."""
    rng = random.Random(606)
    surface = pygame.Surface((TEXTURE_SIZE, TEXTURE_SIZE))
    base = (130, 130, 126)
    surface.fill(base)
    _speckle(surface, rng, 4200, base, spread=22)

    for _ in range(14):
        x = rng.randrange(TEXTURE_SIZE)
        y = rng.randrange(TEXTURE_SIZE)
        radius = rng.randint(6, 22)
        pygame.draw.circle(
            surface, _shade(base, rng.randint(-12, 8)), (x, y), radius,
            max(1, radius // 4),
        )

    for _ in range(5):
        x = rng.randrange(TEXTURE_SIZE)
        y = rng.randrange(TEXTURE_SIZE)
        points = [(x, y)]
        for _ in range(rng.randint(3, 6)):
            x += rng.randint(-18, 18)
            y += rng.randint(10, 26)
            points.append((x, y))
        pygame.draw.lines(surface, (96, 96, 93), False, points, 1)
    return surface


def _make_floor_tiles() -> pygame.Surface:
    """Dark industrial floor plates with bevelled edges."""
    rng = random.Random(707)
    surface = pygame.Surface((TEXTURE_SIZE, TEXTURE_SIZE))
    surface.fill((40, 43, 47))

    tile = 64
    for row, y in enumerate(range(0, TEXTURE_SIZE, tile)):
        for column, x in enumerate(range(0, TEXTURE_SIZE, tile)):
            base = 57 if (row + column) % 2 == 0 else 52
            color = (
                base + rng.randint(-6, 6),
                base + 2 + rng.randint(-6, 6),
                base + 5 + rng.randint(-6, 6),
            )
            pygame.draw.rect(
                surface, color, (x + 2, y + 2, tile - 4, tile - 4)
            )
            pygame.draw.line(
                surface, _shade(color, 15),
                (x + 3, y + 3), (x + tile - 4, y + 3), 1,
            )
            pygame.draw.line(
                surface, _shade(color, -16),
                (x + 3, y + tile - 4), (x + tile - 4, y + tile - 4), 1,
            )
            for _ in range(3):
                px = x + rng.randint(5, tile - 5)
                py = y + rng.randint(5, tile - 5)
                pygame.draw.circle(
                    surface, _shade(color, rng.randint(-12, 12)), (px, py), 1
                )
    return surface


_GENERATORS = {
    "crate": _make_crate,
    "wood_wall": _make_wood_wall,
    "stone_wall": _make_stone_wall,
    "brick_wall": _make_brick_wall,
    "metal_panel": _make_metal_panel,
    "concrete": _make_concrete,
    "floor_tiles": _make_floor_tiles,
}

_materials: dict[str, Material] = {}
_map_materials: dict[str, Material] = {}

#: Generated map textures use a 64 px tile, the classic Quake texture size.
MAP_FALLBACK_TEXTURE_SIZE = 64


def _make_map_texture(name: str) -> pygame.Surface:
    """Deterministic checker tile for a texture that has no asset file."""
    digest = hashlib.md5(name.lower().encode("utf-8")).digest()
    base = (
        58 + digest[0] % 128,
        58 + digest[1] % 128,
        58 + digest[2] % 128,
    )
    surface = pygame.Surface(
        (MAP_FALLBACK_TEXTURE_SIZE, MAP_FALLBACK_TEXTURE_SIZE)
    )
    surface.fill(_shade(base, -18))
    half = MAP_FALLBACK_TEXTURE_SIZE // 2
    for row in range(2):
        for column in range(2):
            color = base if (row + column) % 2 == 0 else _shade(base, 20)
            pygame.draw.rect(
                surface,
                color,
                (column * half + 1, row * half + 1, half - 2, half - 2),
            )
    pygame.draw.rect(
        surface,
        _shade(base, -42),
        (0, 0, MAP_FALLBACK_TEXTURE_SIZE, MAP_FALLBACK_TEXTURE_SIZE),
        2,
    )
    for _ in range(18):
        x = digest[3] * 7 % MAP_FALLBACK_TEXTURE_SIZE
        y = digest[4] * 11 % MAP_FALLBACK_TEXTURE_SIZE
        pygame.draw.circle(
            surface,
            _shade(base, digest[5] % 31 - 15),
            (x, y),
            1,
        )
    return surface


def _normalized_map_name(name: str) -> str:
    cleaned = name.strip().replace("\\", "/")
    parts = [
        part for part in cleaned.split("/")
        if part not in ("", ".", "..")
    ]
    return "/".join(parts)


def _find_map_texture(name: str) -> Path | None:
    """Locate a map texture by its editor name in maps/textures or assets."""
    cleaned = _normalized_map_name(name)
    if not cleaned:
        return None
    without_extension = cleaned
    lowered = cleaned.lower()
    for extension in TEXTURE_EXTENSIONS:
        if lowered.endswith(extension):
            without_extension = cleaned[: -len(extension)]
            break
    base_name = without_extension.rsplit("/", 1)[-1]
    try:
        relative = Path(without_extension)
    except (OSError, ValueError):
        relative = Path(base_name)

    roots = (MAP_TEXTURES_DIR, ASSETS_DIR)
    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / relative)
        candidates.append(root / base_name)
        if cleaned != without_extension:
            candidates.append(root / cleaned)
    for candidate in candidates:
        if candidate.suffix.lower() in TEXTURE_EXTENSIONS and candidate.is_file():
            return candidate
        for extension in TEXTURE_EXTENSIONS:
            path = candidate.with_name(candidate.name + extension)
            if path.is_file():
                return path
    return None


def get_map_material(name: str) -> Material:
    """Resolve a texture name from a ``.map`` file.

    Files in ``engine/maps/textures`` win, then files in ``engine/assets``.
    Built-in procedural materials are reused when the editor name matches one
    of them (``concrete``, ``crate``, ``floor_tiles`` ...).  Every other name
    receives a deterministic checker texture, so an unknown WAD texture never
    prevents a map from loading.
    """
    key = name.strip() or "missing"
    cached = _map_materials.get(key)
    if cached is not None:
        return cached

    normalized = _normalized_map_name(key)
    base_name = normalized.rsplit("/", 1)[-1]
    stem = Path(base_name).stem

    path = _find_map_texture(key)
    surface = _load_asset(path) if path is not None else None
    if surface is None:
        if stem in _GENERATORS:
            material = get_material(stem)
            _map_materials[key] = material
            return material
        surface = _make_map_texture(normalized or key)
        if path is None:
            print(
                f"[map] texture {key!r} was not found; "
                "using a generated placeholder"
            )
    material = Material(
        name=normalized or key,
        surface=surface,
        uv_scale=1.0,
    )
    _map_materials[key] = material
    return material


def _find_asset(name: str) -> Path | None:
    for extension in TEXTURE_EXTENSIONS:
        path = ASSETS_DIR / f"{name}{extension}"
        if path.is_file():
            return path
    return None


def _load_asset(path: Path) -> pygame.Surface | None:
    """Load a user texture, returning ``None`` when the file is unusable."""
    try:
        return pygame.image.load(str(path))
    except (pygame.error, OSError) as exc:
        print(
            f"[assets] could not load {path.name}: {exc}; "
            "using the generated texture"
        )
        return None


def get_material(name: str) -> Material:
    """Return the cached material, loading a file or generating a fallback."""
    material = _materials.get(name)
    if material is not None:
        return material

    path = _find_asset(name)
    surface = _load_asset(path) if path is not None else None
    if surface is None:
        generator = _GENERATORS.get(name)
        if generator is None:
            raise KeyError(f"Unknown material: {name!r}")
        surface = generator()

    material = Material(
        name=name,
        surface=surface,
        uv_scale=UV_SCALES.get(name, 1.0),
    )
    _materials[name] = material
    return material


def clear_material_cache() -> None:
    """Forget generated/loaded materials (useful for tests and hot reloads)."""
    _materials.clear()
    _map_materials.clear()
