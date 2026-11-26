"""Skybox support: equirectangular panorama skies for ``.map`` levels.

The engine looks for a sky texture whose file name matches the map file name.
For a level stored as ``engine/maps/sky_test.map`` the loader picks up
``engine/skybox/sky_test.png`` (``.jpg``, ``.jpeg``, ``.bmp`` and ``.tga`` are
accepted as well).  Nested map folders are mirrored inside ``engine/skybox``,
with a fallback to the plain file name.

A sky texture is a normal equirectangular panorama: the top row is the zenith,
the middle row is the horizon and the bottom row is the nadir.  The renderer
maps it onto a sphere that follows the camera, so the sky never moves relative
to the player and is visible in every direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pygame

from .math3d import Vec3
from .mesh import Mesh

SKYBOX_DIR = Path(__file__).resolve().parent / "skybox"
"""Folder scanned for map sky textures."""

SKYBOX_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tga")
"""Image formats accepted for a sky texture."""

SKYBOX_RADIUS = 100.0
"""Radius of the sky dome, in metres. It only has to fit inside the far plane."""

SKYBOX_SEGMENTS = 64
"""Longitude subdivisions of the generated sky dome."""

SKYBOX_RINGS = 32
"""Latitude subdivisions of the generated sky dome."""


@dataclass(frozen=True)
class Skybox:
    """A loaded sky panorama and the file it came from."""

    name: str
    surface: pygame.Surface
    path: Path | None = None


_skybox_cache: dict[Path, Skybox] = {}


def ensure_skybox_directory() -> Path:
    """Create ``engine/skybox`` if it does not exist yet and return it."""
    try:
        SKYBOX_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return SKYBOX_DIR


def _map_relative_stem(map_path: Path) -> Path:
    """Return the map name without its suffix, relative to the maps folder."""
    map_path = Path(map_path)
    try:
        relative = map_path.resolve().relative_to(
            (SKYBOX_DIR.parent / "maps").resolve()
        )
    except (OSError, ValueError):
        relative = Path(map_path.name)
    return relative.with_suffix("")


def find_skybox_texture(map_path: str | Path) -> Path | None:
    """Find the sky texture that belongs to ``map_path``.

    Exact relative matches are tried first (so ``sub/level.map`` prefers
    ``skybox/sub/level.png``), then the plain file name is tried at the root.
    If neither exists, the folder is scanned case-insensitively, which keeps
    the engine usable on case-sensitive file systems.
    """
    ensure_skybox_directory()
    stem = _map_relative_stem(Path(map_path))
    stem_text = stem.as_posix()
    plain_text = Path(stem.name).as_posix()

    for candidate_stem in (stem_text, plain_text):
        for extension in SKYBOX_EXTENSIONS:
            candidate = SKYBOX_DIR / f"{candidate_stem}{extension}"
            if candidate.is_file():
                return candidate

    wanted = {stem_text.casefold(), plain_text.casefold()}
    try:
        files = sorted(
            path
            for path in SKYBOX_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in SKYBOX_EXTENSIONS
        )
    except OSError:
        return None
    for path in files:
        relative = path.relative_to(SKYBOX_DIR).with_suffix("")
        if relative.as_posix().casefold() in wanted:
            return path
    return None


def load_skybox_for_map(map_path: str | Path) -> Skybox | None:
    """Load the sky texture matching ``map_path``, or return ``None``.

    A missing or unreadable sky texture is not a map error: the level simply
    keeps the engine's dark background colour.
    """
    path = find_skybox_texture(map_path)
    if path is None:
        return None

    cached = _skybox_cache.get(path)
    if cached is not None:
        return cached

    try:
        surface = pygame.image.load(str(path))
    except (pygame.error, OSError) as exc:
        print(f"[skybox] could not load {path.name}: {exc}")
        return None

    skybox = Skybox(name=path.stem, surface=surface, path=path)
    _skybox_cache[path] = skybox
    width, height = surface.get_size()
    print(f"[skybox] {Path(map_path).name} -> {path.name} ({width}x{height})")
    return skybox


def clear_skybox_cache() -> None:
    """Forget loaded panoramas (useful for tests and hot reloads)."""
    _skybox_cache.clear()


def create_sky_dome(
    segments: int = SKYBOX_SEGMENTS,
    rings: int = SKYBOX_RINGS,
) -> Mesh:
    """Build a unit sphere used to draw the panorama around the camera.

    The mesh has no UVs: the sky shader derives equirectangular coordinates
    directly from the direction of every vertex, so the texture seam and the
    poles stay stable regardless of tessellation.
    """
    segments = max(3, int(segments))
    rings = max(2, int(rings))
    mesh = Mesh("skybox:dome")

    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        y = math.cos(phi)
        radius = math.sin(phi)
        for segment in range(segments + 1):
            theta = 2.0 * math.pi * segment / segments
            mesh.add_vertex(
                Vec3(radius * math.cos(theta), y, radius * math.sin(theta))
            )

    stride = segments + 1
    for ring in range(rings):
        for segment in range(segments):
            a = ring * stride + segment
            b = a + 1
            c = a + stride
            d = c + 1
            mesh.add_triangle(a, c, b)
            mesh.add_triangle(b, c, d)
    return mesh
