"""Runtime loader for Quake ``.map`` level files.

The parser understands the classic Quake brush syntax and the Valve 220
variant with explicit U/V axes.  Brushes are stored in the map file as a set of
planes, so the loader rebuilds the convex polygon of every face by clipping a
large quad against all the other planes of the brush.

Maps are authored Z-up (the Quake convention), while the engine is Y-up.  The
loader converts ``(x, y, z)`` to ``(x, z, -y)`` and scales the level to metres
(``0.0254`` by default, i.e. one map unit is one inch).  Texture coordinates
are interpreted with the classic 64-unit texture tile used by Quake-era maps.

Only static geometry is used: every solid brush becomes an AABB collider, and
faces are grouped by material into a handful of meshes.  ``info_player_start``
and ``info_player_deathmatch`` provide the spawn point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .collision import AABB
from .materials import get_map_material
from .math3d import Vec3
from .mesh import Mesh
from .scene import Object3D, Scene
from .skybox import load_skybox_for_map

MAPS_DIR = Path(__file__).resolve().parent / "maps"
"""Folder scanned by the New Game menu."""

MAP_EXTENSION = ".map"
DEFAULT_MAP_SCALE = 0.0254
"""Metres per map unit (one Quake unit is one inch)."""

PLAYER_START_HEIGHT_UNITS = 24.0
"""Quake player origins sit 24 units above the feet."""

MAP_TEXTURE_SIZE = (64.0, 64.0)
"""Nominal texture size used to interpret .map texture coordinates.

Quake-era WAD textures are overwhelmingly 64x64 units, and editors write
texture offsets and scales in that space.  The replacement image can have any
pixel resolution; only the authored UV grid depends on this constant.
"""

_CLIP_EPSILON = 0.02
_HUGE = 131072.0

_INVISIBLE_TEXTURES = {
    "clip",
    "playerclip",
    "monsterclip",
    "nodraw",
    "skip",
    "hint",
    "origin",
    "areaportal",
    "trigger",
    "null",
}

_NON_RENDER_ENTITY_PREFIXES = (
    "trigger_",
    "func_areaportal",
    "func_occluder",
)

_NON_SOLID_ENTITY_PREFIXES = (
    "trigger_",
    "func_areaportal",
    "func_occluder",
    "func_illusionary",
    "func_water",
    "func_fish_pool",
)

_SPAWN_CLASSES = (
    "info_player_start",
    "info_player_deathmatch",
    "info_player_teamspawn",
    "info_player",
)


class MapError(Exception):
    """Raised when a map cannot be read or contains invalid brush data."""


@dataclass(frozen=True)
class MapInfo:
    """One entry shown in the New Game map list."""

    path: Path
    name: str
    detail: str


@dataclass(frozen=True)
class MapMetadata:
    """Summary returned after a successful load."""

    title: str
    brushes: int
    faces: int
    visible_faces: int
    colliders: int
    spawn_found: bool
    warnings: tuple[str, ...] = ()
    skybox: str = ""


@dataclass(frozen=True)
class _Token:
    value: str
    line: int
    quoted: bool = False


@dataclass
class _Face:
    a: Vec3
    b: Vec3
    c: Vec3
    texture: str
    texture_mode: str  # "standard" or "valve"
    rotation: float = 0.0
    u_scale: float = 1.0
    v_scale: float = 1.0
    u_offset: float = 0.0
    v_offset: float = 0.0
    u_axis: Vec3 = field(default_factory=Vec3)
    v_axis: Vec3 = field(default_factory=Vec3)


@dataclass
class _Brush:
    faces: list[_Face]


@dataclass
class _Entity:
    properties: dict[str, str]
    brushes: list[_Brush]


@dataclass(frozen=True)
class _Plane:
    normal: Vec3
    distance: float


@dataclass
class _MapDocument:
    entities: list[_Entity]
    warnings: list[str]

    @property
    def worldspawn(self) -> _Entity | None:
        for entity in self.entities:
            if entity.properties.get("classname", "worldspawn").lower() == "worldspawn":
                return entity
        return None

    @property
    def title(self) -> str:
        world = self.worldspawn
        if world is not None:
            for key in ("message", "title", "_title"):
                value = world.properties.get(key, "").strip()
                if value:
                    return value
        return ""

    @property
    def brush_count(self) -> int:
        return sum(len(entity.brushes) for entity in self.entities)

    @property
    def face_count(self) -> int:
        return sum(
            len(brush.faces)
            for entity in self.entities
            for brush in entity.brushes
        )


# ----------------------------------------------------------------------
# Tokenizer and parser
# ----------------------------------------------------------------------
def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    index = 0
    line = 1
    length = len(text)

    while index < length:
        char = text[index]
        if char == "\n":
            line += 1
            index += 1
            continue
        if char in " \t\r\v\f":
            index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "/":
            index += 2
            while index < length and text[index] != "\n":
                index += 1
            continue
        if char == '"':
            start_line = line
            index += 1
            value: list[str] = []
            while index < length:
                current = text[index]
                if current == "\\" and index + 1 < length:
                    following = text[index + 1]
                    if following in {'"', "\\"}:
                        value.append(following)
                        index += 2
                        continue
                if current == '"':
                    index += 1
                    break
                if current == "\n":
                    line += 1
                value.append(current)
                index += 1
            else:
                raise MapError(f"line {start_line}: unterminated string")
            tokens.append(_Token("".join(value), start_line, True))
            continue
        if char in "{}()[]":
            tokens.append(_Token(char, line))
            index += 1
            continue

        start = index
        while index < length:
            current = text[index]
            if current in " \t\r\n\v\f{}()[]\"":
                break
            if (
                current == "/"
                and index + 1 < length
                and text[index + 1] == "/"
            ):
                break
            index += 1
        value = text[start:index]
        if value:
            tokens.append(_Token(value, line))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.index = 0

    def at_end(self) -> bool:
        return self.index >= len(self.tokens)

    def peek(self) -> _Token | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def take(self) -> _Token:
        token = self.peek()
        if token is None:
            raise MapError("unexpected end of file")
        self.index += 1
        return token

    def expect(self, value: str) -> _Token:
        token = self.take()
        if token.value != value:
            raise MapError(
                f"line {token.line}: expected {value!r}, got {token.value!r}"
            )
        return token

    def parse(self) -> _MapDocument:
        entities: list[_Entity] = []
        warnings: list[str] = []
        while not self.at_end():
            token = self.peek()
            assert token is not None
            if token.value != "{":
                raise MapError(
                    f"line {token.line}: expected '{{' to start an entity, "
                    f"got {token.value!r}"
                )
            self.take()
            entities.append(self._parse_entity(warnings))
        if not entities:
            raise MapError("map contains no entities")
        return _MapDocument(entities, warnings)

    def _parse_entity(self, warnings: list[str]) -> _Entity:
        properties: dict[str, str] = {}
        brushes: list[_Brush] = []
        while True:
            token = self.peek()
            if token is None:
                raise MapError("unexpected end of file inside an entity")
            if token.value == "}":
                self.take()
                break
            if token.value == "{":
                brushes.append(self._parse_brush(warnings))
                continue
            key = self.take().value
            value_token = self.peek()
            if value_token is None or value_token.value in "{}":
                raise MapError(
                    f"line {token.line}: missing value for entity key {key!r}"
                )
            properties[key] = self.take().value
        return _Entity(properties, brushes)

    def _parse_brush(self, warnings: list[str]) -> _Brush:
        start = self.expect("{")
        faces: list[_Face] = []
        while True:
            token = self.peek()
            if token is None:
                raise MapError(
                    f"line {start.line}: unexpected end of file inside a brush"
                )
            if token.value == "}":
                self.take()
                break
            if token.value == "(":
                faces.append(self._parse_face())
                continue
            # Quake 3 brush primitives and patches use an extra keyword and a
            # nested block.  The classic and Valve formats are enough for the
            # editor workflow this engine targets, so skip the block instead
            # of failing the whole map.
            keyword = self.take().value
            if self.peek() is not None and self.peek().value == "{":
                self._skip_block()
                warnings.append(
                    f"line {token.line}: skipped unsupported {keyword} block"
                )
                continue
            raise MapError(
                f"line {token.line}: expected a brush plane, got {keyword!r}"
            )
        if not faces:
            warnings.append(f"line {start.line}: empty brush was skipped")
        return _Brush(faces)

    def _parse_face(self) -> _Face:
        points: list[Vec3] = []
        for _ in range(3):
            self.expect("(")
            values = [self._number() for _ in range(3)]
            self.expect(")")
            points.append(Vec3(values[0], values[1], values[2]))

        texture_token = self.take()
        if texture_token.value in "{}()":
            raise MapError(
                f"line {texture_token.line}: missing texture name in a face"
            )
        texture = texture_token.value

        raw: list[str | list[str]] = []
        while True:
            token = self.peek()
            if token is None or token.value in ("(", "}"):
                break
            if token.value == "[":
                self.take()
                group: list[str] = []
                while True:
                    inner = self.peek()
                    if inner is None:
                        raise MapError(
                            f"line {token.line}: unterminated texture axis"
                        )
                    if inner.value == "]":
                        self.take()
                        break
                    group.append(self.take().value)
                raw.append(group)
                continue
            raw.append(self.take().value)

        groups = [item for item in raw if isinstance(item, list)]
        numbers = [item for item in raw if not isinstance(item, list)]

        if len(groups) >= 2:
            u_values = groups[0]
            v_values = groups[1]
            if len(u_values) < 4 or len(v_values) < 4:
                raise MapError(
                    f"line {texture_token.line}: invalid Valve 220 texture axes"
                )
            u_axis = Vec3(
                self._float(u_values[0]),
                self._float(u_values[1]),
                self._float(u_values[2]),
            )
            v_axis = Vec3(
                self._float(v_values[0]),
                self._float(v_values[1]),
                self._float(v_values[2]),
            )
            u_offset = self._float(u_values[3])
            v_offset = self._float(v_values[3])
            rotation = self._float(numbers[0]) if len(numbers) > 0 else 0.0
            u_scale = self._float(numbers[1]) if len(numbers) > 1 else 1.0
            v_scale = self._float(numbers[2]) if len(numbers) > 2 else 1.0
            return _Face(
                points[0], points[1], points[2], texture, "valve",
                rotation=rotation,
                u_scale=u_scale or 1.0,
                v_scale=v_scale or 1.0,
                u_offset=u_offset,
                v_offset=v_offset,
                u_axis=u_axis,
                v_axis=v_axis,
            )

        values = [self._float(value) for value in numbers]
        u_offset = values[0] if len(values) > 0 else 0.0
        v_offset = values[1] if len(values) > 1 else 0.0
        rotation = values[2] if len(values) > 2 else 0.0
        u_scale = values[3] if len(values) > 3 else 1.0
        v_scale = values[4] if len(values) > 4 else 1.0
        return _Face(
            points[0], points[1], points[2], texture, "standard",
            rotation=rotation,
            u_scale=u_scale or 1.0,
            v_scale=v_scale or 1.0,
            u_offset=u_offset,
            v_offset=v_offset,
        )

    def _number(self) -> float:
        token = self.take()
        try:
            return float(token.value)
        except ValueError as exc:
            raise MapError(
                f"line {token.line}: expected a number, got {token.value!r}"
            ) from exc

    @staticmethod
    def _float(value: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise MapError(f"invalid number {value!r} in texture data") from exc

    def _skip_block(self) -> None:
        self.expect("{")
        depth = 1
        while depth > 0:
            token = self.take()
            if token.value == "{":
                depth += 1
            elif token.value == "}":
                depth -= 1


def _parse_document(path: Path) -> _MapDocument:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise MapError(f"cannot read {path.name}: {exc}") from exc
    if text.startswith("\ufeff"):
        text = text[1:]
    tokens = _tokenize(text)
    if not tokens:
        raise MapError(f"{path.name} is empty")
    try:
        return _Parser(tokens).parse()
    except MapError as exc:
        message = str(exc)
        if message.startswith("line "):
            raise MapError(f"{path.name}: {message}") from None
        raise MapError(f"{path.name}: {message}") from None


# ----------------------------------------------------------------------
# Brush geometry
# ----------------------------------------------------------------------
def _plane_from_points(a: Vec3, b: Vec3, c: Vec3) -> _Plane | None:
    normal = (a - b).cross(c - b)
    length = normal.length()
    if length < 1e-8:
        return None
    normal = normal / length
    return _Plane(normal, normal.dot(a))


def _clip_polygon(polygon: list[Vec3], plane: _Plane) -> list[Vec3]:
    if not polygon:
        return []
    result: list[Vec3] = []
    previous = polygon[-1]
    previous_distance = plane.normal.dot(previous) - plane.distance
    previous_inside = previous_distance <= _CLIP_EPSILON

    for current in polygon:
        current_distance = plane.normal.dot(current) - plane.distance
        current_inside = current_distance <= _CLIP_EPSILON
        if current_inside:
            if not previous_inside:
                result.append(
                    _intersect_plane(
                        previous, current, previous_distance, current_distance
                    )
                )
            result.append(current)
        elif previous_inside:
            result.append(
                _intersect_plane(
                    previous, current, previous_distance, current_distance
                )
            )
        previous = current
        previous_distance = current_distance
        previous_inside = current_inside
    return result


def _intersect_plane(
    start: Vec3, end: Vec3, start_distance: float, end_distance: float
) -> Vec3:
    denominator = start_distance - end_distance
    if abs(denominator) < 1e-12:
        return start
    t = start_distance / denominator
    t = max(0.0, min(1.0, t))
    return start.lerp(end, t)


def _clean_polygon(polygon: list[Vec3]) -> list[Vec3]:
    cleaned: list[Vec3] = []
    for point in polygon:
        if not cleaned or (point - cleaned[-1]).length_squared() > 1e-8:
            cleaned.append(point)
    if len(cleaned) > 1 and (cleaned[0] - cleaned[-1]).length_squared() <= 1e-8:
        cleaned.pop()
    return cleaned


def _face_polygon(
    plane: _Plane, planes: list[_Plane], face_index: int
) -> list[Vec3]:
    helper = Vec3(0.0, 0.0, 1.0)
    if abs(plane.normal.z) > 0.9:
        helper = Vec3(1.0, 0.0, 0.0)
    axis_u = plane.normal.cross(helper)
    if axis_u.length_squared() < 1e-10:
        return []
    axis_u = axis_u.normalized()
    axis_v = plane.normal.cross(axis_u).normalized()
    center = plane.normal * plane.distance
    polygon = [
        center - axis_u * _HUGE - axis_v * _HUGE,
        center + axis_u * _HUGE - axis_v * _HUGE,
        center + axis_u * _HUGE + axis_v * _HUGE,
        center - axis_u * _HUGE + axis_v * _HUGE,
    ]

    for index, other in enumerate(planes):
        if index == face_index:
            continue
        polygon = _clip_polygon(polygon, other)
        if len(polygon) < 3:
            return []

    polygon = _clean_polygon(polygon)
    if len(polygon) < 3:
        return []

    centroid = Vec3()
    for point in polygon:
        centroid += point
    centroid = centroid / len(polygon)

    area = Vec3()
    for index, point in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        area += (point - centroid).cross(following - centroid)
    if area.dot(plane.normal) < 0.0:
        polygon.reverse()
    return polygon


def _texture_axes(normal: Vec3) -> tuple[Vec3, Vec3]:
    """Return the implicit U/V axes of the standard Quake texture projection."""
    if abs(normal.z) > 0.5:
        return Vec3(1.0, 0.0, 0.0), Vec3(0.0, -1.0, 0.0)
    if abs(normal.x) > abs(normal.y):
        return Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, -1.0)
    return Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, -1.0)


def _face_uvs(
    face: _Face,
    plane: _Plane,
    polygon: list[Vec3],
) -> list[tuple[float, float]]:
    width, height = MAP_TEXTURE_SIZE
    u_scale = face.u_scale if abs(face.u_scale) > 1e-6 else 1.0
    v_scale = face.v_scale if abs(face.v_scale) > 1e-6 else 1.0

    if face.texture_mode == "valve":
        u_axis = face.u_axis
        v_axis = face.v_axis
        u_offset = face.u_offset
        v_offset = face.v_offset
    else:
        base_u, base_v = _texture_axes(plane.normal)
        angle = math.radians(face.rotation)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        u_axis = base_u * cosine - base_v * sine
        v_axis = base_u * sine + base_v * cosine
        u_offset = face.u_offset
        v_offset = face.v_offset

    uvs: list[tuple[float, float]] = []
    for point in polygon:
        u = (point.dot(u_axis) / u_scale + u_offset) / width
        v = (point.dot(v_axis) / v_scale + v_offset) / height
        uvs.append((u, v))
    return uvs


def _to_engine(point: Vec3, scale: float) -> Vec3:
    return Vec3(point.x * scale, point.z * scale, -point.y * scale)


def _bounds(points: list[Vec3]) -> tuple[Vec3, Vec3]:
    minimum = Vec3(
        min(point.x for point in points),
        min(point.y for point in points),
        min(point.z for point in points),
    )
    maximum = Vec3(
        max(point.x for point in points),
        max(point.y for point in points),
        max(point.z for point in points),
    )
    return minimum, maximum


def _is_invisible_texture(texture: str) -> bool:
    name = texture.replace("\\", "/").rsplit("/", 1)[-1].lower()
    name = name.rsplit(".", 1)[0]
    return name in _INVISIBLE_TEXTURES


def _entity_is_solid(classname: str, texture: str, face_count: int) -> bool:
    if classname.startswith(_NON_SOLID_ENTITY_PREFIXES):
        return False
    return face_count >= 4


def _build_brush(
    brush: _Brush,
    *,
    render: bool,
    solid: bool,
    scene: Scene,
    scale: float,
    groups: dict[int, tuple[object, Mesh]],
    warnings: list[str],
) -> tuple[int, int, AABB | None]:
    planes: list[_Plane] = []
    valid_faces: list[_Face] = []
    for face in brush.faces:
        plane = _plane_from_points(face.a, face.b, face.c)
        if plane is None:
            warnings.append("degenerate brush plane was skipped")
            continue
        planes.append(plane)
        valid_faces.append(face)
    if len(valid_faces) < 4:
        return 0, 0, None

    world_points: list[Vec3] = []
    visible_faces = 0
    total_faces = 0
    for index, (face, plane) in enumerate(zip(valid_faces, planes)):
        polygon = _face_polygon(plane, planes, index)
        if len(polygon) < 3:
            continue
        total_faces += 1
        world_points.extend(_to_engine(point, scale) for point in polygon)

        if not render or _is_invisible_texture(face.texture):
            continue

        material = get_map_material(face.texture)
        key = id(material)
        group = groups.get(key)
        if group is None:
            mesh = Mesh(f"map:{material.name}")
            group = (material, mesh)
            groups[key] = group
        material, mesh = group

        uvs = _face_uvs(face, plane, polygon)
        base = len(mesh.vertices)
        for point, uv in zip(polygon, uvs):
            mesh.add_vertex(_to_engine(point, scale))
            mesh.uvs.append(uv)

        for offset in range(1, len(polygon) - 1):
            a = base
            b = base + offset
            c = base + offset + 1
            pa = mesh.vertices[a]
            pb = mesh.vertices[b]
            pc = mesh.vertices[c]
            if (pb - pa).cross(pc - pa).length_squared() > 1e-10:
                mesh.add_triangle(a, b, c)
        visible_faces += 1

    collider = None
    if solid and world_points:
        minimum, maximum = _bounds(world_points)
        if (
            maximum.x - minimum.x > 1e-4
            and maximum.y - minimum.y > 1e-4
            and maximum.z - minimum.z > 1e-4
        ):
            collider = AABB(
                minimum.x, minimum.y, minimum.z,
                maximum.x, maximum.y, maximum.z,
            )
            scene.add_collider(collider)
    return visible_faces, total_faces, collider


def _parse_origin(value: str) -> Vec3 | None:
    parts = value.replace(",", " ").split()
    if len(parts) < 3:
        return None
    try:
        return Vec3(float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def _spawn_from_entities(
    document: _MapDocument, scale: float
) -> tuple[Vec3 | None, float, bool]:
    chosen: _Entity | None = None
    for classname in _SPAWN_CLASSES:
        for entity in document.entities:
            if entity.properties.get("classname", "").lower() == classname:
                chosen = entity
                break
        if chosen is not None:
            break
    if chosen is None:
        return None, 0.0, False

    origin = _parse_origin(chosen.properties.get("origin", ""))
    if origin is None:
        return None, 0.0, False

    angle_text = chosen.properties.get("angle")
    if angle_text is None:
        angles = chosen.properties.get("angles", "").split()
        angle_text = angles[0] if angles else "0"
    try:
        angle = float(angle_text)
    except ValueError:
        angle = 0.0

    position = _to_engine(origin, scale)
    position.y -= PLAYER_START_HEIGHT_UNITS * scale
    # Quake angles start at +X and grow counter-clockwise; engine yaw 90° also
    # looks along +X, so a plain subtraction matches the two conventions.
    yaw = math.radians(90.0 - angle)
    return position, yaw, True


def _fallback_spawn(document: _MapDocument, scale: float) -> Vec3:
    points: list[Vec3] = []
    for entity in document.entities:
        for brush in entity.brushes:
            for face in brush.faces:
                points.extend(
                    (
                        _to_engine(face.a, scale),
                        _to_engine(face.b, scale),
                        _to_engine(face.c, scale),
                    )
                )
    if not points:
        return Vec3(0.0, 0.1, 0.0)
    minimum, maximum = _bounds(points)
    return Vec3(
        (minimum.x + maximum.x) * 0.5,
        maximum.y + 1.0,
        (minimum.z + maximum.z) * 0.5,
    )


def load_map_into(
    scene: Scene,
    path: str | Path,
    *,
    scale: float = DEFAULT_MAP_SCALE,
) -> MapMetadata:
    """Replace ``scene`` with the geometry and spawn point of ``path``.

    Raises :class:`MapError` for unreadable files or invalid brush syntax.
    """
    path = Path(path)
    document = _parse_document(path)
    if scale <= 0.0:
        raise MapError("map scale must be positive")

    scene.clear()
    # A map may ship a panorama with the same file name; the skybox folder is
    # optional, so maps without one simply keep the background colour.
    scene.skybox = load_skybox_for_map(path)
    groups: dict[int, tuple[object, Mesh]] = {}
    warnings = list(document.warnings)
    brush_count = 0
    total_faces = 0
    visible_faces = 0
    colliders = 0

    for entity in document.entities:
        properties = entity.properties
        classname = properties.get("classname", "worldspawn").lower()
        render = not classname.startswith(_NON_RENDER_ENTITY_PREFIXES)
        for brush in entity.brushes:
            brush_count += 1
            solid = _entity_is_solid(
                classname, "", len(brush.faces)
            )
            visible, faces, collider = _build_brush(
                brush,
                render=render,
                solid=solid,
                scene=scene,
                scale=scale,
                groups=groups,
                warnings=warnings,
            )
            total_faces += faces
            visible_faces += visible
            if collider is not None:
                colliders += 1

    for material, mesh in groups.values():
        if mesh.triangles:
            scene.add(
                Object3D(
                    mesh,
                    material=material,
                    uv_scale=1.0,
                )
            )

    spawn, yaw, spawn_found = _spawn_from_entities(document, scale)
    if spawn is None:
        spawn = _fallback_spawn(document, scale)
        warnings.append("no info_player_start; using a fallback spawn")
    scene.spawn = spawn
    scene.spawn_yaw = yaw
    scene.title = document.title or path.stem

    metadata = MapMetadata(
        title=scene.title,
        brushes=brush_count,
        faces=total_faces,
        visible_faces=visible_faces,
        colliders=colliders,
        spawn_found=spawn_found,
        warnings=tuple(warnings),
        skybox=scene.skybox.name if scene.skybox is not None else "",
    )
    print(
        f"[map] {path.name}: {metadata.brushes} brushes, "
        f"{metadata.visible_faces}/{metadata.faces} faces, "
        f"{metadata.colliders} colliders"
    )
    for warning in warnings[:8]:
        print(f"[map] warning: {warning}")
    if visible_faces == 0:
        print("[map] warning: the map has no visible surfaces")
    return metadata


# ----------------------------------------------------------------------
# Map browser support
# ----------------------------------------------------------------------
def ensure_maps_directory() -> Path:
    """Create ``engine/maps`` if it does not exist yet and return it."""
    try:
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return MAPS_DIR


def list_maps() -> list[MapInfo]:
    """Return playable ``.map`` files from ``engine/maps``, newest names first."""
    ensure_maps_directory()
    try:
        paths = [
            candidate
            for candidate in MAPS_DIR.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() == MAP_EXTENSION
        ]
    except OSError:
        return []

    entries: list[MapInfo] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        relative = path.relative_to(MAPS_DIR)
        name = relative.with_suffix("").as_posix()
        size_kb = stat.st_size / 1024.0
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%d.%m.%Y %H:%M")
        entries.append(
            MapInfo(
                path=path,
                name=name,
                detail=f"{size_kb:.1f} KB  ·  {modified}",
            )
        )
    entries.sort(key=lambda entry: entry.name.casefold())
    return entries

# В пламенной душе пригорел весь керосин
# И закрыт уже до завтра новый магазин
# В магазине том, мы погавкались с тобою
# И продрав глаза
# Завтра я уйду к другой!!!