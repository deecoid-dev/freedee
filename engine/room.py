"""The built-in test room used by the separate Test map menu entry.

The room is intentionally small and hard-coded: a floor, four walls and a
ceiling with the textured asset showcase lined up along the back.  Real levels
are loaded from ``engine/maps`` instead.
"""

from __future__ import annotations

from .math3d import Vec3
from .objects import (
    BrickWall,
    ConcreteBlock,
    Crate,
    Floor,
    MetalCrate,
    MetalPanel,
    StonePillar,
    StoneWall,
    WoodenWall,
)
from .scene import Scene

#: Left-to-right order of the showcase line, with each prop's x position.
#: The values are spaced by the prop widths plus a small gap so the line
#: stays inside the room without any walls intersecting each other.
SHOWCASE_LINE = (
    (Crate, -6.925),
    (ConcreteBlock, -5.275),
    (WoodenWall, -3.125),
    (StoneWall, -0.575),
    (BrickWall, 1.975),
    (MetalPanel, 4.225),
    (MetalCrate, 5.875),
    (StonePillar, 7.125),
)

ROOM_WIDTH = 18.0
ROOM_DEPTH = 14.0
ROOM_HEIGHT = 4.2
WALL_THICKNESS = 0.4


def _place_box(scene: Scene, obj, center: Vec3) -> None:
    """Place an already constructed box and register its collision box."""
    obj.position = center
    scene.add(obj)
    scene.add_collider(obj.collider())


def build_test_room(scene: Scene) -> None:
    """Populate ``scene`` with the small hard-coded test room."""
    scene.title = "Test room"
    scene.spawn = Vec3(0.0, 0.02, 5.4)
    scene.spawn_yaw = 0.0

    Floor(size=Vec3(ROOM_WIDTH + WALL_THICKNESS, 0.5, ROOM_DEPTH + WALL_THICKNESS)).place(
        scene, 0.0
    )

    half_width = ROOM_WIDTH * 0.5
    half_depth = ROOM_DEPTH * 0.5
    wall_y = ROOM_HEIGHT * 0.5

    _place_box(
        scene,
        BrickWall(size=Vec3(ROOM_WIDTH, ROOM_HEIGHT, WALL_THICKNESS)),
        Vec3(0.0, wall_y, -half_depth),
    )
    _place_box(
        scene,
        WoodenWall(size=Vec3(ROOM_WIDTH, ROOM_HEIGHT, WALL_THICKNESS)),
        Vec3(0.0, wall_y, half_depth),
    )
    _place_box(
        scene,
        StoneWall(size=Vec3(WALL_THICKNESS, ROOM_HEIGHT, ROOM_DEPTH)),
        Vec3(-half_width, wall_y, 0.0),
    )
    _place_box(
        scene,
        StoneWall(size=Vec3(WALL_THICKNESS, ROOM_HEIGHT, ROOM_DEPTH)),
        Vec3(half_width, wall_y, 0.0),
    )
    _place_box(
        scene,
        ConcreteBlock(
            size=Vec3(
                ROOM_WIDTH + WALL_THICKNESS,
                WALL_THICKNESS,
                ROOM_DEPTH + WALL_THICKNESS,
            )
        ),
        Vec3(0.0, ROOM_HEIGHT + WALL_THICKNESS * 0.5, 0.0),
    )

    for object_class, x in SHOWCASE_LINE:
        object_class.spawn(scene, x, -2.8)


#: Backwards-compatible alias for code that still refers to the old demo map.
build_asset_showcase = build_test_room
