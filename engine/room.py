"""The asset showcase map used by the demo.

The old arena is gone: the scene is now a wide floor with a single row of
textured props, each built by its own class in :mod:`engine.objects`.
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
SHOWCASE_LINE = (
    (Crate, -7.35),
    (ConcreteBlock, -5.25),
    (WoodenWall, -3.15),
    (StoneWall, -1.05),
    (BrickWall, 1.05),
    (MetalPanel, 3.15),
    (MetalCrate, 5.25),
    (StonePillar, 7.35),
)


def build_asset_showcase(scene: Scene) -> None:
    """Populate ``scene`` with the textured object gallery."""
    scene.spawn = Vec3(0.0, 0.02, 14.5)
    scene.spawn_yaw = 0.0

    Floor.spawn(scene, 0.0)

    for object_class, x in SHOWCASE_LINE:
        object_class.spawn(scene, x)
