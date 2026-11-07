"""The test room used by the walking demo."""

from __future__ import annotations

from .collision import AABB
from .math3d import Vec3
from .scene import Object3D, Scene
from .shapes import create_box

# Muted blue-grey room, warm crates and a bright central landmark.
WALL_HUE = 0.58
FLOOR_HUE = 0.62
PLATFORM_HUE = 0.55
PEDESTAL_HUE = 0.52
PILLAR_HUE = 0.49
CRATE_HUE = 0.08


def _add_box(
    scene: Scene,
    name: str,
    center: Vec3,
    size: Vec3,
    hue: float,
    saturation: float = 0.28,
    collider: bool = True,
) -> Object3D:
    obj = Object3D(
        create_box(size.x, size.y, size.z, name),
        position=center,
        hue=hue,
        saturation=saturation,
    )
    scene.add(obj)
    if collider:
        scene.add_collider(AABB.from_center_size(center, size))
    return obj


def build_test_room(scene: Scene) -> None:
    """Populate ``scene`` with a closed arena made for movement testing.

    The room is 20 × 20 metres with a 4 metre ceiling, a raised platform with
    steps, crates to jump on and a low barrier for bunny hopping over.
    """
    scene.spawn = Vec3(0.0, 0.02, 8.5)
    scene.spawn_yaw = 0.0

    # Shell -----------------------------------------------------------------
    _add_box(scene, "floor", Vec3(0.0, -0.25, 0.0), Vec3(21.0, 0.5, 21.0),
             FLOOR_HUE, 0.24)
    _add_box(scene, "ceiling", Vec3(0.0, 4.25, 0.0), Vec3(21.0, 0.5, 21.0),
             WALL_HUE, 0.22)
    _add_box(scene, "wall north", Vec3(0.0, 2.25, -10.25),
             Vec3(21.0, 4.5, 0.5), WALL_HUE, 0.30)
    _add_box(scene, "wall south", Vec3(0.0, 2.25, 10.25),
             Vec3(21.0, 4.5, 0.5), WALL_HUE, 0.30)
    _add_box(scene, "wall west", Vec3(-10.25, 2.25, 0.0),
             Vec3(0.5, 4.5, 21.0), WALL_HUE, 0.30)
    _add_box(scene, "wall east", Vec3(10.25, 2.25, 0.0),
             Vec3(0.5, 4.5, 21.0), WALL_HUE, 0.30)

    # Central pedestal and landmark cube ------------------------------------
    _add_box(scene, "pedestal", Vec3(0.0, 0.5, 0.0), Vec3(3.4, 1.0, 3.4),
             PEDESTAL_HUE, 0.26)

    # Raised platform at the back, with three climbable steps --------------
    _add_box(scene, "platform", Vec3(0.0, 0.5, -8.4), Vec3(11.0, 1.0, 3.2),
             PLATFORM_HUE, 0.24)
    _add_box(scene, "step 1", Vec3(0.0, 0.125, -5.3), Vec3(4.4, 0.25, 0.6),
             PLATFORM_HUE, 0.26)
    _add_box(scene, "step 2", Vec3(0.0, 0.25, -5.9), Vec3(4.4, 0.5, 0.6),
             PLATFORM_HUE, 0.26)
    _add_box(scene, "step 3", Vec3(0.0, 0.375, -6.5), Vec3(4.4, 0.75, 0.6),
             PLATFORM_HUE, 0.26)

    # Things to vault and jump on -------------------------------------------
    _add_box(scene, "low barrier", Vec3(6.0, 0.35, 5.0), Vec3(5.0, 0.7, 0.4),
             CRATE_HUE, 0.48)
    _add_box(scene, "crate right", Vec3(6.6, 0.45, 0.0), Vec3(2.2, 0.9, 2.2),
             CRATE_HUE, 0.46)
    _add_box(scene, "crate left", Vec3(-6.6, 0.6, 1.6), Vec3(1.6, 1.2, 1.6),
             CRATE_HUE, 0.42)

    # Corner pillars give the room some depth and occluders.
    for index, (x, z) in enumerate(
        ((-7.8, -7.8), (7.8, -7.8), (-7.8, 7.8), (7.8, 7.8))
    ):
        _add_box(
            scene,
            f"pillar {index + 1}",
            Vec3(x, 2.0, z),
            Vec3(1.0, 4.0, 1.0),
            PILLAR_HUE,
            0.26,
        )


def add_landmark_cube(scene: Scene) -> Object3D:
    """Add the rotating iridescent cube above the pedestal."""
    from .shapes import create_cube

    cube = Object3D(
        create_cube(size=1.7),
        position=Vec3(0.0, 2.15, 0.0),
        rotation=Vec3(0.42, 0.6, 0.0),
    )
    scene.add(cube)
    scene.add_collider(
        AABB.from_center_size(Vec3(0.0, 2.15, 0.0), Vec3(1.7, 1.7, 1.7))
    )
    return cube
