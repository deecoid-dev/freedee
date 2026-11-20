"""Asset-backed object classes for the showcase map.

Every visible prop is a small subclass of :class:`TexturedObject`. The class
chooses the material name, so a crate always receives the crate texture and a
stone wall always receives the stone texture, even when an artist later drops
a replacement file into ``engine/assets``.
"""

from __future__ import annotations

from .collision import AABB
from .materials import get_material
from .math3d import Vec3
from .scene import Object3D, Scene
from .shapes import create_box


class TexturedObject(Object3D):
    """A box-shaped prop whose subclass selects the material and footprint."""

    material_name = "concrete"
    uv_scale: float | None = None
    default_size = Vec3(1.0, 1.0, 1.0)
    #: Props stand with their bottom at y = 0. The floor overrides this so its
    #: top surface is the ground plane instead.
    bottom_at_origin = True

    def __init__(
        self,
        position: Vec3 | None = None,
        size: Vec3 | None = None,
        rotation: Vec3 | None = None,
        name: str | None = None,
    ) -> None:
        size = size or self.default_size
        super().__init__(
            create_box(
                size.x, size.y, size.z,
                name or type(self).__name__,
            ),
            position=position,
            rotation=rotation,
            material=get_material(self.material_name),
            uv_scale=self.uv_scale,
        )
        self.size = size

    def place(
        self, scene: Scene, x: float, z: float = 0.0
    ) -> "TexturedObject":
        """Stand the object on the floor and register its collision box."""
        if self.bottom_at_origin:
            y = self.size.y * 0.5
        else:
            y = -self.size.y * 0.5
        self.position = Vec3(x, y, z)
        scene.add(self)
        scene.add_collider(AABB.from_center_size(self.position, self.size))
        return self

    @classmethod
    def spawn(
        cls,
        scene: Scene,
        x: float,
        z: float = 0.0,
        size: Vec3 | None = None,
    ) -> "TexturedObject":
        return cls(size=size).place(scene, x, z)


class Crate(TexturedObject):
    """Wooden cargo crate."""

    material_name = "crate"
    default_size = Vec3(1.3, 1.3, 1.3)


class WoodenWall(TexturedObject):
    """Timber plank wall section."""

    material_name = "wood_wall"
    default_size = Vec3(2.3, 2.8, 0.36)


class StoneWall(TexturedObject):
    """Heavy stone block wall section."""

    material_name = "stone_wall"
    default_size = Vec3(2.3, 3.2, 0.44)


class BrickWall(TexturedObject):
    """Red brick wall section."""

    material_name = "brick_wall"
    default_size = Vec3(2.3, 2.6, 0.36)


class MetalPanel(TexturedObject):
    """Riveted steel panel standing upright."""

    material_name = "metal_panel"
    default_size = Vec3(1.7, 2.4, 0.30)


class MetalCrate(TexturedObject):
    """Compact steel crate next to the panel."""

    material_name = "metal_panel"
    default_size = Vec3(1.1, 1.1, 1.1)


class ConcreteBlock(TexturedObject):
    """Cast concrete block."""

    material_name = "concrete"
    default_size = Vec3(1.5, 1.5, 1.5)


class StonePillar(TexturedObject):
    """Tall narrow stone pillar."""

    material_name = "stone_wall"
    default_size = Vec3(0.9, 3.2, 0.9)


class Floor(TexturedObject):
    """Large floor plate; its top surface sits on y = 0."""

    material_name = "floor_tiles"
    default_size = Vec3(34.0, 0.5, 34.0)
    bottom_at_origin = False
