"""Scene graph primitives: camera, scene and transformable objects."""

from __future__ import annotations

import math

from .collision import AABB
from .math3d import (
    Mat4,
    Vec3,
    mat4_mul,
    mat4_rotation_x,
    mat4_rotation_y,
    mat4_rotation_z,
    mat4_scale,
    mat4_translation,
)
from .mesh import Mesh


class Camera:
    """A perspective camera with free-look yaw/pitch controls."""

    MAX_PITCH = math.radians(89.0)

    def __init__(
        self,
        position: Vec3 | None = None,
        target: Vec3 | None = None,
        fov: float = 50.0,
    ) -> None:
        self.position = position or Vec3(0.0, 0.0, 5.4)
        self.target = target or Vec3()
        self.up = Vec3(0.0, 1.0, 0.0)
        self.fov = fov
        self.yaw = 0.0
        self.pitch = 0.0
        self.aim_at_target()

    def forward(self) -> Vec3:
        """Unit vector the camera is looking along."""
        cos_pitch = math.cos(self.pitch)
        return Vec3(
            math.sin(self.yaw) * cos_pitch,
            math.sin(self.pitch),
            -math.cos(self.yaw) * cos_pitch,
        )

    def rotate(self, yaw_delta: float, pitch_delta: float) -> None:
        """Turn the camera (radians) and keep the look target in sync."""
        self.yaw = (self.yaw + yaw_delta) % (2.0 * math.pi)
        self.pitch = max(
            -self.MAX_PITCH,
            min(self.MAX_PITCH, self.pitch + pitch_delta),
        )
        self.sync_target()

    def sync_target(self) -> None:
        self.target = self.position + self.forward()

    def aim_at_target(self) -> None:
        """Derive yaw/pitch from the current position and target."""
        direction = self.target - self.position
        if direction.length_squared() < 1e-12:
            return
        direction = direction.normalized()
        self.pitch = math.asin(max(-1.0, min(1.0, direction.y)))
        self.yaw = math.atan2(direction.x, -direction.z)


class Object3D:
    """A mesh placed in the world."""

    def __init__(
        self,
        mesh: Mesh,
        position: Vec3 | None = None,
        rotation: Vec3 | None = None,
        scale: float = 1.0,
        *,
        hue: float | None = None,
        saturation: float = 0.85,
    ) -> None:
        self.mesh = mesh
        self.position = position or Vec3()
        self.rotation = rotation or Vec3()
        self.scale = scale
        #: A fixed hue turns off the animated iridescent palette.
        self.hue = hue
        self.saturation = saturation

    def model_matrix(self) -> Mat4:
        translate = mat4_translation(self.position.x, self.position.y, self.position.z)
        rotate_y = mat4_rotation_y(self.rotation.y)
        rotate_x = mat4_rotation_x(self.rotation.x)
        rotate_z = mat4_rotation_z(self.rotation.z)
        scale = mat4_scale(self.scale)
        return mat4_mul(
            translate,
            mat4_mul(rotate_y, mat4_mul(rotate_x, mat4_mul(rotate_z, scale))),
        )


class Scene:
    """A flat list of renderable objects plus the world collision boxes."""

    def __init__(self) -> None:
        self.objects: list[Object3D] = []
        self.colliders: list[AABB] = []
        self.spawn = Vec3(0.0, 0.0, 7.5)
        self.spawn_yaw = 0.0

    def add(self, obj: Object3D) -> Object3D:
        self.objects.append(obj)
        return obj

    def add_collider(self, collider: AABB) -> AABB:
        self.colliders.append(collider)
        return collider
