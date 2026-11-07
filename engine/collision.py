"""Axis-aligned collision boxes and a Quake-style move-and-slide solver."""

from __future__ import annotations

import math

from .math3d import Vec3


class AABB:
    """An axis-aligned bounding box in world space."""

    __slots__ = (
        "min_x", "min_y", "min_z",
        "max_x", "max_y", "max_z",
    )

    def __init__(
        self,
        min_x: float, min_y: float, min_z: float,
        max_x: float, max_y: float, max_z: float,
    ) -> None:
        self.min_x = min(min_x, max_x)
        self.min_y = min(min_y, max_y)
        self.min_z = min(min_z, max_z)
        self.max_x = max(min_x, max_x)
        self.max_y = max(min_y, max_y)
        self.max_z = max(min_z, max_z)

    @classmethod
    def from_center_size(cls, center: Vec3, size: Vec3) -> "AABB":
        half_x = abs(size.x) * 0.5
        half_y = abs(size.y) * 0.5
        half_z = abs(size.z) * 0.5
        return cls(
            center.x - half_x, center.y - half_y, center.z - half_z,
            center.x + half_x, center.y + half_y, center.z + half_z,
        )

    def overlaps(
        self,
        min_x: float, min_y: float, min_z: float,
        max_x: float, max_y: float, max_z: float,
    ) -> bool:
        return (
            min_x < self.max_x and max_x > self.min_x
            and min_y < self.max_y and max_y > self.min_y
            and min_z < self.max_z and max_z > self.min_z
        )

    def center(self) -> Vec3:
        return Vec3(
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
            (self.min_z + self.max_z) * 0.5,
        )


class CollisionWorld:
    """Static boxes the player collides with.

    The solver moves the player one axis at a time and pushes them out of any
    box they enter. Small ledges are climbed automatically, like the step-up
    logic in Quake. Movement is split into substeps so high speeds cannot
    tunnel through thin walls.
    """

    MAX_SUBSTEP = 0.15

    def __init__(self, colliders: list[AABB] | None = None) -> None:
        self.colliders: list[AABB] = list(colliders or [])

    def add(self, collider: AABB) -> AABB:
        self.colliders.append(collider)
        return collider

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def overlaps(
        self,
        position: Vec3,
        radius: float,
        height: float,
    ) -> bool:
        min_x = position.x - radius
        max_x = position.x + radius
        min_y = position.y
        max_y = position.y + height
        min_z = position.z - radius
        max_z = position.z + radius
        for box in self.colliders:
            if box.overlaps(min_x, min_y, min_z, max_x, max_y, max_z):
                return True
        return False

    def ground_height(
        self,
        position: Vec3,
        radius: float,
        height: float,
    ) -> float | None:
        """Highest box top directly under the player, if any."""
        best: float | None = None
        min_x = position.x - radius
        max_x = position.x + radius
        min_z = position.z - radius
        max_z = position.z + radius
        for box in self.colliders:
            if (
                min_x < box.max_x and max_x > box.min_x
                and min_z < box.max_z and max_z > box.min_z
            ):
                if box.max_y <= position.y + 0.05:
                    if best is None or box.max_y > best:
                        best = box.max_y
        return best

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def move(
        self,
        position: Vec3,
        velocity: Vec3,
        delta: float,
        radius: float,
        height: float,
        step_height: float = 0.34,
        was_on_ground: bool = False,
    ) -> tuple[Vec3, Vec3, bool]:
        """Integrate ``velocity`` for ``delta`` seconds against the boxes.

        Returns the resolved position, the adjusted velocity and whether the
        player ended the move standing on something.
        """
        pos = Vec3(position.x, position.y, position.z)
        vel = Vec3(velocity.x, velocity.y, velocity.z)
        if delta <= 0.0:
            return pos, vel, False

        max_speed = max(abs(vel.x), abs(vel.y), abs(vel.z))
        substeps = max(1, int(math.ceil(max_speed * delta / self.MAX_SUBSTEP)))
        substeps = min(substeps, 64)
        step_delta = delta / substeps

        on_ground = False
        can_step = was_on_ground

        for _ in range(substeps):
            if vel.x:
                self._move_horizontal(
                    pos, vel, "x", vel.x * step_delta,
                    radius, height, can_step, step_height,
                )
            if vel.z:
                self._move_horizontal(
                    pos, vel, "z", vel.z * step_delta,
                    radius, height, can_step, step_height,
                )

            if vel.y or not on_ground:
                landed = self._move_vertical(
                    pos, vel, vel.y * step_delta, radius, height
                )
                if landed:
                    on_ground = True
                    can_step = True

        # A short probe keeps "on ground" stable while standing still.
        if not on_ground and vel.y <= 0.0:
            probe = Vec3(pos.x, pos.y - 0.06, pos.z)
            if self.overlaps(probe, radius, height):
                top = self.ground_height(pos, radius, height)
                if top is not None and pos.y - top <= 0.06:
                    pos.y = top
                    vel.y = 0.0
                    on_ground = True

        return pos, vel, on_ground

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _move_horizontal(
        self,
        pos: Vec3,
        vel: Vec3,
        axis: str,
        amount: float,
        radius: float,
        height: float,
        can_step: bool,
        step_height: float,
    ) -> None:
        if amount == 0.0:
            return
        if axis == "x":
            pos.x += amount
        else:
            pos.z += amount

        for box in self.colliders:
            if not self._overlaps_box(pos, radius, height, box):
                continue

            # Try to step onto low ledges instead of stopping dead.
            if can_step and amount != 0.0:
                rise = box.max_y - pos.y
                if 0.0 < rise <= step_height:
                    saved_y = pos.y
                    pos.y = box.max_y + 0.0001
                    if not self.overlaps(pos, radius, height):
                        continue
                    pos.y = saved_y

            if axis == "x":
                if amount > 0.0:
                    pos.x = box.min_x - radius - 0.0001
                else:
                    pos.x = box.max_x + radius + 0.0001
                vel.x = 0.0
            else:
                if amount > 0.0:
                    pos.z = box.min_z - radius - 0.0001
                else:
                    pos.z = box.max_z + radius + 0.0001
                vel.z = 0.0

    def _move_vertical(
        self,
        pos: Vec3,
        vel: Vec3,
        amount: float,
        radius: float,
        height: float,
    ) -> bool:
        pos.y += amount

        crossings = [
            box
            for box in self.colliders
            if self._overlaps_box(pos, radius, height, box)
        ]
        if not crossings:
            return False

        if amount <= 0.0:
            top = max(box.max_y for box in crossings)
            if pos.y < top:
                pos.y = top
            vel.y = 0.0
            return True

        bottom = min(box.min_y for box in crossings)
        pos.y = bottom - height - 0.0001
        vel.y = 0.0
        return False

    @staticmethod
    def _overlaps_box(
        pos: Vec3,
        radius: float,
        height: float,
        box: AABB,
    ) -> bool:
        return box.overlaps(
            pos.x - radius,
            pos.y,
            pos.z - radius,
            pos.x + radius,
            pos.y + height,
            pos.z + radius,
        )
