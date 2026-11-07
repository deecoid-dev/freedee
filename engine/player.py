"""Half-Life (GoldSrc) style character movement and bunny hopping."""

from __future__ import annotations

import math

from .collision import CollisionWorld
from .math3d import Vec3

PLAYER_RADIUS = 0.36
PLAYER_HEIGHT = 1.78
EYE_HEIGHT = 1.62

# Tuned for a 1 unit == 1 metre room. Ground movement accelerates towards a
# wish speed and friction slows it down. In the air there is no friction, so
# the player keeps momentum; air acceleration follows Half-Life's
# PM_AirAccelerate(), where the wish speed used for the projection test is
# capped to 30/320 of the run speed. That cap is what makes strafe-jumping
# controllable: forward speed is already above it, so only the lateral
# component can still accelerate. Turning the view while holding A/D keeps
# that component pointed across the velocity, so speed builds up smoothly.
GRAVITY = 20.0
JUMP_SPEED = 7.2
GROUND_ACCELERATION = 12.0
AIR_ACCELERATION = 10.0
AIR_WISH_SPEED_RATIO = 30.0 / 320.0
GROUND_FRICTION = 7.5
STOP_SPEED = 2.0
SPRINT_MULTIPLIER = 1.45
TERMINAL_VELOCITY = 50.0
MAX_HORIZONTAL_SPEED = 32.0
JUMP_BUFFER_TIME = 0.14
STEP_HEIGHT = 0.34

WORLD_UP = Vec3(0.0, 1.0, 0.0)

MAX_PITCH = math.radians(89.0)
TURN_SPEED = 2.2


class Player:
    """First-person character controller."""

    def __init__(
        self,
        spawn: Vec3 | None = None,
        yaw: float = 0.0,
        speed: float = 7.0,
        auto_bhop: bool = False,
    ) -> None:
        self.spawn = spawn or Vec3(0.0, 0.0, 7.5)
        self.spawn_yaw = yaw
        self.position = Vec3(self.spawn.x, self.spawn.y, self.spawn.z)
        self.velocity = Vec3()
        self.yaw = yaw
        self.pitch = 0.0
        self.speed = speed
        self.auto_bhop = auto_bhop
        self.on_ground = False
        self.jump_buffer = 0.0
        self.air_time = 0.0

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def eye_position(self) -> Vec3:
        return Vec3(
            self.position.x,
            self.position.y + EYE_HEIGHT,
            self.position.z,
        )

    def forward(self) -> Vec3:
        """Horizontal look direction, built from the raw yaw angle only."""
        return Vec3(math.sin(self.yaw), 0.0, -math.cos(self.yaw))

    def right(self) -> Vec3:
        """Strafe direction, the cross product of forward and world up.

        Keeping the basis orthogonal this way means A/D always stay exactly
        perpendicular to the view, which is what makes air strafing crisp.
        """
        return self.forward().cross(WORLD_UP)

    def look_direction(self) -> Vec3:
        cos_pitch = math.cos(self.pitch)
        return Vec3(
            math.sin(self.yaw) * cos_pitch,
            math.sin(self.pitch),
            -math.cos(self.yaw) * cos_pitch,
        )

    def horizontal_speed(self) -> float:
        return math.hypot(self.velocity.x, self.velocity.z)

    def respawn(self, spawn: Vec3 | None = None, yaw: float | None = None) -> None:
        if spawn is not None:
            self.spawn = Vec3(spawn.x, spawn.y, spawn.z)
        if yaw is not None:
            self.spawn_yaw = yaw
        self.position = Vec3(self.spawn.x, self.spawn.y, self.spawn.z)
        self.velocity = Vec3()
        self.yaw = self.spawn_yaw
        self.pitch = 0.0
        self.on_ground = False
        self.jump_buffer = 0.0
        self.air_time = 0.0

    def rotate(self, yaw_delta: float, pitch_delta: float) -> None:
        self.yaw = (self.yaw + yaw_delta) % (2.0 * math.pi)
        self.pitch = max(-MAX_PITCH, min(MAX_PITCH, self.pitch + pitch_delta))

    def queue_jump(self) -> None:
        """Remember a jump press so it survives a few frames before landing."""
        self.jump_buffer = JUMP_BUFFER_TIME

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def update(
        self,
        delta: float,
        world: CollisionWorld,
        keys: set[str],
    ) -> None:
        if delta <= 0.0:
            return

        if self.jump_buffer > 0.0:
            self.jump_buffer = max(0.0, self.jump_buffer - delta)
        if self.auto_bhop and "space" in keys:
            self.jump_buffer = JUMP_BUFFER_TIME

        wish = self._wish_direction(keys)
        wish_speed = self.speed
        if "shift" in keys:
            wish_speed *= SPRINT_MULTIPLIER

        jumped = False
        if self.jump_buffer > 0.0 and self.on_ground:
            self.velocity.y = JUMP_SPEED
            self.on_ground = False
            self.jump_buffer = 0.0
            jumped = True

        if self.on_ground and not jumped:
            self._apply_friction(delta)

        if wish.length_squared() > 1e-12:
            if self.on_ground:
                self._accelerate(wish, wish_speed, GROUND_ACCELERATION, delta)
            else:
                self._air_accelerate(wish, wish_speed, AIR_ACCELERATION, delta)

        if not self.on_ground:
            self.velocity.y = max(
                -TERMINAL_VELOCITY,
                self.velocity.y - GRAVITY * delta,
            )
            self.air_time += delta
        else:
            self.air_time = 0.0

        # Keep strafe-jumping from reaching absurd speeds.
        horizontal = self.horizontal_speed()
        if horizontal > MAX_HORIZONTAL_SPEED:
            scale = MAX_HORIZONTAL_SPEED / horizontal
            self.velocity.x *= scale
            self.velocity.z *= scale

        position, velocity, on_ground = world.move(
            self.position,
            self.velocity,
            delta,
            PLAYER_RADIUS,
            PLAYER_HEIGHT,
            STEP_HEIGHT,
            self.on_ground,
        )
        self.position = position
        self.velocity = velocity
        self.on_ground = on_ground

    # ------------------------------------------------------------------
    # Movement model
    # ------------------------------------------------------------------
    def _wish_direction(self, keys: set[str]) -> Vec3:
        forward = self.forward()
        right = self.right()
        forward_input = float("w" in keys) - float("s" in keys)
        side_input = float("d" in keys) - float("a" in keys)
        wish = forward * forward_input + right * side_input
        return wish.normalized()

    def _apply_friction(self, delta: float) -> None:
        speed = self.horizontal_speed()
        if speed < 1e-4:
            return
        control = max(speed, STOP_SPEED)
        new_speed = max(0.0, speed - control * GROUND_FRICTION * delta)
        scale = new_speed / speed
        self.velocity.x *= scale
        self.velocity.z *= scale

    def _accelerate(
        self,
        wish_dir: Vec3,
        wish_speed: float,
        acceleration: float,
        delta: float,
    ) -> None:
        current = self.velocity.x * wish_dir.x + self.velocity.z * wish_dir.z
        add_speed = wish_speed - current
        if add_speed <= 0.0:
            return
        accel_speed = acceleration * wish_speed * delta
        if accel_speed > add_speed:
            accel_speed = add_speed
        self.velocity.x += wish_dir.x * accel_speed
        self.velocity.z += wish_dir.z * accel_speed

    def _air_accelerate(
        self,
        wish_dir: Vec3,
        wish_speed: float,
        acceleration: float,
        delta: float,
    ) -> None:
        """Half-Life / GoldSrc ``PM_AirAccelerate``.

        The velocity is projected onto the wish direction, but the wish
        speed used for that test is capped (30 of 320 units in GoldSrc).
        The full, uncapped wish speed is still used for the acceleration
        amount. Because a running player's forward speed is already far
        past the cap, adding W does nothing in the air: only the lateral
        part of the wish direction can still push the velocity. Turning
        the view while holding A/D keeps that lateral part across the
        velocity, so every jump adds a little more speed instead of the
        player drifting sideways as if on ice.
        """
        wish_cap = self.speed * AIR_WISH_SPEED_RATIO
        current = self.velocity.x * wish_dir.x + self.velocity.z * wish_dir.z
        add_speed = min(wish_speed, wish_cap) - current
        if add_speed <= 0.0:
            return
        accel_speed = acceleration * wish_speed * delta
        if accel_speed > add_speed:
            accel_speed = add_speed
        self.velocity.x += wish_dir.x * accel_speed
        self.velocity.z += wish_dir.z * accel_speed
