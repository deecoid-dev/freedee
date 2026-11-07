"""Minimal vector and 4x4 matrix math for the engine.

Matrices are stored as flat, row-major tuples of 16 floats and are applied
to column vectors: ``v' = M @ v``.
"""

from __future__ import annotations

import math

Mat4 = tuple[float, ...]


class Vec3:
    """A three-component vector."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self * scalar

    def __truediv__(self, scalar: float) -> "Vec3":
        inv = 1.0 / scalar
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> "Vec3":
        length = self.length()
        if length < 1e-12:
            return Vec3()
        return self / length

    def lerp(self, other: "Vec3", t: float) -> "Vec3":
        return Vec3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )


def mat4_identity() -> Mat4:
    return (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    """Return ``a @ b`` for two row-major matrices."""
    out = [0.0] * 16
    for row in range(4):
        ai = row * 4
        a0, a1, a2, a3 = a[ai], a[ai + 1], a[ai + 2], a[ai + 3]
        for col in range(4):
            out[ai + col] = (
                a0 * b[col]
                + a1 * b[4 + col]
                + a2 * b[8 + col]
                + a3 * b[12 + col]
            )
    return tuple(out)


def mat4_translation(x: float, y: float, z: float) -> Mat4:
    return (
        1.0, 0.0, 0.0, x,
        0.0, 1.0, 0.0, y,
        0.0, 0.0, 1.0, z,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_scale(sx: float, sy: float | None = None, sz: float | None = None) -> Mat4:
    sy = sx if sy is None else sy
    sz = sx if sz is None else sz
    return (
        sx, 0.0, 0.0, 0.0,
        0.0, sy, 0.0, 0.0,
        0.0, 0.0, sz, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_rotation_x(radians: float) -> Mat4:
    c, s = math.cos(radians), math.sin(radians)
    return (
        1.0, 0.0, 0.0, 0.0,
        0.0, c, -s, 0.0,
        0.0, s, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_rotation_y(radians: float) -> Mat4:
    c, s = math.cos(radians), math.sin(radians)
    return (
        c, 0.0, s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_rotation_z(radians: float) -> Mat4:
    c, s = math.cos(radians), math.sin(radians)
    return (
        c, -s, 0.0, 0.0,
        s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_perspective(fov_y: float, aspect: float, near: float, far: float) -> Mat4:
    """Right-handed perspective projection (camera looks down -Z)."""
    f = 1.0 / math.tan(fov_y * 0.5)
    return (
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far),
        0.0, 0.0, -1.0, 0.0,
    )


def mat4_look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    forward = (target - eye).normalized()
    side = forward.cross(up).normalized()
    true_up = side.cross(forward)
    return (
        side.x, side.y, side.z, -side.dot(eye),
        true_up.x, true_up.y, true_up.z, -true_up.dot(eye),
        -forward.x, -forward.y, -forward.z, forward.dot(eye),
        0.0, 0.0, 0.0, 1.0,
    )


def transform_point(m: Mat4, v: Vec3) -> tuple[float, float, float, float]:
    """Transform a point (w = 1) and return homogeneous ``(x, y, z, w)``."""
    x, y, z = v.x, v.y, v.z
    return (
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
        m[12] * x + m[13] * y + m[14] * z + m[15],
    )


def transform_vec3(m: Mat4, v: Vec3) -> Vec3:
    """Transform a position and drop the homogeneous component."""
    x, y, z = v.x, v.y, v.z
    return Vec3(
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
    )


def transform_direction(m: Mat4, v: Vec3) -> Vec3:
    """Transform a direction (w = 0)."""
    x, y, z = v.x, v.y, v.z
    return Vec3(
        m[0] * x + m[1] * y + m[2] * z,
        m[4] * x + m[5] * y + m[6] * z,
        m[8] * x + m[9] * y + m[10] * z,
    )
