"""Procedural shapes available out of the box."""

from __future__ import annotations

from .math3d import Vec3
from .mesh import Mesh

# Unit-cube corners (size 1, centred on the origin). Multiply by size.
_CUBE_VERTICES = (
    (-1.0, -1.0, -1.0),  # 0
    (1.0, -1.0, -1.0),   # 1
    (1.0, 1.0, -1.0),    # 2
    (-1.0, 1.0, -1.0),   # 3
    (-1.0, -1.0, 1.0),   # 4
    (1.0, -1.0, 1.0),    # 5
    (1.0, 1.0, 1.0),     # 6
    (-1.0, 1.0, 1.0),    # 7
)

# Two triangles per face, wound counter-clockwise when seen from outside.
_CUBE_TRIANGLES = (
    (4, 5, 6), (4, 6, 7),   # +Z
    (1, 0, 3), (1, 3, 2),   # -Z
    (5, 1, 2), (5, 2, 6),   # +X
    (0, 4, 7), (0, 7, 3),   # -X
    (7, 6, 2), (7, 2, 3),   # +Y
    (0, 1, 5), (0, 5, 4),   # -Y
)


def create_box(
    width: float,
    height: float,
    depth: float,
    name: str = "box",
) -> Mesh:
    """Build a box centred on the origin with the given extents."""
    half_x = width * 0.5
    half_y = height * 0.5
    half_z = depth * 0.5
    mesh = Mesh(name)
    for x, y, z in _CUBE_VERTICES:
        mesh.add_vertex(Vec3(x * half_x, y * half_y, z * half_z))
    for a, b, c in _CUBE_TRIANGLES:
        mesh.add_triangle(a, b, c)
    return mesh


def create_cube(size: float = 2.0, name: str = "cube") -> Mesh:
    """Build a cube with half-extent ``size / 2``."""
    return create_box(size, size, size, name)
