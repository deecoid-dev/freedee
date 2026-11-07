"""Triangle mesh storage and edge helpers."""

from __future__ import annotations

from .math3d import Vec3


class Mesh:
    """An indexed triangle mesh.

    Vertices are stored in object space. Face normals and a triangle-edge
    adjacency table are derived lazily on first use.
    """

    def __init__(self, name: str = "mesh") -> None:
        self.name = name
        self.vertices: list[Vec3] = []
        self.triangles: list[tuple[int, int, int]] = []
        self._normals: list[Vec3] = []
        self._edges: list[tuple[int, int, tuple[int, ...]]] | None = None

    def add_vertex(self, vertex: Vec3) -> int:
        self.vertices.append(vertex)
        return len(self.vertices) - 1

    def add_triangle(self, a: int, b: int, c: int) -> None:
        self.triangles.append((a, b, c))
        self._normals.clear()
        self._edges = None

    def face_normals(self) -> list[Vec3]:
        """Unit normals for every base triangle (counter-clockwise winding)."""
        if len(self._normals) != len(self.triangles):
            normals: list[Vec3] = []
            for a, b, c in self.triangles:
                p0 = self.vertices[a]
                p1 = self.vertices[b]
                p2 = self.vertices[c]
                normals.append((p1 - p0).cross(p2 - p0).normalized())
            self._normals = normals
        return self._normals

    def edge_faces(self) -> list[tuple[int, int, tuple[int, ...]]]:
        """Unique base edges with the indices of the triangles using them."""
        if self._edges is None:
            order: list[tuple[int, int]] = []
            faces: dict[tuple[int, int], list[int]] = {}
            for triangle_index, triangle in enumerate(self.triangles):
                for a, b in (
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                ):
                    key = (a, b) if a < b else (b, a)
                    if key not in faces:
                        faces[key] = []
                        order.append(key)
                    faces[key].append(triangle_index)
            self._edges = [
                (a, b, tuple(faces[(a, b)])) for a, b in order
            ]
        return self._edges

