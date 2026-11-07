"""OpenGL renderer: GPU pipeline with a real depth buffer.

Replaces the old Tk-canvas software renderer. Every object is uploaded once
as a vertex buffer and drawn with a small GLSL program, so there is no
per-frame triangle sorting, no painter's algorithm and no near-plane
projection artefacts: the GPU resolves visibility with a 24-bit depth buffer.
"""

from __future__ import annotations

import array
import ctypes
import logging
import math
from enum import Enum

import pygame

# PyOpenGL logs a warning when its optional numpy handler is missing; the
# engine only uses ctypes arrays, so silence that noisy import message.
logging.getLogger("OpenGL.plugins").setLevel(logging.ERROR)

from OpenGL import GL

from .math3d import Mat4, mat4_look_at, mat4_mul, mat4_perspective
from .scene import Camera, Object3D, Scene

BACKGROUND = (0.027, 0.039, 0.063)
"""Clear colour (RGB, 0..1) matching the old #070a10."""

_FLOATS_PER_VERTEX = 7  # position(3) + normal(3) + triangle id(1)


class RenderMode(Enum):
    """Display modes cycled with F7."""

    SHADED = "Shaded"
    SHADED_WIREFRAME = "Shaded + Wireframe"
    WIREFRAME = "Wireframe"
    TRIANGLES = "Triangles"
    NORMALS = "Normals"

    def next(self) -> "RenderMode":
        members = list(type(self))
        return members[(members.index(self) + 1) % len(members)]


_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in float a_triangle;

uniform mat4 u_model;
uniform mat4 u_view_projection;
uniform mat3 u_normal_matrix;

out vec3 v_local_position;
out vec3 v_local_normal;
out vec3 v_normal;
out float v_triangle;

void main() {
    v_local_position = a_position;
    v_local_normal = a_normal;
    v_normal = normalize(u_normal_matrix * a_normal);
    v_triangle = a_triangle;
    gl_Position = u_view_projection * u_model * vec4(a_position, 1.0);
}
"""

_FRAGMENT_SHADER = """#version 330 core
in vec3 v_local_position;
in vec3 v_local_normal;
in vec3 v_normal;
in float v_triangle;

uniform int u_mode;              // 0 shaded, 2 triangles, 3 normals
uniform float u_time;
uniform vec3 u_light;
uniform float u_hue;
uniform float u_saturation;
uniform float u_iridescent;
uniform float u_triangle_count;

out vec4 frag_color;

vec3 hsv_to_rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

void main() {
    vec3 normal = normalize(v_normal);
    float diffuse = max(dot(normal, normalize(u_light)), 0.0);
    vec3 color;

    if (u_mode == 0) {
        float hue;
        float saturation;
        float value;
        if (u_iridescent > 0.5) {
            vec3 n = v_local_normal;
            vec3 p = v_local_position;
            hue = 0.5 + u_time * 0.10
                + 0.10 * n.x + 0.16 * n.y + 0.22 * n.z
                + 0.055 * p.x + 0.045 * p.y + 0.055 * p.z;
            saturation = 0.85;
            value = 0.38 + 0.62 * diffuse;
        } else {
            hue = u_hue
                + 0.022 * v_local_normal.x
                + 0.014 * v_local_normal.z
                + 0.012 * v_local_position.y;
            saturation = u_saturation;
            value = 0.32 + 0.58 * diffuse;
        }
        color = hsv_to_rgb(vec3(fract(hue), saturation, value));
    } else if (u_mode == 2) {
        float hue = fract(
            v_triangle / max(u_triangle_count, 1.0) + u_time * 0.10 + 0.02
        );
        color = hsv_to_rgb(vec3(hue, 0.78, 0.42 + 0.55 * diffuse));
    } else if (u_mode == 3) {
        color = (normal * 0.5 + 0.5) * (0.55 + 0.45 * diffuse);
    } else {
        color = vec3(1.0, 0.0, 1.0);
    }

    frag_color = vec4(color, 1.0);
}
"""

_LINE_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec3 a_position;

uniform mat4 u_model;
uniform mat4 u_view_projection;

void main() {
    gl_Position = u_view_projection * u_model * vec4(a_position, 1.0);
}
"""

_LINE_FRAGMENT_SHADER = """#version 330 core
uniform vec3 u_color;
out vec4 frag_color;

void main() {
    frag_color = vec4(u_color, 1.0);
}
"""

_OVERLAY_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec2 a_position;
layout (location = 1) in vec2 a_uv;

out vec2 v_uv;

void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

_OVERLAY_FRAGMENT_SHADER = """#version 330 core
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 frag_color;

void main() {
    frag_color = texture(u_texture, v_uv);
}
"""


def _hex_rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


class GLRenderer:
    """Draws a :class:`Scene` with OpenGL 3.3 core."""

    def __init__(self) -> None:
        self.mode = RenderMode.SHADED
        self.stats: dict[str, int] = {
            "base_tris": 0,
            "drawn_tris": 0,
            "vertices": 0,
            "object_edges": 0,
            "edges": 0,
            "draw_calls": 0,
        }
        self.info = "OpenGL"
        self._ready = False
        self._program = 0
        self._line_program = 0
        self._overlay_program = 0
        self._uniforms: dict[str, int] = {}
        self._line_uniforms: dict[str, int] = {}
        self._overlay_uniforms: dict[str, int] = {}
        self._mesh_cache: dict[int, dict] = {}
        self._overlay = {
            "texture": 0,
            "vao": 0,
            "vbo": 0,
            "size": (0, 0),
        }
        self._light = (-0.45, 0.8, 0.55)

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Forget GPU handles after the window (and its context) is rebuilt."""
        self._ready = False
        self._mesh_cache.clear()
        self._overlay = {"texture": 0, "vao": 0, "vbo": 0, "size": (0, 0)}
        self._program = 0
        self._line_program = 0
        self._overlay_program = 0

    def close(self) -> None:
        if not self._ready:
            return
        try:
            for entry in self._mesh_cache.values():
                GL.glDeleteVertexArrays(1, [entry["vao"]])
                GL.glDeleteBuffers(1, [entry["vbo"]])
                if entry["edge_vao"]:
                    GL.glDeleteVertexArrays(1, [entry["edge_vao"]])
                    GL.glDeleteBuffers(1, [entry["edge_vbo"]])
            GL.glDeleteProgram(self._program)
            GL.glDeleteProgram(self._line_program)
            GL.glDeleteProgram(self._overlay_program)
            if self._overlay["texture"]:
                GL.glDeleteTextures([self._overlay["texture"]])
        except Exception:
            pass
        self._ready = False

    def _ensure_gl(self) -> None:
        if self._ready:
            return

        self._program = self._build_program(
            _VERTEX_SHADER, _FRAGMENT_SHADER
        )
        self._line_program = self._build_program(
            _LINE_VERTEX_SHADER, _LINE_FRAGMENT_SHADER
        )
        self._overlay_program = self._build_program(
            _OVERLAY_VERTEX_SHADER, _OVERLAY_FRAGMENT_SHADER
        )
        self._uniforms = {
            name: GL.glGetUniformLocation(self._program, name)
            for name in (
                "u_model", "u_view_projection", "u_normal_matrix",
                "u_mode", "u_time", "u_light", "u_hue", "u_saturation",
                "u_iridescent", "u_triangle_count",
            )
        }
        self._line_uniforms = {
            name: GL.glGetUniformLocation(self._line_program, name)
            for name in ("u_model", "u_view_projection", "u_color")
        }
        self._overlay_uniforms = {
            "u_texture": GL.glGetUniformLocation(
                self._overlay_program, "u_texture"
            )
        }
        self._mesh_cache.clear()
        self._create_overlay_buffers()

        version = GL.glGetString(GL.GL_VERSION)
        renderer = GL.glGetString(GL.GL_RENDERER)
        version = version.decode() if isinstance(version, bytes) else str(version)
        renderer = renderer.decode() if isinstance(renderer, bytes) else str(renderer)
        self.info = f"OpenGL {version.split()[0]} · {renderer}"
        self._ready = True

    @staticmethod
    def _compile_shader(source: str, kind: int) -> int:
        shader = GL.glCreateShader(kind)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(shader)
            if isinstance(log, bytes):
                log = log.decode(errors="replace")
            raise RuntimeError(f"Shader compilation failed:\n{log}")
        return shader

    def _build_program(self, vertex: str, fragment: str) -> int:
        vs = self._compile_shader(vertex, GL.GL_VERTEX_SHADER)
        fs = self._compile_shader(fragment, GL.GL_FRAGMENT_SHADER)
        program = GL.glCreateProgram()
        GL.glAttachShader(program, vs)
        GL.glAttachShader(program, fs)
        GL.glLinkProgram(program)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(program)
            if isinstance(log, bytes):
                log = log.decode(errors="replace")
            raise RuntimeError(f"Shader link failed:\n{log}")
        return program

    # ------------------------------------------------------------------
    # Mesh upload
    # ------------------------------------------------------------------
    def _mesh_entry(self, obj: Object3D) -> dict:
        key = id(obj.mesh)
        entry = self._mesh_cache.get(key)
        if entry is not None:
            return entry

        mesh = obj.mesh
        normals = mesh.face_normals()
        values = array.array("f")
        for triangle_index, (a, b, c) in enumerate(mesh.triangles):
            normal = normals[triangle_index]
            for vertex_index in (a, b, c):
                vertex = mesh.vertices[vertex_index]
                values.extend(
                    (
                        vertex.x, vertex.y, vertex.z,
                        normal.x, normal.y, normal.z,
                        float(triangle_index),
                    )
                )

        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        upload = (ctypes.c_float * len(values))(*values)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, ctypes.sizeof(upload), upload, GL.GL_STATIC_DRAW
        )
        stride = _FLOATS_PER_VERTEX * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, None
        )
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
            ctypes.c_void_p(3 * 4),
        )
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(
            2, 1, GL.GL_FLOAT, GL.GL_FALSE, stride,
            ctypes.c_void_p(6 * 4),
        )
        GL.glBindVertexArray(0)

        edge_vao = 0
        edge_vbo = 0
        edge_count = 0
        edges = mesh.edge_faces()
        if edges:
            line_values = array.array("f")
            for a, b, _faces in edges:
                va = mesh.vertices[a]
                vb = mesh.vertices[b]
                line_values.extend((va.x, va.y, va.z, vb.x, vb.y, vb.z))
            edge_count = len(edges) * 2

            edge_vao = GL.glGenVertexArrays(1)
            edge_vbo = GL.glGenBuffers(1)
            GL.glBindVertexArray(edge_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, edge_vbo)
            upload_lines = (ctypes.c_float * len(line_values))(*line_values)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, ctypes.sizeof(upload_lines), upload_lines,
                GL.GL_STATIC_DRAW,
            )
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(
                0, 3, GL.GL_FLOAT, GL.GL_FALSE, 3 * 4, None
            )
            GL.glBindVertexArray(0)

        entry = {
            "vao": vao,
            "vbo": vbo,
            "vertex_count": len(mesh.triangles) * 3,
            "triangle_count": len(mesh.triangles),
            "vertex_total": len(mesh.vertices),
            "edge_vao": edge_vao,
            "edge_vbo": edge_vbo,
            "edge_count": edge_count,
            "edge_total": len(edges),
        }
        self._mesh_cache[key] = entry
        return entry

    # ------------------------------------------------------------------
    # Public rendering
    # ------------------------------------------------------------------
    def render(
        self,
        scene: Scene,
        camera: Camera,
        width: int,
        height: int,
        time: float,
    ) -> None:
        if width < 2 or height < 2:
            return
        self._ensure_gl()

        self.stats = {
            "base_tris": 0,
            "drawn_tris": 0,
            "vertices": 0,
            "object_edges": 0,
            "edges": 0,
            "draw_calls": 0,
        }

        GL.glViewport(0, 0, width, height)
        GL.glClearColor(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glLineWidth(1.4)

        projection = mat4_perspective(
            math.radians(camera.fov), width / height, 0.05, 500.0
        )
        view = mat4_look_at(camera.position, camera.target, camera.up)
        view_projection = mat4_mul(projection, view)

        mode = self.mode

        if mode == RenderMode.WIREFRAME:
            for obj in scene.objects:
                self._account(obj)
            self._draw_depth_prepass(scene, view_projection)
            self._draw_edges(scene, view_projection, "#7fe9ff")
            return

        # Filled geometry first. Polygon offset keeps a small gap between a
        # face and its outline so the wireframe never z-fights.
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)
        GL.glUseProgram(self._program)
        self._set_uniform_matrix(
            self._uniforms["u_view_projection"], view_projection
        )
        GL.glUniform3f(
            self._uniforms["u_light"],
            self._light[0], self._light[1], self._light[2],
        )
        GL.glUniform1f(self._uniforms["u_time"], time)

        if mode == RenderMode.TRIANGLES:
            shader_mode = 2
        elif mode == RenderMode.NORMALS:
            shader_mode = 3
        else:
            # Shaded and shaded + wireframe both use the lit material shader.
            shader_mode = 0
        GL.glUniform1i(self._uniforms["u_mode"], shader_mode)

        for obj in scene.objects:
            self._draw_faces(obj, view_projection, time)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glUseProgram(0)

        if mode == RenderMode.SHADED_WIREFRAME:
            self._draw_edges(scene, view_projection, "#0b1017")
        elif mode in (RenderMode.TRIANGLES, RenderMode.NORMALS):
            self._draw_edges(scene, view_projection, "#05080d")

    # ------------------------------------------------------------------
    # Passes
    # ------------------------------------------------------------------
    def _account(self, obj: Object3D) -> dict:
        entry = self._mesh_entry(obj)
        self.stats["base_tris"] += entry["triangle_count"]
        self.stats["vertices"] += entry["vertex_total"]
        self.stats["object_edges"] += entry["edge_total"]
        return entry

    def _draw_faces(
        self,
        obj: Object3D,
        view_projection: Mat4,
        time: float,
    ) -> None:
        entry = self._account(obj)
        model = obj.model_matrix()
        self._set_uniform_matrix(self._uniforms["u_model"], model)
        self._set_uniform_normal_matrix(
            self._uniforms["u_normal_matrix"], model
        )
        iridescent = 1.0 if obj.hue is None else 0.0
        GL.glUniform1f(self._uniforms["u_hue"], obj.hue if obj.hue is not None else 0.5)
        GL.glUniform1f(self._uniforms["u_saturation"], obj.saturation)
        GL.glUniform1f(self._uniforms["u_iridescent"], iridescent)
        GL.glUniform1f(
            self._uniforms["u_triangle_count"],
            float(max(1, entry["triangle_count"])),
        )

        GL.glBindVertexArray(entry["vao"])
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, entry["vertex_count"])
        GL.glBindVertexArray(0)
        self.stats["drawn_tris"] += entry["triangle_count"]
        self.stats["draw_calls"] += 1

    def _draw_depth_prepass(
        self, scene: Scene, view_projection: Mat4
    ) -> None:
        """Fill the depth buffer only, so wireframe hides occluded edges."""
        GL.glUseProgram(self._line_program)
        self._set_uniform_matrix(
            self._line_uniforms["u_view_projection"], view_projection
        )
        GL.glUniform3f(self._line_uniforms["u_color"], 0.0, 0.0, 0.0)
        GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)
        for obj in scene.objects:
            entry = self._mesh_entry(obj)
            self._set_uniform_matrix(
                self._line_uniforms["u_model"], obj.model_matrix()
            )
            GL.glBindVertexArray(entry["vao"])
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, entry["vertex_count"])
            GL.glBindVertexArray(0)
            self.stats["draw_calls"] += 1
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glUseProgram(0)

    def _draw_edges(
        self,
        scene: Scene,
        view_projection: Mat4,
        color: str,
    ) -> None:
        red, green, blue = _hex_rgb(color)
        GL.glUseProgram(self._line_program)
        self._set_uniform_matrix(
            self._line_uniforms["u_view_projection"], view_projection
        )
        GL.glUniform3f(
            self._line_uniforms["u_color"], red, green, blue
        )
        GL.glLineWidth(1.6 if self.mode == RenderMode.WIREFRAME else 1.0)

        for obj in scene.objects:
            entry = self._mesh_entry(obj)
            if not entry["edge_count"]:
                continue
            self._set_uniform_matrix(
                self._line_uniforms["u_model"], obj.model_matrix()
            )
            GL.glBindVertexArray(entry["edge_vao"])
            GL.glDrawArrays(GL.GL_LINES, 0, entry["edge_count"])
            GL.glBindVertexArray(0)
            self.stats["edges"] += entry["edge_count"] // 2
            self.stats["draw_calls"] += 1
        GL.glUseProgram(0)

    # ------------------------------------------------------------------
    # UI overlay
    # ------------------------------------------------------------------
    def _create_overlay_buffers(self) -> None:
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        # Two triangles in clip space; v grows downwards so pygame surfaces
        # (origin top-left) upload without an extra flip.
        data = array.array(
            "f",
            (
                -1.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 0.0,
                1.0, -1.0, 1.0, 1.0,
                -1.0, 1.0, 0.0, 0.0,
                1.0, -1.0, 1.0, 1.0,
                -1.0, -1.0, 0.0, 1.0,
            ),
        )
        upload_quad = (ctypes.c_float * len(data))(*data)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, ctypes.sizeof(upload_quad), upload_quad,
            GL.GL_STATIC_DRAW,
        )
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4)
        )
        GL.glBindVertexArray(0)

        texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self._overlay = {
            "texture": texture,
            "vao": vao,
            "vbo": vbo,
            "size": (0, 0),
        }

    def draw_overlay(self, surface: pygame.Surface) -> None:
        """Upload a pygame UI surface and blend it over the 3D frame."""
        if not self._ready:
            return
        width, height = surface.get_size()
        if width < 2 or height < 2:
            return
        to_bytes = getattr(pygame.image, "tobytes", None)
        if to_bytes is not None:
            data = to_bytes(surface, "RGBA", False)
        else:  # pragma: no cover - older pygame
            data = pygame.image.tostring(surface, "RGBA", False)

        overlay = self._overlay
        GL.glBindTexture(GL.GL_TEXTURE_2D, overlay["texture"])
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if overlay["size"] != (width, height):
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data,
            )
            overlay["size"] = (width, height)
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0, width, height,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data,
            )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glUseProgram(self._overlay_program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, overlay["texture"])
        GL.glUniform1i(self._overlay_uniforms["u_texture"], 0)
        GL.glBindVertexArray(overlay["vao"])
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)
        GL.glEnable(GL.GL_DEPTH_TEST)

    # ------------------------------------------------------------------
    # Uniform helpers (our matrices are row-major, hence transpose=True)
    # ------------------------------------------------------------------
    @staticmethod
    def _set_uniform_matrix(location: int, matrix: Mat4) -> None:
        values = (ctypes.c_float * 16)(*matrix)
        GL.glUniformMatrix4fv(location, 1, GL.GL_TRUE, values)

    @staticmethod
    def _set_uniform_normal_matrix(location: int, model: Mat4) -> None:
        values = (ctypes.c_float * 9)(
            model[0], model[1], model[2],
            model[4], model[5], model[6],
            model[8], model[9], model[10],
        )
        GL.glUniformMatrix3fv(location, 1, GL.GL_TRUE, values)
