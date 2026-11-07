"""A small immediate-mode 2D canvas for drawing UI over the 3D frame.

Everything is expressed in screen pixels with the origin at the top-left
corner, exactly like a pygame surface. The canvas turns each item into two
textured triangles and draws the whole batch with an orthographic projection,
so sprites, panels and text all live in the same GPU pass instead of being
composited into a software surface.

OpenGL 3.3 core no longer has the old fixed-function ``glOrtho`` call; the
equivalent matrix is built by :func:`engine.math3d.mat4_ortho` and uploaded to
the shader.
"""

from __future__ import annotations

import array
import ctypes
from pathlib import Path

import pygame
from OpenGL import GL

from .math3d import Mat4, mat4_ortho

#: position(2) + uv(2) + rgba(4)
_FLOATS_PER_VERTEX = 8
_VERTEX_STRIDE = _FLOATS_PER_VERTEX * 4

_ANCHORS = {
    "topleft": (0.0, 0.0),
    "topright": (1.0, 0.0),
    "bottomleft": (0.0, 1.0),
    "bottomright": (1.0, 1.0),
    "center": (0.5, 0.5),
    "midtop": (0.5, 0.0),
    "midbottom": (0.5, 1.0),
    "midleft": (0.0, 0.5),
    "midright": (1.0, 0.5),
}

_UI_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec2 a_position;
layout (location = 1) in vec2 a_uv;
layout (location = 2) in vec4 a_color;

uniform mat4 u_projection;

out vec2 v_uv;
out vec4 v_color;

void main() {
    v_uv = a_uv;
    v_color = a_color;
    gl_Position = u_projection * vec4(a_position, 0.0, 1.0);
}
"""

_UI_FRAGMENT_SHADER = """#version 330 core
in vec2 v_uv;
in vec4 v_color;

uniform sampler2D u_texture;

out vec4 frag_color;

void main() {
    frag_color = texture(u_texture, v_uv) * v_color;
}
"""


class _Texture:
    """A GPU texture together with its pixel dimensions."""

    __slots__ = ("id", "width", "height")

    def __init__(self, texture_id: int, width: int, height: int) -> None:
        self.id = texture_id
        self.width = width
        self.height = height


def _to_rgba(color: tuple[int, ...], alpha: int | None = None) -> tuple[float, ...]:
    if len(color) == 3:
        red, green, blue = color
        old_alpha = 255
    else:
        red, green, blue, old_alpha = color[:4]
    if alpha is None:
        alpha = old_alpha
    return (
        red / 255.0,
        green / 255.0,
        blue / 255.0,
        alpha / 255.0,
    )


class UICanvas:
    """Immediate-mode 2D layer drawn over the scene with an ortho projection."""

    def __init__(self) -> None:
        self._ready = False
        self._program = 0
        self._projection_uniform = -1
        self._texture_uniform = -1
        self._vao = 0
        self._vbo = 0
        self._white: _Texture | None = None

        # Pygame surfaces stay cached across GL context rebuilds; only the
        # GPU handles are discarded by reset().
        self._surfaces: dict[tuple, pygame.Surface] = {}
        self._textures: dict[tuple, _Texture] = {}
        self._commands: list[tuple[_Texture, array.array]] = []

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Forget GPU handles after the window and its context are rebuilt."""
        self._ready = False
        self._program = 0
        self._vao = 0
        self._vbo = 0
        self._white = None
        self._textures.clear()
        self._commands.clear()

    def close(self) -> None:
        if not self._ready:
            return
        try:
            for texture in self._textures.values():
                GL.glDeleteTextures([texture.id])
            if self._white is not None:
                GL.glDeleteTextures([self._white.id])
            GL.glDeleteVertexArrays(1, [self._vao])
            GL.glDeleteBuffers(1, [self._vbo])
            GL.glDeleteProgram(self._program)
        except Exception:
            pass
        self._surfaces.clear()
        self.reset()

    def _ensure_gl(self) -> None:
        if self._ready:
            return

        self._program = self._build_program(
            _UI_VERTEX_SHADER, _UI_FRAGMENT_SHADER
        )
        self._projection_uniform = GL.glGetUniformLocation(
            self._program, "u_projection"
        )
        self._texture_uniform = GL.glGetUniformLocation(
            self._program, "u_texture"
        )

        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 2, GL.GL_FLOAT, GL.GL_FALSE, _VERTEX_STRIDE, None
        )
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, _VERTEX_STRIDE,
            ctypes.c_void_p(2 * 4),
        )
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(
            2, 4, GL.GL_FLOAT, GL.GL_FALSE, _VERTEX_STRIDE,
            ctypes.c_void_p(4 * 4),
        )
        GL.glBindVertexArray(0)

        self._white = self._upload_surface(self._white_surface())
        self._ready = True

    @staticmethod
    def _white_surface() -> pygame.Surface:
        surface = pygame.Surface((1, 1), pygame.SRCALPHA)
        surface.fill((255, 255, 255, 255))
        return surface

    @staticmethod
    def _compile_shader(source: str, kind: int) -> int:
        shader = GL.glCreateShader(kind)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(shader)
            if isinstance(log, bytes):
                log = log.decode(errors="replace")
            raise RuntimeError(f"UI shader compilation failed:\n{log}")
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
            raise RuntimeError(f"UI shader link failed:\n{log}")
        return program

    # ------------------------------------------------------------------
    # Textures
    # ------------------------------------------------------------------
    def _upload_surface(self, surface: pygame.Surface) -> _Texture:
        width, height = surface.get_size()
        to_bytes = getattr(pygame.image, "tobytes", None)
        if to_bytes is not None:
            data = to_bytes(surface, "RGBA", False)
        else:  # pragma: no cover - older pygame
            data = pygame.image.tostring(surface, "RGBA", False)
        texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return _Texture(texture_id, width, height)

    def _sprite_surface(self, source: str | Path) -> pygame.Surface:
        key = ("sprite", str(Path(source)))
        surface = self._surfaces.get(key)
        if surface is None:
            surface = pygame.image.load(str(source))
            self._surfaces[key] = surface
        return surface

    def _text_surface(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, ...],
    ) -> pygame.Surface:
        key = ("text", id(font), text, color[:3])
        surface = self._surfaces.get(key)
        if surface is None:
            surface = font.render(text, True, color[:3])
            self._surfaces[key] = surface
        return surface

    def _texture_for(self, key: tuple, surface: pygame.Surface) -> _Texture:
        texture = self._textures.get(key)
        if texture is None:
            texture = self._upload_surface(surface)
            self._textures[key] = texture
        return texture

    # ------------------------------------------------------------------
    # Public drawing API
    # ------------------------------------------------------------------
    def begin(self) -> None:
        """Start a new UI frame."""
        self._commands.clear()

    def image_size(self, source: str | Path) -> tuple[int, int]:
        """Return the pixel size of a sprite without drawing it."""
        return self._sprite_surface(source).get_size()

    def text_size(
        self, text: str, font: pygame.font.Font
    ) -> tuple[int, int]:
        return font.size(text)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: tuple[int, ...] = (255, 255, 255, 255),
        alpha: int | None = None,
    ) -> None:
        """Draw a solid (optionally translucent) rectangle."""
        if self._white is None:
            self._ensure_gl()
        assert self._white is not None
        self._push_quad(
            self._white, x, y, width, height, _to_rgba(color, alpha)
        )

    def panel(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: tuple[int, ...] = (11, 17, 25, 235),
        border: tuple[int, ...] | None = None,
        border_width: float = 1.0,
    ) -> None:
        """Draw a filled panel and an optional four-sided border."""
        self.rect(x, y, width, height, fill)
        if border is None or border_width <= 0.0:
            return
        edge = border_width
        self.rect(x, y, width, edge, border)
        self.rect(x, y + height - edge, width, edge, border)
        self.rect(x, y, edge, height, border)
        self.rect(x + width - edge, y, edge, height, border)

    def sprite(
        self,
        source: str | Path,
        position: tuple[float, float],
        width: float | None = None,
        height: float | None = None,
        anchor: str = "topleft",
        tint: tuple[int, ...] = (255, 255, 255, 255),
    ) -> tuple[float, float, float, float]:
        """Draw an image file as a textured quad, preserving its aspect ratio."""
        surface = self._sprite_surface(source)
        source_width, source_height = surface.get_size()
        if width is None and height is None:
            width, height = float(source_width), float(source_height)
        elif width is None:
            assert height is not None
            width = height * source_width / source_height
        elif height is None:
            height = width * source_height / source_width

        texture = self._texture_for(("sprite", str(Path(source))), surface)
        return self._place_and_push(
            texture, position, width, height, anchor, _to_rgba(tint)
        )

    def text(
        self,
        text: str,
        position: tuple[float, float],
        font: pygame.font.Font,
        color: tuple[int, ...] = (219, 232, 245, 255),
        anchor: str = "topleft",
        alpha: int | None = None,
    ) -> tuple[float, float, float, float]:
        """Draw a line of text using any pygame font."""
        surface = self._text_surface(text, font, color)
        texture = self._texture_for(
            ("text", id(font), text, color[:3]), surface
        )
        width, height = surface.get_size()
        base_alpha = color[3] if len(color) > 3 else 255
        if alpha is not None:
            base_alpha = alpha
        return self._place_and_push(
            texture,
            position,
            float(width),
            float(height),
            anchor,
            _to_rgba((255, 255, 255, 255), base_alpha),
        )

    # ------------------------------------------------------------------
    # Geometry and flush
    # ------------------------------------------------------------------
    def _place_and_push(
        self,
        texture: _Texture,
        position: tuple[float, float],
        width: float,
        height: float,
        anchor: str,
        tint: tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        anchor_x, anchor_y = _ANCHORS.get(anchor, _ANCHORS["topleft"])
        x = position[0] - width * anchor_x
        y = position[1] - height * anchor_y
        self._push_quad(texture, x, y, width, height, tint)
        return (x, y, width, height)

    def _push_quad(
        self,
        texture: _Texture,
        x: float,
        y: float,
        width: float,
        height: float,
        tint: tuple[float, ...],
    ) -> None:
        if width == 0.0 or height == 0.0:
            return
        x0 = float(x)
        y0 = float(y)
        x1 = x0 + float(width)
        y1 = y0 + float(height)
        red, green, blue, alpha = tint
        self._commands.append(
            (
                texture,
                array.array(
                    "f",
                    (
                        x0, y0, 0.0, 0.0, red, green, blue, alpha,
                        x1, y0, 1.0, 0.0, red, green, blue, alpha,
                        x1, y1, 1.0, 1.0, red, green, blue, alpha,
                        x0, y0, 0.0, 0.0, red, green, blue, alpha,
                        x1, y1, 1.0, 1.0, red, green, blue, alpha,
                        x0, y1, 0.0, 1.0, red, green, blue, alpha,
                    ),
                ),
            )
        )

    def render(self, width: int, height: int) -> None:
        """Flush the queued items into the current framebuffer."""
        if width < 2 or height < 2:
            self._commands.clear()
            return
        if not self._commands:
            return
        self._ensure_gl()

        GL.glViewport(0, 0, width, height)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glUseProgram(self._program)

        projection = mat4_ortho(0.0, float(width), float(height), 0.0, -1.0, 1.0)
        self._set_uniform_matrix(self._projection_uniform, projection)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glUniform1i(self._texture_uniform, 0)
        GL.glBindVertexArray(self._vao)

        for texture, vertices in self._batches():
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture.id)
            self._upload_vertices(vertices)
            GL.glDrawArrays(
                GL.GL_TRIANGLES, 0, len(vertices) // _FLOATS_PER_VERTEX
            )

        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self._commands.clear()

    def _batches(self) -> list[tuple[_Texture, array.array]]:
        batches: list[tuple[_Texture, array.array]] = []
        for texture, vertices in self._commands:
            if batches and batches[-1][0] is texture:
                batches[-1][1].extend(vertices)
            else:
                batches.append((texture, array.array("f", vertices)))
        return batches

    def _upload_vertices(self, vertices: array.array) -> None:
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        upload = (ctypes.c_float * len(vertices))(*vertices)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            ctypes.sizeof(upload),
            upload,
            GL.GL_DYNAMIC_DRAW,
        )

    @staticmethod
    def _set_uniform_matrix(location: int, matrix: Mat4) -> None:
        values = (ctypes.c_float * 16)(*matrix)
        GL.glUniformMatrix4fv(location, 1, GL.GL_TRUE, values)
