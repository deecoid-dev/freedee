"""A tiny 3D engine for Python, rendered with OpenGL.

The engine keeps a small dependency-free math layer (vectors, matrices,
projection), stores meshes as indexed triangles and draws them with a
PyOpenGL 3.3 core pipeline: vertex buffers, GLSL shaders, MSAA and a real
depth buffer.
"""

import os

# Keep pygame's import banner out of the console.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

__version__ = "0.2.0"
