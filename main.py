"""Demo entry point: a walkable test room with an iridescent cube.

Run with::

    python main.py
"""

from __future__ import annotations

import sys

try:
    from engine.app import Engine
    from engine.math3d import Vec3
    from engine.room import add_landmark_cube, build_test_room
    from engine.scene import Scene
except ModuleNotFoundError as exc:  # pragma: no cover - friendly setup path
    if exc.name in {"OpenGL", "pygame"}:
        sys.exit(
            "Missing dependency: "
            f"{exc.name}.\nInstall the renderer requirements with:\n"
            "    python -m pip install -r requirements.txt"
        )
    raise


def main() -> None:
    scene = Scene()
    build_test_room(scene)
    cube = add_landmark_cube(scene)

    def animate(_scene: Scene, delta: float, _time: float) -> None:
        cube.rotation.y += 0.85 * delta
        cube.rotation.x += 0.33 * delta

    Engine(scene, update=animate).run()


if __name__ == "__main__":
    main()
