"""Entry point: launch the FreeDee engine and its main menu.

Run with::

    python main.py
"""

from __future__ import annotations

import sys

try:
    from engine.app import Engine
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
    Engine(Scene()).run()


if __name__ == "__main__":
    main()
