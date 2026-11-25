"""Verbose console diagnostics for tracking down engine crashes.

The F3 overlay only shows the current state.  This tracer prints a compact
state line to stdout every couple of simulation ticks, keeps a rolling history
of recent frames and input events, and dumps that history plus a detailed
snapshot when the main loop raises.  ``faulthandler`` is enabled as well, so a
native crash (for example inside a graphics driver) still prints a Python
traceback.
"""

from __future__ import annotations

import faulthandler
import math
import platform
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import pygame

from . import __version__

if TYPE_CHECKING:
    from .app import Engine

LOG_FILE_NAME = "freedee_debug.log"
"""Debugger output is mirrored here, next to this module."""

LOG_FILE = Path(__file__).resolve().parent / LOG_FILE_NAME

REPORT_EVERY_TICKS = 2
"""Print one state line every N completed frames."""

HISTORY_TICKS = 240
"""How many recent report lines are kept for the crash dump."""

CRASH_HISTORY_TICKS = 30
EVENT_HISTORY = 40
CRASH_EVENT_HISTORY = 15

_KEY_ORDER = (
    "w", "a", "s", "d", "q", "e", "r", "c",
    "space", "shift", "ctrl", "up", "down", "left", "right",
)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute without letting an incomplete object hide a crash."""
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _one_line(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value).split())
    if not text:
        return "-"
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _fmt_vec(vector: Any, digits: int = 2) -> str:
    if vector is None:
        return "-"
    try:
        return (
            f"({vector.x:+.{digits}f},{vector.y:+.{digits}f},"
            f"{vector.z:+.{digits}f})"
        )
    except Exception:
        return str(vector)


def _fmt_yaw(value: Any) -> str:
    try:
        return f"{math.degrees(float(value)) % 360.0:5.1f}deg"
    except (TypeError, ValueError):
        return "-"


def _fmt_pitch(value: Any) -> str:
    try:
        return f"{math.degrees(float(value)):+6.1f}deg"
    except (TypeError, ValueError):
        return "-"


def _fmt_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pygame_version() -> str:
    version = _get(pygame.version, "ver")
    if not version:
        version = _get(pygame, "__version__", "?")
    name = "pygame-ce" if _get(pygame, "IS_CE", False) else "pygame"
    sdl = _get(pygame.version, "SDL")
    if sdl:
        return f"{name} {version} (SDL {sdl})"
    return f"{name} {version}"


class ConsoleDebugger:
    """Periodic stdout diagnostics and a rolling crash report."""

    def __init__(
        self,
        interval: int = REPORT_EVERY_TICKS,
        *,
        enabled: bool = True,
        log_path: str | Path | None = None,
    ) -> None:
        self.interval = max(1, int(interval))
        self.enabled = enabled
        self.stage = "init"
        self.frame = 0

        self.log_path = Path(log_path) if log_path is not None else LOG_FILE
        self._log_file: TextIO | None = None
        self._log_error = ""

        self._started = time.perf_counter()
        self._history: deque[str] = deque(maxlen=HISTORY_TICKS)
        self._events: deque[str] = deque(maxlen=EVENT_HISTORY)

    # ------------------------------------------------------------------
    # Log file
    # ------------------------------------------------------------------
    def _open_log(self) -> None:
        """Create a fresh log next to the engine; never stop the game."""
        self._close_log()
        self._log_error = ""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open(
                "w",
                encoding="utf-8",
                errors="replace",
                buffering=1,
            )
        except OSError as exc:
            self._log_file = None
            self._log_error = str(exc)

    def _close_log(self) -> None:
        if self._log_file is None:
            return
        try:
            self._log_file.close()
        except Exception:
            pass
        self._log_file = None

    def _write_log(self, text: str) -> None:
        """Write raw text to the log file, disabling it if the disk fails."""
        if self._log_file is None:
            return
        try:
            self._log_file.write(text)
            self._log_file.flush()
        except Exception:
            self._close_log()

    def _emit(self, message: str = "") -> None:
        """Write one line to both stdout and the log file."""
        try:
            print(message, flush=True)
        except Exception:
            pass
        self._write_log(message + "\n")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, engine: "Engine") -> None:
        """Open the log, enable the crash handler and print the banner."""
        self._open_log()
        try:
            faulthandler.enable(
                file=self._log_file if self._log_file is not None else sys.stderr
            )
        except Exception:
            pass

        self._started = time.perf_counter()
        settings = _get(engine, "settings")
        settings_path = _get(settings, "path", "?")
        self._emit("=" * 78)
        self._emit(
            f"[freedee] engine {__version__} | "
            f"Python {platform.python_version()} "
            f"({platform.architecture()[0]}) | {_pygame_version()}"
        )
        self._emit(f"[freedee] cwd={Path.cwd()} | settings={settings_path}")
        if self._log_file is not None:
            self._emit(
                f"[freedee] log file: {self.log_path} "
                f"(recreated on every launch)"
            )
        else:
            self._emit(
                f"[freedee] log file unavailable: {self.log_path} "
                f"({self._log_error or 'unknown error'})"
            )
        self._emit(
            f"[freedee] console debugger "
            f"{'ON' if self.enabled else 'OFF'}: one state line every "
            f"{self.interval} ticks; F2 toggles, F3 is the in-game overlay"
        )
        self._emit(
            f"[freedee] crash history: last {HISTORY_TICKS} ticks "
            f"and {EVENT_HISTORY} input events"
        )
        self._emit("=" * 78)

    def stop(self) -> None:
        elapsed = time.perf_counter() - self._started
        self._emit(
            f"[freedee] stopped after {self.frame} ticks "
            f"({elapsed:.2f}s)"
        )
        try:
            faulthandler.disable()
        except Exception:
            pass
        self._close_log()

    def toggle(self) -> bool:
        """Flip periodic reporting and tell the user what happened."""
        self.enabled = not self.enabled
        self._emit(
            f"[freedee] console debugger "
            f"{'ON' if self.enabled else 'OFF'}"
            + (
                f" (one state line every {self.interval} ticks)"
                if self.enabled
                else ""
            )
        )
        return self.enabled

    # ------------------------------------------------------------------
    # Per-frame tracing
    # ------------------------------------------------------------------
    def record_event(self, event: pygame.event.Event) -> None:
        """Keep a short history of input and window events for the crash dump."""
        try:
            try:
                name = pygame.event.event_name(event.type)
            except Exception:
                name = str(event.type)

            parts: list[str] = []
            key = getattr(event, "key", None)
            if key is not None:
                try:
                    key_name = pygame.key.name(key)
                except Exception:
                    key_name = str(key)
                parts.append(f"key={key_name}")
            button = getattr(event, "button", None)
            if button is not None:
                parts.append(f"button={button}")
            position = getattr(event, "pos", None)
            if position is not None:
                parts.append(f"pos={position}")
            relative = getattr(event, "rel", None)
            if relative is not None:
                parts.append(f"rel={relative}")
            wheel_x = getattr(event, "x", None)
            wheel_y = getattr(event, "y", None)
            if event.type == pygame.MOUSEWHEEL:
                parts.append(f"wheel=({wheel_x},{wheel_y})")
            mods = getattr(event, "mod", None)
            if mods:
                parts.append(f"mod=0x{mods:x}")

            detail = (" " + " ".join(parts)) if parts else ""
            self._events.append(f"{self.frame:06d} {name}{detail}")
        except Exception:
            # Tracing must never be the reason the engine stops.
            pass

    def tick(self, engine: "Engine", delta: float) -> None:
        """Capture the current state and print it every ``interval`` ticks."""
        self.frame += 1
        try:
            snapshot = self.capture(engine, delta)
            line = self.format_line(snapshot)
            self._history.append(line)
            if self.enabled and self.frame % self.interval == 0:
                self._emit(line)
        except Exception:
            self._emit(traceback.format_exc().rstrip())

    # ------------------------------------------------------------------
    # Snapshot and formatting
    # ------------------------------------------------------------------
    def capture(self, engine: "Engine", delta: float) -> dict[str, Any]:
        scene = _get(engine, "scene")
        player = _get(engine, "player")
        camera = _get(engine, "camera")
        renderer = _get(engine, "renderer")
        collision_world = _get(engine, "collision_world")
        state = _get(engine, "state")
        stats = dict(_get(renderer, "stats", {}) or {})

        try:
            speed = float(player.horizontal_speed())
        except Exception:
            speed = 0.0
        try:
            width, height = engine.screen.get_size()
        except Exception:
            width = height = 0
        try:
            focused = bool(pygame.key.get_focused())
        except Exception:
            focused = False

        keys = _get(engine, "_keys", set()) or set()
        ordered_keys = [name for name in _KEY_ORDER if name in keys]
        ordered_keys.extend(
            sorted(str(name) for name in keys if name not in _KEY_ORDER)
        )

        objects = _get(scene, "objects", []) or []
        colliders = _get(scene, "colliders", []) or []
        world_colliders = _get(collision_world, "colliders", []) or []

        return {
            "frame": self.frame,
            "elapsed": time.perf_counter() - self._started,
            "stage": self.stage,
            "delta": _fmt_float(delta),
            "fps": _fmt_float(_get(engine, "fps")),
            "frame_ms": _fmt_float(_get(engine, "frame_ms")),
            "sim_time": _fmt_float(_get(engine, "time")),
            "state": getattr(state, "name", str(state)),
            "world": _one_line(
                _get(engine, "world_title", "")
                or _get(scene, "title", "")
                or "-"
            ),
            "objects": len(objects),
            "colliders": len(colliders),
            "world_colliders": len(world_colliders),
            "spawn": _fmt_vec(_get(scene, "spawn")),
            "player_pos": _fmt_vec(_get(player, "position")),
            "player_vel": _fmt_vec(_get(player, "velocity")),
            "player_spawn": _fmt_vec(_get(player, "spawn")),
            "speed": speed,
            "ground": bool(_get(player, "on_ground", False)),
            "air": _fmt_float(_get(player, "air_time")),
            "jump": _fmt_float(_get(player, "jump_buffer")),
            "player_yaw": _fmt_yaw(_get(player, "yaw")),
            "player_pitch": _fmt_pitch(_get(player, "pitch")),
            "camera_pos": _fmt_vec(_get(camera, "position")),
            "camera_yaw": _fmt_yaw(_get(camera, "yaw")),
            "camera_pitch": _fmt_pitch(_get(camera, "pitch")),
            "fov": _fmt_float(_get(camera, "fov")),
            "keys": " ".join(ordered_keys) or "-",
            "mouse": (
                "captured"
                if _get(engine, "mouse_captured", False)
                else "released"
            ),
            "focus": focused,
            "walk": bool(_get(engine, "walk_mode", True)),
            "settings": bool(_get(engine, "settings_open", False)),
            "map_browser": bool(_get(engine, "map_browser_open", False)),
            "debug_overlay": bool(_get(engine, "debug_visible", False)),
            "menu_cursor": _get(engine, "menu_cursor", 0),
            "settings_tab": _get(engine, "settings_tab", 0),
            "settings_cursor": _get(engine, "settings_cursor", 0),
            "map_cursor": _get(engine, "map_cursor", 0),
            "map_count": len(_get(engine, "map_entries", []) or []),
            "render_mode": getattr(
                _get(engine, "render_mode"), "value", "?"
            ),
            "gpu": _one_line(
                str(_get(renderer, "info", "OpenGL")).replace("·", "|")
            ),
            "base_tris": stats.get("base_tris", 0),
            "drawn_tris": stats.get("drawn_tris", 0),
            "vertices": stats.get("vertices", 0),
            "edges": stats.get("edges", 0),
            "draw_calls": stats.get("draw_calls", 0),
            "window": f"{width}x{height}",
            "fullscreen": bool(_get(engine, "fullscreen", False)),
            "vsync": bool(_get(engine, "vsync", False)),
            "world_error": _one_line(_get(engine, "world_error", "")),
            "map_error": _one_line(_get(engine, "map_error", "")),
        }

    def format_line(self, snapshot: dict[str, Any]) -> str:
        """One greppable line with the frame, player, render and UI state."""
        return (
            f"[dbg {snapshot['frame']:06d}] "
            f"t={snapshot['elapsed']:8.2f}s "
            f"dt={snapshot['delta'] * 1000.0:6.2f}ms "
            f"fps={snapshot['fps']:5.1f} "
            f"frame={snapshot['frame_ms']:6.2f}ms "
            f"state={snapshot['state']} stage={snapshot['stage']} | "
            f"world='{snapshot['world']}' "
            f"objs={snapshot['objects']} "
            f"cols={snapshot['colliders']}/{snapshot['world_colliders']} "
            f"spawn={snapshot['spawn']} sim={snapshot['sim_time']:7.2f}s | "
            f"player pos={snapshot['player_pos']} "
            f"vel={snapshot['player_vel']} "
            f"spd={snapshot['speed']:5.1f} "
            f"ground={'yes' if snapshot['ground'] else 'air'} "
            f"air={snapshot['air']:4.2f} jump={snapshot['jump']:4.2f} "
            f"yaw={snapshot['player_yaw']} "
            f"pitch={snapshot['player_pitch']} | "
            f"camera pos={snapshot['camera_pos']} "
            f"yaw={snapshot['camera_yaw']} "
            f"pitch={snapshot['camera_pitch']} "
            f"fov={snapshot['fov']:.0f} | "
            f"input keys={snapshot['keys']} "
            f"mouse={snapshot['mouse']} "
            f"focus={'yes' if snapshot['focus'] else 'no'} "
            f"walk={'yes' if snapshot['walk'] else 'no'} | "
            f"ui settings={'yes' if snapshot['settings'] else 'no'} "
            f"map={'yes' if snapshot['map_browser'] else 'no'} "
            f"debug={'yes' if snapshot['debug_overlay'] else 'no'} "
            f"menu={snapshot['menu_cursor']} | "
            f"render {snapshot['render_mode']} "
            f"gpu='{snapshot['gpu']}' "
            f"tris={snapshot['drawn_tris']}/{snapshot['base_tris']} "
            f"draws={snapshot['draw_calls']} "
            f"verts={snapshot['vertices']} "
            f"edges={snapshot['edges']} | "
            f"window {snapshot['window']} "
            f"{'fullscreen' if snapshot['fullscreen'] else 'windowed'} "
            f"vsync={'on' if snapshot['vsync'] else 'off'} | "
            f"errors world='{snapshot['world_error']}' "
            f"map='{snapshot['map_error']}'"
        )

    def format_detail(self, snapshot: dict[str, Any]) -> list[str]:
        """A multi-line snapshot used when the loop crashes."""
        return [
            (
                f"  frame    : {snapshot['frame']} | stage="
                f"{snapshot['stage']} | state={snapshot['state']} | "
                f"elapsed={snapshot['elapsed']:.3f}s"
            ),
            (
                f"  timing   : sim={snapshot['sim_time']:.3f}s "
                f"delta={snapshot['delta'] * 1000.0:.3f}ms "
                f"fps={snapshot['fps']:.2f} "
                f"frame={snapshot['frame_ms']:.2f}ms"
            ),
            (
                f"  window   : {snapshot['window']} "
                f"{'fullscreen' if snapshot['fullscreen'] else 'windowed'} "
                f"vsync={'on' if snapshot['vsync'] else 'off'}"
            ),
            (
                f"  world    : '{snapshot['world']}' "
                f"objects={snapshot['objects']} "
                f"colliders={snapshot['colliders']} "
                f"world_colliders={snapshot['world_colliders']} "
                f"spawn={snapshot['spawn']} "
                f"sim_time={snapshot['sim_time']:.3f}s"
            ),
            (
                f"  player   : pos={snapshot['player_pos']} "
                f"spawn={snapshot['player_spawn']} "
                f"vel={snapshot['player_vel']} "
                f"speed={snapshot['speed']:.3f} "
                f"ground={'yes' if snapshot['ground'] else 'air'} "
                f"air={snapshot['air']:.3f} "
                f"jump_buffer={snapshot['jump']:.3f} "
                f"yaw={snapshot['player_yaw']} "
                f"pitch={snapshot['player_pitch']}"
            ),
            (
                f"  camera   : pos={snapshot['camera_pos']} "
                f"yaw={snapshot['camera_yaw']} "
                f"pitch={snapshot['camera_pitch']} "
                f"fov={snapshot['fov']:.1f}"
            ),
            (
                f"  input    : keys={snapshot['keys']} "
                f"mouse={snapshot['mouse']} "
                f"focus={'yes' if snapshot['focus'] else 'no'} "
                f"walk={'yes' if snapshot['walk'] else 'no'}"
            ),
            (
                f"  ui       : settings={'yes' if snapshot['settings'] else 'no'} "
                f"map_browser={'yes' if snapshot['map_browser'] else 'no'} "
                f"debug_overlay={'yes' if snapshot['debug_overlay'] else 'no'} "
                f"menu_cursor={snapshot['menu_cursor']} "
                f"settings_tab={snapshot['settings_tab']} "
                f"settings_cursor={snapshot['settings_cursor']} "
                f"map_cursor={snapshot['map_cursor']}/{snapshot['map_count']}"
            ),
            (
                f"  renderer : mode={snapshot['render_mode']} "
                f"gpu='{snapshot['gpu']}' "
                f"tris={snapshot['drawn_tris']}/{snapshot['base_tris']} "
                f"draw_calls={snapshot['draw_calls']} "
                f"vertices={snapshot['vertices']} "
                f"edges={snapshot['edges']}"
            ),
            (
                f"  errors   : world='{snapshot['world_error']}' "
                f"map='{snapshot['map_error']}'"
            ),
        ]

    # ------------------------------------------------------------------
    # Crash report
    # ------------------------------------------------------------------
    def report_crash(
        self, engine: "Engine", delta: float, exc: BaseException
    ) -> None:
        """Dump the recent history and current state before re-raising."""
        try:
            snapshot = self.capture(engine, delta)
            self._emit()
            self._emit("!" * 78)
            self._emit(
                f"[freedee] CRASH at frame {self.frame} during "
                f"'{self.stage}': {type(exc).__name__}: {exc}"
            )
            self._emit("!" * 78)

            self._emit(
                f"[freedee] last {CRASH_EVENT_HISTORY} input events:"
            )
            events = list(self._events)[-CRASH_EVENT_HISTORY:]
            if events:
                for event in events:
                    self._emit(f"  {event}")
            else:
                self._emit("  (none)")

            self._emit(
                f"[freedee] last {CRASH_HISTORY_TICKS} completed ticks:"
            )
            ticks = list(self._history)[-CRASH_HISTORY_TICKS:]
            if ticks:
                for tick in ticks:
                    self._emit(f"  {tick}")
            else:
                self._emit("  (none)")

            self._emit("[freedee] state at crash:")
            for line in self.format_detail(snapshot):
                self._emit(line)
            self._emit("!" * 78)

            # Python prints the traceback to stderr after the exception
            # propagates; keep a copy in the log as well.
            self._write_log("[freedee] traceback:\n")
            self._write_log(
                "".join(
                    traceback.format_exception(
                        type(exc), exc, exc.__traceback__
                    )
                )
            )
        except Exception:
            # Never hide the original failure behind a debugger failure.
            self._emit(traceback.format_exc().rstrip())
