"""Application shell: pygame window, OpenGL frame loop and all hotkeys."""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import pygame

from .canvas import UICanvas
from .collision import CollisionWorld
from .crosshair import Crosshair
from .debug import ConsoleDebugger
from .game_state import GameState
from .mapfile import MapInfo, ensure_maps_directory, list_maps, load_map_into
from .math3d import Vec3
from .player import Player, TURN_SPEED as PLAYER_TURN_SPEED
from .config import Settings
from .renderer import GLRenderer, RenderMode
from .room import build_test_room
from .scene import Camera, Scene
from .ui import UI

WINDOW_TITLE = "FreeDee engine"
TARGET_FPS = 60

CAT_SPRITE_PATH = (
    Path(__file__).resolve().parent.parent / "uisprites" / "cat.jpg"
)

CAMERA_MOVE_SPEED = 3.0
CAMERA_SPRINT_SPEED = 8.0
CAMERA_TURN_SPEED = 1.8
MOUSE_SENSITIVITY_SCALE = 0.001
DEFAULT_MOUSE_SENSITIVITY = 4.5
MAX_MOUSE_DELTA = 320
WHEEL_STEP = 0.55

DEFAULT_PLAYER_SPEED = 7.0

# Window sizes offered in the settings screen (filtered by the real display).
RESOLUTION_CANDIDATES = (
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1366, 768),
    (1440, 900),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
)

_GL_FLAGS = pygame.OPENGL | pygame.DOUBLEBUF

# Keyboard mapping for the gameplay layer.
_MOVEMENT_KEYS = {
    pygame.K_w: "w",
    pygame.K_a: "a",
    pygame.K_s: "s",
    pygame.K_d: "d",
    pygame.K_q: "q",
    pygame.K_e: "e",
    pygame.K_r: "r",
    pygame.K_c: "c",
    pygame.K_SPACE: "space",
    pygame.K_LSHIFT: "shift",
    pygame.K_RSHIFT: "shift",
    pygame.K_LCTRL: "ctrl",
    pygame.K_RCTRL: "ctrl",
    pygame.K_UP: "up",
    pygame.K_DOWN: "down",
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
}


def _color_component(value) -> int:
    """Clamp a stored RGB component to the 0-255 byte range."""
    try:
        return max(0, min(255, int(value)))
    except (TypeError, ValueError):
        return 255


def _enable_high_dpi() -> None:
    """Ask Windows to give us real pixels instead of a stretched bitmap."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class Engine:
    """Owns the window, renders the scene and dispatches input."""

    def __init__(
        self,
        scene: Scene,
        update=None,
        title: str = WINDOW_TITLE,
        width: int = 800,
        height: int = 600,
        *,
        scene_loader: Callable[[Scene], None] | None = None,
    ) -> None:
        self.scene = scene
        self.scene_loader = scene_loader
        self.update_callback = update
        self.title = title
        self.windowed_width = width
        self.windowed_height = height

        self.fullscreen = False
        self.vsync = False
        self.render_mode = RenderMode.SHADED
        self.debug_visible = False
        self.show_hud = True
        self.settings_open = False
        self.map_browser_open = False
        self.map_entries: list[MapInfo] = []
        self.map_cursor = 0
        self.map_scroll = 0
        self.map_error = ""
        self.world_error = ""
        self.world_title = ""
        self._map_last_click: tuple[int, float] | None = None
        self.state = GameState.MAIN_MENU
        self.menu_cursor = 0
        self.settings_tab = 0
        self.settings_cursor = 0
        self.settings_draft: dict = {}
        self._slider_drag: str | None = None
        self.walk_mode = True
        self.player_speed = DEFAULT_PLAYER_SPEED
        self.auto_bunnyhop = False
        self.mouse_sensitivity = DEFAULT_MOUSE_SENSITIVITY
        self.invert_y = False
        self.cat_visible = False
        self.crosshair = Crosshair()
        self.console_debug = ConsoleDebugger()

        self.toast_text = ""
        self.toast_timer = 0.0
        self.time = 0.0
        self.fps = 0.0
        self.frame_ms = 0.0
        self._frame_intervals: deque[float] = deque(maxlen=60)
        self._render_times: deque[float] = deque(maxlen=60)
        self._running = False
        self._keys: set[str] = set()
        self.mouse_captured = False
        self._captured_since = 0.0
        self._clock: pygame.time.Clock | None = None
        self._overlay_surface: pygame.Surface | None = None

        _enable_high_dpi()
        pygame.init()
        pygame.font.init()
        pygame.key.set_repeat(0)
        ensure_maps_directory()

        self.renderer = GLRenderer()
        self._init_window(width, height)

        self.camera = Camera()
        self.ui = UI()
        self.ui_canvas = UICanvas()
        self.resolutions = self._build_resolutions()
        self.resolution_index = 0

        self.collision_world = CollisionWorld(self.scene.colliders)
        self.player = Player(
            self.scene.spawn,
            self.scene.spawn_yaw,
            speed=self.player_speed,
            auto_bhop=self.auto_bunnyhop,
        )
        self.camera.position = self.player.eye_position()
        self.camera.yaw = self.player.yaw
        self.camera.pitch = self.player.pitch
        self.camera.sync_target()

        self._home_position = Vec3(
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z,
        )
        self._home_yaw = self.camera.yaw
        self._home_pitch = self.camera.pitch

        self.settings = Settings()
        self._apply_loaded_settings()

        # Open the window the way it was left: apply the saved resolution and
        # fullscreen mode once so a restarted engine opens identically.
        if self.fullscreen:
            self._recreate_display(
                self._desktop_size(), _GL_FLAGS | pygame.FULLSCREEN
            )
        else:
            width, height = self.resolutions[self.resolution_index]
            self.windowed_width = width
            self.windowed_height = height
            self._recreate_display((width, height), _GL_FLAGS)

        self.console_debug.start(self)

    def _apply_loaded_settings(self) -> None:
        settings = self.settings
        self.render_mode = list(RenderMode)[
            int(settings.get("render_mode", 0)) % len(list(RenderMode))
        ]
        self.fullscreen = not bool(settings.get("windowed", True))
        self.vsync = bool(settings.get("vsync", False))
        self.renderer.brightness = float(settings.get("brightness", 1.0))
        self.renderer.gamma = float(settings.get("gamma", 1.0))
        self.camera.fov = float(settings.get("fov", 50.0))
        self.mouse_sensitivity = float(
            settings.get("sensitivity", DEFAULT_MOUSE_SENSITIVITY)
        )
        self.invert_y = bool(settings.get("invert_y", False))
        self.player_speed = float(
            settings.get("player_speed", DEFAULT_PLAYER_SPEED)
        )
        self.player.speed = self.player_speed
        self.auto_bunnyhop = bool(settings.get("auto_bunnyhop", False))
        self.player.auto_bhop = self.auto_bunnyhop
        self.ui.set_theme(int(settings.get("menu_theme", 0)))
        self.ui.set_accent(int(settings.get("accent", 0)))
        self.ui.menu_opacity = float(settings.get("menu_opacity", 0.96))
        self.show_hud = bool(settings.get("show_hud", True))
        self.debug_visible = bool(settings.get("debug_overlay", False))
        self.crosshair.enabled = bool(settings.get("crosshair_enabled", True))
        self.crosshair.color_index = max(
            0,
            min(
                len(Crosshair.COLORS),
                int(settings.get("crosshair_color", 0)),
            ),
        )
        self.crosshair.custom_color = (
            _color_component(settings.get("crosshair_custom_r", 255)),
            _color_component(settings.get("crosshair_custom_g", 255)),
            _color_component(settings.get("crosshair_custom_b", 255)),
        )
        self.crosshair.size = int(settings.get("crosshair_size", 10))
        self.crosshair.gap = int(settings.get("crosshair_gap", 6))
        self.crosshair.thickness = int(settings.get("crosshair_thickness", 2))
        self.crosshair.dot = bool(settings.get("crosshair_dot", True))
        self.crosshair.dot_size = int(settings.get("crosshair_dot_size", 3))
        self.crosshair.outline = bool(settings.get("crosshair_outline", True))
        self.crosshair.outline_thickness = int(
            settings.get("crosshair_outline_width", 1)
        )
        self.crosshair.opacity = int(settings.get("crosshair_opacity", 100))

        self.resolution_index = int(
            settings.get("resolution", 0)
        ) % len(self.resolutions)

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------
    def _set_gl_attributes(
        self, version: tuple[int, int] | None, multisample: bool
    ) -> None:
        if version is not None:
            major, minor = version
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_MAJOR_VERSION, major
            )
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_MINOR_VERSION, minor
            )
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
            )
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(
            pygame.GL_MULTISAMPLEBUFFERS, 1 if multisample else 0
        )
        pygame.display.gl_set_attribute(
            pygame.GL_MULTISAMPLESAMPLES, 4 if multisample else 0
        )

    def _create_screen(
        self,
        size: tuple[int, int],
        flags: int,
        vsync: bool | None = None,
    ) -> pygame.Surface:
        # SDL will not resize an existing OpenGL drawable reliably, so the
        # display is torn down and rebuilt for every resolution change.
        if vsync is None:
            vsync = self.vsync
        attempts = (
            ((3, 3), True),
            ((3, 3), False),
            ((3, 2), False),
            (None, False),
        )
        last_error: Exception | None = None
        for version, multisample in attempts:
            pygame.display.quit()
            pygame.display.init()
            pygame.display.set_caption(self.title)
            try:
                self._set_gl_attributes(version, multisample)
                try:
                    return pygame.display.set_mode(
                        size, flags, vsync=1 if vsync else 0
                    )
                except Exception:
                    # Some drivers reject the vsync argument; the game still
                    # runs, only the option falls back to the driver default.
                    return pygame.display.set_mode(size, flags)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to create OpenGL window")

    def _init_window(self, width: int, height: int) -> None:
        self.screen = self._create_screen((width, height), _GL_FLAGS)
        pygame.display.set_caption(self.title)
        self._center_window((width, height))

    def _center_window(self, size: tuple[int, int]) -> None:
        desktop_width, desktop_height = self._desktop_size()
        x = max(0, (desktop_width - size[0]) // 2)
        y = max(0, (desktop_height - size[1]) // 2 - 24)
        try:
            pygame.display.set_window_position((x, y))
        except Exception:
            pass
    def _build_resolutions(self) -> list[tuple[int, int]]:
        try:
            screen_width, screen_height = pygame.display.get_desktop_sizes()[0]
        except Exception:
            screen_width, screen_height = 1366, 768
        max_width = screen_width - 16
        max_height = screen_height - 48
        sizes = [
            size
            for size in RESOLUTION_CANDIDATES
            if size[0] <= max_width and size[1] <= max_height
        ]
        if (800, 600) not in sizes:
            sizes.insert(0, (800, 600))
        return sizes

    def _desktop_size(self) -> tuple[int, int]:
        try:
            return pygame.display.get_desktop_sizes()[0]
        except Exception:
            return (1366, 768)

    def _recreate_display(
        self, size: tuple[int, int], flags: int
    ) -> None:
        was_captured = self.mouse_captured
        if was_captured:
            self._release_mouse_capture()
        self.screen = self._create_screen(size, flags)
        pygame.display.set_caption(self.title)
        if not (flags & pygame.FULLSCREEN):
            self._center_window(size)
        self._overlay_surface = None
        # The old GL context is gone with the window; all GPU handles must be
        # rebuilt for the new one.
        self.renderer.reset()
        self.ui_canvas.reset()
        if was_captured:
            self._capture_mouse()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._running = True
        self._clock = pygame.time.Clock()
        delta = 0.0
        try:
            while self._running:
                delta = self._clock.tick(TARGET_FPS) / 1000.0
                if delta <= 0.0:
                    delta = 1.0 / TARGET_FPS
                delta = min(delta, 0.1)

                self.console_debug.stage = "events"
                self._handle_events()
                if not self._running:
                    break

                self.console_debug.stage = "update"
                self._update(delta)
                self.console_debug.stage = "render"
                self._render_frame(delta)
                self.console_debug.stage = "frame"
                self.console_debug.tick(self, delta)
        except BaseException as exc:
            self.console_debug.report_crash(self, delta, exc)
            raise
        finally:
            self.console_debug.stop()
            self.close()

    def close(self) -> None:
        if not self._running and not pygame.get_init():
            return
        self._running = False
        if self.mouse_captured:
            self._release_mouse_capture()
        canvas = getattr(self, "ui_canvas", None)
        if canvas is not None:
            canvas.close()
        self.renderer.close()
        if pygame.get_init():
            pygame.quit()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            self.console_debug.record_event(event)
            if event.type == pygame.QUIT:
                self._running = False
                return

            if event.type == pygame.KEYDOWN:
                self._on_key_down(event)
            elif event.type == pygame.KEYUP:
                name = _MOVEMENT_KEYS.get(event.key)
                if name is not None:
                    self._keys.discard(name)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.settings_open:
                    self._settings_mouse_down(event.pos, event.button)
                elif self.map_browser_open:
                    self._map_browser_mouse_down(event.pos, event.button)
                elif self._menu_visible():
                    self._menu_mouse_down(event.pos, event.button)
                elif event.button == 1 and not self.mouse_captured:
                    self._capture_mouse()
            elif event.type == pygame.MOUSEBUTTONUP:
                if self.settings_open:
                    self._settings_mouse_up(event.pos, event.button)
            elif event.type == pygame.MOUSEMOTION:
                if self.settings_open:
                    self._settings_mouse_motion(event.pos)
                elif self.map_browser_open:
                    self._map_browser_mouse_motion(event.pos)
                elif self._menu_visible():
                    self._menu_mouse_motion(event.pos)
            elif event.type == pygame.MOUSEWHEEL:
                if self.settings_open:
                    self._settings_mouse_wheel(event.y)
                elif self.map_browser_open:
                    self._move_map_selection(-event.y)
                elif self._menu_visible():
                    items = self._menu_items()
                    if items:
                        self.menu_cursor = (
                            self.menu_cursor - event.y
                        ) % len(items)
                elif not self.walk_mode:
                    self.camera.position = (
                        self.camera.position
                        + self.camera.forward() * (event.y * WHEEL_STEP)
                    )
                    self.camera.sync_target()

            focus_lost = getattr(pygame, "WINDOWFOCUSLOST", -1)
            if event.type == focus_lost:
                self._keys.clear()
                self._release_mouse_capture()

    def _on_key_down(self, event: pygame.event.Event) -> None:
        key = event.key

        if key == pygame.K_F2:
            enabled = self.console_debug.toggle()
            self.toast_text = (
                "CONSOLE DEBUG ON" if enabled else "CONSOLE DEBUG OFF"
            )
            self.toast_timer = 1.6
            return

        if self.settings_open:
            if key == pygame.K_ESCAPE:
                if self.ui.open_dropdown is not None:
                    self.ui.open_dropdown = None
                else:
                    self._close_settings(apply=False)
            else:
                self._handle_settings_key(key, event.mod)
            return

        if self.map_browser_open:
            self._handle_map_browser_key(key)
            return

        if self._menu_visible():
            self._handle_menu_key(key)
            return

        if key == pygame.K_F3:
            self.debug_visible = not self.debug_visible
            return
        if key == pygame.K_F7:
            self._cycle_render_mode()
            return
        if key == pygame.K_F11:
            self._toggle_fullscreen()
            return
        if key == pygame.K_TAB:
            self._toggle_walk_mode()
            return
        if key == pygame.K_ESCAPE:
            self._open_pause_menu()
            return

        if key == pygame.K_m:
            self.cat_visible = not self.cat_visible
            self.toast_text = "CAT OVERLAY" if self.cat_visible else "CAT HIDDEN"
            self.toast_timer = 1.6
            return

        name = _MOVEMENT_KEYS.get(key)
        if name is None:
            return

        if name == "space" and self.walk_mode and name not in self._keys:
            self.player.queue_jump()
        if name == "r" and self.walk_mode and name not in self._keys:
            self._respawn_player()
        if name == "c" and name not in self._keys:
            if self.walk_mode:
                self._reset_player_view()
            else:
                self._reset_camera()
        self._keys.add(name)

    # ------------------------------------------------------------------
    # Front-end menu and pause menu
    # ------------------------------------------------------------------
    @property
    def in_game(self) -> bool:
        """True while a world is loaded (playing or paused)."""
        return self.state is not GameState.MAIN_MENU

    @property
    def pause_open(self) -> bool:
        return self.state is GameState.PAUSED

    def _set_state(self, state: GameState) -> bool:
        """Move the engine through its three top-level states.

        All state-dependent side effects live here: the world is built when
        entering ``PLAYING`` from the main menu, torn down when returning to
        ``MAIN_MENU``, and the mouse is only captured while playing.  A loader
        failure leaves the engine in the previous state and returns ``False``.
        """
        if state is self.state:
            return True
        previous = self.state
        self.state = state
        self._keys.clear()
        self.ui.open_dropdown = None
        self.ui.end_drag()
        self.player.jump_buffer = 0.0

        if state is GameState.MAIN_MENU:
            self._release_mouse_capture()
            self._unload_world()
        elif state is GameState.PAUSED:
            self._release_mouse_capture()
        elif state is GameState.PLAYING:
            if previous is GameState.MAIN_MENU:
                try:
                    self._load_world()
                except Exception as exc:
                    self.world_error = str(exc)
                    self.state = previous
                    self.scene.clear()
                    self.collision_world = CollisionWorld(self.scene.colliders)
                    self.player.respawn(self.scene.spawn, self.scene.spawn_yaw)
                    self.world_title = ""
                    self.toast_text = ""
                    self.toast_timer = 0.0
                    print(f"[world] failed to load: {exc}")
                    return False
            self._capture_mouse()
        return True

    def _load_world(self) -> None:
        """Build the playable world from the scene loader."""
        self.renderer.release_scene()
        if self.scene_loader is not None:
            self.scene.clear()
            self.scene_loader(self.scene)
        self.world_title = self.scene.title
        self.collision_world = CollisionWorld(self.scene.colliders)
        self.player.respawn(self.scene.spawn, self.scene.spawn_yaw)
        self._sync_camera_to_player()
        self._home_position = Vec3(
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z,
        )
        self._home_yaw = self.camera.yaw
        self._home_pitch = self.camera.pitch
        self.time = 0.0
        self._frame_intervals.clear()
        self._render_times.clear()
        self.fps = 0.0
        self.frame_ms = 0.0

    def _unload_world(self) -> None:
        """Drop the world and every GPU resource that belonged to it."""
        self.renderer.release_scene()
        self.world_title = ""
        if self.scene_loader is None:
            return
        self.scene.clear()
        self.collision_world = CollisionWorld(self.scene.colliders)
        self.player.respawn(self.scene.spawn, self.scene.spawn_yaw)
        self.time = 0.0

    def _menu_visible(self) -> bool:
        return self.state is not GameState.PLAYING

    def _menu_items(self) -> list[tuple[str, str]]:
        if self.state is GameState.PAUSED:
            return [
                ("disconnect", "Disconnect"),
                ("settings", "Settings"),
                ("exit", "Exit"),
            ]
        return [
            ("new_game", "New game"),
            ("test_room", "Test room"),
            ("settings", "Settings"),
            ("exit", "Exit"),
        ]

    def _menu_state(self) -> dict:
        return {
            "items": self._menu_items(),
            "cursor": self.menu_cursor,
            "pause": self.state is GameState.PAUSED,
            "dim": not self.settings_open,
        }

    def _menu_layout(self) -> dict:
        width, height = self.screen.get_size()
        return self.ui.menu_layout(width, height, self._menu_items())

    def _handle_menu_key(self, key: int) -> None:
        items = self._menu_items()
        if not items:
            return
        if key in (pygame.K_UP, pygame.K_w):
            self.menu_cursor = (self.menu_cursor - 1) % len(items)
            return
        if key in (pygame.K_DOWN, pygame.K_s):
            self.menu_cursor = (self.menu_cursor + 1) % len(items)
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._activate_menu_item(
                items[self.menu_cursor % len(items)][0]
            )
            return
        if key == pygame.K_ESCAPE and self.state is GameState.PAUSED:
            self._resume_game()

    def _menu_mouse_down(
        self, position: tuple[int, int], button: int
    ) -> None:
        if button != 1:
            return
        for index, (item_id, rect) in enumerate(
            self._menu_layout()["buttons"]
        ):
            if rect.collidepoint(position):
                self.menu_cursor = index
                self._activate_menu_item(item_id)
                return

    def _menu_mouse_motion(self, position: tuple[int, int]) -> None:
        for index, (_item_id, rect) in enumerate(
            self._menu_layout()["buttons"]
        ):
            if rect.collidepoint(position):
                self.menu_cursor = index
                return

    def _activate_menu_item(self, item_id: str) -> None:
        if item_id == "new_game":
            self._open_map_browser()
        elif item_id == "test_room":
            self._start_test_room()
        elif item_id == "disconnect":
            self._return_to_main_menu()
        elif item_id == "settings":
            self._open_settings()
        elif item_id == "exit":
            self._running = False

    def _start_game(self) -> bool:
        self.menu_cursor = 0
        self.walk_mode = True
        self.toast_text = ""
        self.toast_timer = 0.0
        self.map_browser_open = False
        self._map_last_click = None
        return self._set_state(GameState.PLAYING)

    def _start_test_room(self) -> None:
        """Play the small hard-coded room instead of a ``.map`` file."""
        self.scene_loader = build_test_room
        self.world_error = ""
        self.map_error = ""
        self._start_game()

    def _open_map_browser(self) -> None:
        """Show the New Game window with every map from ``engine/maps``."""
        if self.settings_open:
            return
        self.map_error = ""
        self._refresh_maps()
        self.map_browser_open = True
        self.menu_cursor = 0
        self._map_last_click = None
        self._keys.clear()
        self.player.jump_buffer = 0.0
        self._release_mouse_capture()

    def _close_map_browser(self) -> None:
        self.map_browser_open = False
        self.map_error = ""
        self._map_last_click = None
        self._keys.clear()

    # ------------------------------------------------------------------
    # New Game map browser: state, layout and input
    # ------------------------------------------------------------------
    def _map_browser_state(self) -> dict:
        return {
            "title": "New game",
            "subtitle": "Maps in engine/maps",
            "entries": [
                {"name": entry.name, "detail": entry.detail}
                for entry in self.map_entries
            ],
            "cursor": self.map_cursor,
            "scroll": self.map_scroll,
            "error": self.map_error,
        }

    def _map_browser_layout(self) -> dict:
        width, height = self.screen.get_size()
        return self.ui.map_browser_layout(
            width, height, self._map_browser_state()
        )

    def _refresh_maps(self, preserve: Path | None = None) -> None:
        """Rescan the maps folder while keeping the selected file if possible."""
        if preserve is None and self.map_entries:
            preserve = self.map_entries[
                self.map_cursor % len(self.map_entries)
            ].path
        self.map_entries = list_maps()
        self.map_cursor = 0
        self.map_scroll = 0
        if not self.map_entries:
            return
        if preserve is not None:
            for index, entry in enumerate(self.map_entries):
                if entry.path == preserve:
                    self.map_cursor = index
                    break
        layout = self._map_browser_layout()
        visible = max(1, layout["visible_rows"])
        if self.map_cursor >= visible:
            self.map_scroll = min(
                self.map_cursor - visible + 1, layout["max_scroll"]
            )

    def _move_map_selection(self, amount: int) -> None:
        if not self.map_entries:
            return
        layout = self._map_browser_layout()
        visible = max(1, layout["visible_rows"])
        self.map_cursor = (
            self.map_cursor + amount
        ) % len(self.map_entries)
        if self.map_cursor < self.map_scroll:
            self.map_scroll = self.map_cursor
        elif self.map_cursor >= self.map_scroll + visible:
            self.map_scroll = self.map_cursor - visible + 1
        self.map_scroll = max(
            0, min(layout["max_scroll"], self.map_scroll)
        )

    def _launch_selected_map(self) -> None:
        if not self.map_entries:
            return
        entry = self.map_entries[
            self.map_cursor % len(self.map_entries)
        ]
        path = entry.path

        def loader(scene: Scene, path: Path = path) -> None:
            load_map_into(scene, path)

        self.scene_loader = loader
        self.map_error = ""
        self.world_error = ""
        if self._start_game():
            return
        self.map_error = self.world_error or "Could not load the map"
        self.map_browser_open = True
        self._refresh_maps(preserve=path)

    def _handle_map_browser_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self._close_map_browser()
            return
        if key in (pygame.K_UP, pygame.K_w):
            self._move_map_selection(-1)
            return
        if key in (pygame.K_DOWN, pygame.K_s):
            self._move_map_selection(1)
            return
        if key == pygame.K_PAGEUP:
            layout = self._map_browser_layout()
            self._move_map_selection(-layout["visible_rows"])
            return
        if key == pygame.K_PAGEDOWN:
            layout = self._map_browser_layout()
            self._move_map_selection(layout["visible_rows"])
            return
        if key == pygame.K_HOME and self.map_entries:
            self.map_cursor = 0
            self.map_scroll = 0
            return
        if key == pygame.K_END and self.map_entries:
            self.map_cursor = len(self.map_entries) - 1
            layout = self._map_browser_layout()
            visible = max(1, layout["visible_rows"])
            self.map_scroll = min(
                layout["max_scroll"], max(0, self.map_cursor - visible + 1)
            )
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._launch_selected_map()
            return
        if key == pygame.K_r:
            self.map_error = ""
            self._refresh_maps()

    def _map_browser_mouse_down(
        self, position: tuple[int, int], button: int
    ) -> None:
        if button != 1:
            return
        layout = self._map_browser_layout()

        if layout["close"].collidepoint(position):
            self._close_map_browser()
            return

        buttons = layout["buttons"]
        if buttons["cancel"].collidepoint(position):
            self._close_map_browser()
            return
        if buttons["refresh"].collidepoint(position):
            self.map_error = ""
            self._refresh_maps()
            return
        if buttons["launch"].collidepoint(position):
            self._launch_selected_map()
            return

        for index, row in layout["rows"]:
            if not row.collidepoint(position):
                continue
            now = time.perf_counter()
            previous = self._map_last_click
            self.map_cursor = index
            self._map_last_click = (index, now)
            if (
                previous is not None
                and previous[0] == index
                and now - previous[1] <= 0.4
            ):
                self._map_last_click = None
                self._launch_selected_map()
            return

    def _map_browser_mouse_motion(self, position: tuple[int, int]) -> None:
        for index, row in self._map_browser_layout()["rows"]:
            if row.collidepoint(position):
                self.map_cursor = index
                return

    def _return_to_main_menu(self) -> None:
        self.menu_cursor = 0
        self.toast_text = ""
        self.toast_timer = 0.0
        self._set_state(GameState.MAIN_MENU)

    def _open_pause_menu(self) -> None:
        if self.state is not GameState.PLAYING:
            return
        self.menu_cursor = 0
        self._set_state(GameState.PAUSED)

    def _resume_game(self) -> None:
        if self.state is not GameState.PAUSED:
            return
        self.menu_cursor = 0
        self._set_state(GameState.PLAYING)

    # ------------------------------------------------------------------
    # Options window: state and layout
    # ------------------------------------------------------------------
    def _settings_state(self) -> dict:
        draft = self.settings_draft
        return {
            "tab": self.settings_tab,
            "cursor": self.settings_cursor,
            "values": draft,
            "options": {
                "render_mode": [mode.value for mode in RenderMode],
                "resolution": [
                    f"{width} x {height}"
                    for width, height in self.resolutions
                ],
                "menu_theme": list(self.ui.theme_names),
                "accent": list(self.ui.accent_names),
                "crosshair_color": [
                    name for name, _color in Crosshair.COLORS
                ] + ["Custom"],
            },
            "crosshair": self._draft_crosshair(),
            "window": (self.windowed_width, self.windowed_height),
            "screen": self._desktop_size(),
        }

    def _settings_layout(self) -> dict:
        width, height = self.screen.get_size()
        return self.ui.settings_layout(
            width, height, self._settings_state()
        )

    def _draft_crosshair(self) -> Crosshair:
        draft = self.settings_draft
        crosshair = Crosshair()
        crosshair.enabled = bool(draft.get("crosshair_enabled", True))
        crosshair.color_index = max(
            0,
            min(
                len(Crosshair.COLORS),
                int(draft.get("crosshair_color", 0)),
            ),
        )
        crosshair.custom_color = (
            _color_component(draft.get("crosshair_custom_r", 255)),
            _color_component(draft.get("crosshair_custom_g", 255)),
            _color_component(draft.get("crosshair_custom_b", 255)),
        )
        crosshair.size = int(draft.get("crosshair_size", 10))
        crosshair.gap = int(draft.get("crosshair_gap", 6))
        crosshair.thickness = int(draft.get("crosshair_thickness", 2))
        crosshair.dot = bool(draft.get("crosshair_dot", True))
        crosshair.dot_size = int(draft.get("crosshair_dot_size", 3))
        crosshair.outline = bool(draft.get("crosshair_outline", True))
        crosshair.outline_thickness = int(
            draft.get("crosshair_outline_width", 1)
        )
        crosshair.opacity = int(draft.get("crosshair_opacity", 100))
        return crosshair

    def _focus_settings_widget(self, widget_id: str, layout: dict) -> None:
        try:
            self.settings_cursor = layout["order"].index(widget_id)
        except ValueError:
            pass

    def _set_settings_tab(self, index: int) -> None:
        self.settings_tab = index % len(UI.TABS)
        self.settings_cursor = 0
        self.ui.open_dropdown = None
        self._slider_drag = None

    def _cycle_settings_tab(self, direction: int) -> None:
        self._set_settings_tab(self.settings_tab + direction)

    # ------------------------------------------------------------------
    # Options window: keyboard
    # ------------------------------------------------------------------
    def _handle_settings_key(self, key: int, mods: int = 0) -> None:
        layout = self._settings_layout()

        dropdown_id = self.ui.open_dropdown
        if dropdown_id is not None:
            widget = layout["widgets"].get(dropdown_id)
            if widget is not None and widget["kind"] == "dropdown":
                if key in (pygame.K_UP, pygame.K_w, pygame.K_LEFT, pygame.K_a):
                    self._cycle_dropdown(dropdown_id, -1)
                    return
                if key in (
                    pygame.K_DOWN, pygame.K_s,
                    pygame.K_RIGHT, pygame.K_d,
                ):
                    self._cycle_dropdown(dropdown_id, 1)
                    return
                if key in (
                    pygame.K_RETURN, pygame.K_KP_ENTER,
                    pygame.K_SPACE, pygame.K_ESCAPE,
                ):
                    self.ui.open_dropdown = None
                    return
            else:
                self.ui.open_dropdown = None

        if key == pygame.K_TAB:
            direction = -1 if mods & pygame.KMOD_SHIFT else 1
            self._cycle_settings_tab(direction)
            return

        order = layout["order"]
        if not order:
            return
        if key in (pygame.K_UP, pygame.K_w):
            self.settings_cursor = (
                self.settings_cursor - 1
            ) % len(order)
            return
        if key in (pygame.K_DOWN, pygame.K_s):
            self.settings_cursor = (
                self.settings_cursor + 1
            ) % len(order)
            return
        if key in (pygame.K_LEFT, pygame.K_a):
            self._step_focused(-1, layout)
            return
        if key in (pygame.K_RIGHT, pygame.K_d):
            self._step_focused(1, layout)
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._activate_focused(layout)

    def _step_focused(self, direction: int, layout: dict) -> None:
        order = layout["order"]
        if not order:
            return
        widget_id = order[self.settings_cursor % len(order)]
        widget = layout["widgets"][widget_id]
        kind = widget["kind"]
        if kind == "checkbox":
            self.settings_draft[widget_id] = not bool(
                self.settings_draft.get(widget_id, False)
            )
        elif kind == "dropdown":
            self._cycle_dropdown(widget_id, direction)
        elif kind == "slider":
            current = float(
                self.settings_draft.get(widget_id, widget["min"])
            )
            self._set_slider_value(
                widget_id, widget, current + direction * float(widget["step"])
            )

    def _activate_focused(self, layout: dict) -> None:
        order = layout["order"]
        if not order:
            return
        widget_id = order[self.settings_cursor % len(order)]
        widget = layout["widgets"][widget_id]
        kind = widget["kind"]
        if kind == "checkbox":
            self.settings_draft[widget_id] = not bool(
                self.settings_draft.get(widget_id, False)
            )
        elif kind == "dropdown":
            self.ui.open_dropdown = widget_id
        elif kind == "button":
            self._settings_button_action(widget_id)

    def _cycle_dropdown(self, widget_id: str, direction: int) -> None:
        layout = self._settings_layout()
        widget = layout["widgets"].get(widget_id)
        if widget is None or widget["kind"] != "dropdown":
            return
        count = len(widget["options"])
        if count <= 0:
            return
        current = int(self.settings_draft.get(widget_id, 0))
        self.settings_draft[widget_id] = (current + direction) % count

    # ------------------------------------------------------------------
    # Options window: mouse
    # ------------------------------------------------------------------
    def _settings_mouse_down(
        self, position: tuple[int, int], button: int
    ) -> None:
        if button == 3:
            self.ui.open_dropdown = None
            return
        if button != 1:
            return

        layout = self._settings_layout()
        open_id = self.ui.open_dropdown
        if open_id is not None:
            dropdown = layout.get("dropdown")
            if dropdown is not None and dropdown["id"] == open_id:
                for index, rect in enumerate(dropdown["rects"]):
                    if rect.collidepoint(position):
                        self.settings_draft[open_id] = index
                        self.ui.open_dropdown = None
                        self._focus_settings_widget(open_id, layout)
                        return
                field = layout["widgets"][open_id]["field"]
                if field.collidepoint(position):
                    self.ui.open_dropdown = None
                    return
            self.ui.open_dropdown = None

        if layout["close"].collidepoint(position):
            self._close_settings(apply=False)
            return
        if layout["title"].collidepoint(position):
            self.ui.start_drag(position)
            return

        for index, (_tab_id, _label, rect) in enumerate(layout["tabs"]):
            if rect.collidepoint(position):
                self._set_settings_tab(index)
                return

        for widget_id in layout["order"]:
            widget = layout["widgets"][widget_id]
            if not widget["rect"].collidepoint(position):
                continue
            self._focus_settings_widget(widget_id, layout)
            kind = widget["kind"]
            if kind == "checkbox":
                self.settings_draft[widget_id] = not bool(
                    self.settings_draft.get(widget_id, False)
                )
            elif kind == "dropdown":
                self.ui.open_dropdown = widget_id
            elif kind == "slider":
                self._slider_drag = widget_id
                self._set_slider_value(
                    widget_id,
                    widget,
                    self._slider_value_from_position(widget, position),
                )
            elif kind == "button":
                self._settings_button_action(widget_id)
            return

    def _settings_mouse_up(
        self, position: tuple[int, int], button: int
    ) -> None:
        if button == 1:
            self.ui.end_drag()
            self._slider_drag = None

    def _settings_mouse_motion(self, position: tuple[int, int]) -> None:
        if self.ui.dragging:
            self.ui.drag_to(position)
            return
        if self._slider_drag is not None:
            layout = self._settings_layout()
            widget = layout["widgets"].get(self._slider_drag)
            if widget is not None and widget["kind"] == "slider":
                self._set_slider_value(
                    self._slider_drag,
                    widget,
                    self._slider_value_from_position(widget, position),
                )

    def _settings_mouse_wheel(self, direction: int) -> None:
        layout = self._settings_layout()
        position = pygame.mouse.get_pos()
        for widget_id in layout["order"]:
            widget = layout["widgets"][widget_id]
            if widget["kind"] != "slider":
                continue
            if not widget["rect"].collidepoint(position):
                continue
            current = float(
                self.settings_draft.get(widget_id, widget["min"])
            )
            self._set_slider_value(
                widget_id,
                widget,
                current + direction * float(widget["step"]),
            )
            return

    def _slider_value_from_position(
        self, widget: dict, position: tuple[int, int]
    ) -> float:
        track = widget["track"]
        fraction = (position[0] - track.left) / max(1, track.width)
        fraction = max(0.0, min(1.0, fraction))
        return float(widget["min"]) + fraction * (
            float(widget["max"]) - float(widget["min"])
        )

    def _set_slider_value(
        self, widget_id: str, widget: dict, value: float
    ) -> None:
        minimum = float(widget["min"])
        maximum = float(widget["max"])
        value = max(minimum, min(maximum, float(value)))
        step = float(widget.get("step", 0.0))
        if step > 0.0:
            value = minimum + round((value - minimum) / step) * step
            value = max(minimum, min(maximum, value))
        if widget.get("integer"):
            self.settings_draft[widget_id] = int(round(value))
        else:
            self.settings_draft[widget_id] = round(value, 4)
        if widget_id.startswith("crosshair_custom_"):
            # Touching an RGB slider implies the custom entry in the colour
            # list, otherwise the change would have no visible effect.
            self.settings_draft["crosshair_color"] = len(Crosshair.COLORS)

    def _settings_button_action(self, button_id: str) -> None:
        if button_id == "ok":
            self._close_settings(apply=True)
        elif button_id == "cancel":
            self._close_settings(apply=False)
        elif button_id == "apply":
            self._apply_settings()
        elif button_id == "open_crosshair":
            for index, (tab_id, _label) in enumerate(UI.TABS):
                if tab_id == "crosshair":
                    self._set_settings_tab(index)
                    break
        elif button_id == "defaults":
            self._reset_settings_draft()

    def _reset_settings_draft(self) -> None:
        """Reset every option in the draft; Apply still has to commit it."""
        self.settings_draft.update(
            {
                "render_mode": list(RenderMode).index(RenderMode.SHADED),
                "resolution": 0,
                "windowed": True,
                "vsync": False,
                "brightness": 1.0,
                "gamma": 1.0,
                "fov": 50.0,
                "sensitivity": DEFAULT_MOUSE_SENSITIVITY,
                "invert_y": False,
                "player_speed": DEFAULT_PLAYER_SPEED,
                "auto_bunnyhop": False,
                "menu_theme": 0,
                "accent": 0,
                "menu_opacity": 0.96,
                "show_hud": True,
                "debug_overlay": False,
                "crosshair_enabled": True,
                "crosshair_color": 0,
                "crosshair_custom_r": 255,
                "crosshair_custom_g": 255,
                "crosshair_custom_b": 255,
                "crosshair_size": 10,
                "crosshair_gap": 6,
                "crosshair_thickness": 2,
                "crosshair_dot": True,
                "crosshair_dot_size": 3,
                "crosshair_outline": True,
                "crosshair_outline_width": 1,
                "crosshair_opacity": 100,
            }
        )

    # ------------------------------------------------------------------
    # Options window: applying values
    # ------------------------------------------------------------------
    def _apply_settings(self) -> None:
        draft = self.settings_draft
        if not draft:
            return

        modes = list(RenderMode)
        self.render_mode = modes[
            int(draft.get("render_mode", 0)) % len(modes)
        ]
        self.player_speed = float(draft.get("player_speed", self.player_speed))
        self.player.speed = self.player_speed
        self.auto_bunnyhop = bool(
            draft.get("auto_bunnyhop", self.auto_bunnyhop)
        )
        self.player.auto_bhop = self.auto_bunnyhop
        self.mouse_sensitivity = float(
            draft.get("sensitivity", self.mouse_sensitivity)
        )
        self.invert_y = bool(draft.get("invert_y", self.invert_y))
        self.renderer.brightness = float(
            draft.get("brightness", self.renderer.brightness)
        )
        self.renderer.gamma = float(
            draft.get("gamma", self.renderer.gamma)
        )
        self.camera.fov = float(draft.get("fov", self.camera.fov))
        self.ui.set_theme(int(draft.get("menu_theme", self.ui.theme_index)))
        self.ui.set_accent(int(draft.get("accent", self.ui.accent_index)))
        self.ui.menu_opacity = float(
            draft.get("menu_opacity", self.ui.menu_opacity)
        )
        self.show_hud = bool(draft.get("show_hud", self.show_hud))
        self.debug_visible = bool(
            draft.get("debug_overlay", self.debug_visible)
        )

        crosshair = self.crosshair
        crosshair.enabled = bool(
            draft.get("crosshair_enabled", crosshair.enabled)
        )
        crosshair.color_index = max(
            0,
            min(
                len(Crosshair.COLORS),
                int(draft.get("crosshair_color", crosshair.color_index)),
            ),
        )
        crosshair.custom_color = (
            _color_component(
                draft.get("crosshair_custom_r", crosshair.custom_color[0])
            ),
            _color_component(
                draft.get("crosshair_custom_g", crosshair.custom_color[1])
            ),
            _color_component(
                draft.get("crosshair_custom_b", crosshair.custom_color[2])
            ),
        )
        crosshair.size = int(draft.get("crosshair_size", crosshair.size))
        crosshair.gap = int(draft.get("crosshair_gap", crosshair.gap))
        crosshair.thickness = int(
            draft.get("crosshair_thickness", crosshair.thickness)
        )
        crosshair.dot = bool(draft.get("crosshair_dot", crosshair.dot))
        crosshair.dot_size = int(
            draft.get("crosshair_dot_size", crosshair.dot_size)
        )
        crosshair.outline = bool(
            draft.get("crosshair_outline", crosshair.outline)
        )
        crosshair.outline_thickness = int(
            draft.get(
                "crosshair_outline_width", crosshair.outline_thickness
            )
        )
        crosshair.opacity = int(
            draft.get("crosshair_opacity", crosshair.opacity)
        )

        new_resolution = int(
            draft.get("resolution", self.resolution_index)
        ) % len(self.resolutions)
        new_fullscreen = not bool(draft.get("windowed", not self.fullscreen))
        new_vsync = bool(draft.get("vsync", self.vsync))
        display_changed = (
            new_resolution != self.resolution_index
            or new_fullscreen != self.fullscreen
            or new_vsync != self.vsync
        )
        self.vsync = new_vsync
        if display_changed:
            self.resolution_index = new_resolution
            self.fullscreen = new_fullscreen
            if new_fullscreen:
                self._recreate_display(
                    self._desktop_size(),
                    _GL_FLAGS | pygame.FULLSCREEN,
                )
            else:
                width, height = self.resolutions[new_resolution]
                self.windowed_width = width
                self.windowed_height = height
                self._recreate_display((width, height), _GL_FLAGS)

        # Apply is a commit as well: write the values out immediately so they
        # survive a restart even if the game is closed without pressing OK.
        self.settings.data.update(draft)
        self.settings.save()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def _update(self, delta: float) -> None:
        if not self.in_game or self.pause_open:
            # The front end and pause menu freeze the scene so the backdrop
            # stays stable while the player picks an option.
            return
        self.time += delta
        if self.toast_timer > 0.0:
            self.toast_timer -= delta

        if (
            self.mouse_captured
            and not self.settings_open
            and time.perf_counter() - self._captured_since > 0.3
            and not pygame.key.get_focused()
        ):
            self._release_mouse_capture()

        if self.mouse_captured and not self.settings_open:
            delta_x, delta_y = pygame.mouse.get_rel()
            if delta_x or delta_y:
                delta_x = max(-MAX_MOUSE_DELTA, min(MAX_MOUSE_DELTA, delta_x))
                delta_y = max(-MAX_MOUSE_DELTA, min(MAX_MOUSE_DELTA, delta_y))
                self._apply_mouse_look(delta_x, delta_y)

        if self.update_callback is not None:
            self.update_callback(self.scene, delta, self.time)
        if not self.settings_open:
            if self.walk_mode:
                self._update_player(delta)
            else:
                self._update_camera(delta)

        self._frame_intervals.append(delta)
        if len(self._frame_intervals) >= 2:
            self.fps = len(self._frame_intervals) / sum(self._frame_intervals)

    def _render_frame(self, delta: float) -> None:
        width, height = self.screen.get_size()
        if width < 8 or height < 8:
            return

        started = time.perf_counter()
        self.renderer.mode = self.render_mode
        self.renderer.render(
            self.scene, self.camera, width, height, self.time
        )
        self._render_ui_canvas(width, height)
        surface = self._draw_overlays(width, height)
        self.renderer.draw_overlay(surface)
        pygame.display.flip()

        self._render_times.append((time.perf_counter() - started) * 1000.0)
        self.frame_ms = sum(self._render_times) / len(self._render_times)

    # ------------------------------------------------------------------
    # 2D UI canvas
    # ------------------------------------------------------------------
    def _render_ui_canvas(self, width: int, height: int) -> None:
        canvas = self.ui_canvas
        canvas.begin()
        if self.cat_visible and self.in_game and not self.pause_open:
            self._draw_cat_overlay(canvas, width, height)
        canvas.render(width, height)

    def _draw_cat_overlay(
        self, canvas: UICanvas, width: int, height: int
    ) -> None:
        try:
            source_width, source_height = canvas.image_size(CAT_SPRITE_PATH)
        except (pygame.error, OSError):
            canvas.panel(
                width / 2 - 190, height / 2 - 34, 380, 68,
                fill=(*self.ui.theme.dark, 235),
                border=(255, 107, 107, 255),
                border_width=2,
            )
            canvas.text(
                "uisprites/cat.jpg not found",
                (width / 2, height / 2),
                self.ui.ui_bold, (255, 107, 107, 255), anchor="center",
            )
            return

        max_size = min(560.0, width * 0.52, height * 0.66)
        scale = min(max_size / source_width, max_size / source_height)
        image_width = max(1, int(source_width * scale))
        image_height = max(1, int(source_height * scale))

        padding = 14
        caption_height = 38
        panel_width = image_width + padding * 2
        panel_height = image_height + padding * 2 + caption_height
        panel_x = (width - panel_width) / 2
        panel_y = (height - panel_height) / 2

        # Translucent backdrop, then a framed sprite and a text caption. This
        # is the whole demo of the 2D canvas: rect + panel + sprite + text.
        canvas.rect(0, 0, width, height, (0, 0, 0, 150))
        canvas.panel(
            panel_x, panel_y, panel_width, panel_height,
            fill=(*self.ui.theme.dark, 238),
            border=(*self.ui.accent, 255),
            border_width=2,
        )
        canvas.sprite(
            CAT_SPRITE_PATH,
            (panel_x + padding, panel_y + padding),
            width=image_width,
            height=image_height,
        )

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------
    def _draw_overlays(
        self, width: int, height: int
    ) -> pygame.Surface:
        surface = self._overlay_surface
        if surface is None or surface.get_size() != (width, height):
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            self._overlay_surface = surface
        surface.fill((0, 0, 0, 0))

        menu_visible = self._menu_visible()
        if menu_visible:
            self.ui.draw_main_menu(
                surface, width, height, self._menu_state()
            )

        if self.settings_open:
            if menu_visible:
                # The menu is already opaque; darken it with a blended layer
                # instead of letting draw_settings overwrite it.
                self.ui.draw_dim(surface, width, height)
                self.ui.draw_settings(
                    surface, width, height, self._settings_state(),
                    dim=False,
                )
            else:
                self.ui.draw_settings(
                    surface, width, height, self._settings_state()
                )

        if self.map_browser_open and not self.settings_open:
            self.ui.draw_map_browser(
                surface, width, height, self._map_browser_state()
            )

        if menu_visible:
            return surface

        if self.debug_visible and not self.settings_open:
            self.ui.draw_debug(surface, self._debug_rows(width, height))

        if self.toast_timer > 0.0 and self.toast_text:
            self.ui.draw_toast(surface, width, self.toast_text)

        if (
            not self.settings_open
            and not self.mouse_captured
            and self.toast_timer <= 0.0
        ):
            self.ui.draw_mouse_hint(surface, width, height)

        if not self.settings_open:
            if self.show_hud:
                self.ui.draw_hud(surface, width, height, self.walk_mode)
            if not self.cat_visible:
                self.ui.draw_crosshair(
                    surface, width, height, self.crosshair
                )

        return surface

    def _debug_rows(self, width: int, height: int) -> list[tuple[str, str]]:
        stats = self.renderer.stats
        rotation = "—"
        cube = next(
            (obj for obj in self.scene.objects if obj.hue is None),
            self.scene.objects[0] if self.scene.objects else None,
        )
        if cube is not None:
            rot = cube.rotation
            rotation = (
                f"X {(math.degrees(rot.x) % 360):5.1f}°   "
                f"Y {(math.degrees(rot.y) % 360):5.1f}°"
            )
        camera = self.camera.position
        window_state = "fullscreen" if self.fullscreen else "windowed"
        gpu = self.renderer.info
        if len(gpu) > 40:
            gpu = gpu[:39] + "…"

        rows = [
            ("fps", f"{self.fps:5.1f}"),
            ("frame", f"{self.frame_ms:5.2f} ms"),
            ("resolution", f"{width}×{height}  {window_state}"),
            ("world", self.world_title[:40] or "main menu"),
            ("render mode", self.render_mode.value),
            (
                "console debug",
                f"{'on' if self.console_debug.enabled else 'off'} / "
                f"every {self.console_debug.interval} ticks",
            ),
            ("movement", "walk" if self.walk_mode else "free camera"),
            (
                "mouse",
                "captured" if self.mouse_captured else "released",
            ),
            ("triangles", f"{stats['drawn_tris']} / {stats['base_tris']} base"),
            ("vertices", str(stats["vertices"])),
            (
                "edges",
                f"{stats['object_edges']} base / {stats['edges']} drawn",
            ),
            ("draw calls", str(stats["draw_calls"])),
            ("gpu", gpu),
            ("rotation", rotation),
            ("camera", f"{camera.x:.1f}, {camera.y:.1f}, {camera.z:.1f}"),
            (
                "view yaw/pitch",
                f"{(math.degrees(self.camera.yaw) % 360):5.1f}° / "
                f"{math.degrees(self.camera.pitch):5.1f}°",
            ),
        ]

        if self.walk_mode:
            player = self.player
            position = player.position
            velocity = player.velocity
            rows.extend(
                [
                    (
                        "player",
                        f"{position.x:.1f}, {position.y:.1f}, {position.z:.1f}",
                    ),
                    (
                        "velocity",
                        f"{velocity.x:+.1f}, {velocity.y:+.1f}, {velocity.z:+.1f}",
                    ),
                    (
                        "speed",
                        f"{player.horizontal_speed():.1f} m/s  /  "
                        f"{self.player_speed:.1f} base",
                    ),
                    ("ground", "yes" if player.on_ground else "air"),
                    (
                        "bunnyhop",
                        f"auto {'on' if self.auto_bunnyhop else 'off'}  /  "
                        f"air {player.air_time:.2f}s",
                    ),
                ]
            )
        return rows

    # ------------------------------------------------------------------
    # Mouse capture
    # ------------------------------------------------------------------
    def _apply_mouse_look(self, delta_x: float, delta_y: float) -> None:
        sensitivity = self.mouse_sensitivity * MOUSE_SENSITIVITY_SCALE
        yaw_delta = delta_x * sensitivity
        pitch_delta = -delta_y * sensitivity
        if self.invert_y:
            pitch_delta = -pitch_delta
        if self.walk_mode:
            self.player.rotate(yaw_delta, pitch_delta)
            self._sync_camera_to_player()
        else:
            self.camera.rotate(yaw_delta, pitch_delta)

    def _capture_mouse(self) -> None:
        if (
            self.settings_open
            or self.pause_open
            or not self.in_game
            or self.mouse_captured
        ):
            return
        pygame.mouse.set_relative_mode(True)
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()  # discard motion accumulated while released
        self.mouse_captured = True
        self._captured_since = time.perf_counter()

    def _release_mouse_capture(self) -> None:
        if not self.mouse_captured:
            return
        self.mouse_captured = False
        pygame.mouse.set_relative_mode(False)
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)

    def _open_settings(self) -> None:
        self.settings_open = True
        self.settings_tab = 0
        self.settings_cursor = 0
        self._slider_drag = None
        self.ui.open_dropdown = None
        self.ui.end_drag()
        self.settings_draft = {
            "render_mode": list(RenderMode).index(self.render_mode),
            "resolution": self.resolution_index,
            "windowed": not self.fullscreen,
            "vsync": self.vsync,
            "brightness": self.renderer.brightness,
            "gamma": self.renderer.gamma,
            "fov": self.camera.fov,
            "sensitivity": self.mouse_sensitivity,
            "invert_y": self.invert_y,
            "player_speed": self.player_speed,
            "auto_bunnyhop": self.auto_bunnyhop,
            "menu_theme": self.ui.theme_index,
            "accent": self.ui.accent_index,
            "menu_opacity": self.ui.menu_opacity,
            "show_hud": self.show_hud,
            "debug_overlay": self.debug_visible,
            "crosshair_enabled": self.crosshair.enabled,
            "crosshair_color": self.crosshair.color_index,
            "crosshair_custom_r": self.crosshair.custom_color[0],
            "crosshair_custom_g": self.crosshair.custom_color[1],
            "crosshair_custom_b": self.crosshair.custom_color[2],
            "crosshair_size": self.crosshair.size,
            "crosshair_gap": self.crosshair.gap,
            "crosshair_thickness": self.crosshair.thickness,
            "crosshair_dot": self.crosshair.dot,
            "crosshair_dot_size": self.crosshair.dot_size,
            "crosshair_outline": self.crosshair.outline,
            "crosshair_outline_width": self.crosshair.outline_thickness,
            "crosshair_opacity": self.crosshair.opacity,
        }
        self._keys.clear()
        self._release_mouse_capture()
        self.player.jump_buffer = 0.0

    def _close_settings(self, apply: bool = False) -> None:
        if apply:
            # _apply_settings() persists the draft as part of the commit.
            self._apply_settings()
        self.settings_open = False
        self.settings_draft = {}
        self._slider_drag = None
        self.ui.open_dropdown = None
        self.ui.end_drag()
        self._keys.clear()
        if self.in_game and not self.pause_open:
            self._capture_mouse()

    # ------------------------------------------------------------------
    # Player
    # ------------------------------------------------------------------
    def _update_player(self, delta: float) -> None:
        keys = self._keys
        turn = PLAYER_TURN_SPEED * delta
        yaw_delta = 0.0
        pitch_delta = 0.0
        if "left" in keys:
            yaw_delta -= turn
        if "right" in keys:
            yaw_delta += turn
        if "up" in keys:
            pitch_delta += turn
        if "down" in keys:
            pitch_delta -= turn
        if yaw_delta or pitch_delta:
            self.player.rotate(yaw_delta, pitch_delta)

        self.player.update(delta, self.collision_world, keys)
        self._sync_camera_to_player()

    def _sync_camera_to_player(self) -> None:
        self.camera.position = self.player.eye_position()
        self.camera.yaw = self.player.yaw
        self.camera.pitch = self.player.pitch
        self.camera.sync_target()

    def _toggle_walk_mode(self) -> None:
        self.walk_mode = not self.walk_mode
        self._keys.clear()
        self.player.jump_buffer = 0.0
        self._sync_camera_to_player()
        self.toast_text = "WALK MODE" if self.walk_mode else "FREE CAMERA"
        self.toast_timer = 1.6

    def _respawn_player(self) -> None:
        self.player.respawn(self.scene.spawn, self.scene.spawn_yaw)
        self._sync_camera_to_player()
        self.toast_text = "RESPAWN"
        self.toast_timer = 1.2

    def _reset_player_view(self) -> None:
        self.player.yaw = self.player.spawn_yaw
        self.player.pitch = 0.0
        self._sync_camera_to_player()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _update_camera(self, delta: float) -> None:
        camera = self.camera
        keys = self._keys
        turn = CAMERA_TURN_SPEED * delta
        yaw_delta = 0.0
        pitch_delta = 0.0
        if "left" in keys:
            yaw_delta -= turn
        if "right" in keys:
            yaw_delta += turn
        if "up" in keys:
            pitch_delta += turn
        if "down" in keys:
            pitch_delta -= turn
        if yaw_delta or pitch_delta:
            camera.rotate(yaw_delta, pitch_delta)

        forward = camera.forward()
        strafe = Vec3(math.cos(camera.yaw), 0.0, math.sin(camera.yaw))
        move = Vec3()
        if "w" in keys:
            move += forward
        if "s" in keys:
            move -= forward
        if "d" in keys:
            move += strafe
        if "a" in keys:
            move -= strafe
        if "e" in keys or "space" in keys:
            move += Vec3(0.0, 1.0, 0.0)
        if "q" in keys or "ctrl" in keys:
            move -= Vec3(0.0, 1.0, 0.0)

        if move.length_squared() > 1e-10:
            speed = (
                CAMERA_SPRINT_SPEED if "shift" in keys else CAMERA_MOVE_SPEED
            )
            camera.position = camera.position + move.normalized() * (
                speed * delta
            )

        camera.sync_target()

    def _reset_camera(self) -> None:
        self.camera.position = Vec3(
            self._home_position.x,
            self._home_position.y,
            self._home_position.z,
        )
        self.camera.yaw = self._home_yaw
        self.camera.pitch = self._home_pitch
        self.camera.sync_target()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _cycle_render_mode(self) -> None:
        self.render_mode = self.render_mode.next()
        self.toast_text = self.render_mode.value.upper()
        self.toast_timer = 1.8

    def _toggle_fullscreen(self) -> None:
        self._set_fullscreen(not self.fullscreen)

    def _set_fullscreen(self, enabled: bool) -> None:
        if enabled == self.fullscreen:
            return
        self.fullscreen = enabled
        if enabled:
            flags = _GL_FLAGS | pygame.FULLSCREEN
            self._recreate_display(self._desktop_size(), flags)
        else:
            self._recreate_display(
                (self.windowed_width, self.windowed_height), _GL_FLAGS
            )
