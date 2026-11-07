"""Application shell: pygame window, OpenGL frame loop and all hotkeys."""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path

import pygame

from .canvas import UICanvas
from .collision import CollisionWorld
from .math3d import Vec3
from .player import Player, TURN_SPEED as PLAYER_TURN_SPEED
from .renderer import GLRenderer, RenderMode
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
MOUSE_SENSITIVITY = 0.0045
MAX_MOUSE_DELTA = 320
WHEEL_STEP = 0.55

PLAYER_SPEED_MIN = 3.0
PLAYER_SPEED_MAX = 15.0
PLAYER_SPEED_STEP = 0.5
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
    ) -> None:
        self.scene = scene
        self.update_callback = update
        self.title = title
        self.windowed_width = width
        self.windowed_height = height

        self.fullscreen = False
        self.render_mode = RenderMode.SHADED
        self.debug_visible = False
        self.settings_open = False
        self.settings_cursor = 0
        self.walk_mode = True
        self.player_speed = DEFAULT_PLAYER_SPEED
        self.auto_bunnyhop = False
        self.cat_visible = False

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

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------
    def _set_gl_attributes(self, multisample: bool) -> None:
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_MAJOR_VERSION, 3
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_MINOR_VERSION, 3
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
        self, size: tuple[int, int], flags: int
    ) -> pygame.Surface:
        # SDL will not resize an existing OpenGL drawable reliably, so the
        # display is torn down and rebuilt for every resolution change.
        pygame.display.quit()
        pygame.display.init()
        pygame.display.set_caption(self.title)
        try:
            self._set_gl_attributes(True)
            return pygame.display.set_mode(size, flags)
        except pygame.error:
            # Some drivers reject 4x MSAA; retry without it.
            pygame.display.quit()
            pygame.display.init()
            pygame.display.set_caption(self.title)
            self._set_gl_attributes(False)
            return pygame.display.set_mode(size, flags)

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
        self._capture_mouse()
        try:
            while self._running:
                delta = self._clock.tick(TARGET_FPS) / 1000.0
                if delta <= 0.0:
                    delta = 1.0 / TARGET_FPS
                delta = min(delta, 0.1)
                self._handle_events()
                if not self._running:
                    break
                self._update(delta)
                self._render_frame(delta)
        finally:
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
                if (
                    event.button == 1
                    and not self.settings_open
                    and not self.mouse_captured
                ):
                    self._capture_mouse()
            elif event.type == pygame.MOUSEWHEEL:
                if not self.settings_open and not self.walk_mode:
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
            if not self.settings_open:
                self._toggle_walk_mode()
            return
        if key == pygame.K_ESCAPE:
            if self.settings_open:
                self._close_settings()
            else:
                self._open_settings()
            return

        if self.settings_open:
            self._handle_settings_key(key)
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

    def _handle_settings_key(self, key: int) -> None:
        rows = UI.ROWS
        if key in (pygame.K_UP, pygame.K_w):
            self.settings_cursor = (self.settings_cursor - 1) % len(rows)
            return
        if key in (pygame.K_DOWN, pygame.K_s):
            self.settings_cursor = (self.settings_cursor + 1) % len(rows)
            return

        direction = 0
        if key in (pygame.K_LEFT, pygame.K_a):
            direction = -1
        elif key in (pygame.K_RIGHT, pygame.K_d):
            direction = 1

        row = rows[self.settings_cursor]
        if direction:
            self._change_setting(row, direction)
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            if row == "back":
                self._close_settings()
            else:
                self._change_setting(row, 1)

    def _change_setting(self, row: str, direction: int) -> None:
        if row == "resolution":
            self._step_resolution(direction)
        elif row == "fullscreen":
            self._toggle_fullscreen()
        elif row == "player_speed":
            self._step_player_speed(direction)
        elif row == "auto_bunnyhop":
            self._toggle_auto_bunnyhop()
        elif row == "accent_color":
            self._step_accent_color(direction)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def _update(self, delta: float) -> None:
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
        if self.cat_visible:
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
                fill=(11, 17, 25, 235),
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
        canvas.rect(0, 0, width, height, (2, 5, 10, 150))
        canvas.panel(
            panel_x, panel_y, panel_width, panel_height,
            fill=(11, 17, 25, 238),
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

        if self.settings_open:
            self.ui.draw_settings(
                surface,
                width,
                height,
                {
                    "cursor": self.settings_cursor,
                    "resolution": self._resolution_text(),
                    "fullscreen": self.fullscreen,
                    "player_speed": self.player_speed,
                    "auto_bunnyhop": self.auto_bunnyhop,
                    "accent_name": self.ui.accent_name,
                    "accent_hex": self.ui.accent_hex,
                    "window": (self.windowed_width, self.windowed_height),
                    "screen": self._desktop_size(),
                },
            )

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
            self.ui.draw_hud(surface, width, height, self.walk_mode)

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
            ("render mode", self.render_mode.value),
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
        yaw_delta = delta_x * MOUSE_SENSITIVITY
        pitch_delta = -delta_y * MOUSE_SENSITIVITY
        if self.walk_mode:
            self.player.rotate(yaw_delta, pitch_delta)
            self._sync_camera_to_player()
        else:
            self.camera.rotate(yaw_delta, pitch_delta)

    def _capture_mouse(self) -> None:
        if self.settings_open or self.mouse_captured:
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
        self.settings_cursor = 0
        self._keys.clear()
        self._release_mouse_capture()
        self.player.jump_buffer = 0.0

    def _close_settings(self) -> None:
        self.settings_open = False
        self._keys.clear()
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

    def _step_player_speed(self, direction: int) -> None:
        value = self.player_speed + direction * PLAYER_SPEED_STEP
        value = round(
            max(PLAYER_SPEED_MIN, min(PLAYER_SPEED_MAX, value)),
            1,
        )
        self.player_speed = value
        self.player.speed = value

    def _toggle_auto_bunnyhop(self) -> None:
        self.auto_bunnyhop = not self.auto_bunnyhop
        self.player.auto_bhop = self.auto_bunnyhop

    def _step_accent_color(self, direction: int) -> None:
        self.ui.cycle_accent(direction)
        self.toast_text = f"ACCENT: {self.ui.accent_name.upper()}"
        self.toast_timer = 1.4

    def _resolution_text(self) -> str:
        width, height = self.resolutions[self.resolution_index]
        return f"{width} × {height}"

    def _step_resolution(self, direction: int) -> None:
        self.resolution_index = (
            self.resolution_index + direction
        ) % len(self.resolutions)
        width, height = self.resolutions[self.resolution_index]
        self._apply_resolution(width, height)

    def _apply_resolution(self, width: int, height: int) -> None:
        self.windowed_width = width
        self.windowed_height = height
        if not self.fullscreen:
            self._recreate_display((width, height), _GL_FLAGS)

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
