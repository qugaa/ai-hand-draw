"""The annotation session: camera in, annotated page out.

Design notes
------------
* One OpenCV window.  The old second "Webcam Preview" window is now a
  picture-in-picture panel, so there is nothing left that can be orphaned.
* The window's native close button is honoured (``WND_PROP_VISIBLE``), as are
  Esc and Q.
* Every resource (camera, MediaPipe graph, window, page buffers) is released in
  a ``finally`` block, so an exception mid-loop cannot leave the webcam locked.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import overlay
from .camera import Camera
from .document import PdfDocument
from .errors import AppError
from .export import Exporter
from .gestures import Gesture, GestureRecognizer, GestureStabilizer
from .mapping import PointerMapper, Viewport
from .overlay import Hud, ToolButton
from .settings import PEN_PRESETS, AppSettings
from .state import SessionState
from .tracking import HandLandmarks, HandTracker

WINDOW_NAME = "AI Hand Draw"
_REFERENCE_PAGE_WIDTH = 1240.0  # A4 at 150 DPI; tool sizes scale from this


class AnnotationSession:
    """Runs the gesture annotation loop for one document."""

    def __init__(
        self,
        settings: AppSettings,
        document: PdfDocument,
        model_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.document = document
        self.model_path = model_path
        self.state = SessionState(document)
        self.hud = Hud(settings.theme)
        self.recognizer = GestureRecognizer()
        self.stabilizer = GestureStabilizer(settings.gesture_stability_frames)
        self.mapper = PointerMapper(
            margin=settings.mapping_margin,
            smoothing_min=settings.smoothing_min,
            smoothing_max=settings.smoothing_max,
            smoothing_speed=settings.smoothing_speed,
        )
        self.exporter = Exporter(document, settings.output_dir or Path.cwd())

        self._camera: Camera | None = None
        self._tracker: HandTracker | None = None
        self._window_open = False

        # per-frame working state
        self._last_point: tuple[int, int] | None = None
        self._suppress_stroke = False
        self._hold_started: float | None = None
        self._hold_consumed = False
        self._swipe: deque[tuple[float, float]] = deque(maxlen=32)
        self._swipe_ready_at = 0.0
        self._toast_text = ""
        self._toast_tone = "info"
        self._toast_until = 0.0
        self._show_help = settings.show_help
        self._show_preview = settings.show_preview
        self._fps = 0.0
        self._mapper_key: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._base_key: tuple[int, int, int, int] | None = None
        self._base_image: np.ndarray | None = None
        self._canvas: np.ndarray | None = None
        self._background: np.ndarray | None = None
        self._window_size = (1280, 860)
        self.saved_files: list[Path] = []

    # --------------------------------------------------------------- public
    def run(self) -> list[Path]:
        """Open the window and block until the user closes it."""
        try:
            # The window comes up first: opening a camera takes a couple of
            # seconds, and a splash beats staring at nothing.
            self._create_window()
            self._splash("Starting the camera...")
            self._camera = Camera(
                self.settings.camera_index,
                self.settings.camera_width,
                self.settings.camera_height,
            ).open()
            self._splash("Loading hand tracking...")
            self._tracker = HandTracker(
                detection_confidence=self.settings.detection_confidence,
                tracking_confidence=self.settings.tracking_confidence,
                detection_width=self.settings.detection_width,
                model_complexity=self.settings.model_complexity,
                model_path=self.model_path,
            )
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()
        return self.saved_files

    # --------------------------------------------------------------- window
    def _create_window(self) -> None:
        page_w, page_h = self.state.layer.size
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        self._window_open = True

        # Leave room for the chrome so the page itself opens at a usable size.
        scale = min(900 / max(page_w, 1), 760 / max(page_h, 1), 1.0)
        width = min(1600, max(860, int(page_w * scale) + 230))
        height = min(1000, max(620, int(page_h * scale) + 190))
        cv2.resizeWindow(WINDOW_NAME, width, height)
        self._window_size = (width, height)

    def _splash(self, message: str) -> None:
        """Paint a loading card while the hardware warms up."""
        width, height = self._current_window_size()
        canvas = np.empty((height, width, 3), dtype=np.uint8)
        canvas[:] = self.settings.theme.background

        title = self.document.title
        scale = min(max(height / 900.0, 0.7), 1.6)
        card_w, card_h = int(min(width * 0.7, 520 * scale)), int(150 * scale)
        x1, y1 = (width - card_w) // 2, (height - card_h) // 2
        overlay.panel(canvas, (x1, y1, x1 + card_w, y1 + card_h),
                      self.settings.theme.surface, 0.95, int(18 * scale))

        for text, dy, size, color in (
            (title, 0.38, 0.62 * scale, self.settings.theme.text),
            (message, 0.68, 0.5 * scale, self.settings.theme.accent),
        ):
            text_w = overlay.text_size(text, size, 1)[0]
            overlay.draw_text(canvas, text, (x1 + (card_w - text_w) // 2, y1 + int(card_h * dy)),
                              size, color, 1, shadow=False)

        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(1)

    def _window_is_closed(self) -> bool:
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _current_window_size(self) -> tuple[int, int]:
        try:
            _, _, width, height = cv2.getWindowImageRect(WINDOW_NAME)
            if width > 1 and height > 1:
                self._window_size = (width, height)
        except cv2.error:
            pass
        return self._window_size

    # ----------------------------------------------------------------- loop
    def _loop(self) -> None:
        assert self._camera is not None and self._tracker is not None
        previous_time = time.perf_counter()

        while True:
            if self._window_is_closed():
                break

            frame = self._camera.read(mirror=self.settings.mirror)
            if frame is None:
                self._toast("Lost the camera feed.", "error", 3.0)
                self._render(None, Gesture.NONE, None, 0.0)
                cv2.waitKey(1200)
                break

            now = time.perf_counter()
            delta = now - previous_time
            previous_time = now
            if delta > 0:
                self._fps = self._fps * 0.9 + (1.0 / delta) * 0.1 if self._fps else 1.0 / delta

            layer = self.state.layer
            frame_size = (frame.shape[1], frame.shape[0])
            page_size = layer.size

            window_size = self._current_window_size()
            self.hud.layout(window_size)
            viewport = Viewport.fit(
                page_size,
                window_size,
                self.hud.content_insets(self._show_help, self._show_preview),
            )

            # The pointer addresses the whole window, not just the page, so the
            # toolbar stays reachable. The page is scaled uniformly inside the
            # window, so mapping the zone to the window keeps strokes undistorted.
            key = (frame_size, window_size)
            if key != self._mapper_key:
                self.mapper.configure(frame_size, window_size)
                self._mapper_key = key

            landmarks = self._tracker.process(frame)
            gesture = Gesture.NONE
            point: tuple[int, int] | None = None
            hold_progress = 0.0

            if landmarks is None:
                self._on_hand_lost()
            else:
                reading = self.recognizer.analyse(landmarks, frame_size[0] / frame_size[1])
                gesture = self.stabilizer.update(reading.gesture)
                point = self.mapper.update(*reading.pointer)
                hold_progress = self._dispatch(gesture, point, viewport, now)

            self._render(frame, gesture, point, hold_progress, viewport, landmarks)

            if not self._handle_key(cv2.waitKey(1) & 0xFF):
                break

    # ------------------------------------------------------------- gestures
    def _on_hand_lost(self) -> None:
        self.stabilizer.reset()
        self.recognizer.reset()
        self.mapper.reset()
        self._last_point = None
        self._suppress_stroke = False
        self._hold_started = None
        self._hold_consumed = False
        self._swipe.clear()

    def _dispatch(
        self,
        gesture: Gesture,
        point: tuple[int, int],
        viewport: Viewport,
        now: float,
    ) -> float:
        if gesture is not Gesture.DRAW:
            self._last_point = None
            self._suppress_stroke = False
        if gesture is not Gesture.SAVE:
            self._hold_started = None
            self._hold_consumed = False
        if gesture is not Gesture.NAVIGATE:
            self._swipe.clear()

        if gesture is Gesture.DRAW:
            self._handle_draw(point, viewport)
        elif gesture is Gesture.ERASE:
            self._handle_erase(point, viewport)
        elif gesture is Gesture.SAVE:
            return self._handle_hold_save(now)
        elif gesture is Gesture.NAVIGATE:
            self._handle_swipe(now)
        return 0.0

    def _tool_scale(self) -> float:
        return max(0.35, self.state.layer.size[0] / _REFERENCE_PAGE_WIDTH)

    def _handle_draw(self, window_point: tuple[int, int], viewport: Viewport) -> None:
        button = self.hud.hit(window_point)

        if button is not None:
            # Fresh pinch over a control activates it; otherwise just skip
            # drawing so strokes never end up hidden under the toolbar.
            if self._last_point is None and not self._suppress_stroke:
                self._activate(button)
            self._suppress_stroke = True
            self._last_point = None
            return

        if self._suppress_stroke:
            return

        if not viewport.holds(window_point):
            # Off the page (margins, status bar): lift the pen instead of
            # smearing a line across the gap when it comes back.
            self._last_point = None
            return

        point = viewport.window_to_page(window_point)
        width = max(2, int(round(self.settings.pen_width * self._tool_scale())))
        layer = self.state.layer
        if self._last_point is None:
            layer.dot(point, self.state.pen_color, max(1, width // 2))
        else:
            layer.stroke(self._last_point, point, self.state.pen_color, width)
        self._last_point = point

    def _handle_erase(self, window_point: tuple[int, int], viewport: Viewport) -> None:
        if not viewport.holds(window_point):
            return
        radius = max(6, int(round(self.settings.eraser_radius * self._tool_scale())))
        self.state.layer.erase(viewport.window_to_page(window_point), radius)

    def _handle_hold_save(self, now: float) -> float:
        if self._hold_started is None:
            self._hold_started = now
        elapsed = now - self._hold_started
        progress = elapsed / max(self.settings.save_hold_seconds, 0.05)
        if progress >= 1.0 and not self._hold_consumed:
            self._hold_consumed = True
            self._save_page()
            return 1.0
        return 0.0 if self._hold_consumed else min(progress, 1.0)

    def _handle_swipe(self, now: float) -> None:
        if now < self._swipe_ready_at:
            self._swipe.clear()
            return

        self._swipe.append((now, self.mapper.raw_zone_u))
        window = self.settings.swipe_window
        while self._swipe and now - self._swipe[0][0] > window:
            self._swipe.popleft()
        if len(self._swipe) < 3:
            return

        # Measure the largest run of travel in a single direction inside the
        # window rather than just first-vs-last, so a swipe still registers even
        # if the hand drifts back a little before the pen shape is released.
        values = [u for _, u in self._swipe]
        running_min = running_max = values[0]
        rise = fall = 0.0
        for u in values:
            rise = max(rise, u - running_min)   # rightward travel
            fall = max(fall, running_max - u)   # leftward travel
            running_min = min(running_min, u)
            running_max = max(running_max, u)
        delta = rise if rise >= fall else -fall
        if abs(delta) < self.settings.swipe_distance:
            return

        if delta > 0:
            moved = self.state.previous_page()
            self._toast("Previous page" if moved else "First page", "info" if moved else "error")
        else:
            moved = self.state.next_page()
            self._toast("Next page" if moved else "Last page", "info" if moved else "error")

        self._swipe.clear()
        self._swipe_ready_at = now + self.settings.swipe_cooldown
        self._last_point = None

    def _activate(self, button: ToolButton) -> None:
        if button.kind == "pen":
            self.state.set_pen(button.key)
            self._toast(f"{button.label} pen", "info")
        elif button.key == "erase_page":
            self.state.layer.clear()
            self._toast("Page cleared", "info")
        elif button.key == "save_page":
            self._save_page()

    # ---------------------------------------------------------------- saving
    def _save_page(self) -> None:
        try:
            path = self.exporter.save_page(self.state)
        except AppError as exc:
            self._toast(exc.message, "error", 4.0)
            return
        self.saved_files.append(path)
        self._toast(f"Saved {path.name}", "success", 2.5)

    def _save_pdf(self) -> None:
        try:
            path = self.exporter.save_pdf(self.state)
        except AppError as exc:
            self._toast(exc.message, "error", 4.0)
            return
        self.saved_files.append(path)
        self._toast(f"Saved {path.name}", "success", 3.0)

    # ------------------------------------------------------------------ keys
    def _handle_key(self, key: int) -> bool:
        """Returns False when the session should end."""
        if key in (27, ord("q")):
            return False
        if key == 255 or key <= 0:
            return True

        char = chr(key).lower() if 32 <= key < 127 else ""
        if char == "s":
            self._save_page()
        elif char == "d":
            self._save_pdf()
        elif char == "c":
            self.state.layer.clear()
            self._toast("Page cleared", "info")
        elif char == "n":
            self._toast("Next page" if self.state.next_page() else "Last page", "info")
        elif char == "p":
            self._toast("Previous page" if self.state.previous_page() else "First page", "info")
        elif char == "h":
            self._show_help = not self._show_help
        elif char == "v":
            self._show_preview = not self._show_preview
        elif char in "1234":
            index = int(char) - 1
            if index < len(PEN_PRESETS):
                self.state.set_pen(PEN_PRESETS[index].key)
                self._toast(f"{PEN_PRESETS[index].label} pen", "info")
        return True

    # --------------------------------------------------------------- drawing
    def _toast(self, text: str, tone: str = "info", seconds: float = 1.6) -> None:
        self._toast_text = text
        self._toast_tone = tone
        self._toast_until = time.perf_counter() + seconds

    def _page_base(self, viewport: Viewport) -> np.ndarray:
        """Scaled copy of the page, refreshed only where it changed.

        A full rescale of a 150 DPI page costs ~5 ms; while drawing, only a few
        hundred pixels change per frame, so only that region is rescaled.
        """
        layer = self.state.layer
        draw_w, draw_h = viewport.draw_size
        key = (self.state.page_index, id(layer), draw_w, draw_h)
        interpolation = cv2.INTER_AREA if viewport.scale < 1.0 else cv2.INTER_LINEAR

        if key != self._base_key or self._base_image is None:
            layer.take_dirty()
            self._base_image = cv2.resize(layer.image, (draw_w, draw_h), interpolation=interpolation)
            self._base_key = key
            return self._base_image

        dirty = layer.take_dirty()
        if dirty is None:
            return self._base_image

        page_w, page_h = layer.size
        scale_x, scale_y = draw_w / page_w, draw_h / page_h
        sx0, sy0, sx1, sy1 = dirty
        dx0, dy0 = int(sx0 * scale_x), int(sy0 * scale_y)
        dx1 = min(draw_w, int(sx1 * scale_x) + 1)
        dy1 = min(draw_h, int(sy1 * scale_y) + 1)
        if dx1 <= dx0 or dy1 <= dy0:
            return self._base_image

        # Derive the source window back from the destination window so the
        # rescaled patch lines up exactly with the surrounding pixels.
        rx0, ry0 = int(dx0 / scale_x), int(dy0 / scale_y)
        rx1 = min(page_w, int(dx1 / scale_x) + 1)
        ry1 = min(page_h, int(dy1 / scale_y) + 1)
        if rx1 <= rx0 or ry1 <= ry0:
            return self._base_image

        self._base_image[dy0:dy1, dx0:dx1] = cv2.resize(
            layer.image[ry0:ry1, rx0:rx1], (dx1 - dx0, dy1 - dy0), interpolation=interpolation
        )
        return self._base_image

    def _render(
        self,
        frame: np.ndarray | None,
        gesture: Gesture,
        point: tuple[int, int] | None,
        hold_progress: float,
        viewport: Viewport | None = None,
        landmarks: HandLandmarks | None = None,
    ) -> None:
        if viewport is None:
            window_size = self._current_window_size()
            self.hud.layout(window_size)
            viewport = Viewport.fit(
                self.state.layer.size,
                window_size,
                self.hud.content_insets(self._show_help, self._show_preview),
            )
        self.hud.begin_frame()

        width, height = viewport.window_size
        if self._canvas is None or self._canvas.shape[:2] != (height, width):
            self._canvas = np.empty((height, width, 3), dtype=np.uint8)
            self._background = np.empty((height, width, 3), dtype=np.uint8)
            self._background[:] = self.settings.theme.background
        canvas = self._canvas
        # memcpy from a prepared background beats re-broadcasting the colour.
        np.copyto(canvas, self._background)

        base = self._page_base(viewport)
        x, y = viewport.offset_x, viewport.offset_y
        canvas[y : y + base.shape[0], x : x + base.shape[1]] = base
        self.hud.draw_zone_frame(canvas, (x, y, x + base.shape[1], y + base.shape[0]))

        hovered = None
        if point is not None:
            hovered_button = self.hud.hit(point)
            hovered = hovered_button.key if hovered_button else None

        self.hud.draw_toolbar(canvas, self.state.pen_key, hovered)
        self.hud.draw_status(
            canvas,
            self.document.title,
            f"Page {self.state.page_index + 1} / {self.state.page_count}",
            gesture.label,
            self._fps,
        )

        if self._show_preview and frame is not None:
            annotate = (
                (lambda thumb: HandTracker.draw_skeleton(thumb, landmarks))
                if landmarks is not None
                else None
            )
            self.hud.draw_preview(canvas, frame, annotate)
        if self._show_help:
            self.hud.draw_help(canvas)

        remaining = self._toast_until - time.perf_counter()
        if remaining > 0 and self._toast_text:
            self.hud.draw_toast(canvas, self._toast_text, self._toast_tone, min(1.0, remaining / 0.45))

        if point is not None:
            cursor_color = {
                Gesture.DRAW: self.state.pen_color,
                Gesture.ERASE: self.settings.theme.danger,
                Gesture.SAVE: self.settings.theme.success,
            }.get(gesture, self.settings.theme.cursor)
            radius = 9 if gesture is not Gesture.ERASE else max(
                9, viewport.scaled(self.settings.eraser_radius * self._tool_scale())
            )
            self.hud.draw_cursor(canvas, point, cursor_color, radius, hold_progress)

        cv2.imshow(WINDOW_NAME, canvas)

    # -------------------------------------------------------------- teardown
    def _cleanup(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
        if self._window_open:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass
            # Pump the GUI queue so the window actually disappears on Windows.
            for _ in range(4):
                cv2.waitKey(1)
            self._window_open = False
        self.state.release()
        self._base_image = None
        self._canvas = None
        self._background = None
