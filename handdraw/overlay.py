"""The on-screen interface drawn over the page.

Everything is laid out in *window* space and recomputed whenever the window is
resized, so the controls stay usable at any size.  Panels are drawn as
anti-aliased rounded rectangles blended over the page, which keeps text legible
on top of dense documents - the old flat white text on white paper was not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .settings import BGR, PEN_PRESETS, Theme

_FONT = cv2.FONT_HERSHEY_DUPLEX
_FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------- shapes
@lru_cache(maxsize=32)
def _corner_alpha(radius: int) -> np.ndarray:
    """Top-left quarter-disc coverage (float32), used only at panel corners."""
    supersample = 4
    size = radius * supersample
    big = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(big, (size, size), size, 255, -1, cv2.LINE_AA)
    small = cv2.resize(big, (radius, radius), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def _clip_rect(rect: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = rect
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def panel(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: BGR,
    alpha: float = 0.78,
    radius: int = 14,
) -> None:
    """Blend a translucent rounded panel into ``image``.

    The interior is blended with integer SIMD ops (``addWeighted``) and only the
    four corner squares pay for float coverage math, which is roughly 8x faster
    than blending the whole panel in float - and it runs every frame.
    """
    x1, y1, x2, y2 = _clip_rect(rect, image.shape)
    width, height = x2 - x1, y2 - y1
    if width < 2 or height < 2:
        return

    roi = image[y1:y2, x1:x2]
    fill = np.empty_like(roi)
    fill[:] = color
    blended = cv2.addWeighted(fill, float(alpha), roi, 1.0 - float(alpha), 0.0)

    radius = max(0, min(radius, width // 2, height // 2))
    if radius == 0:
        roi[:] = blended
        return

    roi[:, radius : width - radius] = blended[:, radius : width - radius]
    roi[radius : height - radius, :radius] = blended[radius : height - radius, :radius]
    roi[radius : height - radius, width - radius :] = blended[radius : height - radius, width - radius :]

    quarter = _corner_alpha(radius)
    corners = (
        (slice(0, radius), slice(0, radius), quarter),
        (slice(0, radius), slice(width - radius, width), quarter[:, ::-1]),
        (slice(height - radius, height), slice(0, radius), quarter[::-1, :]),
        (slice(height - radius, height), slice(width - radius, width), quarter[::-1, ::-1]),
    )
    for rows, cols, coverage in corners:
        mask = coverage[:, :, None]
        patch = roi[rows, cols].astype(np.float32) * (1.0 - mask)
        patch += blended[rows, cols].astype(np.float32) * mask
        roi[rows, cols] = patch.astype(np.uint8)


def outline(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: BGR,
    thickness: int = 2,
    radius: int = 14,
) -> None:
    """Anti-aliased rounded outline."""
    x1, y1, x2, y2 = rect
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    for center, angle in (
        ((x1 + radius, y1 + radius), 180),
        ((x2 - radius, y1 + radius), 270),
        ((x2 - radius, y2 - radius), 0),
        ((x1 + radius, y2 - radius), 90),
    ):
        cv2.ellipse(image, center, (radius, radius), angle, 0, 90, color, thickness, cv2.LINE_AA)


def text_size(text: str, scale: float, thickness: int, font: int = _FONT) -> tuple[int, int]:
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    return width, height


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: BGR,
    thickness: int = 1,
    font: int = _FONT,
    shadow: bool = True,
) -> None:
    if shadow:
        cv2.putText(image, text, (origin[0] + 1, origin[1] + 1), font, scale, (0, 0, 0),
                    thickness + 1, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, scale, color, thickness, cv2.LINE_AA)


# -------------------------------------------------------------------- buttons
@dataclass(frozen=True)
class ToolButton:
    key: str
    label: str
    kind: str  # "pen" | "action"
    color: BGR
    rect: tuple[int, int, int, int]

    def contains(self, point: tuple[int, int]) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


class Hud:
    """Builds and paints the whole on-screen interface."""

    def __init__(self, theme: Theme) -> None:
        self.theme = theme
        self.buttons: tuple[ToolButton, ...] = ()
        self._rail_rect = (0, 0, 0, 0)
        self._size = (0, 0)
        self.unit = 1.0
        self._preview_rect: tuple[int, int, int, int] | None = None

    def begin_frame(self) -> None:
        """Reset per-frame layout state before painting."""
        self._preview_rect = None

    # ------------------------------------------------------------- reserving
    def content_insets(self, show_help: bool, show_preview: bool) -> tuple[int, int, int, int]:
        """Space claimed by the chrome, as (left, top, right, bottom).

        The page is fitted inside what is left, so the toolbar, status bar,
        help strip and camera preview never cover the document.
        """
        width, _ = self._size
        margin = int(14 * self.unit)
        left = (self._rail_rect[2] + margin) if self.buttons else margin
        top = 2 * margin + int(46 * self.unit)
        bottom = margin

        if show_help:
            bottom = max(bottom, int(26 * self.unit) * 2 + int(34 * self.unit) + margin)
        if show_preview:
            preview_w = int(min(max(width * 0.19, 168), 320))
            bottom = max(bottom, int(preview_w * 0.78) + 2 * int(18 * self.unit))
        return left, top, margin, bottom

    # ------------------------------------------------------------ layout
    def layout(self, size: tuple[int, int]) -> None:
        """Recompute control geometry for the current window size."""
        if size == self._size and self.buttons:
            return
        self._size = size
        width, height = size
        self.unit = min(max(height / 900.0, 0.62), 1.8)

        items: list[tuple[str, str, str, BGR]] = [
            (p.key, p.label, "pen", p.color) for p in PEN_PRESETS
        ]
        items.append(("erase_page", "Clear", "action", self.theme.surface_strong))
        items.append(("save_page", "Save", "action", self.theme.accent))

        count = len(items)
        button = int(min(max(height * 0.082, 42), 92))
        gap = max(8, int(button * 0.22))
        pad = max(10, int(button * 0.20))

        # Shrink the whole rail proportionally until it fits the window; on a
        # window too small for even a minimal rail, hide it rather than emit
        # out-of-bounds rectangles.
        needed = count * button + (count - 1) * gap + 2 * pad
        if needed > height:
            factor = height / needed
            button = max(10, int(button * factor))
            gap = max(2, int(gap * factor))
            pad = max(2, int(pad * factor))
            needed = count * button + (count - 1) * gap + 2 * pad
        if needed > height or button + 2 * pad > width * 0.4:
            self.buttons = ()
            self._rail_rect = (0, 0, 0, 0)
            return

        total = count * button + (count - 1) * gap
        rail_x = max(6, int(width * 0.012))
        rail_y = max(0, (height - total) // 2 - pad)
        rail_y = min(rail_y, max(0, height - total - 2 * pad))
        self._rail_rect = (rail_x, rail_y, rail_x + button + 2 * pad, rail_y + total + 2 * pad)

        buttons: list[ToolButton] = []
        for index, (key, label, kind, color) in enumerate(items):
            top = rail_y + pad + index * (button + gap)
            left = rail_x + pad
            buttons.append(
                ToolButton(key, label, kind, color, (left, top, left + button, top + button))
            )
        self.buttons = tuple(buttons)

    def hit(self, point: tuple[int, int]) -> ToolButton | None:
        for button in self.buttons:
            if button.contains(point):
                return button
        return None

    # ----------------------------------------------------------- painting
    def draw_toolbar(self, canvas: np.ndarray, active_pen: str, hovered: str | None) -> None:
        theme = self.theme
        panel(canvas, self._rail_rect, theme.surface, theme.panel_alpha, int(20 * self.unit))
        radius = int(14 * self.unit)

        for button in self.buttons:
            x1, y1, x2, y2 = button.rect
            is_active = button.kind == "pen" and button.key == active_pen
            is_hovered = hovered == button.key

            panel(canvas, button.rect, button.color, 0.95 if button.kind == "pen" else 0.85, radius)

            if button.kind == "action":
                label_scale = 0.42 * self.unit
                width, height = text_size(button.label, label_scale, 1, _FONT_SMALL)
                # Dark text on bright fills, light text on dark fills.
                luminance = 0.114 * button.color[0] + 0.587 * button.color[1] + 0.299 * button.color[2]
                label_color = (24, 22, 20) if luminance > 140 else theme.text
                draw_text(
                    canvas,
                    button.label,
                    ((x1 + x2 - width) // 2, (y1 + y2 + height) // 2),
                    label_scale,
                    label_color,
                    1,
                    _FONT_SMALL,
                    shadow=False,
                )

            if is_active:
                outline(canvas, (x1 - 4, y1 - 4, x2 + 4, y2 + 4), theme.text, 2, radius + 4)
            elif is_hovered:
                outline(canvas, (x1 - 3, y1 - 3, x2 + 3, y2 + 3), theme.accent, 2, radius + 3)
            else:
                outline(canvas, button.rect, theme.outline, 1, radius)

    def draw_status(
        self,
        canvas: np.ndarray,
        title: str,
        page_text: str,
        mode: str,
        fps: float,
    ) -> None:
        theme = self.theme
        height_px = int(46 * self.unit)
        margin = int(14 * self.unit)
        left = self._rail_rect[2] + margin
        right = canvas.shape[1] - margin
        if right - left < 120:
            left, right = margin, canvas.shape[1] - margin
        rect = (left, margin, right, margin + height_px)
        panel(canvas, rect, theme.surface, theme.panel_alpha, int(height_px * 0.42))

        baseline = margin + int(height_px * 0.64)
        scale = 0.52 * self.unit
        draw_text(canvas, title, (left + int(18 * self.unit), baseline), scale, theme.text, 1,
                  shadow=False)

        title_width = text_size(title, scale, 1)[0]
        info_x = left + int(18 * self.unit) + title_width + int(18 * self.unit)
        draw_text(canvas, page_text, (info_x, baseline), scale * 0.92, theme.text_muted, 1,
                  _FONT_SMALL, shadow=False)

        right_text = f"{mode}   {fps:4.0f} fps"
        right_width = text_size(right_text, scale * 0.92, 1, _FONT_SMALL)[0]
        draw_text(canvas, right_text, (right - right_width - int(18 * self.unit), baseline),
                  scale * 0.92, theme.accent, 1, _FONT_SMALL, shadow=False)

    def draw_help(self, canvas: np.ndarray) -> None:
        theme = self.theme
        lines = (
            "Pinch = draw    Open palm = erase    Point + swipe = turn page    OK sign = save page",
            "Keys: S save page   D save PDF   C clear   N/P pages   1-4 colours   H help   Q quit",
        )
        scale = 0.44 * self.unit
        widths = [text_size(line, scale, 1, _FONT_SMALL)[0] for line in lines]
        box_w = max(widths) + int(40 * self.unit)
        line_h = int(26 * self.unit)
        box_h = line_h * len(lines) + int(18 * self.unit)

        # Keep clear of the camera preview when it is showing.
        margin = int(16 * self.unit)
        right_limit = self._preview_rect[0] - margin if self._preview_rect else canvas.shape[1]
        centre = max(box_w // 2 + margin, (right_limit + margin) // 2)
        x1 = max(margin, min(centre - box_w // 2, right_limit - box_w))
        y2 = canvas.shape[0] - margin
        y1 = y2 - box_h
        panel(canvas, (x1, y1, x1 + box_w, y2), theme.surface, 0.7, margin)
        for index, line in enumerate(lines):
            draw_text(
                canvas,
                line,
                (x1 + (box_w - widths[index]) // 2, y1 + int(30 * self.unit) + index * line_h),
                scale,
                theme.text_muted if index else theme.text,
                1,
                _FONT_SMALL,
                shadow=False,
            )

    def draw_toast(self, canvas: np.ndarray, text: str, tone: str, strength: float) -> None:
        """Fading pill message. ``strength`` runs 1 -> 0 as it expires."""
        theme = self.theme
        colors = {"info": theme.accent, "success": theme.success, "error": theme.danger}
        accent = colors.get(tone, theme.accent)
        scale = 0.55 * self.unit
        width, height = text_size(text, scale, 1)
        pad_x, pad_y = int(26 * self.unit), int(16 * self.unit)
        box_w, box_h = width + 2 * pad_x, height + 2 * pad_y
        x1 = (canvas.shape[1] - box_w) // 2
        y1 = int(canvas.shape[0] * 0.72)
        alpha = max(0.0, min(1.0, strength))
        panel(canvas, (x1, y1, x1 + box_w, y1 + box_h), theme.surface_strong, 0.85 * alpha,
              int(box_h * 0.5))
        bar_w = max(3, int(4 * self.unit))
        panel(canvas, (x1 + pad_x // 3, y1 + pad_y // 2, x1 + pad_x // 3 + bar_w, y1 + box_h - pad_y // 2),
              accent, alpha, bar_w // 2)
        blended = tuple(
            int(c * alpha + b * (1 - alpha))
            for c, b in zip(theme.text, theme.surface, strict=True)
        )
        draw_text(canvas, text, (x1 + pad_x, y1 + pad_y + height), scale, blended, 1, shadow=False)

    def draw_cursor(
        self,
        canvas: np.ndarray,
        point: tuple[int, int],
        color: BGR,
        radius: int,
        progress: float = 0.0,
    ) -> None:
        """Pen cursor: filled core, contrasting ring, optional progress arc."""
        radius = max(4, radius)
        cv2.circle(canvas, point, radius + 3, (12, 12, 12), 2, cv2.LINE_AA)
        cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point, radius, (250, 250, 250), 1, cv2.LINE_AA)
        if progress > 0.0:
            ring = radius + int(12 * self.unit)
            cv2.ellipse(canvas, point, (ring, ring), -90, 0, 360, self.theme.outline, 2, cv2.LINE_AA)
            cv2.ellipse(canvas, point, (ring, ring), -90, 0, int(360 * min(progress, 1.0)),
                        self.theme.success, 3, cv2.LINE_AA)

    def draw_zone_frame(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        """Corner ticks showing where the page area is."""
        x1, y1, x2, y2 = rect
        length = int(min(x2 - x1, y2 - y1) * 0.06)
        color = self.theme.outline
        for (cx, cy), (dx, dy) in (
            ((x1, y1), (1, 1)),
            ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)),
            ((x2, y2), (-1, -1)),
        ):
            cv2.line(canvas, (cx, cy), (cx + dx * length, cy), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * length), color, 2, cv2.LINE_AA)

    def draw_preview(
        self,
        canvas: np.ndarray,
        frame: np.ndarray,
        annotate: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """Picture-in-picture webcam view, bottom right.

        ``annotate`` is invoked on the thumbnail before it is pasted, which lets
        the caller draw the hand skeleton without the full-size frame ever
        being copied.
        """
        theme = self.theme
        target_w = int(min(max(canvas.shape[1] * 0.19, 168), 320))
        scale = target_w / frame.shape[1]
        target_h = max(1, int(frame.shape[0] * scale))
        margin = int(18 * self.unit)
        x2 = canvas.shape[1] - margin
        y2 = canvas.shape[0] - margin
        x1, y1 = x2 - target_w, y2 - target_h
        if x1 < 0 or y1 < 0:
            return

        pad = max(4, int(6 * self.unit))
        self._preview_rect = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
        panel(canvas, self._preview_rect, theme.surface, 0.85, int(14 * self.unit))
        # Decimating first makes the area-average resize ~6x cheaper with no
        # visible difference at thumbnail size.
        source = frame
        step = frame.shape[1] // (target_w * 2)
        if step >= 2:
            source = frame[::step, ::step]
        thumb = cv2.resize(source, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if annotate is not None:
            annotate(thumb)
        canvas[y1:y2, x1:x2] = thumb
        outline(canvas, (x1 - pad, y1 - pad, x2 + pad, y2 + pad), theme.outline, 1,
                int(14 * self.unit))
