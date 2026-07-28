"""Coordinate mapping: camera space -> page space -> window space.

Two distortions are corrected here:

* **Aspect ratio.**  A 16:9 camera mapped straight onto a 1:1.41 page squashes
  horizontal motion.  Instead an *active zone* is carved out of the camera
  frame with exactly the page's aspect ratio, and only that zone is mapped, so
  a circle drawn in the air stays a circle on the page.
* **Window scaling.**  The page is letterboxed into whatever size the user
  drags the window to, and :class:`Viewport` converts both ways so the toolbar
  can live in window space while strokes live in page space.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Zone:
    """The camera sub-rectangle that maps onto the page, normalised 0..1."""

    x: float
    y: float
    width: float
    height: float

    def contains(self, nx: float, ny: float) -> bool:
        return self.x <= nx <= self.x + self.width and self.y <= ny <= self.y + self.height


class PointerMapper:
    """Turns a normalised fingertip position into a page pixel position."""

    def __init__(
        self,
        margin: float = 0.86,
        smoothing_min: float = 0.18,
        smoothing_max: float = 0.85,
        smoothing_speed: float = 0.045,
    ) -> None:
        self.margin = min(max(margin, 0.2), 1.0)
        self.smoothing_min = smoothing_min
        self.smoothing_max = smoothing_max
        self.smoothing_speed = max(smoothing_speed, 1e-4)
        self.zone = Zone(0.0, 0.0, 1.0, 1.0)
        self._page_size = (1, 1)
        self._smoothed: tuple[float, float] | None = None
        self._raw_u: float | None = None

    # ------------------------------------------------------------- configure
    def configure(self, frame_size: tuple[int, int], page_size: tuple[int, int]) -> None:
        """Recompute the active zone for a camera/page size pair."""
        frame_w, frame_h = max(1, frame_size[0]), max(1, frame_size[1])
        page_w, page_h = max(1, page_size[0]), max(1, page_size[1])
        self._page_size = (page_w, page_h)

        page_aspect = page_w / page_h  # physical width / height of the target

        # Start from the tallest allowed zone and derive the width that gives
        # the zone the same *physical* aspect ratio as the page.
        height = self.margin
        width = page_aspect * (height * frame_h) / frame_w
        if width > self.margin:
            width = self.margin
            height = (width * frame_w) / (page_aspect * frame_h)

        self.zone = Zone((1.0 - width) / 2.0, (1.0 - height) / 2.0, width, height)
        self._smoothed = None
        self._raw_u = None

    def reset(self) -> None:
        self._smoothed = None
        self._raw_u = None

    # ----------------------------------------------------------------- apply
    def to_zone(self, nx: float, ny: float) -> tuple[float, float]:
        """Normalised frame point -> 0..1 inside the active zone (clamped)."""
        u = (nx - self.zone.x) / self.zone.width
        v = (ny - self.zone.y) / self.zone.height
        return min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)

    def _smooth(self, u: float, v: float) -> tuple[float, float]:
        """Speed-adaptive EMA: steady when still, responsive when moving."""
        if self._smoothed is None:
            self._smoothed = (u, v)
            return self._smoothed
        prev_u, prev_v = self._smoothed
        speed = hypot(u - prev_u, v - prev_v)
        blend = min(1.0, speed / self.smoothing_speed)
        alpha = self.smoothing_min + (self.smoothing_max - self.smoothing_min) * blend
        self._smoothed = (prev_u + alpha * (u - prev_u), prev_v + alpha * (v - prev_v))
        return self._smoothed

    def update(self, nx: float, ny: float) -> tuple[int, int]:
        """Fingertip in frame coords -> smoothed pixel position on the page."""
        raw_u, raw_v = self.to_zone(nx, ny)
        self._raw_u = raw_u
        u, v = self._smooth(raw_u, raw_v)
        page_w, page_h = self._page_size
        x = int(round(u * (page_w - 1)))
        y = int(round(v * (page_h - 1)))
        return min(max(x, 0), page_w - 1), min(max(y, 0), page_h - 1)

    @property
    def zone_u(self) -> float:
        """Last smoothed horizontal position inside the zone (for swipes)."""
        return 0.0 if self._smoothed is None else self._smoothed[0]

    @property
    def raw_zone_u(self) -> float:
        """Last *unsmoothed* horizontal position inside the zone.

        Swipe detection uses this instead of :attr:`zone_u` so a quick flick is
        not damped away by the pointer smoothing before it can be measured.
        """
        return 0.0 if self._raw_u is None else self._raw_u


@dataclass(frozen=True)
class Viewport:
    """Letterboxed placement of a page inside the window."""

    scale: float
    offset_x: int
    offset_y: int
    page_size: tuple[int, int]
    window_size: tuple[int, int]

    @classmethod
    def fit(
        cls,
        page_size: tuple[int, int],
        window_size: tuple[int, int],
        insets: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Viewport:
        """Fit the page inside the window, keeping clear of reserved chrome.

        ``insets`` is (left, top, right, bottom) space claimed by the toolbar,
        status bar and other panels, so the page is never hidden underneath
        them.
        """
        page_w, page_h = max(1, page_size[0]), max(1, page_size[1])
        win_w, win_h = max(1, window_size[0]), max(1, window_size[1])

        left, top, right, bottom = insets
        avail_w = win_w - left - right
        avail_h = win_h - top - bottom
        if avail_w < win_w * 0.35 or avail_h < win_h * 0.35:
            # Window too small to honour the insets; use the whole surface.
            left = top = right = bottom = 0
            avail_w, avail_h = win_w, win_h

        scale = min(avail_w / page_w, avail_h / page_h)
        draw_w, draw_h = int(page_w * scale), int(page_h * scale)
        return cls(
            scale=scale,
            offset_x=left + (avail_w - draw_w) // 2,
            offset_y=top + (avail_h - draw_h) // 2,
            page_size=(page_w, page_h),
            window_size=(win_w, win_h),
        )

    @property
    def draw_size(self) -> tuple[int, int]:
        return (
            max(1, int(self.page_size[0] * self.scale)),
            max(1, int(self.page_size[1] * self.scale)),
        )

    def page_to_window(self, point: tuple[int, int]) -> tuple[int, int]:
        return (
            int(round(point[0] * self.scale)) + self.offset_x,
            int(round(point[1] * self.scale)) + self.offset_y,
        )

    def window_to_page(self, point: tuple[int, int]) -> tuple[int, int]:
        if self.scale <= 0:
            return 0, 0
        return (
            int(round((point[0] - self.offset_x) / self.scale)),
            int(round((point[1] - self.offset_y) / self.scale)),
        )

    def scaled(self, value: float) -> int:
        """Convert a page-space length into window-space pixels."""
        return max(1, int(round(value * self.scale)))

    def holds(self, window_point: tuple[int, int]) -> bool:
        """True when a window-space point falls on the page itself."""
        draw_w, draw_h = self.draw_size
        return (
            self.offset_x <= window_point[0] < self.offset_x + draw_w
            and self.offset_y <= window_point[1] < self.offset_y + draw_h
        )
