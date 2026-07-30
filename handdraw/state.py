"""Session state: pages, strokes, tools.

Replaces the module-level globals of the old ``config.py``.  All mutation goes
through these classes, so a session can be created, reset, and thrown away
without leaving anything behind.

Performance note: each page keeps a single *composite* image that is edited
incrementally.  The old loop rebuilt the composite from scratch every frame
(~36 MB of copies per frame at 150 DPI); here a frame that changes nothing
costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .document import PdfDocument
from .settings import BGR, PEN_PRESETS


class PageLayer:
    """The annotated version of one page.

    Holds the composited image (page + strokes) plus a 1-channel stroke mask.
    The mask lets the eraser restore original pixels and lets the exporter
    rebuild a transparent overlay without storing a second full colour buffer.
    """

    def __init__(self, page: np.ndarray) -> None:
        self._page = page
        self.image = page.copy()
        self.mask = np.zeros(page.shape[:2], dtype=np.uint8)
        self.version = 0
        self.has_marks = False
        self._dirty: tuple[int, int, int, int] | None = None

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    # ----------------------------------------------------------- dirty region
    def _mark_dirty(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Grow the region that changed since the last render."""
        height, width = self.image.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x0 >= x1 or y0 >= y1:
            return
        if self._dirty is None:
            self._dirty = (x0, y0, x1, y1)
        else:
            px0, py0, px1, py1 = self._dirty
            self._dirty = (min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1))

    def take_dirty(self) -> tuple[int, int, int, int] | None:
        """Return and clear the pending dirty region."""
        dirty, self._dirty = self._dirty, None
        return dirty

    # ------------------------------------------------------------------ draw
    def stroke(self, start: tuple[int, int], end: tuple[int, int], color: BGR, width: int) -> None:
        cv2.line(self.image, start, end, color, width, cv2.LINE_AA)
        cv2.line(self.mask, start, end, 255, width, cv2.LINE_AA)
        pad = width // 2 + 2
        self._mark_dirty(
            min(start[0], end[0]) - pad, min(start[1], end[1]) - pad,
            max(start[0], end[0]) + pad + 1, max(start[1], end[1]) + pad + 1,
        )
        self.has_marks = True
        self.version += 1

    def dot(self, center: tuple[int, int], color: BGR, radius: int) -> None:
        radius = max(1, radius)
        cv2.circle(self.image, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(self.mask, center, radius, 255, -1, cv2.LINE_AA)
        self._mark_dirty(center[0] - radius - 2, center[1] - radius - 2,
                         center[0] + radius + 3, center[1] + radius + 3)
        self.has_marks = True
        self.version += 1

    # ----------------------------------------------------------------- erase
    def erase(self, center: tuple[int, int], radius: int) -> None:
        """Restore original page pixels inside a circle."""
        height, width = self.image.shape[:2]
        cx, cy = center
        radius = max(1, radius)
        x0, y0 = max(0, cx - radius), max(0, cy - radius)
        x1, y1 = min(width, cx + radius + 1), min(height, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return

        stamp = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.circle(stamp, (cx - x0, cy - y0), radius, 255, -1, cv2.LINE_AA)
        selection = stamp.astype(bool)

        np.copyto(self.image[y0:y1, x0:x1], self._page[y0:y1, x0:x1], where=selection[:, :, None])
        self.mask[y0:y1, x0:x1][selection] = 0
        self._mark_dirty(x0, y0, x1, y1)
        self.has_marks = bool(self.mask.any())
        self.version += 1

    def clear(self) -> None:
        np.copyto(self.image, self._page)
        self.mask[:] = 0
        self._mark_dirty(0, 0, self.image.shape[1], self.image.shape[0])
        self.has_marks = False
        self.version += 1

    # ---------------------------------------------------------------- export
    def overlay_rgba(self) -> np.ndarray:
        """Strokes only, as a BGRA image with a transparent background."""
        bgra = np.zeros((*self.mask.shape, 4), dtype=np.uint8)
        bgra[:, :, :3] = self.image
        bgra[:, :, 3] = self.mask
        return bgra


@dataclass
class SessionState:
    """Everything the annotation loop mutates."""

    document: PdfDocument
    pen_color: BGR = PEN_PRESETS[0].color
    pen_key: str = PEN_PRESETS[0].key
    _layers: dict[int, PageLayer] = field(default_factory=dict, repr=False)
    _index: int = 0

    # ----------------------------------------------------------------- pages
    @property
    def page_index(self) -> int:
        return self._index

    @property
    def page_count(self) -> int:
        return self.document.page_count

    @property
    def layer(self) -> PageLayer:
        """The current page's layer, rendered and created on demand.

        Each layer is sized from *its own* page, so mixed-orientation PDFs no
        longer stretch annotations across pages.
        """
        existing = self._layers.get(self._index)
        if existing is not None:
            return existing
        layer = PageLayer(self.document.render(self._index))
        self._layers[self._index] = layer
        return layer

    def go_to(self, index: int) -> bool:
        """Jump to a page; returns False when already at that bound."""
        target = min(max(index, 0), self.page_count - 1)
        if target == self._index:
            return False
        self._index = target
        return True

    def next_page(self) -> bool:
        return self.go_to(self._index + 1)

    def previous_page(self) -> bool:
        return self.go_to(self._index - 1)

    # ----------------------------------------------------------------- tools
    def set_pen(self, key: str) -> None:
        for preset in PEN_PRESETS:
            if preset.key == key:
                self.pen_key = preset.key
                self.pen_color = preset.color
                return

    # ---------------------------------------------------------------- export
    @property
    def annotated_pages(self) -> dict[int, PageLayer]:
        return {index: layer for index, layer in self._layers.items() if layer.has_marks}

    def release(self) -> None:
        self._layers.clear()
