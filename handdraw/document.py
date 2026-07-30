"""PDF access built on PyMuPDF (fitz).

Replaces pdf2image/Poppler: no external binary, no subprocess, and pages are
rendered lazily so a 400-page file opens as fast as a 2-page one.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PyMuPDF is required. Install it with:  pip install pymupdf"
    ) from exc

from .errors import DocumentError


class PdfDocument:
    """A rendered view over a PDF file.

    Pages are rasterised on demand and kept in a small LRU cache, so memory
    stays bounded no matter how large the document is.
    """

    def __init__(self, path: Path | str, dpi: int = 150, max_pixels: int = 2600,
                 cache_size: int = 6) -> None:
        self.path = Path(path)
        self.dpi = int(dpi)
        self.max_pixels = int(max_pixels)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = max(1, cache_size)
        self._doc: fitz.Document | None = None
        self._open()

    # ------------------------------------------------------------------ open
    def _open(self) -> None:
        if not self.path.is_file():
            raise DocumentError("That PDF no longer exists.", str(self.path))
        try:
            doc = fitz.open(self.path)
        except Exception as exc:
            raise DocumentError("This file could not be opened as a PDF.", str(exc)) from exc

        if doc.needs_pass:
            doc.close()
            raise DocumentError(
                "This PDF is password protected.",
                "Remove the password or export an unlocked copy, then try again.",
            )
        if doc.page_count == 0:
            doc.close()
            raise DocumentError("This PDF has no pages.", str(self.path))

        self._doc = doc

    # ------------------------------------------------------------- properties
    @property
    def page_count(self) -> int:
        return 0 if self._doc is None else self._doc.page_count

    @property
    def title(self) -> str:
        return self.path.stem

    @property
    def is_open(self) -> bool:
        return self._doc is not None and not self._doc.is_closed

    # ---------------------------------------------------------------- render
    def _effective_dpi(self, page: fitz.Page) -> float:
        """Clamp DPI so no page exceeds ``max_pixels`` on its longest edge."""
        rect = page.rect
        longest_inches = max(rect.width, rect.height) / 72.0
        if longest_inches <= 0:
            return float(self.dpi)
        return min(float(self.dpi), self.max_pixels / longest_inches)

    def render(self, index: int) -> np.ndarray:
        """Return page ``index`` as a BGR uint8 array."""
        if not self.is_open or self._doc is None:
            raise DocumentError("The document is closed.")
        if not 0 <= index < self.page_count:
            raise DocumentError(f"Page {index + 1} does not exist.")

        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached

        try:
            page = self._doc.load_page(index)
            # A float zoom matrix (rather than an integer dpi) keeps the
            # max_pixels ceiling exact instead of rounding past it.
            zoom = self._effective_dpi(page) / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        except Exception as exc:
            raise DocumentError(f"Page {index + 1} could not be rendered.", str(exc)) from exc

        # pixmap.samples is row-padded to `stride`; slice the padding away.
        buffer = np.frombuffer(pixmap.samples, dtype=np.uint8)
        buffer = buffer.reshape(pixmap.height, pixmap.stride)
        rgb = buffer[:, : pixmap.width * pixmap.n].reshape(pixmap.height, pixmap.width, pixmap.n)
        if pixmap.n == 1:
            image = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        elif pixmap.n == 4:
            image = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image = np.ascontiguousarray(image)

        self._cache[index] = image
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return image

    def page_size(self, index: int = 0) -> tuple[int, int]:
        """(width, height) in pixels of the rendered page."""
        image = self.render(index)
        return image.shape[1], image.shape[0]

    # ---------------------------------------------------------------- export
    def export_with_overlays(self, overlays: dict[int, bytes], destination: Path) -> Path:
        """Write a copy of the PDF with RGBA stroke overlays stamped on top.

        The original page content stays vector/searchable; only the annotation
        layer is an image.
        """
        if not self.is_open or self._doc is None:
            raise DocumentError("The document is closed.")
        out: fitz.Document | None = None
        try:
            out = fitz.open(self.path)
            for index, png_bytes in overlays.items():
                if 0 <= index < out.page_count:
                    page = out.load_page(index)
                    page.insert_image(page.rect, stream=png_bytes, overlay=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            out.save(str(destination), garbage=3, deflate=True)
        except Exception as exc:
            raise DocumentError("The annotated PDF could not be saved.", str(exc)) from exc
        finally:
            if out is not None:
                out.close()
        return destination

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        self._cache.clear()
        if self._doc is not None and not self._doc.is_closed:
            self._doc.close()
        self._doc = None

    def __enter__(self) -> PdfDocument:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        try:
            self.close()
        except Exception:
            pass
