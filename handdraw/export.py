"""Saving annotated work to disk."""

from __future__ import annotations

from pathlib import Path

from . import imagefile, paths
from .document import PdfDocument
from .errors import StorageError
from .state import SessionState


class Exporter:
    """Writes PNG snapshots and annotated PDF copies."""

    def __init__(self, document: PdfDocument, output_dir: Path) -> None:
        self.document = document
        self.output_dir = Path(output_dir)
        self.stem = paths.safe_stem(document.title)

    def _ensure_dir(self) -> Path:
        if not paths.is_writable(self.output_dir):
            fallback = paths.default_output_dir()
            if not paths.is_writable(fallback):
                raise StorageError(
                    "No writable folder is available for saving.",
                    f"Tried {self.output_dir} and {fallback}.",
                )
            self.output_dir = fallback
        return self.output_dir

    # ------------------------------------------------------------------ PNG
    def save_page(self, state: SessionState) -> Path:
        """Save the current page (page + strokes) as a PNG."""
        directory = self._ensure_dir()
        layer = state.layer
        name = f"{self.stem}-p{state.page_index + 1:03d}-{paths.timestamp()}"
        return imagefile.write(layer.image, paths.unique_path(directory, name, ".png"))

    # ------------------------------------------------------------------ PDF
    def save_pdf(self, state: SessionState) -> Path:
        """Save a copy of the PDF with the strokes stamped over the pages.

        Only the annotation layer is rasterised; the original page content stays
        vector and searchable.
        """
        directory = self._ensure_dir()
        annotated = state.annotated_pages
        if not annotated:
            raise StorageError("There is nothing to save yet.", "Draw something first.")

        overlays = {
            index: imagefile.encode(layer.overlay_rgba(), ".png")
            for index, layer in annotated.items()
        }
        destination = paths.unique_path(directory, f"{self.stem}-annotated-{paths.timestamp()}", ".pdf")
        return self.document.export_with_overlays(overlays, destination)
