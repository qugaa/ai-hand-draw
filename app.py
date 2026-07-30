"""AI Hand Draw entry point.

    python app.py                      # launch the desktop UI
    python app.py --pdf notes.pdf      # skip the UI and start annotating
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from handdraw import paths
from handdraw.errors import AppError
from handdraw.settings import AppSettings


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ai-hand-draw", description=__doc__)
    parser.add_argument("--pdf", type=Path, help="open this PDF directly, without the launcher")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--dpi", type=int, default=150, help="page render DPI (default: 150)")
    parser.add_argument("--output", type=Path, help="folder for saved annotations")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the camera")
    parser.add_argument("--no-preview", action="store_true", help="hide the camera preview")
    return parser.parse_args(argv)


def _run_headless(args: argparse.Namespace) -> int:
    from handdraw.document import PdfDocument
    from handdraw.session import AnnotationSession

    settings = AppSettings(
        pdf_path=args.pdf,
        render_dpi=args.dpi,
        camera_index=args.camera,
        mirror=not args.no_mirror,
        show_preview=not args.no_preview,
        output_dir=args.output or paths.default_output_dir(),
    )

    document = PdfDocument(args.pdf, dpi=settings.render_dpi, max_pixels=settings.max_page_pixels)
    try:
        saved = AnnotationSession(settings, document).run()
    finally:
        document.close()

    for path in saved:
        print(f"saved: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.pdf is not None:
            return _run_headless(args)
        from handdraw.ui.launcher import main as launch

        return launch()
    except AppError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
