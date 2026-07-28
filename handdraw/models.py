"""Hand-tracking model asset management.

MediaPipe >= 0.10.30 ships without the legacy ``mp.solutions`` graphs, so the
Tasks API needs an explicit ``hand_landmarker.task`` bundle.  It is resolved
from (in order) an environment variable, a ``models/`` folder next to the repo,
the per-user cache, and finally a one-time download.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .errors import AppError

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_MIN_BYTES = 1_000_000
ENV_OVERRIDE = "HANDDRAW_HAND_MODEL"

ProgressFn = Callable[[float], None]


def cache_dir() -> Path:
    """Per-user location for downloaded assets."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "AI Hand Draw" / "models"


def _candidates() -> list[Path]:
    found: list[Path] = []
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        found.append(Path(override))
    repo_root = Path(__file__).resolve().parent.parent
    found.append(repo_root / "models" / MODEL_FILENAME)
    found.append(cache_dir() / MODEL_FILENAME)
    return found


def find_model() -> Path | None:
    for candidate in _candidates():
        if candidate.is_file() and candidate.stat().st_size >= MODEL_MIN_BYTES:
            return candidate
    return None


def download_model(destination: Path | None = None, progress: ProgressFn | None = None) -> Path:
    """Fetch the model bundle (~7.5 MB) and store it atomically."""
    target = destination or (cache_dir() / MODEL_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)

    handle = None
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "ai-hand-draw"})
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            handle = tempfile.NamedTemporaryFile(
                delete=False, dir=target.parent, suffix=".part"
            )
            downloaded = 0
            while True:
                chunk = response.read(262_144)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if progress and total:
                    progress(min(1.0, downloaded / total))
            handle.close()
            if downloaded < MODEL_MIN_BYTES:
                raise AppError("The downloaded hand-tracking model looks incomplete.")
            shutil.move(handle.name, target)
            handle = None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise AppError(
            "The hand-tracking model could not be downloaded.",
            f"Download it manually from\n{MODEL_URL}\nand save it as\n{target}\n\n({exc})",
        ) from exc
    finally:
        if handle is not None:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)

    return target


def ensure_model(progress: ProgressFn | None = None, allow_download: bool = True) -> Path:
    """Return a usable model path, downloading it once if necessary."""
    existing = find_model()
    if existing is not None:
        return existing
    if not allow_download:
        raise AppError(
            "The hand-tracking model is missing.",
            f"Place {MODEL_FILENAME} in {cache_dir()} or set {ENV_OVERRIDE}.",
        )
    return download_model(progress=progress)
