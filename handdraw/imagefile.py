"""Unicode-safe image encoding helpers.

``cv2.imwrite`` silently fails on Windows when the path contains non-ASCII
characters (a very common case: user names, OneDrive folders).  Everything here
encodes to memory first and writes bytes through pathlib instead.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .errors import StorageError


def encode(image: np.ndarray, suffix: str = ".png") -> bytes:
    """Encode a BGR/BGRA array into image bytes."""
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise StorageError("Could not encode the image.", f"format: {suffix}")
    return buffer.tobytes()


def write(image: np.ndarray, path: Path) -> Path:
    """Write a BGR/BGRA array to ``path``, creating parent folders."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encode(image, path.suffix or ".png"))
    except OSError as exc:
        raise StorageError(f"Could not write {path.name}.", str(exc)) from exc
    return path
