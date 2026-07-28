"""Webcam ownership with deterministic release.

Nothing here happens at import time: the device is only touched when
:meth:`Camera.open` is called, and it is always released, including on
exceptions, because the class is a context manager.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from .errors import CameraError


def _backends() -> tuple[int, ...]:
    """Preferred capture backends, best throughput first."""
    if sys.platform == "win32":
        # DirectShow negotiates uncompressed YUY2 on most UVC webcams, which
        # caps 720p at ~10 fps; Media Foundation reaches the sensor's full
        # 30 fps. DSHOW stays as the fallback for devices MSMF cannot open.
        return (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)
    if sys.platform == "darwin":
        return (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY)
    return (cv2.CAP_V4L2, cv2.CAP_ANY)


class Camera:
    """A single webcam, opened lazily and released exactly once."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self._capture: cv2.VideoCapture | None = None
        self._failed_reads = 0

    # ------------------------------------------------------------------ open
    def open(self) -> Camera:
        if self._capture is not None and self._capture.isOpened():
            return self

        last_error = ""
        for backend in _backends():
            capture = cv2.VideoCapture(self.index, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                capture.set(cv2.CAP_PROP_FPS, 30)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low
                ok, _ = capture.read()
                if ok:
                    self._capture = capture
                    self._failed_reads = 0
                    return self
                last_error = "the device opened but delivered no frames"
            capture.release()

        raise CameraError(
            f"Camera {self.index} is not available.",
            last_error
            or "Check that no other app (Teams, Zoom, Camera) is using it, and that "
            "camera access is allowed in your privacy settings.",
        )

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def frame_size(self) -> tuple[int, int]:
        """Actual (width, height) negotiated with the device."""
        if self._capture is None:
            return self.width, self.height
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        return width, height

    # ------------------------------------------------------------------ read
    def read(self, mirror: bool = True) -> np.ndarray | None:
        """Grab one frame, or None after a few consecutive failures."""
        if self._capture is None:
            raise CameraError("The camera is not open.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._failed_reads += 1
            if self._failed_reads >= 15:
                return None
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._failed_reads = 0
        return cv2.flip(frame, 1) if mirror else frame

    # --------------------------------------------------------------- release
    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Camera:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __del__(self) -> None:  # pragma: no cover - safety net
        try:
            self.release()
        except Exception:
            pass


def probe_cameras(limit: int = 4) -> list[int]:
    """Indices that respond to a quick open/read probe."""
    found: list[int] = []
    backend = _backends()[0]
    for index in range(limit):
        capture = cv2.VideoCapture(index, backend)
        try:
            if capture.isOpened() and capture.read()[0]:
                found.append(index)
        finally:
            capture.release()
    return found
