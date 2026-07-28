"""Hand tracking, wrapped so it is always closed.

Two MediaPipe generations are supported behind one interface:

* the legacy ``mp.solutions.hands`` graph (MediaPipe < 0.10.30), and
* the Tasks API ``HandLandmarker`` (0.10.30+, where ``mp.solutions`` was
  removed), which needs a downloadable model bundle.

Whichever backend is active, :meth:`HandTracker.process` returns the same
lightweight ``HandLandmarks`` object, so the gesture code never has to care.

Inference runs on a downscaled copy of the frame - landmarks are normalised, so
they stay valid for the full-resolution frame - which roughly halves the
per-frame cost at 720p.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .errors import AppError
from .models import ensure_model

# Skeleton drawn on the preview thumbnail; avoids depending on mediapipe's
# drawing_utils, which the Tasks-only builds no longer ship.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class HandLandmarks:
    """Backend-neutral landmark set (mirrors the legacy attribute name)."""

    landmark: tuple[Landmark, ...]


def _legacy_available() -> bool:
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")  # type: ignore[attr-defined]


class HandTracker:
    """Owns the MediaPipe graph and guarantees it gets closed."""

    def __init__(
        self,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
        detection_width: int = 480,
        model_complexity: int = 0,
        model_path: Path | None = None,
    ) -> None:
        self.detection_width = max(160, int(detection_width))
        self._closed = False
        self._legacy = None
        self._tasks = None
        self._timestamp_ms = 0

        if _legacy_available():
            self._legacy = mp.solutions.hands.Hands(  # type: ignore[attr-defined]
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=int(model_complexity),
                min_detection_confidence=float(detection_confidence),
                min_tracking_confidence=float(tracking_confidence),
            )
            self.backend = "solutions"
        else:
            self._tasks = self._create_tasks_landmarker(
                model_path or ensure_model(), detection_confidence, tracking_confidence
            )
            self.backend = "tasks"

    @staticmethod
    def _create_tasks_landmarker(model_path: Path, detection: float, tracking: float):
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - broken install
            raise AppError("MediaPipe is installed but unusable.", str(exc)) from exc

        try:
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=float(detection),
                min_hand_presence_confidence=float(detection),
                min_tracking_confidence=float(tracking),
            )
            return vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise AppError(
                "The hand-tracking model could not be loaded.",
                f"{model_path}\n\n{exc}",
            ) from exc

    # --------------------------------------------------------------- process
    def _downscale(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        if width <= self.detection_width:
            return frame_bgr
        scale = self.detection_width / width
        return cv2.resize(
            frame_bgr,
            (self.detection_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def process(self, frame_bgr: np.ndarray) -> HandLandmarks | None:
        """Return the first detected hand's landmarks, or None."""
        if self._closed:
            return None

        rgb = cv2.cvtColor(self._downscale(frame_bgr), cv2.COLOR_BGR2RGB)

        if self._legacy is not None:
            rgb.flags.writeable = False
            results = self._legacy.process(rgb)
            if not results.multi_hand_landmarks:
                return None
            return HandLandmarks(
                tuple(Landmark(p.x, p.y, p.z) for p in results.multi_hand_landmarks[0].landmark)
            )

        assert self._tasks is not None
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        # Tasks VIDEO mode demands strictly increasing timestamps.
        self._timestamp_ms = max(self._timestamp_ms + 1, int(time.perf_counter() * 1000))
        result = self._tasks.detect_for_video(image, self._timestamp_ms)
        if not result.hand_landmarks:
            return None
        return HandLandmarks(tuple(Landmark(p.x, p.y, p.z) for p in result.hand_landmarks[0]))

    # ------------------------------------------------------------ rendering
    @staticmethod
    def draw_skeleton(image: np.ndarray, hand: HandLandmarks,
                      color: tuple[int, int, int] = (120, 255, 160)) -> None:
        """Draw the hand graph onto ``image`` (used for the preview panel)."""
        height, width = image.shape[:2]
        points = [
            (int(p.x * width), int(p.y * height)) for p in hand.landmark
        ]
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(image, points[start], points[end], color, 1, cv2.LINE_AA)
        for point in points:
            cv2.circle(image, point, 2, (255, 255, 255), -1, cv2.LINE_AA)

    # ---------------------------------------------------------------- close
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resource in (self._legacy, self._tasks):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception:  # pragma: no cover - teardown noise
                pass
        self._legacy = None
        self._tasks = None

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - safety net
        try:
            self.close()
        except Exception:
            pass
