"""Rotation-invariant gesture recognition.

The previous implementation compared raw ``landmark.y`` values, so tilting or
rotating the hand broke every gesture.  This module instead works on 3D
distances between joints, normalised by palm size, which is invariant to
rotation, distance from the camera, and hand size.

Landmark x/y arrive normalised to *image* dimensions, so they are pre-scaled by
the frame aspect ratio; otherwise a 16:9 frame stretches the hand horizontally
and skews every distance ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# --- MediaPipe hand landmark indices ------------------------------------
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# (mcp, pip, tip) for index, middle, ring, pinky
_FINGERS: tuple[tuple[int, int, int], ...] = (
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)

_EXTENDED_RATIO = 1.16  # |wrist->tip| / |wrist->pip| when a finger is straight
_CURLED_RATIO = 1.02
# Thresholds calibrated against MediaPipe reference hands: fingertips actually
# touching measure 0.05-0.25, while an open hand with the thumb resting near
# the index measures ~0.41, so 0.32 separates them with margin either way.
_PINCH_ON = 0.32  # |thumb_tip - index_tip| / palm
_PINCH_OFF = 0.45  # hysteresis: wider threshold to release a pinch
# In a closed fist the thumb also lands near the index tip, so a pinch
# additionally requires the index finger to be reaching away from the wrist.
_PINCH_INDEX_REACH = 1.25  # |wrist - index_tip| / |wrist - index_mcp|


class Gesture(Enum):
    """What the hand is currently asking the app to do."""

    NONE = "none"
    DRAW = "draw"
    ERASE = "erase"
    SAVE = "save"
    NAVIGATE = "navigate"

    @property
    def label(self) -> str:
        return {
            Gesture.NONE: "Idle",
            Gesture.DRAW: "Drawing",
            Gesture.ERASE: "Erasing",
            Gesture.SAVE: "Hold to save",
            Gesture.NAVIGATE: "Swipe to turn page",
        }[self]


@dataclass(frozen=True)
class HandReading:
    """One analysed frame of hand data."""

    gesture: Gesture
    pointer: tuple[float, float]  # index tip, normalised frame coords (0..1)
    palm_size: float
    fingers_extended: tuple[bool, bool, bool, bool]
    thumb_extended: bool
    pinching: bool


def _to_array(landmarks, aspect: float) -> np.ndarray:
    """(21, 3) float array in aspect-corrected, palm-normalisable units."""
    points = np.empty((len(landmarks.landmark), 3), dtype=np.float32)
    for i, lm in enumerate(landmarks.landmark):
        points[i] = (lm.x * aspect, lm.y, lm.z * aspect)
    return points


def _distance(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a] - points[b]))


class GestureRecognizer:
    """Classifies a landmark set into a :class:`Gesture`.

    Keeps a single bit of history - whether a pinch is currently held - so the
    pinch threshold has hysteresis and drawing does not stutter.
    """

    def __init__(self) -> None:
        self._pinch_held = False

    def reset(self) -> None:
        self._pinch_held = False

    # ---------------------------------------------------------------- helpers
    def _is_extended(self, points: np.ndarray, finger: tuple[int, int, int]) -> bool:
        """A finger is extended when its tip is further from the wrist than its
        middle joint - true from any hand orientation."""
        _, pip, tip = finger
        pip_distance = _distance(points, WRIST, pip)
        if pip_distance <= 1e-6:
            return False
        return _distance(points, WRIST, tip) / pip_distance > _EXTENDED_RATIO

    def _is_curled(self, points: np.ndarray, finger: tuple[int, int, int]) -> bool:
        _, pip, tip = finger
        pip_distance = _distance(points, WRIST, pip)
        if pip_distance <= 1e-6:
            return False
        return _distance(points, WRIST, tip) / pip_distance < _CURLED_RATIO

    def _thumb_extended(self, points: np.ndarray, palm: float) -> bool:
        if palm <= 1e-6:
            return False
        reach = _distance(points, WRIST, THUMB_TIP) / max(
            _distance(points, WRIST, THUMB_IP), 1e-6
        )
        spread = _distance(points, THUMB_TIP, INDEX_MCP) / palm
        return reach > 1.10 and spread > 0.55

    # ------------------------------------------------------------------ main
    def analyse(self, landmarks, aspect: float) -> HandReading:
        points = _to_array(landmarks, aspect)

        # Palm size: wrist -> middle knuckle. Rotation invariant, scales with
        # distance from the camera, so every ratio below is scale free.
        palm = _distance(points, WRIST, MIDDLE_MCP)
        if palm <= 1e-6:
            palm = 1e-6

        pinch_distance = _distance(points, THUMB_TIP, INDEX_TIP) / palm
        threshold = _PINCH_OFF if self._pinch_held else _PINCH_ON
        index_reach = _distance(points, WRIST, INDEX_TIP) / max(
            _distance(points, WRIST, INDEX_MCP), 1e-6
        )
        pinching = pinch_distance < threshold and index_reach > _PINCH_INDEX_REACH
        self._pinch_held = pinching

        extended = tuple(self._is_extended(points, f) for f in _FINGERS)
        curled = tuple(self._is_curled(points, f) for f in _FINGERS)
        thumb_out = self._thumb_extended(points, palm)

        index_tip = landmarks.landmark[INDEX_TIP]
        pointer = (float(index_tip.x), float(index_tip.y))

        gesture = self._classify(pinching, extended, curled, thumb_out)

        return HandReading(
            gesture=gesture,
            pointer=pointer,
            palm_size=palm,
            fingers_extended=extended,  # type: ignore[arg-type]
            thumb_extended=thumb_out,
            pinching=pinching,
        )

    @staticmethod
    def _classify(
        pinching: bool,
        extended: tuple[bool, ...],
        curled: tuple[bool, ...],
        thumb_out: bool,
    ) -> Gesture:
        index_ext, middle_ext, ring_ext, pinky_ext = extended
        others_extended = middle_ext and ring_ext and pinky_ext

        # OK sign: thumb and index pinched while the other three stay straight.
        if pinching and others_extended:
            return Gesture.SAVE
        # Pinch with the hand closed up: pen down.
        if pinching:
            return Gesture.DRAW
        # Open palm: eraser.
        if index_ext and others_extended:
            return Gesture.ERASE
        # Index only: page navigation.
        if index_ext and curled[1] and curled[2] and curled[3] and not thumb_out:
            return Gesture.NAVIGATE
        return Gesture.NONE


class GestureStabilizer:
    """Debounces raw per-frame classifications.

    A new gesture must be seen ``patience`` frames in a row before it is
    committed, which removes the single-frame flicker that used to drop strokes.
    """

    def __init__(self, patience: int = 2) -> None:
        self.patience = max(1, patience)
        self._committed = Gesture.NONE
        self._candidate = Gesture.NONE
        self._streak = 0

    @property
    def current(self) -> Gesture:
        return self._committed

    def update(self, gesture: Gesture) -> Gesture:
        if gesture == self._committed:
            self._candidate = gesture
            self._streak = 0
            return self._committed
        if gesture == self._candidate:
            self._streak += 1
        else:
            self._candidate = gesture
            self._streak = 1
        if self._streak >= self.patience:
            self._committed = gesture
            self._streak = 0
        return self._committed

    def reset(self) -> None:
        self._committed = Gesture.NONE
        self._candidate = Gesture.NONE
        self._streak = 0
