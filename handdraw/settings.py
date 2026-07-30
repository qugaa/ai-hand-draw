"""Immutable configuration objects.

This module replaces the old ``config.py`` global-mutable-state module.  It
holds *values only*: importing it never touches the camera, mediapipe, or the
filesystem, so the app starts instantly and unit tests stay hermetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# All colours are BGR because they are consumed by OpenCV.
BGR = tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    """Colours and metrics for the on-screen (OpenCV) interface."""

    background: BGR = (18, 16, 15)
    surface: BGR = (34, 30, 28)
    surface_strong: BGR = (48, 43, 40)
    outline: BGR = (86, 78, 72)
    text: BGR = (245, 245, 245)
    text_muted: BGR = (176, 170, 166)
    accent: BGR = (62, 154, 240)  # amber #F09A3E, matching the launcher
    danger: BGR = (72, 72, 235)
    success: BGR = (120, 214, 122)
    cursor: BGR = (140, 255, 120)

    panel_alpha: float = 0.78
    corner_radius: int = 14


@dataclass(frozen=True)
class PenPreset:
    """A selectable pen colour."""

    key: str
    label: str
    color: BGR


PEN_PRESETS: tuple[PenPreset, ...] = (
    PenPreset("red", "Red", (60, 60, 235)),
    PenPreset("blue", "Blue", (235, 150, 60)),
    PenPreset("green", "Green", (90, 200, 90)),
    PenPreset("ink", "Ink", (24, 24, 24)),
)


@dataclass(frozen=True)
class AppSettings:
    """Everything tunable, in one immutable value object.

    Use :meth:`with_changes` to derive a modified copy; nothing mutates a
    settings instance in place, which keeps the session loop deterministic.
    """

    # --- source document -------------------------------------------------
    pdf_path: Path | None = None
    render_dpi: int = 150
    max_page_pixels: int = 2600  # longest page edge after rendering

    # --- camera ----------------------------------------------------------
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    mirror: bool = True

    # --- hand tracking ---------------------------------------------------
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.6
    detection_width: int = 480  # frames are downscaled to this before inference
    model_complexity: int = 0  # 0 = fastest, 1 = more accurate

    # --- pointer ---------------------------------------------------------
    mapping_margin: float = 0.86  # fraction of the frame usable for pointing
    smoothing_min: float = 0.18  # EMA alpha when the hand is still
    smoothing_max: float = 0.85  # EMA alpha when the hand moves fast
    smoothing_speed: float = 0.045  # normalised speed reaching smoothing_max

    # --- tools -----------------------------------------------------------
    pen_width: int = 8
    eraser_radius: int = 34
    gesture_stability_frames: int = 2
    save_hold_seconds: float = 1.2
    swipe_distance: float = 0.22  # fraction of the mapping zone width
    swipe_window: float = 0.8  # seconds
    swipe_cooldown: float = 0.8  # seconds

    # --- output ----------------------------------------------------------
    output_dir: Path | None = None

    # --- presentation ----------------------------------------------------
    show_preview: bool = True
    show_help: bool = True
    theme: Theme = field(default_factory=Theme)

    def with_changes(self, **kwargs: object) -> AppSettings:
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)  # type: ignore[arg-type]
