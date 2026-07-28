"""Exception hierarchy shared by every subsystem.

Every recoverable failure is surfaced as an AppError so the GUI can show a
readable message instead of a traceback.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for user-facing errors."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        return self.message if not self.detail else f"{self.message}\n\n{self.detail}"


class DocumentError(AppError):
    """The PDF could not be opened or rendered."""


class CameraError(AppError):
    """The webcam could not be opened or stopped delivering frames."""


class StorageError(AppError):
    """Output files could not be written."""
