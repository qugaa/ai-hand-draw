"""Cross-platform output-path resolution.

The old code hard-coded ``~/Desktop``, which does not exist on many Windows
installs (OneDrive redirects it, some locales rename it) and is wrong on Linux.
Here the real Desktop is asked for via the Win32 known-folder API, and every
candidate is probed for writability before being handed out.
"""

from __future__ import annotations

import ctypes
import datetime as _dt
import os
import sys
import tempfile
from pathlib import Path

APP_FOLDER_NAME = "AI Hand Draw"

_FOLDERID_DESKTOP = "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}"
_FOLDERID_DOCUMENTS = "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}"


def _known_folder(folder_id: str) -> Path | None:
    """Resolve a Windows known folder, following OneDrive redirection."""
    if sys.platform != "win32":
        return None
    try:

        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_uint32),
                ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        ole32 = ctypes.windll.ole32  # type: ignore[attr-defined]
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]

        guid = _GUID()
        if ole32.CLSIDFromString(ctypes.c_wchar_p(folder_id), ctypes.byref(guid)) != 0:
            return None

        out = ctypes.c_wchar_p()
        if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out)) != 0:
            return None
        try:
            return Path(out.value) if out.value else None
        finally:
            ole32.CoTaskMemFree(out)
    except Exception:  # pragma: no cover - defensive, any COM failure is non-fatal
        return None


def desktop_dir() -> Path | None:
    """Best guess at the user's Desktop, or None when there isn't one."""
    known = _known_folder(_FOLDERID_DESKTOP)
    if known and known.is_dir():
        return known
    fallback = Path.home() / "Desktop"
    return fallback if fallback.is_dir() else None


def documents_dir() -> Path | None:
    known = _known_folder(_FOLDERID_DOCUMENTS)
    if known and known.is_dir():
        return known
    fallback = Path.home() / "Documents"
    return fallback if fallback.is_dir() else None


def is_writable(directory: Path) -> bool:
    """True when a file can actually be created inside ``directory``."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    probe = directory / f".handdraw-write-test-{os.getpid()}"
    try:
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def default_output_dir() -> Path:
    """First writable location among Desktop, Documents, home, temp."""
    candidates: list[Path] = []
    for base in (desktop_dir(), documents_dir(), Path.home()):
        if base is not None:
            candidates.append(base / APP_FOLDER_NAME)
    candidates.append(Path(tempfile.gettempdir()) / APP_FOLDER_NAME)

    for candidate in candidates:
        if is_writable(candidate):
            return candidate
    # Every candidate failed; hand back temp and let the caller report the error.
    return Path(tempfile.gettempdir()) / APP_FOLDER_NAME


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """A non-colliding path inside ``directory``."""
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def safe_stem(name: str, fallback: str = "document") -> str:
    """Strip characters Windows refuses in filenames."""
    cleaned = "".join("-" if ch in '<>:"/\\|?*' else ch for ch in name).strip(" .")
    cleaned = cleaned[:60].strip()
    return cleaned or fallback
