# AI Hand Draw

Annotate any PDF in mid‑air. Pinch to draw, open your palm to erase, point and swipe to turn the
page, hold an OK sign to save. Your webcam is the pen.

---

## ✨ Highlights

- **Hands‑free workflow** — draw, erase, navigate and save without touching mouse or keyboard.
- **Rotation‑invariant gestures** — tilt or turn your hand however you like; recognition uses 3D
  joint geometry normalised by palm size, not raw screen coordinates.
- **No distortion** — a mapping zone matching the page's aspect ratio is carved out of the camera
  frame, so a circle drawn in the air is a circle on the page.
- **No Poppler, no external binaries** — pages are rendered by PyMuPDF.
- **Dark, modern UI** — a CustomTkinter launcher plus a translucent on‑screen HUD that stays legible
  over dense documents and re‑lays itself out whenever you resize the window.
- **Exports** — PNG snapshots, or an annotated copy of the PDF where the original pages stay vector
  and searchable.

---

## 🎯 Gestures & keys

| Gesture | Action |
| --- | --- |
| Pinch (thumb + index) | Draw; pinch over a toolbar button to press it |
| Open palm | Erase |
| Index finger up, then swipe | Previous / next page |
| OK sign, held ~1.2 s | Save the current page as PNG (a ring shows the progress) |

| Key | Action |
| --- | --- |
| `S` / `D` | Save page as PNG / save annotated PDF |
| `C` | Clear the page |
| `N` / `P` | Next / previous page |
| `1`–`4` | Pen colour |
| `H` / `V` | Toggle help strip / camera preview |
| `Esc`, `Q`, or the window's ✕ | Quit |

---

## 🛠️ Installation

```bash
git clone https://github.com/qugaa/ai-hand-draw.git
cd ai-hand-draw
pip install -r requirements.txt
```

Requires Python 3.10+. Install `opencv-contrib-python` (pulled in by `mediapipe`) rather than
`opencv-python` — having both installed breaks `cv2`.

On MediaPipe 0.10.30 and newer the legacy `mp.solutions` graphs are gone, so the app uses the Tasks
API and downloads `hand_landmarker.task` (7.5 MB) once, into
`%LOCALAPPDATA%\AI Hand Draw\models` (or `~/.cache/AI Hand Draw/models`). To supply it yourself, drop
it in `models/` next to this README or point `HANDDRAW_HAND_MODEL` at it. Older MediaPipe releases
keep working through the legacy backend, with no model download.

---

## 🚀 Usage

```bash
python app.py                       # dark‑mode launcher
python app.py --pdf notes.pdf       # skip the launcher
python app.py --pdf notes.pdf --camera 1 --dpi 200 --output D:\scans --no-preview
```

Annotations are saved to a writable folder chosen at startup — your real Desktop (resolved through
the Windows known‑folder API, so OneDrive redirection is handled), else Documents, else your home
directory. Pick a different one in the launcher at any time.

---

## 🧭 Architecture

```
app.py                 entry point (launcher or --pdf headless)
handdraw/
  settings.py          immutable AppSettings + Theme; importing it touches no hardware
  paths.py             Desktop/Documents resolution, writability probing, safe filenames
  document.py          PyMuPDF rendering, LRU page cache, annotated‑PDF export
  camera.py            webcam ownership; context manager, deterministic release
  tracking.py          MediaPipe wrapper (Tasks API + legacy solutions), always closed
  models.py            one‑time model download with progress and atomic install
  gestures.py          rotation‑invariant recognition + temporal debouncing
  mapping.py           aspect‑correct camera→page zone, adaptive smoothing, viewport
  state.py             per‑page annotation layers with dirty‑region tracking
  overlay.py           translucent rounded HUD: toolbar, status, help, toasts, preview
  export.py            PNG and PDF output
  session.py           the annotation loop and all resource teardown
  ui/launcher.py       CustomTkinter dark launcher
```

Design notes:

- **No global mutable state.** Everything lives in `AppSettings` (frozen) and `SessionState`;
  a session can be created, run and discarded without leaving anything behind.
- **Nothing at import time.** The camera and MediaPipe are constructed inside the session and
  released in a `finally` block, so an exception can never leave the webcam locked.
- **One window.** The camera preview is a picture‑in‑picture panel, and the native ✕ button is
  honoured via `WND_PROP_VISIBLE`, so no window can be orphaned.
- **Incremental rendering.** Each page keeps one composited image, edited in place; only the
  changed region is rescaled per frame. Frames that change nothing cost nothing.

---

Found a bug or have an idea? Open an issue. Happy annotating! 🎉
