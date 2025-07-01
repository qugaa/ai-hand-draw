# PDF Gesture Annotation Tool

Welcome to the **PDF Gesture Annotation Tool**, a Python application that lets you annotate PDF documents using intuitive hand gestures via your webcam. Draw, erase, change colors, clear pages, save your work, and navigate between pages—all without touching the keyboard or mouse!

---

## 🔍 Features

- **Gesture‑Driven Drawing**: Pinch (thumb + index) to draw, with smoothed strokes using a deque buffer of recent fingertip positions.
- **Erase Mode**: Open hand (four or more fingers extended) to erase with a large circular brush.
- **Gesture Buttons**: Tap (pen-up → pinch) over on-screen color/clear buttons to switch pen color or clear the current page:
  - **Red**, **Blue**, **Black**, and **Clear**.
- **Save with OK Sign**: Hold the “OK” gesture (pinch + other three fingers extended) for 2 seconds to save the current annotated page as a timestamped PNG on your Desktop.
- **Swipe Navigation**: Swipe left/right with an open hand to move forward/back between PDF pages.
- **Responsive Canvas Overlay**: Drawings are overlaid on PDF pages using a binary mask, ensuring true-black strokes and perfect layering.
- **File Picker**: Choose any PDF via a standard Tkinter file dialog; all pages are converted to images at 150 DPI.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **pip** package manager

### Install Dependencies

Clone the repo and install the required packages:

```bash
git clone https://github.com/yourusername/pdf-gesture-annotation.git
cd pdf-gesture-annotation
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```
mediapipe
opencv-python
numpy
pdf2image
pillow
PyMuPDF
PySimpleGUI

```

> On Windows, you may also need to install poppler utilities for `pdf2image`. See its documentation for installation steps.

### Run the Tool

```bash
python main.py
```

1. A file dialog will appear. Select the PDF you want to annotate.
2. A window titled **"PDF Annotation"** will open, showing the first page overlaid with on-screen buttons.
3. Use gestures to annotate:
   - **Pinch**: Draw
   - **Open Hand**: Erase
   - **Pinch over Button**: Change pen color or clear canvas
   - **Hold OK Sign**: Save annotated page
   - **Swipe Left/Right**: Change pages
4. Press `` at any time to quit.

---

## 🖐️ Gesture Controls Cheat Sheet

| Gesture          | Action                                          |
| ---------------- | ----------------------------------------------- |
| Pinch (👆+👌)    | Draw (smoothed line)                            |
| Open Hand (🖐️)  | Erase with circular brush                       |
| Pen-Up → Pinch   | Tap button under cursor to switch color / clear |
| Hold OK (👌 + 3) | Save current page as PNG after 2s hold          |
| Swipe Left/Right | Navigate to next/previous PDF page              |
| Keyboard `q`     | Quit the application                            |

---

## ⚙️ Configuration

- **SMOOTHING\_WINDOW**: Number of frames to average for smoothing strokes (default: 5).
- **BUTTON COLORS & POSITIONS**: Modify the `labels` list in `main.py` to add or change buttons.
- **Save Directory**: By default, pages are saved to your Desktop. Change `save_dir` in `drawing()` to customize.
- **Save Hold Duration**: Adjust `HOLD_DURATION` if you want a shorter/longer OK‑hold.

---

## 🛠️ Troubleshooting

- **Black strokes not appearing?** Ensure you’re using the separate mask logic or switch your “Black” button to a near‑black RGB like `(10, 10, 10)`.
- **Poppler errors on PDF conversion**? Install poppler and add it to your PATH.
- **Webcam not detected**? Verify your camera index (0, 1, etc.) in `cv2.VideoCapture()`.
- **Gesture not recognized reliably**? Tweak the threshold constants (`rel_thresh`, `min_detection_confidence`, `min_tracking_confidence`).

---

## 📝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---


---

*Made with ❤️ using MediaPipe, OpenCV, and pure Python.*

