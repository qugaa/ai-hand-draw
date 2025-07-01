# ai-hand-draw

Welcome to the **PDF Gesture Annotation Tool**, where annotating PDF pages is as simple as waving your hand. No more fumbling with menus or hunting for the right button—just point, pinch, or swipe in front of your webcam and let the magic happen.

---

## ✨ Why You’ll Love It

- **Hands‑free workflow**: Draw, erase, navigate, and save—without touching your mouse or keyboard.
- **Intuitive gestures**: Pinch to draw, open palm to erase, swipe to turn pages, and hold an OK sign to save. It’s like using a digital pen, only cooler.
- **Works on any PDF**: Whether you’re marking up lecture slides, reviewing contracts, or sketching ideas on your favorite eBook, it’s ready.

---

## 🎯 Key Features

- **Draw Anywhere**: Pinch your index finger and thumb together, then move to sketch lines or dots on the page.
- **Erase Easily**: Open your palm and hover over any marks to erase them like a digital eraser.
- **Navigate Pages**: Swipe left or right in the air to flip through your document.
- **Save with a Gesture**: Form an OK sign and hold for two seconds to snapshot your annotated page.
- **Color Picker & Clear**: Pinch on the on‑screen color buttons to switch pen colors or clear the page entirely.

---

## 🛠️ Installation

1. **Clone this repository**

```bash
git clone https://https://github.com/qugaa/ai-hand-draw.git
cd ai-hand-draw
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
```

> On Windows, you may also need to install poppler utilities for `pdf2image`. See its documentation for installation steps.

### Run the Tool

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

1. **Launch the GUI launcher**

   ```bash
   python gui.py
   ```

2. Click **Open & Annotate PDF** and choose your file.

3. The annotation window will open—start waving your hand:

   - **Pinch** to draw or select a button.
   - **Open palm** to erase.
   - **Swipe** left/right to change pages.
   - **Hold OK sign** for two seconds to save a snapshot.

4. To exit, press **Esc** or **‘q’**, or click the window’s close button.

---

## 🤝 Contributing

We’d love your help!\
Whether it’s improving gesture accuracy, adding new features, or polishing the UI, your pull requests are welcome.

1. Fork the repo.
2. Create a feature branch: `git checkout -b my-new-feature`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin my-new-feature`.
5. Open a Pull Request.

---

Thanks for trying out the PDF Gesture Annotation Tool! 🎉\
If you run into any issues or have feedback, drop an issue on GitHub or say hi in the discussion board. Happy annotating!

