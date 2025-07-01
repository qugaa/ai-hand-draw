# -*- coding: utf-8 -*-
"""
pdf_loader.py: Handles file selection and conversion of PDF pages
into OpenCV-compatible images and initializes blank annotation canvases.
"""

# --- Standard library imports ---
from tkinter import Tk, filedialog  # Use Tkinter for a native file open dialog

# --- Third-party library imports ---
from pdf2image import convert_from_path  # Convert PDF pages into PIL images
import numpy as np                       # NumPy for array operations (image data handling)
import cv2                               # OpenCV for image format conversion

# --- Import config module for shared state ---
import config

def load_pdf():
    """
    Prompts the user to select a PDF file, then loads and prepares the PDF pages:
      1. Converts each page of the PDF to an OpenCV BGR image.
      2. Initializes a blank canvas (annotation layer) for each page.
      3. Resets current_page index to 0 (first page).
    
    Updates global state in the config module. Returns True if a PDF was loaded, or False if no file was selected.
    """
    # Hide the root Tk window (we only want the file dialog to appear)
    root = Tk()
    root.withdraw()  # Prevent the root window from appearing

    # Open a file dialog for the user to choose a PDF file
    file_path = filedialog.askopenfilename(
        title="Select PDF to annotate",
        filetypes=[("PDF Files", "*.pdf")]
    )
    if not file_path:
        # If the user closed the dialog or pressed cancel, no file was selected
        print("No PDF selected.")  # Notify in console that no file was chosen
        return False               # Return False to indicate that loading was aborted

    # Convert PDF pages to PIL images at 150 DPI resolution
    pil_pages = convert_from_path(file_path, dpi=150)

    # Convert each PIL image to an OpenCV BGR image array and store in config.pdf_pages
    config.pdf_pages = [
        cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        for pil_image in pil_pages
    ]

    # Create a blank (black) canvas for each page, matching that page's dimensions
    template = np.zeros_like(config.pdf_pages[0])            # Black image same size as first page
    config.page_canvases = [template.copy() for _ in config.pdf_pages]  # One blank canvas per page

    # Reset the current page index to the first page
    config.current_page = 0

    # Reset drawing state defaults for the new PDF
    config.pen_color = (0, 0, 255)         # Revert pen color to default (red)
    config.prev_x, config.prev_y = None, None  # Clear any previous drawing positions

    return True  # Indicate that the PDF was successfully loaded and prepared
