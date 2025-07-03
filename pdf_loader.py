from tkinter import Tk, filedialog
from pdf2image import convert_from_path
import numpy as np
import cv2
import config

def load_pdf():

    # Hide the root Tk window (we only want the file dialog to appear)
    root = Tk()
    root.withdraw()

    # Open a file dialog for the user to choose a PDF file
    file_path = filedialog.askopenfilename(
        title="Select PDF to annotate",
        filetypes=[("PDF Files", "*.pdf")]
    )
    if not file_path:
        # If the user closed the dialog or pressed cancel, no file was selected
        print("No PDF selected.")
        return False

    # Convert PDF pages to PIL images at 150 DPI resolution
    pil_pages = convert_from_path(file_path, dpi=150)

    # Convert each PIL image to an OpenCV BGR image array and store in config.pdf_pages
    config.pdf_pages = [
        cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        for pil_image in pil_pages
    ]

    # Create a blank canvas for each page, matching that page's dimensions
    template = np.zeros_like(config.pdf_pages[0])
    config.page_canvases = [template.copy() for _ in config.pdf_pages]

    # Reset the current page index to the first page
    config.current_page = 0

    # Reset drawing state defaults for the new PDF
    config.pen_color = (0, 0, 255)
    config.prev_x, config.prev_y = None, None

    return True
