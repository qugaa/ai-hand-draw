# -*- coding: utf-8 -*-
"""
ui_helpers.py: Provides helper functions to define and draw UI buttons
and to handle saving the annotated pages.
"""

# --- Standard library imports ---
import os
import datetime

# --- Third-party imports ---
import cv2

# --- Import global state from config ---
import config  # Import the config module to access shared state variables (pdf_pages, page_canvases, current_page)

def get_button_specs(page_img_shape) -> list:
    """
    Computes the specifications for UI buttons (positions, labels, colors) based on the current page size.
    
    Returns:
        A list of dictionaries, each describing a button with keys:
        'label': text label of the button,
        'color': BGR color tuple for the button background,
        'pos': (x1, y1, x2, y2) coordinates defining the button rectangle on the image.
    """
    h, w = page_img_shape[:2]  # Height and width of the page image
    # Define button labels and their base colors (in BGR format)
    labels = [
        ("Red",   (0, 0, 255)),   # Red color button (for pen color)
        ("Blue",  (255, 0, 0)),   # Blue color button (for pen color)
        ("Black", (10, 10, 10)),  # Black color button (almost black)
        ("Clear", (50, 50, 50))   # Clear button (gray color, used to clear annotations)
    ]
    btn_size = 100    # Width and height of each square button in pixels
    spacing = 20      # Vertical spacing between buttons in pixels
    total_h = len(labels) * btn_size + (len(labels) - 1) * spacing  # Total vertical span of all buttons
    # Center the button stack vertically on the page:
    y_start = h // 2 - total_h // 2  # Starting y-coordinate for the first button
    x_start = 20                     # x-coordinate for all buttons (offset from left edge of page)
    specs = []
    for i, (lbl, col) in enumerate(labels):
        # Calculate the position of the button rectangle
        y1 = y_start + i * (btn_size + spacing)
        y2 = y1 + btn_size
        x1 = x_start
        x2 = x_start + btn_size
        specs.append({
            'label': lbl,
            'color': col,
            'pos': (x1, y1, x2, y2)
        })
    return specs

def draw_buttons(image, specs):
    """
    Draws the UI buttons defined in specs onto the given image.
    
    Each button is drawn with a semi-transparent colored background, a contrasting border, and centered text.
    
    Args:
        image: The OpenCV BGR image on which to draw the buttons (this should be the page image with annotations).
        specs: List of button specification dictionaries (as returned by get_button_specs).
    """
    for btn in specs:
        x1, y1, x2, y2 = btn['pos']
        # Determine the region of the image corresponding to this button
        button_region = image[y1:y2, x1:x2]  # Region of interest (ROI) for the button on the image

        # Create an overlay of the button color (same size as the button region)
        overlay = button_region.copy()
        overlay[:, :] = btn['color']  # Fill overlay with the button's base color

        # Blend the overlay with the underlying image region to achieve transparency (40% color, 60% original)
        cv2.addWeighted(overlay, 0.4, button_region, 0.6, 0, button_region)

        # Decide border color for contrast: white border on dark buttons, black border on bright buttons
        color_bgr = btn['color']
        if max(color_bgr) - min(color_bgr) > 200:
            border_color = (0, 0, 0)    # High color contrast (saturated color) -> use black border
        else:
            border_color = (255, 255, 255)  # Low contrast or dark color -> use white border

        # Draw a rectangle border around the button (thickness=2, anti-aliased for smooth edges)
        cv2.rectangle(image, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)

        # Prepare the button label text
        label = btn['label']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 1
        # Measure the size of the text to center it within the button
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        # Compute coordinates so that the text is centered in the button region
        text_x = x1 + ( (x2 - x1) - text_width ) // 2
        text_y = y1 + ((y2 - y1) + text_height) // 2 - baseline // 2
        # Draw the text label on the button (white text for all buttons for consistency)
        cv2.putText(image, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def save_annotated_page():
    """
    Saves the currently displayed PDF page with its annotations as an image file (PNG) on the user's Desktop.
    
    The output filename includes the page number and a timestamp.
    """
    # Ensure there is a PDF loaded and a valid page to save
    if not config.pdf_pages or config.current_page >= len(config.pdf_pages):
        print("No PDF page available to save.")
        return

    # Construct a filepath on the user's Desktop with a timestamp
    desktop_dir = os.path.expanduser("~/Desktop")  # Path to Desktop
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Current time as YYYYMMDD_HHMMSS
    filename = f"annotated_page_{config.current_page + 1}_{timestamp}.png"
    save_path = os.path.join(desktop_dir, filename)

    # Retrieve the current page image and its annotation canvas
    page_img = config.pdf_pages[config.current_page]
    canvas = config.page_canvases[config.current_page]
    # Create a mask of the drawn areas on the canvas (where pixel value > 0)
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)  # mask: white (255) where canvas has drawings
    mask_inv = cv2.bitwise_not(mask)                                 # inverse mask: white where no drawing

    # Use the mask to combine the original page image and the annotations
    background = cv2.bitwise_and(page_img, page_img, mask=mask_inv)  # background where there are no annotations
    foreground = cv2.bitwise_and(canvas, canvas, mask=mask)          # the drawn annotation content
    combined = cv2.add(background, foreground)                       # overlay annotations on the page

    # Save the combined image to disk as a PNG file
    cv2.imwrite(save_path, combined)
    print(f"Saved annotated page to {save_path}")  # Console output confirming save location
