import os
import datetime
import cv2
import config

def get_button_specs(page_img_shape) -> list:

    h, w = page_img_shape[:2]
    labels = [
        ("Red",   (0, 0, 255)),
        ("Blue",  (255, 0, 0)),
        ("Black", (10, 10, 10)),
        ("Clear", (50, 50, 50))
    ]
    btn_size = 100
    spacing = 20
    total_h = len(labels) * btn_size + (len(labels) - 1) * spacing

    y_start = h // 2 - total_h // 2
    x_start = 20
    specs = []
    for i, (lbl, col) in enumerate(labels):

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
    for btn in specs:
        x1, y1, x2, y2 = btn['pos']
        button_region = image[y1:y2, x1:x2]

        overlay = button_region.copy()
        overlay[:, :] = btn['color']

        cv2.addWeighted(overlay, 0.4, button_region, 0.6, 0, button_region)

        color_bgr = btn['color']
        if max(color_bgr) - min(color_bgr) > 200:
            border_color = (0, 0, 0)
        else:
            border_color = (255, 255, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)

        label = btn['label']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        text_x = x1 + ( (x2 - x1) - text_width ) // 2
        text_y = y1 + ((y2 - y1) + text_height) // 2 - baseline // 2

        cv2.putText(image, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def save_annotated_page():

    # Ensure there is a PDF loaded and a valid page to save
    if not config.pdf_pages or config.current_page >= len(config.pdf_pages):
        print("No PDF page available to save.")
        return

    # Construct a filepath on the user's Desktop with a timestamp
    desktop_dir = os.path.expanduser("~/Desktop")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"annotated_page_{config.current_page + 1}_{timestamp}.png"
    save_path = os.path.join(desktop_dir, filename)

    # Retrieve the current page image and its annotation canvas
    page_img = config.pdf_pages[config.current_page]
    canvas = config.page_canvases[config.current_page]
    
    # Create a mask of the drawn areas on the canvas (where pixel value > 0)
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask) 

    # Use the mask to combine the original page image and the annotations
    background = cv2.bitwise_and(page_img, page_img, mask=mask_inv)
    foreground = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(background, foreground)

    cv2.imwrite(save_path, combined)
    print(f"Saved annotated page to {save_path}")
