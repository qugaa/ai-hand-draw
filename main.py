import os
import datetime
import time
import mediapipe as mp         # For hand tracking and gesture recognition
import cv2                     # For camera handling and image processing
import numpy as np             # For array/image manipulations
from collections import deque  # For smoothing finger movement
import fitz                    # PyMuPDF, for advanced PDF manipulation (not used for drawing, but can be extended)
from pdf2image import convert_from_path # For converting PDF pages into images

# Import Tkinter just for file dialog (lets user pick a PDF visually)
from tkinter import Tk, filedialog

# ----- GLOBAL CONFIGURATION AND VARIABLES -----

SMOOTHING_WINDOW = 5  # Number of fingertip points to average for smoother lines
point_buffer = deque(maxlen=SMOOTHING_WINDOW)  # Holds last N fingertip points

# Initialize Mediapipe's hand tracking solution
mp_hands = mp.solutions.hands  # type: ignore[attr-defined] getting rid of annoyind type checking error
hands = mp_hands.Hands(
    static_image_mode=False,      # Live video (not single image mode)
    max_num_hands=1,              # Track only one hand at a time
    min_detection_confidence=0.5, # Minimum confidence for detection
    min_tracking_confidence=0.5   # Minimum confidence for tracking
)
mp_draw = mp.solutions.drawing_utils # Utility to draw hand landmarks on frames #type: ignore[attr-defined] again, ignore type checking error

# OpenCV: Start webcam capture (device 0 is default camera)
cap = cv2.VideoCapture(0)

# Canvas: Will hold the user's drawing strokes (overlay)
canvas = None
prev_x, prev_y = None, None # To remember the previous finger position (for continuous lines)

# ----- PDF-related GLOBALS -----

pdf_pages = []        # Will store each PDF page as an image (list of np.arrays)
page_canvases = []    # For each PDF page, a corresponding blank canvas for drawings
current_page = 0      # Which page is currently being annotated/viewed

# ----- FILE PICKER AND PDF LOADING -----

def load_pdf():
    """
    Allows the user to pick a PDF file using a dialog, converts each page of that PDF to an image,
    and prepares a blank drawing canvas for each page.
    """
    global pdf_pages, page_canvases, current_page

    # Hide the main Tkinter window (we only want the file dialog)
    root = Tk()
    root.withdraw()

    # Open a file dialog, allowing user to choose a PDF file
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if not file_path:
        print("No PDF selected")
        exit()  # Quit the script if no file is chosen

    # Convert each page of the selected PDF into a PIL Image at 150 DPI (good balance quality/speed)
    pil_pages = convert_from_path(file_path, dpi=150)

    # Convert each PIL image to an OpenCV BGR image (for drawing and display)
    pdf_pages = [cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR) for pil in pil_pages]

    # For each page, create a blank (black) canvas for user's annotations (same size as the page)
    page_canvases = [np.zeros_like(pdf_pages[0]) for _ in pdf_pages]

    current_page = 0  # Start at the first page

# ----- HAND GESTURE UTILITIES -----

def is_hand_open(hand_landmarks):
    """
    Returns True if most fingers are extended (open hand), False otherwise.
    Used for erasing gesture.
    """
    finger_tips = [8, 12, 16, 20]     # Landmark indices for finger tips
    finger_mcps = [5, 9, 13, 17]      # Landmark indices for finger MCPs (knuckles/base)

    open_fingers = 0
    for tip_id, mcp_id in zip(finger_tips, finger_mcps):
        tip = hand_landmarks.landmark[tip_id]
        mcp = hand_landmarks.landmark[mcp_id]
        dy = mcp.y - tip.y           # Finger is "up" if tip is above mcp (y-axis, normalized coords)
        dz = abs(mcp.z - tip.z)      # Small difference means finger is not tilted much

        if dy > 0.01 or dz < 0.02:
            open_fingers += 1
    return open_fingers >= 4  # True if at least 4 fingers open

def is_pinch(hand_pos, rel_thresh=0.35):
    """
    Returns True if index finger tip and thumb tip are close together (pinch gesture), False otherwise.
    Used for "pen down" (drawing).
    """
    it = hand_pos.landmark[8]   # Index fingertip
    tt = hand_pos.landmark[4]   # Thumb tip

    dx = it.x - tt.x
    dy = it.y - tt.y
    pinch_dist_2d = (dx*dx + dy*dy) ** 0.5

    # Hand size normalization: distance from wrist (0) to middle finger MCP (9)
    w = hand_pos.landmark[0]
    m = hand_pos.landmark[9]
    hx = w.x - m.x
    hy = w.y - m.y
    hand_size_2d = (hx*hx + hy*hy) ** 0.5

    return pinch_dist_2d < (hand_size_2d * rel_thresh)

def is_ok_sign(hand_pos, rel_thresh=0.35):
    """
    Returns True if the "OK" sign is shown: index and thumb touch, other fingers extended.
    Used to trigger save action.
    """
    # Must be pinched
    if not is_pinch(hand_pos, rel_thresh):
        return False

    # Check if other three fingers are extended
    other_tips = [12, 16, 20]
    other_mcps = [9, 13, 17]
    open_count = 0
    for tip_id, mcp_id in zip(other_tips, other_mcps):
        tip = hand_pos.landmark[tip_id]
        mcp = hand_pos.landmark[mcp_id]
        if mcp.y - tip.y > 0.01:
            open_count += 1
    return open_count >= 3  # True if middle, ring, pinky are open

# ----- MAIN DRAWING/ANNOTATION LOOP -----

def drawing():
    """
    Main loop for live hand gesture recognition, annotation, erasing, and navigation.
    Now also shows a fingertip cursor and ensures layering is always correct.
    """
    global canvas, prev_x, prev_y, point_buffer, pdf_pages, page_canvases, current_page

    ok_saved     = False          # Prevents multiple saves per OK gesture
    pen_color    = (0, 0, 255)    # initial pen color Red
    cursor_color = (0, 255, 0)    # Green for fingertip cursor
    # ——— gesture-click state ———
    prev_gesture = "pen_up"       # to detect pen-up → pinch transitions
    click_feedback = ""           # feedback string shown on screen

    from collections import deque
    SWIPE_BUFFER_SIZE     = 16
    SWIPE_THRESHOLD_PX    = 400
    SWIPE_COOLDOWN_FRAMES = 20
    swipe_buffer   = deque(maxlen=SWIPE_BUFFER_SIZE)
    swipe_cooldown = 0

    # --- OK-sign hold detection setup ---
    hold_start          = None
    HOLD_DURATION       = 2.0
    HOLD_FAULT_TOLERANCE = 0.2

    #  make window resizable by the user
    cv2.namedWindow("PDF Annotation", cv2.WINDOW_NORMAL)

    #  size the window to the first page
    page_img = pdf_pages[current_page]
    h, w     = page_img.shape[:2]
    cv2.resizeWindow("PDF Annotation", w, h)

    # ——— DEFINE YOUR BUTTONS ONCE, BASED ON PAGE SIZE ———
    labels = [("Red",   (0,0,255)),
             ("Blue",  (255,0,0)),
             ("Black", (10,10,10)),   # “almost black,” but will pass gray>1
             ("Clear", (50,50,50))]
    button_size = 100
    spacing     = 20
    total_h     = len(labels)*button_size + (len(labels)-1)*spacing
    y_start     = h//2 - total_h//2
    x1_left     = 20

    button_specs = []
    for idx, (lbl, col) in enumerate(labels):
        y1 = y_start + idx*(button_size + spacing)
        button_specs.append({
            "label": lbl,
            "color": col,
            "pos":   (x1_left, y1, x1_left+button_size, y1+button_size)
        })

    # track previous gesture state for click detection
    prev_gesture = "pen_up"

    while True:
        exists_frame, frame = cap.read()
        if not exists_frame:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_where = hands.process(rgb)

        # === CHANGE: Always use the current PDF page and its canvas ===
        # This ensures you never draw behind or "lose" your drawing.
        # canvas is always the one associated with the current PDF page.
        if pdf_pages:
            page_img = pdf_pages[current_page].copy()
            canvas = page_canvases[current_page]

        else:
            # If no PDF loaded (shouldn't happen in normal use), use a dummy frame.
            if canvas is None:
                canvas = np.zeros_like(frame)
            page_img = np.zeros_like(frame)
        
        if canvas.shape != page_img.shape:
            canvas = cv2.resize(
                canvas,
                (page_img.shape[1], page_img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        


        # Prepare the mode text to display on the screen
        mode_text = f"Page: {current_page+1}/{len(pdf_pages)}  "

        # This flag enables the cursor drawing logic
        draw_cursor = False
        avg_x, avg_y = 0, 0  # Defaults if no hand is detected
        ok_now = False # Tracks whether OK sign is currently detected

        if hands_where.multi_hand_landmarks:
            hands_exactly_there = hands_where.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hands_exactly_there, mp_hands.HAND_CONNECTIONS)

            # === CHANGE: Use PDF's page size for all coordinates ===
            # Fixes bugs with different webcam/PDF sizes!
            frame_h, frame_w, _ = page_img.shape
            index_finger = hands_exactly_there.landmark[8]
            cx, cy = int(index_finger.x * frame_w), int(index_finger.y * frame_h)

            # === CHANGE: Always smooth fingertip motion for less jittery drawing ===
            point_buffer.append((cx, cy))
            avg_x = int(sum(p[0] for p in point_buffer) / len(point_buffer))
            avg_y = int(sum(p[1] for p in point_buffer) / len(point_buffer))
            draw_cursor = True  # Only show cursor when a hand is detected

            # === CHANGE: All drawing/writing happens on the canvas for the current page ===
            if is_ok_sign(hands_exactly_there):
                 ok_now = True
                 mode_text += "OK held"
            elif is_pinch(hands_exactly_there):
                hold_start = None
                ok_saved   = False
                mode_text += "Drawing"

                # DETECT a “click” when going from pen_up → drawing over a button:
                cur_gesture = "drawing"
                if prev_gesture == "pen_up":
                    for btn in button_specs:
                        x1,y1,x2,y2 = btn["pos"]
                        if x1 <= avg_x <= x2 and y1 <= avg_y <= y2:
                            # perform button action
                            click_feedback = f"Clicked: {btn['label']}"
                            if btn["label"] == "Clear":
                                page_canvases[current_page][:] = 0
                            else:
                                pen_color = btn["color"]
                            # skip drawing this frame
                            prev_x, prev_y = None, None
                            break
                    else:
                        # no button hit → normal drawing
                        if prev_x is not None and prev_y is not None:
                            cv2.line(canvas, (prev_x, prev_y), (avg_x, avg_y), pen_color, 5)
                        else:
                            cv2.circle(canvas, (avg_x, avg_y), 8, pen_color, -1)
                        prev_x, prev_y = avg_x, avg_y
                else:
                    # continued pinch (holding down): normal drawing
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (avg_x, avg_y), pen_color, 5)
                    else:
                        cv2.circle(canvas, (avg_x, avg_y), 8, pen_color, -1)
                    prev_x, prev_y = avg_x, avg_y

                prev_gesture = cur_gesture


            elif is_hand_open(hands_exactly_there):
                # Erase with a big black circle
                cv2.circle(canvas, (avg_x, avg_y), 30, (0, 0, 0), -1)
                prev_x, prev_y = None, None
                prev_gesture = "pen_up"
                ok_saved = False
                mode_text += "Erasing"
            else:
                # Hand is not in any active drawing state
                prev_x, prev_y = None, None
                prev_gesture = "pen_up"
                ok_saved = False
                mode_text += "Pen Up"
            # ——— Timed OK‐hold save logic ———
            now = time.time()
            if ok_now:
                if hold_start is None:
                    hold_start = now
                elif (now - hold_start + HOLD_FAULT_TOLERANCE) >= HOLD_DURATION and not ok_saved:
                    # 1) Build a timestamped filename
                    save_dir = os.path.expanduser("~/Desktop")
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pdf_annotated_page{current_page+1}_{ts}.png"
                    path = os.path.join(save_dir, filename)

                    # 2) Compose overlay and save exactly what’s on screen
                    overlay = page_img.copy()
                    gray_tmp = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                    _, mask_tmp = cv2.threshold(gray_tmp, 1, 255, cv2.THRESH_BINARY)
                    bg_tmp = cv2.bitwise_and(overlay, overlay, mask=cv2.bitwise_not(mask_tmp))
                    fg_tmp = cv2.bitwise_and(canvas,   canvas,   mask=mask_tmp)
                    cv2.imwrite(path, cv2.add(bg_tmp, fg_tmp))

                    ok_saved = True
                    mode_text += f" Saved {filename}"
            else:
                hold_start = None

        # If the cursor is visible (i.e., we have a tracked fingertip), add its X position to the swipe buffer.
        # Otherwise, clear the buffer so we only detect swipes when the hand is present.
        if hands_where.multi_hand_landmarks and is_hand_open(hands_exactly_there):
            swipe_buffer.append(avg_x)
        else:
            swipe_buffer.clear()

        # Once the buffer is full and we're not in a cooldown period, check for a horizontal swipe.
        if swipe_cooldown == 0 and len(swipe_buffer) == SWIPE_BUFFER_SIZE:
            # Compute the net movement from the oldest to the newest point
            dx = swipe_buffer[-1] - swipe_buffer[0]
             # If the movement exceeds our threshold, count it as a swipe
            if abs(dx) > SWIPE_THRESHOLD_PX:
                if dx < 0:
                   # Negative dx → finger moved left → go to next page
                    current_page = (current_page + 1) % len(pdf_pages)
                else:
                    # Positive dx → finger moved right → go to previous page
                    current_page = (current_page - 1) % len(pdf_pages)
                # Reset drawing state so you don’t continue drawing when you change page
                prev_x, prev_y = None, None
                # Start a short cooldown to avoid multiple flips from one swipe
                swipe_cooldown = SWIPE_COOLDOWN_FRAMES
                swipe_buffer.clear()  # Clear the buffer so you start fresh after the swipe

        # If we’re in a cooldown, decrement it each frame until it reaches zero
        if swipe_cooldown > 0:
            swipe_cooldown -= 1


        # Use a mask-overlay instead of raw addWeighted to draw on the PDF page
        # Make a copy of the PDF page
        output = page_img.copy()

        # 1) Create a binary mask of where the canvas is non-zero
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # 2) Resize mask (and its inverse) to match output dimensions
        mask     = cv2.resize(mask,     (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_inv = cv2.resize(255 - mask, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 3) Apply bitwise-and overlays
        bg     = cv2.bitwise_and(output, output, mask=mask_inv)
        fg     = cv2.bitwise_and(canvas, canvas,   mask=mask)
        output = cv2.add(bg, fg)

         # ——— compute mode-text size & top-right position ———
        (text_size, _) = cv2.getTextSize(
            mode_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,    # font scale
            2       # thickness
        )
        tw, th = text_size
        x_mode = output.shape[1] - tw - 10
        y_mode = 30

        # === NEW FEATURE: Draw a green fingertip cursor, so user always knows where writing will occur ===
        if draw_cursor:
            cv2.circle(output, (avg_x, avg_y), 15, cursor_color, 3)  # Large, green ring
            cv2.circle(output, (avg_x, avg_y), 4, cursor_color, -1)  # Small green dot

        # draw mode text top-right
        cv2.putText(
            output,
            mode_text,
            (x_mode, y_mode),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,                # font scale
            (0, 255, 0),        # green
            2,                  # thickness
            cv2.LINE_AA
        )

        # draw click-feedback just below the mode text
        if click_feedback:
            cv2.putText(
                output,
                click_feedback,
                (x_mode, y_mode + th + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,                # slightly smaller
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # draw UI buttons on top
        for btn in button_specs:
            x1, y1, x2, y2 = btn["pos"]
            # filled box
            cv2.rectangle(output, (x1, y1), (x2, y2), btn["color"], -1)
            # label in white
            cv2.putText(
                output,
                btn["label"],
                (x1 + 5, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow("PDF Annotation", output)


        # Key events: 'd' for next, 'a' for previous page, 'q' to quit
        # Added capital letters for convenience 
        # Key events: 'd' for next, 'a' for previous page, 'q' to quit
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break
        elif pdf_pages and key in (ord('d'), ord('D')):
            current_page = (current_page + 1) % len(pdf_pages)
            prev_x, prev_y = None, None
        elif pdf_pages and key in (ord('a'), ord('A')):
            current_page = (current_page - 1) % len(pdf_pages)
            prev_x, prev_y = None, None


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_pdf()   # Ask user for PDF, prepare images/canvases
    drawing()    # Enter main annotation loop

