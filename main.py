# -*- coding: utf-8 -*-
"""
main.py: Core application orchestrator for the PDF Gesture Annotation Tool.
Loads PDFs, initializes gesture/canvas state, and runs the main annotation loop.
Shared state (pages, canvases, etc.) is maintained in the config module.
"""

# --- Standard library imports ---
import time  # Time functions for measuring durations (e.g., OK gesture hold time)

# --- Third-party imports ---
import cv2             # OpenCV for video capture and image display
import numpy as np     # NumPy for image array operations

# --- Local module imports ---
import config                                   # Import the config module containing global state
from pdf_loader import load_pdf                 # Function to load PDF pages into memory
from gestures import is_hand_open, is_pinch, is_ok_sign, is_point_up  # Gesture detection helper functions
from ui_helpers import get_button_specs, draw_buttons, save_annotated_page  # UI helper functions

def run_annotation_loop():
    """
    Runs the main real-time annotation loop:
      1. Captures video frames from the webcam.
      2. Detects hand landmarks and identifies gestures.
      3. Draws or erases on the canvas based on gestures.
      4. Navigates pages when swipe gestures are detected.
      5. Saves the annotated page when an OK gesture is held.
      6. Overlays annotations and UI on the PDF page and displays it.
      7. Handle the OK-sign hold logic for saving the page
      8. Swipe detection for page navigation (point-and-swipe)
      9. Prepare the final image for display by combining the PDF page with the annotations
      10. Show the final composed image in the window
    """
    # Ensure a PDF is loaded before starting the loop
    if not config.pdf_pages:
        print("No PDF loaded. Exiting.")
        return

    # If the webcam is not open (e.g., after a previous run), open it
    if not config.cap.isOpened():
        config.cap = cv2.VideoCapture(0)
    if not config.cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    # Reset drawing state and buffers at the start of the loop
    config.prev_x, config.prev_y = None, None
    config.point_buffer.clear()

    # Prepare the display window (match window size to the first PDF page dimensions)
    cv2.namedWindow("PDF Annotation", cv2.WINDOW_NORMAL)
    page_h, page_w = config.pdf_pages[0].shape[:2]        # Height and width of the first page image
    cv2.resizeWindow("PDF Annotation", page_w, page_h)    # Set the OpenCV window to the PDF page size

    # Mirror preview of your webcam
    cv2.namedWindow("Webcam Preview", cv2.WINDOW_NORMAL)
    # Optional: force a compact preview size
    cv2.resizeWindow("Webcam Preview", 320, 240)

    # --- Initialize gesture state flags and variables ---
    ok_saved = False           # Indicates if a save action was performed after detecting OK gesture (to prevent multiple saves per hold)
    prev_gesture = "pen_up"    # Tracks the previous gesture state ("pen_up", "drawing", etc.)
    click_feedback = ""       # Text feedback for UI button clicks or page changes
    feedback_frames = 0       # Counter for how many frames to display the feedback text
    hold_start = None         # Start time when an OK-sign gesture is first detected (for save hold timing)
    HOLD_DURATION = 2.0       # Time in seconds to hold the OK gesture to trigger save
    HOLD_FAULT_TOLERANCE = 0.2# Additional time buffer to ensure reliable hold detection

    # --- Swipe gesture detection setup ---
    from collections import deque
    SWIPE_BUFFER_SIZE = 8    # Number of recent frames to consider for swipe movement
    SWIPE_THRESHOLD_PX = 200  # Minimum horizontal movement (in pixels) to qualify as a swipe
    SWIPE_COOLDOWN = 30       # Cooldown period (in frames) after a swipe to avoid immediate repeat
    swipe_buffer = deque(maxlen=SWIPE_BUFFER_SIZE)  # Buffer to store recent finger x-coordinates for swipe analysis
    swipe_cooldown = 0        # Counter for swipe cooldown frames remaining

    # --- Main loop for reading camera frames and handling gestures ---
    while True:
        # 1. Capture a frame from the webcam
        ret, frame = config.cap.read()
        if not ret:
            # If frame capture failed (camera disconnected or end of stream), exit the loop
            print("Camera capture failed or ended.")
            break
        frame = cv2.flip(frame, 1)  # Mirror the frame horizontally for natural interaction (like a mirror)        
        cv2.imshow("Webcam Preview", frame) # Show the raw webcam feed in its own window
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR frame to RGB for Mediapipe processing

        # 2. Hand landmark detection using Mediapipe
        results = config.hands.process(rgb_frame)  # Process the frame to detect hand landmarks

        # 3. Prepare the current PDF page image and its annotation canvas
        page_img = config.pdf_pages[config.current_page].copy()   # Copy of the current page (to draw annotations onto for display)
        canvas = config.page_canvases[config.current_page]        # Current annotation canvas for this page
        # Ensure the canvas matches the page dimensions (in case of page size differences)
        if canvas.shape[:2] != page_img.shape[:2]:
            # Resize the canvas to match the page, using nearest-neighbor for pixel alignment
            canvas = cv2.resize(canvas, (page_img.shape[1], page_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            config.page_canvases[config.current_page] = canvas  # Update stored canvas to the new size

        # 4. Determine UI button layout based on the page size
        button_specs = get_button_specs(page_img.shape)

        # 5. Initialize status text and cursor visibility for this frame
        mode_text = f"Page: {config.current_page + 1}/{len(config.pdf_pages)}"  # Start with page number info
        draw_cursor = False  # Will set to True if a fingertip position is available to draw a cursor
        avg_x = avg_y = None # Smoothed fingertip coordinates (if available)
        ok_now = False       # True if an OK-sign gesture is detected in the current frame

        # 6. Process hand landmarks and gestures if a hand is detected in the frame
        if results.multi_hand_landmarks:
            # Take the first detected hand (we only track one hand at a time)
            hand_landmarks = results.multi_hand_landmarks[0]
            # (Optional) Draw the hand landmarks on the original camera frame for debugging
            config.mp_draw.draw_landmarks(frame, hand_landmarks, config.mp_hands.HAND_CONNECTIONS)

            # Compute the finger tip (index finger tip) coordinates in image pixels
            img_h, img_w = page_img.shape[:2]                  # Page image dimensions
            # Normalized coordinates (0 to 1) of the index fingertip (Mediapipe landmark index 8)
            norm_x = hand_landmarks.landmark[8].x
            norm_y = hand_landmarks.landmark[8].y
            # Convert normalized coordinates to image pixel coordinates
            cx = int(norm_x * img_w)
            cy = int(norm_y * img_h)

            # Smooth the cursor motion by averaging over the last few positions
            config.point_buffer.append((cx, cy))
            avg_x = sum(pt[0] for pt in config.point_buffer) // len(config.point_buffer)
            avg_y = sum(pt[1] for pt in config.point_buffer) // len(config.point_buffer)
            draw_cursor = True  # Indicate that we have a cursor position to draw

            # Detect all gestures once
            ok_now    = is_ok_sign(hand_landmarks)
            pinch_now = is_pinch(hand_landmarks)
            open_now  = is_hand_open(hand_landmarks)
            point_now  = is_point_up(hand_landmarks)

            # 6a) OK-sign branch (highest priority)
            if ok_now:
                mode_text += " | OK held"
                # Start or continue the hold timer
                if hold_start is None:
                    hold_start = time.time()
                elif (time.time() - hold_start) >= HOLD_DURATION:
                    # Only save once per hold
                    if not ok_saved:
                        save_annotated_page()
                        ok_saved = True
                        mode_text += " | Saved"
                        # Reset drawing state so no immediate stroke
                        prev_gesture = "pen_up"
                        config.prev_x = config.prev_y = None

            # 6b) Drawing branch (pinch), only if NOT OK-sign
            elif pinch_now:
                # Cancel any pending save
                hold_start = None
                ok_saved = False
                mode_text += " | Drawing"

                # On initial transition from pen_up → drawing, check button clicks
                if prev_gesture == "pen_up":
                    for btn in button_specs:
                        x1,y1,x2,y2 = btn['pos']
                        if x1 <= avg_x <= x2 and y1 <= avg_y <= y2:
                            click_feedback = f"Clicked: {btn['label']}"
                            feedback_frames = 60
                            if btn['label'] == 'Clear':
                                config.page_canvases[config.current_page][:] = 0
                            else:
                                config.pen_color = btn['color']
                            config.prev_x, config.prev_y = None, None
                            break

                # ALWAYS draw a stroke segment (or dot) each frame of the pinch
                if config.prev_x is not None and config.prev_y is not None:
                    cv2.line(canvas,
                            (config.prev_x, config.prev_y),
                            (avg_x, avg_y),
                            config.pen_color,
                            5)
                else:
                    cv2.circle(canvas,
                            (avg_x, avg_y),
                            8,
                            config.pen_color,
                            -1)
                config.prev_x, config.prev_y = avg_x, avg_y
                prev_gesture = "drawing"

            # 6c) Erasing branch (open palm)
            elif open_now:
                mode_text += " | Erasing"
                cv2.circle(canvas, (avg_x, avg_y), 30, (0,0,0), -1)
                config.prev_x = config.prev_y = None
                prev_gesture = "pen_up"
                ok_saved = False

            # 6d) No gesture → pen up
            else:
                mode_text += " | Pen Up"
                config.prev_x = config.prev_y = None
                prev_gesture = "pen_up"
                hold_start = None
                ok_saved = False

            # 7. Handle the OK-sign hold logic for saving the page
            current_time = time.time()
            if ok_now and not ok_saved:
                # If currently in OK gesture
                if hold_start is None:
                    hold_start = current_time  # Start timing the hold
                elif (current_time - hold_start) >= HOLD_DURATION - HOLD_FAULT_TOLERANCE and not ok_saved:
                    # If the OK gesture has been held long enough and save not done yet:
                    save_annotated_page()  # Save the current page with annotations
                    ok_saved = True
                    mode_text += " | Saved"  # Append confirmation in status text
                    # after saving, reset drawing state so we don't immediately start drawing
                    prev_gesture = "pen_up"
                    config.prev_x = config.prev_y = None
            else:
                hold_start = None  # Reset the hold timer if OK gesture is not active

             #8. Swipe detection for page navigation (point-and-swipe)
            if point_now and not (ok_now or pinch_now or open_now):
                mode_text += " | Swipe to change page"
                swipe_buffer.append(avg_x)
                if swipe_cooldown > 0:
                    swipe_cooldown -= 1
                elif len(swipe_buffer) == SWIPE_BUFFER_SIZE:
                    delta_x = swipe_buffer[-1] - swipe_buffer[0]
                    if delta_x > SWIPE_THRESHOLD_PX:
                        # Swipe right → Previous page
                        if config.current_page > 0:
                            config.current_page -= 1
                            click_feedback = "Previous Page"
                        else:
                            click_feedback = "First Page"
                        feedback_frames = 60
                        swipe_cooldown = SWIPE_COOLDOWN
                        swipe_buffer.clear()
                    elif delta_x < -SWIPE_THRESHOLD_PX:
                        # Swipe left → Next page
                        if config.current_page < len(config.pdf_pages) - 1:
                            config.current_page += 1
                            click_feedback = "Next Page"
                        else:
                            click_feedback = "Last Page"
                        feedback_frames = 60
                        swipe_cooldown = SWIPE_COOLDOWN
                        swipe_buffer.clear()
            else:
                swipe_buffer.clear()

        else:
            # No hand detected in this frame
            prev_gesture = "pen_up"
            hold_start = None
            ok_saved = False
            # Clear swipe tracking if no hand
            swipe_buffer.clear()

        # 9. Prepare the final image for display by combining the PDF page with the annotations
        # Create a mask of where the canvas has drawings (non-zero pixels)
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Combine page image and canvas using the mask
        page_without_annots = cv2.bitwise_and(page_img, page_img, mask=mask_inv)  # areas with no annotations
        annots_only = cv2.bitwise_and(canvas, canvas, mask=mask)                  # the drawn annotations only
        display_img = cv2.add(page_without_annots, annots_only)                  # page with annotations overlaid

        # Draw the UI buttons onto the display image
        draw_buttons(display_img, button_specs)

        # If a fingertip cursor is available, draw it on the display image
        if draw_cursor and avg_x is not None and avg_y is not None:
            cv2.circle(display_img, (avg_x, avg_y), 10, config.cursor_color, -1)

        # Append any feedback text (button click or page navigation) to the mode_text if active
        if feedback_frames > 0 and click_feedback:
            mode_text += f" | {click_feedback}"
            feedback_frames -= 1
            if feedback_frames == 0:
                click_feedback = ""  # Clear feedback text after it has been shown for the desired frames

        # Overlay the status text (mode_text) onto the display image (at the top-left of the page for visibility)
        cv2.putText(display_img, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display_img, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # 10. Show the final composed image in the window
        cv2.imshow("PDF Annotation", display_img)
        # Wait briefly for a key press (and to allow image to render). Capture any pressed key.
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            # Exit if ESC (27) or 'q' is pressed
            break
        # Also break out if the window was closed by the user
        if cv2.getWindowProperty("PDF Annotation", cv2.WND_PROP_AUTOSIZE) < 0:
            break

    # End of loop - cleanup resources
    config.cap.release()    # Release the webcam
    cv2.destroyAllWindows() # Close the OpenCV window(s)
