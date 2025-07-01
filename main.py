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
from gestures import is_hand_open, is_pinch, is_ok_sign  # Gesture detection helper functions
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
    SWIPE_BUFFER_SIZE = 16    # Number of recent frames to consider for swipe movement
    SWIPE_THRESHOLD_PX = 400  # Minimum horizontal movement (in pixels) to qualify as a swipe
    SWIPE_COOLDOWN = 20       # Cooldown period (in frames) after a swipe to avoid immediate repeat
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

            # 6a. OK-sign gesture (index finger and thumb touching, other fingers up) for save action
            if is_ok_sign(hand_landmarks):
                ok_now = True
                mode_text += " | OK held"  # Add status text indicating the OK gesture is being held

            # 6b. Pinch gesture for drawing or UI button clicks
            if is_pinch(hand_landmarks):
                # Cancel any ongoing OK-sign save timing
                hold_start = None
                ok_saved = False
                mode_text += " | Drawing"

                # On the very first frame of a new pinch (prev_gesture was pen_up),
                # check if the finger is “clicking” on a UI button:
                if prev_gesture == "pen_up":
                    for btn in button_specs:
                        x1, y1, x2, y2 = btn['pos']
                        if x1 <= avg_x <= x2 and y1 <= avg_y <= y2:
                            # We clicked a button—give feedback and perform action:
                            click_feedback = f"Clicked: {btn['label']}"
                            feedback_frames = 60  # show for ~2 seconds
                            if btn['label'] == 'Clear':
                                # Erase entire canvas for this page
                                config.page_canvases[config.current_page][:] = 0
                            else:
                                # Change pen color to the button’s color
                                config.pen_color = btn['color']
                            # Reset previous line endpoint so stroke restarts next time
                            config.prev_x, config.prev_y = None, None
                            break

                # --- ALWAYS draw a stroke (line or dot) on every frame of pinch ---
                if config.prev_x is not None and config.prev_y is not None:
                    # Continue the line from last point to this frame’s avg_x,avg_y
                    cv2.line(canvas,
                             (config.prev_x, config.prev_y),
                             (avg_x, avg_y),
                             config.pen_color,
                             5)
                else:
                    # First point of a new stroke: draw a dot
                    cv2.circle(canvas,
                               (avg_x, avg_y),
                               8,
                               config.pen_color,
                               -1)
                # Update prev_x, prev_y for the next frame
                config.prev_x, config.prev_y = avg_x, avg_y

                prev_gesture = "drawing"

            # 6c. Open-palm gesture (all fingers extended) for erasing
            elif is_hand_open(hand_landmarks):
                mode_text += " | Erasing"  # Update status to indicate erase mode
                # Erase by drawing a large circle of "black" (0 pixel value) on the canvas at the fingertip position
                cv2.circle(canvas, (avg_x, avg_y), 30, (0, 0, 0), -1)
                # Reset drawing continuity (lifting the pen up)
                config.prev_x = config.prev_y = None
                prev_gesture = "pen_up"
                ok_saved = False  # Reset save flag since hand is now open (not saving)

            # 6d. No specific gesture detected (hand is present but neither pinch, open, nor OK)
            else:
                mode_text += " | Pen Up"  # Indicate that the "pen" (fingertip) is not touching (no drawing)
                config.prev_x = config.prev_y = None
                prev_gesture = "pen_up"
                ok_saved = False

            # 7. Handle the OK-sign hold logic for saving the page
            current_time = time.time()
            if ok_now:
                # If currently in OK gesture
                if hold_start is None:
                    hold_start = current_time  # Start timing the hold
                elif (current_time - hold_start) >= HOLD_DURATION - HOLD_FAULT_TOLERANCE and not ok_saved:
                    # If the OK gesture has been held long enough and save not done yet:
                    save_annotated_page()  # Save the current page with annotations
                    ok_saved = True
                    mode_text += " | Saved"  # Append confirmation in status text
            else:
                hold_start = None  # Reset the hold timer if OK gesture is not active

            # 8. Swipe detection for page navigation (based on open hand horizontal movement)
            if is_hand_open(hand_landmarks):
                # Collect the current x-position for swipe analysis
                swipe_buffer.append(avg_x)
                if swipe_cooldown > 0:
                    swipe_cooldown -= 1  # Countdown the cooldown if it's active
                else:
                    # Only attempt to detect a swipe if not in cooldown
                    if len(swipe_buffer) == SWIPE_BUFFER_SIZE:
                        # Check the overall horizontal movement across the buffered positions
                        delta_x = swipe_buffer[-1] - swipe_buffer[0]
                        if delta_x > SWIPE_THRESHOLD_PX:
                            # Significant movement to the right (hand moved rightward) -> Navigate to previous page
                            if config.current_page > 0:
                                config.current_page -= 1
                                click_feedback = "Previous Page"
                            else:
                                # Already at the first page, cannot go further left
                                click_feedback = "First Page"
                            feedback_frames = 60
                            swipe_cooldown = SWIPE_COOLDOWN
                            swipe_buffer.clear()
                        elif delta_x < -SWIPE_THRESHOLD_PX:
                            # Significant movement to the left (hand moved leftward) -> Navigate to next page
                            if config.current_page < len(config.pdf_pages) - 1:
                                config.current_page += 1
                                click_feedback = "Next Page"
                            else:
                                # Already at the last page, cannot go further right
                                click_feedback = "Last Page"
                            feedback_frames = 60
                            swipe_cooldown = SWIPE_COOLDOWN
                            swipe_buffer.clear()
            else:
                # If hand is not open (or no hand present), reset swipe tracking
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
