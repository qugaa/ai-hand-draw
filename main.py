import time
import cv2 
import numpy as np
import config
from pdf_loader import load_pdf
from gestures import is_hand_open, is_pinch, is_ok_sign, is_point_up
from ui_helpers import get_button_specs, draw_buttons, save_annotated_page

def run_annotation_loop():

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
    page_h, page_w = config.pdf_pages[0].shape[:2]        
    cv2.resizeWindow("PDF Annotation", page_w, page_h)  
    cv2.namedWindow("Webcam Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam Preview", 320, 240)

    # Initialize gesture state flags and variables
    ok_saved = False           
    prev_gesture = "pen_up"    
    click_feedback = ""       
    feedback_frames = 0       
    hold_start = None         
    HOLD_DURATION = 2.0       
    HOLD_FAULT_TOLERANCE = 0.2

    # Swipe gesture detection setup
    from collections import deque
    SWIPE_BUFFER_SIZE = 8    
    SWIPE_THRESHOLD_PX = 200  
    SWIPE_COOLDOWN = 30      
    swipe_buffer = deque(maxlen=SWIPE_BUFFER_SIZE) 
    swipe_cooldown = 0       

    while True:
        # 1. Capture a frame from the webcam
        ret, frame = config.cap.read()
        if not ret:
            print("Camera capture failed or ended.")
            break
        frame = cv2.flip(frame, 1) 
        cv2.imshow("Webcam Preview", frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Hand landmark detection using Mediapipe
        results = config.hands.process(rgb_frame) 

        # 3. Prepare the current PDF page image and its annotation canvas
        page_img = config.pdf_pages[config.current_page].copy() 
        canvas = config.page_canvases[config.current_page]       
      
        if canvas.shape[:2] != page_img.shape[:2]:

            canvas = cv2.resize(canvas, (page_img.shape[1], page_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            config.page_canvases[config.current_page] = canvas  

        # 4. Determine UI button layout based on the page size
        button_specs = get_button_specs(page_img.shape)

        # 5. Initialize status text and cursor visibility for this frame
        mode_text = f"Page: {config.current_page + 1}/{len(config.pdf_pages)}"
        draw_cursor = False
        avg_x = avg_y = None
        ok_now = False

        # 6. Process hand landmarks and gestures if a hand is detected in the frame
        if results.multi_hand_landmarks:           
            hand_landmarks = results.multi_hand_landmarks[0]
            config.mp_draw.draw_landmarks(frame, hand_landmarks, config.mp_hands.HAND_CONNECTIONS)

            img_h, img_w = page_img.shape[:2]
            norm_x = hand_landmarks.landmark[8].x
            norm_y = hand_landmarks.landmark[8].y
            cx = int(norm_x * img_w)
            cy = int(norm_y * img_h)

            config.point_buffer.append((cx, cy))
            avg_x = sum(pt[0] for pt in config.point_buffer) // len(config.point_buffer)
            avg_y = sum(pt[1] for pt in config.point_buffer) // len(config.point_buffer)

            # Detect all gestures once
            ok_now    = is_ok_sign(hand_landmarks)
            pinch_now = is_pinch(hand_landmarks)
            open_now  = is_hand_open(hand_landmarks)
            point_now  = is_point_up(hand_landmarks)

            # 6a) OK-sign branch (highest priority)
            if ok_now:
                mode_text += " | OK held"
                if hold_start is None:
                    hold_start = time.time()
                elif (time.time() - hold_start) >= HOLD_DURATION:
                    if not ok_saved:
                        save_annotated_page()
                        ok_saved = True
                        mode_text += " | Saved"
                        prev_gesture = "pen_up"
                        config.prev_x = config.prev_y = None

            # 6b) Drawing branch (pinch), only if NOT OK-sign
            elif pinch_now:
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
                if hold_start is None:
                    hold_start = current_time 
                elif (current_time - hold_start) >= HOLD_DURATION - HOLD_FAULT_TOLERANCE and not ok_saved:
                    save_annotated_page()
                    ok_saved = True
                    mode_text += " | Saved"
                    prev_gesture = "pen_up"
                    config.prev_x = config.prev_y = None
            else:
                hold_start = None

             #8. Swipe detection for page navigation (point-and-swipe)
            if point_now and not (ok_now or pinch_now or open_now):
                mode_text += " | Swipe to change page"
                swipe_buffer.append(avg_x)
                if swipe_cooldown > 0:
                    swipe_cooldown -= 1
                elif len(swipe_buffer) == SWIPE_BUFFER_SIZE:
                    delta_x = swipe_buffer[-1] - swipe_buffer[0]
                    if delta_x > SWIPE_THRESHOLD_PX:
                        if config.current_page > 0:
                            config.current_page -= 1
                            click_feedback = "Previous Page"
                        else:
                            click_feedback = "First Page"
                        feedback_frames = 60
                        swipe_cooldown = SWIPE_COOLDOWN
                        swipe_buffer.clear()
                    elif delta_x < -SWIPE_THRESHOLD_PX:
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
            prev_gesture = "pen_up"
            hold_start = None
            ok_saved = False
            swipe_buffer.clear()

        # 9. Prepare the final image for display by combining the PDF page with the annotations
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        page_without_annots = cv2.bitwise_and(page_img, page_img, mask=mask_inv)
        annots_only = cv2.bitwise_and(canvas, canvas, mask=mask)
        display_img = cv2.add(page_without_annots, annots_only)

        draw_buttons(display_img, button_specs)

        if draw_cursor and avg_x is not None and avg_y is not None:
            cv2.circle(display_img, (avg_x, avg_y), 10, config.cursor_color, -1)

        if feedback_frames > 0 and click_feedback:
            mode_text += f" | {click_feedback}"
            feedback_frames -= 1
            if feedback_frames == 0:
                click_feedback = ""

        cv2.putText(display_img, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display_img, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # 10. Show the final composed image in the window
        cv2.imshow("PDF Annotation", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    config.cap.release()
    cv2.destroyAllWindows()
