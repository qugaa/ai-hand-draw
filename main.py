import os
import datetime
import mediapipe as mp
import cv2
import numpy as np
from collections import deque #for smoothing the drawing

# flag to ensure saving only once per OK sign
ok_saved = False

''' A fixed-size buffer holding the last N fingertip points for smoothing
 This is used to smoth the drawing by averaging the last few points'''
SMOOTHING_WINDOW = 5                    # ← define window size
point_buffer = deque(maxlen=SMOOTHING_WINDOW)  # ← initialize buffer

# Initialize Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
     static_image_mode=False,
     max_num_hands=1,
     min_detection_confidence=0.5,
     min_tracking_confidence=0.5)

mp_draw = mp.solutions.drawing_utils

# webcam
cap = cv2.VideoCapture(0)

#initialize Canvas (new added, I started up with the drawing logic)
canvas = None

#this is for tracking 
prev_x, prev_y = None, None

def is_hand_open(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_mcps = [5, 9, 13, 17]

    open_fingers = 0

    for tip_id, mcp_id in zip(finger_tips, finger_mcps):
        tip = hand_landmarks.landmark[tip_id]
        mcp = hand_landmarks.landmark[mcp_id]
        
        '''trying to combine vertical (Y) and depth (Z) to check angled hands'''
        dy = mcp.y - tip.y            # positive if tip is above mcp in image
        dz = abs(mcp.z - tip.z)       # small if finger not tilted out of palm plane

        if dy > 0.01 or dz < 0.02:
            open_fingers += 1
    return open_fingers >= 4

'''trying pinch detection relative to hand size'''
def is_pinch(hand_pos, rel_thresh=0.35):

    it = hand_pos.landmark[8]   # index fingertip
    tt = hand_pos.landmark[4]   # thumb tip
    dx = it.x - tt.x
    dy = it.y - tt.y
    pinch_dist_2d = (dx*dx + dy*dy) ** 0.5

    # measure normalized hand size via wrist (0) to middle_finger MCP (9), 2D only
    w = hand_pos.landmark[0]
    m = hand_pos.landmark[9]
    hx = w.x - m.x
    hy = w.y - m.y
    hand_size_2d = (hx*hx + hy*hy) ** 0.5

    return pinch_dist_2d < (hand_size_2d * rel_thresh)


'''detect OK sign: thumb and index touch and other fingers extended'''
def is_ok_sign(hand_pos, rel_thresh=0.35):
    # pinch condition
    if not is_pinch(hand_pos, rel_thresh):
        return False
    # count open for middle, ring, pinky
    other_tips = [12, 16, 20]
    other_mcps = [9, 13, 17]
    open_count = 0
    for tip_id, mcp_id in zip(other_tips, other_mcps):
        tip = hand_pos.landmark[tip_id]
        mcp = hand_pos.landmark[mcp_id]
        if mcp.y - tip.y > 0.01:
            open_count += 1
    return open_count >= 3
    
#I'll try drawing here
def drawing():
    global canvas, prev_x, prev_y, point_buffer, ok_saved
    while True:
        exists_frame, frame = cap.read()
        if not exists_frame:
            print("no frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_where = hands.process(rgb)

        #if there is nothing on the canvas, we create a black one
        if canvas is None:
            canvas = np.zeros_like(frame)

        mode_text = ""
        if hands_where.multi_hand_landmarks:
            #I realized that hands_where, we can't use it, since it's more like if hands detected or not
            #So now, I have hands_exactly_there, which contains the position of hands
            hands_exactly_there = hands_where.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hands_exactly_there, mp_hands.HAND_CONNECTIONS)
        
            #here I got some help, also lernt that mediapipe gives us positions 0,1, so I had to multiply them with actual lenghts 
            #so thet I had actual coorinates for the index finger
            frame_h, frame_w, _ = frame.shape
            index_finger = hands_exactly_there.landmark[8]
            cx, cy = int(index_finger.x * frame_w), int(index_finger.y * frame_h)

            point_buffer.append((cx, cy))           
            ''' we add the newest fingertip point'''

            avg_x = int(sum(p[0] for p in point_buffer) / len(point_buffer))
            avg_y = int(sum(p[1] for p in point_buffer) / len(point_buffer))
            ''' we calculate the average of the last N points in the buffer'''


            ''' now we Check OK sign to save canvas, we use (avg_x, avg_y) instead of (cx, cy) for smoothing
            and open hand now erases'''
            
            if is_ok_sign(hands_exactly_there):
                if not ok_saved:
                    save_dir = os.path.expanduser("~/Desktop")
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"board_{timestamp}.png"
                    path = os.path.join(save_dir, filename)
                    cv2.imwrite(path, canvas)
                    ok_saved = True
                mode_text = f"Saved: {filename}"
                # small delay to avoid multiple saves, allow quitting
                delay_key = cv2.waitKey(2000) & 0xFF
                if delay_key == ord('q'):
                    break
            elif is_pinch(hands_exactly_there):

                # draw smooth line
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (avg_x, avg_y), (255, 255, 255), 5)
                else:
                    cv2.circle(canvas, (avg_x, avg_y), 8, (255, 255, 255), -1)
                prev_x, prev_y = avg_x, avg_y
                ok_saved = False
                mode_text = "Drawing"
            elif is_hand_open(hands_exactly_there):
                # erase with big black circle at smoothed point
                cv2.circle(canvas, (avg_x, avg_y), 30, (0, 0, 0), -1)
                prev_x, prev_y = None, None
                ok_saved = False
                mode_text = "Erasing"
            else:
                # pen up: reset tracking
                prev_x, prev_y = None, None
                ok_saved = False
                mode_text = "Pen Up"
                        
                   
        #putting them together was not as hard as we expected...
        output = cv2.addWeighted(frame, 1, canvas, 1, 0)

        '''overlay mode text in corner, comment out to remove'''
        cv2.putText(output, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)  

        #also, showing them was not as hard
        cv2.imshow("Virtual Drawing", output)

        #This part, I just stole it from your code
        '''i stole it from gpt anyways :D but I rewrote it more human also it's all our code now'''
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
       
    cap.release()
    cv2.destroyAllWindows()


#Here I just tweaked it a bit so I could try the bit I just wrote
if __name__ == "__main__":
    drawing()


