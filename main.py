import mediapipe as mp
import cv2
import numpy as np
from collections import deque 
'''for smoothing the drawing'''

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
    # Tip and MCP landmarks for: index, middle, ring, pinky
    finger_tips = [8, 12, 16, 20]
    finger_mcps = [5, 9, 13, 17]

    open_fingers = 0

    for tip_id, mcp_id in zip(finger_tips, finger_mcps):
        tip_y = hand_landmarks.landmark[tip_id].y
        mcp_y = hand_landmarks.landmark[mcp_id].y
        
        if tip_y < mcp_y:  # Finger is "up"
            open_fingers += 1

    return open_fingers >= 3

#For optimization 1, I started by defining a function to detect the pinch

def is_pinch(hand_pos):
    
    index_tip = hand_pos.landmark[8]
    thumb_tip = hand_pos.landmark[4]

    dist = ((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2) ** 0.5

    return dist < 0.1
#if returns true, means that its a pinch



    
#I'll try drawing here

def drawing():
    global canvas, prev_x, prev_y
    while True:
        exists_frame, frame = cap.read()
        if exists_frame == False:
            print("no frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_where = hands.process(rgb)

        #if there is nothing on the canvas, we create a black one
        if canvas is None:
            canvas = np.zeros_like(frame)
        
        if hands_where.multi_hand_landmarks:
            #I realized that hands_where, we can't use it, since it's more like if hands detected or not
            #So now, I have hands_exactly_there, which contains the position of hands
            hands_exactly_there = hands_where.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hands_exactly_there, mp_hands.HAND_CONNECTIONS)
        
            #here I got some help, also lernt that mediapipe gives us positions 0,1, so I had to multiply them with actual lenghts 
            #so thet I had actual coorinates for the index finger
            h, w, _ = frame.shape
            index_finger = hands_exactly_there.landmark[8]
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)

            point_buffer.append((cx, cy))           
            ''' add the newest fingertip point'''

            avg_x = int(sum(p[0] for p in point_buffer) / len(point_buffer))
            avg_y = int(sum(p[1] for p in point_buffer) / len(point_buffer))
            ''' calculate the average of the last N points in the buffer'''


            ''' now use (avg_x, avg_y) instead of (cx, cy) below'''
            if is_pinch(hands_exactly_there):
                if is_hand_open(hands_exactly_there):
                    cv2.circle(canvas, (avg_x, avg_y), 30, (0, 0, 0), -1)
                    prev_x, prev_y = None, None
                else:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (avg_x, avg_y), (255, 255, 255), 5)
                    else:
                        cv2.circle(canvas, (avg_x, avg_y), 8, (255, 255, 255), -1)
                    prev_x, prev_y = avg_x, avg_y
            else:
                prev_x, prev_y = None, None
        
        #putting them together was not as hard as we expected...
        output = cv2.addWeighted(frame, 1, canvas, 1, 0)

        #also, showing them was not as hard
        cv2.imshow("Virtual Drawing", output)

        #This part, I just stole it from your code

                # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#Here I just tweaked it a bit so I could try the bit I just wrote
if __name__ == "__main__":
    drawing()


