import mediapipe as mp
import cv2
import numpy as np

# Initialize Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       )
mp_draw = mp.solutions.drawing_utils

# webcam
cap = cv2.VideoCapture(0)

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

    if open_fingers >= 3:
	    return True
    







#boom


