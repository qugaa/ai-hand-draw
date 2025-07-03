import os               
import datetime         
import time           
import cv2          
import mediapipe as mp  
from collections import deque 

SMOOTHING_WINDOW = 5  # Amount of frame history to smooth cursor movement
point_buffer = deque(maxlen=SMOOTHING_WINDOW)  

mp_hands = mp.solutions.hands  # type: ignore[reportUndefinedVariable](falsepositive)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,               # Amount of hands to detect
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils  # type: ignore[reportUndefinedVariable](falsepositive)

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = None, None

pdf_pages = []
page_canvases = []
current_page = 0

pen_color = (0, 0, 255)    # Default color (red)
cursor_color = (0, 255, 0) # Cursor color (green)
