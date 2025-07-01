# -*- coding: utf-8 -*-
"""
config.py: Centralizes all global settings, imports, and initial objects
required by the PDF Gesture Annotation Tool.
"""

# --- Standard library imports ---
import os               # Provides OS filesystem operations (e.g., for path handling)
import datetime         # Used for generating timestamped filenames (e.g., for saved images)
import time             # Used for measuring elapsed time (e.g., OK gesture hold duration)

# --- Third-party imports ---
import cv2              # OpenCV for webcam capture and image processing
import mediapipe as mp  # Mediapipe for real-time hand detection and tracking
from collections import deque  # deque provides an efficient fixed-length queue for smoothing points

# ----- SMOOTHING CONFIG -----
SMOOTHING_WINDOW = 5  # Number of recent fingertip points to average for smoothing cursor movement
point_buffer = deque(maxlen=SMOOTHING_WINDOW)  # Buffer to store the last few (x, y) points of the fingertip

# ----- MEDIAPIPE HANDS SETUP -----
mp_hands = mp.solutions.hands  # Reference to Mediapipe's Hands solution class (ignore IDE type-checking errors) # type: ignore[reportUndefinedVariable](falsepositive)
# Initialize the Mediapipe Hands model for live video processing:
hands = mp_hands.Hands(
    static_image_mode=False,       # Process input as a continuous video stream (not static images)
    max_num_hands=1,               # Detect and track at most one hand at a time
    min_detection_confidence=0.5,  # Minimum confidence value (0-1) for the hand detection to be considered successful
    min_tracking_confidence=0.5    # Minimum confidence value (0-1) for hand landmark tracking to be considered successful
)
mp_draw = mp.solutions.drawing_utils  # Utility for drawing hand landmarks on images (ignore IDE type-checking errors) # type: ignore[reportUndefinedVariable](falsepositive)

# ----- OPENCV CAMERA INITIALIZATION -----
cap = cv2.VideoCapture(0)  # Open a connection to the default webcam (device index 0)

# ----- CANVAS AND CURSOR STATE -----
canvas = None               # Will hold the current page's annotation overlay image (initialized when a PDF is loaded)
prev_x, prev_y = None, None # Last known fingertip coordinates (for drawing continuous lines)

# ----- PDF NAVIGATION STATE -----
pdf_pages = []      # List of images (as numpy arrays) for each page of the loaded PDF (initially empty)
page_canvases = []  # Parallel list of blank canvas images corresponding to each PDF page (for annotations)
current_page = 0    # Index of the currently displayed PDF page (0-based)

# ----- DRAWING STATE DEFAULTS -----
pen_color = (0, 0, 255)    # Default drawing color for annotations (BGR format: red)
cursor_color = (0, 255, 0) # Color for the fingertip cursor indicator (BGR format: green)
