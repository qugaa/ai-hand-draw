# -*- coding: utf-8 -*-
"""
gestures.py: Defines functions to detect user hand gestures
(pinch, open palm, OK sign) using Mediapipe hand landmarks.
Each function returns a boolean indicating whether the gesture is active.
"""

# --- Third-party imports ---
import mediapipe as mp  # Import Mediapipe library (for access to landmark data structures)

# Mediapipe hand landmark indices for fingertips and their base (MCP) joints:
_FINGER_TIPS = [8, 12, 16, 20]  # Indices of tip landmarks for Index, Middle, Ring, Pinky fingers
_FINGER_MCPS = [5, 9, 13, 17]   # Indices of corresponding MCP (knuckle) landmarks for those fingers

def is_hand_open(landmarks) -> bool:
    """
    Determines if the hand is open (flat palm) by checking if most fingers are extended (not curled).
    
    Args:
        landmarks: Mediapipe landmarks object for a detected hand.
    Returns:
        True if at least 4 fingers are extended (open), False otherwise.
    """
    open_count = 0
    # Check each finger (index, middle, ring, pinky) by comparing tip and MCP heights:
    for tip_id, mcp_id in zip(_FINGER_TIPS, _FINGER_MCPS):
        tip = landmarks.landmark[tip_id]   # Landmark of the finger's tip
        mcp = landmarks.landmark[mcp_id]   # Landmark of the finger's base (MCP joint)
        # A finger is considered extended if the tip's y-coordinate is higher (smaller value) than the MCP's y-coordinate by a margin
        if mcp.y - tip.y > 0.01:           # (In Mediapipe, lower y means higher in the image frame)
            open_count += 1               # Count this finger as extended (finger up)
    return open_count >= 4                # Hand is open if 4 or more fingers are extended

def is_pinch(landmarks, rel_thresh=0.35) -> bool:
    """
    Detects a pinch gesture (thumb and index finger very close together).
    
    Args:
        landmarks: Mediapipe landmarks object for a detected hand.
        rel_thresh: Relative threshold (fraction of hand size) below which thumb-index distance is considered a pinch.
    Returns:
        True if the distance between thumb tip and index finger tip is below the threshold, False otherwise.
    """
    # Extract relevant landmarks:
    idx_tip = landmarks.landmark[8]    # Index finger tip
    thumb_tip = landmarks.landmark[4]  # Thumb tip
    wrist = landmarks.landmark[0]      # Wrist (for scale reference)
    mid_mcp = landmarks.landmark[9]    # Middle finger MCP (for hand size reference)

    # Compute 2D distance between index tip and thumb tip (in normalized coordinates 0..1):
    dx = idx_tip.x - thumb_tip.x
    dy = idx_tip.y - thumb_tip.y
    tip_dist = (dx*dx + dy*dy) ** 0.5  # Euclidean distance between index tip and thumb tip

    # Compute an approximate hand size for normalization (distance from wrist to middle finger MCP joint):
    hx = wrist.x - mid_mcp.x
    hy = wrist.y - mid_mcp.y
    hand_size = (hx*hx + hy*hy) ** 0.5

    # Consider it a pinch if thumb-index distance is less than the specified fraction of hand size
    return tip_dist < (hand_size * rel_thresh)

def is_ok_sign(landmarks, rel_thresh=0.35) -> bool:
    """
    Detects the 'OK' gesture: thumb and index finger pinched together, with the other fingers extended.
    
    Args:
        landmarks: Mediapipe landmarks object.
        rel_thresh: Same pinch distance threshold factor as used in is_pinch.
    Returns:
        True if thumb & index are pinched and all other three fingers are extended (OK sign), False otherwise.
    """
    # First, verify the thumb-index pinch condition:
    if not is_pinch(landmarks, rel_thresh):
        return False

    # If pinched, check that the other three fingers (middle, ring, pinky) are extended:
    extended = 0
    for tip_id, mcp_id in zip(_FINGER_TIPS[1:], _FINGER_MCPS[1:]):  # Skip index finger (already pinched)
        tip = landmarks.landmark[tip_id]
        mcp = landmarks.landmark[mcp_id]
        # Count finger as extended if tip is higher than MCP by the threshold:
        if mcp.y - tip.y > 0.01:
            extended += 1
    return extended == 3  # True if all three non-pinching fingers are extended
