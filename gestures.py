import mediapipe as mp 
from math import hypot

# Mediapipe hand landmark indices for fingertips and their base (MCP) joints:
_FINGER_TIPS = [8, 12, 16, 20]  # Indices of tip landmarks for Index, Middle, Ring, Pinky fingers
_FINGER_MCPS = [5, 9, 13, 17]   # Indices of corresponding MCP (knuckle) landmarks for those fingers

def is_hand_open(landmarks) -> bool:

    open_count = 0
    # Check each finger (index, middle, ring, pinky) by comparing tip and MCP heights:
    for tip_id, mcp_id in zip(_FINGER_TIPS, _FINGER_MCPS):
        tip = landmarks.landmark[tip_id]
        mcp = landmarks.landmark[mcp_id]
        if mcp.y - tip.y > 0.01:
            open_count += 1
    return open_count >= 4

def is_pinch(landmarks, rel_thresh=0.35) -> bool:

    # Extract relevant landmarks:
    idx_tip = landmarks.landmark[8]
    thumb_tip = landmarks.landmark[4]
    wrist = landmarks.landmark[0]
    mid_mcp = landmarks.landmark[9]

    # Compute 2D distance between index tip and thumb tip (in normalized coordinates 0..1):
    dx = idx_tip.x - thumb_tip.x
    dy = idx_tip.y - thumb_tip.y
    tip_dist = (dx*dx + dy*dy) ** 0.5

    # Compute an approximate hand size for normalization (distance from wrist to middle finger MCP joint):
    hx = wrist.x - mid_mcp.x
    hy = wrist.y - mid_mcp.y
    hand_size = (hx*hx + hy*hy) ** 0.5

    # Consider it a pinch if thumb-index distance is less than the specified fraction of hand size
    return tip_dist < (hand_size * rel_thresh)

def is_ok_sign(landmarks, rel_thresh=0.35) -> bool:

    # First, verify the thumb-index pinch condition:
    if not is_pinch(landmarks, rel_thresh):
        return False

    # If pinched, check that the other three fingers (middle, ring, pinky) are extended:
    extended = 0
    for tip_id, mcp_id in zip(_FINGER_TIPS[1:], _FINGER_MCPS[1:]):
        tip = landmarks.landmark[tip_id]
        mcp = landmarks.landmark[mcp_id]
        if mcp.y - tip.y > 0.01:
            extended += 1
    return extended == 3

def is_point_up(landmarks, rel_frac=0.12, thumb_frac=0.95) -> bool:

    # 1) hand size = distance wrist→middle-MCP
    wrist   = landmarks.landmark[0]
    mid_mcp = landmarks.landmark[9]
    hand_size = hypot(wrist.x - mid_mcp.x, wrist.y - mid_mcp.y)
    y_thresh  = hand_size * rel_frac

    # 2) index must poke up
    idx_tip = landmarks.landmark[8]
    idx_mcp = landmarks.landmark[5]
    if (idx_mcp.y - idx_tip.y) < y_thresh:
        return False

    # 3) middle/ring/pinky must be down
    for tip_id, mcp_id in zip(_FINGER_TIPS[1:], _FINGER_MCPS[1:]):
        tip = landmarks.landmark[tip_id]
        mcp = landmarks.landmark[mcp_id]
        if (mcp.y - tip.y) > y_thresh:
            return False

    # 4) thumb can’t stick out too far
    thumb_tip = landmarks.landmark[4]
    thumb_dist = hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
    if thumb_dist > hand_size * thumb_frac:
        return False

    return True

