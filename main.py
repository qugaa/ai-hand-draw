import mediapipe as mp
import cv2
import numpy as np

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
    

#boom


def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip and convert to RGB for Mediapipe
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        status_text = "No hand"

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            # Draw skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Decide open vs. closed
            if is_hand_open(hand_landmarks):
                status_text = "Hand Open"
                color = (0, 255, 0)  # green
            else:
                status_text = "Hand Closed"
                color = (0, 0, 255)  # red

            # Optional: do something when open/closed
            # e.g. trigger an action here

        # Overlay status
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color if result.multi_hand_landmarks else (200, 200, 200),
            2
        )

        cv2.imshow("Hand Tracking", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


