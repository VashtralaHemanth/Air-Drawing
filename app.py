import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0

# Default color
color = (0, 0, 255)
color_name = "Red"

# Drawing toggle
drawing_enabled = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# it is to dectect the hand landmarks and contains in result
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm_list = hand.landmark

        index_tip = lm_list[8]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        # Draw line from previous point to current
        if prev_x == 0 and prev_y == 0:
            prev_x, prev_y = x, y

        if drawing_enabled:
            cv2.line(canvas, (prev_x, prev_y), (x, y), color, 6)

        prev_x, prev_y = x, y

        # Draw hand landmarks
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    else:
        prev_x, prev_y = 0, 0 # if no hand will visble it will reset the cordinates

    # Overlay canvas
    output = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    # here we can Show current color and drawing status
    cv2.rectangle(output, (0, 0), (350, 60), (0, 0, 0), -1)
    cv2.putText(output, f"Color: {color_name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(output, f"Drawing: {'ON' if drawing_enabled else 'OFF'}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Air Drawing", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        color = (0, 0, 255)
        color_name = "Red"
    elif key == ord('g'):
        color = (0, 255, 0)
        color_name = "Green"
    elif key == ord('b'):
        color = (0, 0, 0)
        color_name = "Black"
    elif key == ord('w'):
        color = (255, 255, 255)
        color_name = "White"
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('s'):
        drawing_enabled = not drawing_enabled
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
