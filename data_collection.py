# data_collection.py
import cv2
import mediapipe as mp
import csv
import os

def collect_data(label):
    filename = f"datasets/{label}_data.csv"
    os.makedirs("datasets", exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    data = []

    print(f"[INFO] Collecting data for '{label}'. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                data.append([label] + landmarks)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Data Collection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"[INFO] Saved {len(data)} samples to {filename}")
