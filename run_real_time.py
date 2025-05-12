# run_real_time.py
import cv2
import numpy as np
import mediapipe as mp
import joblib
import os

def run_detection():
    # Load trained SVM model and label encoder
    model = joblib.load('model/svm_model.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    mp_draw = mp.solutions.drawing_utils

    # Start webcam
    cap = cv2.VideoCapture(0)

    print("[INFO] Starting real-time detection. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)

        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == model.n_features_in_:
                    input_data = np.array(landmarks).reshape(1, -1)
                    prediction = model.predict(input_data)
                    label = label_encoder.inverse_transform(prediction)[0]

                    # Draw landmarks
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Display label text for each hand at different Y positions
                    y_position = 60 + idx * 60
                    color = (0, 255 - idx * 100, 255)  # Greenish for first hand, pinkish for second
                    cv2.putText(frame, f"Hand {idx + 1}: {label}", (10, y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Show result
        cv2.imshow("Real-Time Detection", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Run if this script is executed
if __name__ == "__main__":
    run_detection()
