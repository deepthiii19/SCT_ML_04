"""
realtime_inference.py

Runs real-time hand gesture recognition from the webcam using:
  - MediaPipe Hands for landmark detection
  - The trained classifier (models/gesture_model.pkl) for gesture classification

Press 'q' to quit.

Usage:
    python realtime_inference.py
"""

import argparse
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def normalize_landmarks(landmarks):
    pts = np.array(landmarks, dtype=np.float32)
    wrist = pts[0].copy()
    pts -= wrist
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 1e-6:
        pts /= max_dist
    return pts.flatten().reshape(1, -1)


def run(model_path, encoder_path, cam_index=0, conf_threshold=0.6):
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --cam index.")

    prev_time = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            label_text = "No hand detected"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                feats = normalize_landmarks(coords)

                # Use predict_proba if available for a confidence score
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(feats)[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = probs[pred_idx]
                else:
                    pred_idx = int(model.predict(feats)[0])
                    confidence = 1.0

                gesture = le.inverse_transform([pred_idx])[0]

                if confidence >= conf_threshold:
                    label_text = f"{gesture} ({confidence*100:.1f}%)"
                else:
                    label_text = f"Uncertain ({confidence*100:.1f}%)"

            # FPS counter
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            cv2.putText(
                frame, label_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA
            )

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gesture_model.pkl")
    parser.add_argument("--encoder", default="models/label_encoder.pkl")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    args = parser.parse_args()

    run(args.model, args.encoder, args.cam, args.conf)