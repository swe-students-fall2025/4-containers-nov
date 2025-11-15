from pathlib import Path
from datetime import datetime, timezone
import time
import cv2
import mediapipe as mp
import numpy as np
import torch
from pymongo import MongoClient

MODEL_PATH = Path("models/gesture_mlp.pt")

# MongoDB connection (localhost:27017 → handsense.gesture_events)
mongo_client = MongoClient("mongodb://localhost:27017")
mongo_client.admin.command("ping")
print("[INFO] MongoDB connected successfully!")
mongo_db = mongo_client["handsense"]
gesture_collection = mongo_db["gesture_events"]


class GestureMLP(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    input_dim = checkpoint["input_dim"]
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]

    model = GestureMLP(input_dim, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("[INFO] Loaded model from:", MODEL_PATH)
    print("[INFO] Classes:", class_names)
    return model, class_names


def extract_keypoints(hand_landmarks):
    # 21 x (x,y,z) → 63
    kp = []
    for lm in hand_landmarks.landmark:
        kp.extend([lm.x, lm.y, lm.z])
    return np.array(kp, dtype=np.float32)


def main():
    # ---- Load ML model ----
    model, class_names = load_model()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"[INFO] Using device: {device}")

    # ---- MediaPipe Hands ----
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    # ---- Webcam ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("[INFO] Running live gesture recognition... Press 'q' to quit.")

    # rate-limiting
    last_label = None
    last_logged_at = 0.0
    MIN_INTERVAL = 1.0        # minimum time (sec) between logging same gesture
    MIN_CONFIDENCE = 0.8      # ignore predictions below this confidence threshold

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        pred_label = "No Hand"
        confidence = 0.0
        handedness = "Unknown"

        # BGR → RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                # --- Extract 63-dim landmark vector ---
                kp = extract_keypoints(hand_landmarks)
                kp_t = torch.from_numpy(kp).float().unsqueeze(0).to(device)

                # --- Model prediction ---
                with torch.no_grad():
                    logits = model(kp_t)
                    prob = torch.softmax(logits, dim=1)
                    conf, pred_idx = prob.max(dim=1)
                    pred_label = class_names[pred_idx.item()]
                    confidence = float(conf.item())

                # --- Handedness ("Left" / "Right") ---
                handedness = hand_info.classification[0].label

                # Draw visual hand skeleton
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- Rate-limit DB writes: log only on change OR after MIN_INTERVAL ---
                now = time.time()
                if (
                    pred_label != "No Hand"
                    and confidence >= MIN_CONFIDENCE
                    and (
                        pred_label != last_label
                        or (now - last_logged_at) > MIN_INTERVAL
                    )
                ):
                    event = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "gesture": pred_label,
                        "confidence": confidence,
                        "handedness": handedness,
                    }
                    gesture_collection.insert_one(event)
                    print("[DB] Inserted:", event)

                    last_label = pred_label
                    last_logged_at = now

        # Display the gesture on screen
        cv2.putText(
            frame,
            f"Gesture: {pred_label} ({confidence:.2f})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

        cv2.imshow("Live Gesture Recognition (PyTorch + MediaPipe)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
