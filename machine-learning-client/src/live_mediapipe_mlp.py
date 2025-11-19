from pathlib import Path
from datetime import datetime, timezone
import time
import os

import cv2
import mediapipe as mp
import numpy as np
import torch
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# pylint: disable=global-statement

MODEL_PATH = Path("models/gesture_mlp.pt")

# Add environment variable-based configuration above the global variables.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "handsense")

mongo_client = None
mongo_db = None
gesture_collection = None
controls_collection = None


def init_db() -> None:
    """Initialize MongoDB connection (handsense DB).

    Uses MONGO_URI and MONGO_DB_NAME from environment variables so it works
    both on localhost and inside Docker (where Mongo runs as `mongodb`).
    """
    global mongo_client, mongo_db, gesture_collection, controls_collection

    if mongo_client is not None:
        return

    try:
        # Shorter timeout so we fail fast if Mongo is not available
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # Simple ping to check the connection
        mongo_client.admin.command("ping")
    except PyMongoError as e:
        print("[ERROR] Could not connect to MongoDB:", e)
        return

    print(f"[INFO] MongoDB connected successfully at {MONGO_URI}")

    mongo_db = mongo_client[MONGO_DB_NAME]
    gesture_collection = mongo_db["gesture_events"]
    controls_collection = mongo_db["controls"]

    # Make sure we always have a default capture state document
    if controls_collection.find_one({"_id": "capture_control"}) is None:
        controls_collection.insert_one({"_id": "capture_control", "enabled": False})
        print("[INFO] Initialized capture_control = False")


# def should_capture() -> bool:
#     """Read capture state from DB (Flask updates this)."""
#     doc = controls_collection.find_one({"_id": "capture_control"})
#     if doc is None:
#         return False
#     return bool(doc.get("enabled", False))

# --- Cache capture state to avoid hitting DB every frame ---
_last_check_time = 0
_cached_capture_state = False

def should_capture(rate_limit=0.5):
    """Only hit DB at most once every rate_limit seconds."""
    global _last_check_time, _cached_capture_state

    now = time.time()
    if now - _last_check_time < rate_limit:
        return _cached_capture_state

    doc = controls_collection.find_one(
        {"_id": "capture_control"},
        {"enabled": 1}
    )
    _cached_capture_state = bool(doc.get("enabled", False)) if doc else False
    _last_check_time = now
    return _cached_capture_state



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
    """Extract 21 (x, y, z) hand landmarks into a 63-d vector."""
    kp = []
    for lm in hand_landmarks.landmark:
        kp.extend([lm.x, lm.y, lm.z])
    return np.array(kp, dtype=np.float32)


def main():
    # pylint: disable=too-many-statements

    # ---- Initialize MongoDB ----
    init_db()

    # ---- Load ML model ----
    model, class_names = load_model()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"[INFO] Using device: {device}")

    # ---- MediaPipe ----
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

    last_label = None
    last_logged_at = 0.0
    min_interval = 1.0
    min_confidence = 0.8

    while True:
        # --------------------------
        # 1. Check capture toggle
        # --------------------------
        if not should_capture():
            cv2.imshow(
                "Live Gesture Recognition (PyTorch + MediaPipe)",
                cv2.putText(
                    np.zeros((480, 640, 3), dtype=np.uint8),
                    "Capture OFF",
                    (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    4,
                ),
            )
            if cv2.waitKey(200) & 0xFF == ord("q"):
                break
            continue

        # --------------------------
        # 2. If capture ON → Process
        # --------------------------
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        pred_label = "No Hand"
        confidence = 0.0
        handedness = "Unknown"

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                kp = extract_keypoints(hand_landmarks)
                kp_t = torch.from_numpy(kp).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(kp_t)
                    prob = torch.softmax(logits, dim=1)
                    conf, pred_idx = prob.max(dim=1)
                    pred_label = class_names[pred_idx.item()]
                    confidence = float(conf.item())

                handedness = hand_info.classification[0].label
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # rate limiting writes
                now = time.time()
                if confidence >= min_confidence and (
                    pred_label != last_label or (now - last_logged_at) > min_interval
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
