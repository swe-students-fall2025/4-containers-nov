from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


HAGRID_ROOT = Path("/Users/yfanw./Downloads/hagrid-classification-512p")

USED_CLASSES = [
    "palm",
    "fist",
    "like",
    "stop",
    "ok",
    "one",
    "two_up",
    "three",
]

MAX_IMAGES_PER_CLASS = 1200

DATA_DIR = Path("data")
X_PATH = DATA_DIR / "hagrid_keypoints_X.npy"
Y_PATH = DATA_DIR / "hagrid_keypoints_y.npy"
CLASSES_PATH = DATA_DIR / "hagrid_classes.json"


mp_hands = mp.solutions.hands

NUM_LANDMARKS = 21
DIM_PER_LM = 3


def collect_image_paths(
    root: Path, classes: List[str], max_per_class: int
) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    for label_idx, cls in enumerate(classes):
        cls_dir = root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")

        img_paths = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix in exts]

        if not img_paths:
            raise RuntimeError(f"No images found in {cls_dir}")

        random.shuffle(img_paths)
        take = min(len(img_paths), max_per_class)

        for p in img_paths[:take]:
            samples.append((p, label_idx))

        print(f"[INFO] Class '{cls}' -> using {take} images")

    random.shuffle(samples)
    print(f"[INFO] Total images collected: {len(samples)}")
    return samples


def extract_landmark_vector(hand_landmarks) -> np.ndarray:
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    samples = collect_image_paths(HAGRID_ROOT, USED_CLASSES, MAX_IMAGES_PER_CLASS)

    X_list = []
    y_list = []

    num_no_hand = 0
    num_ok = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        for idx, (img_path, label_idx) in enumerate(samples, start=1):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                num_no_hand += 1
                continue

            hand = results.multi_hand_landmarks[0]
            vec = extract_landmark_vector(hand)

            if vec.shape[0] != NUM_LANDMARKS * DIM_PER_LM:
                print(f"[WARN] Unexpected vector shape for {img_path}: {vec.shape}")
                continue

            X_list.append(vec)
            y_list.append(label_idx)
            num_ok += 1

            if idx % 200 == 0:
                print(f"[PROGRESS] Processed {idx}/{len(samples)} images")

    if not X_list:
        raise RuntimeError("No samples with detected hands. Check your settings.")

    X = np.stack(X_list, axis=0)  # shape: (N, 63)
    y = np.array(y_list, dtype=np.int64)

    print(f"[DONE] Extracted {X.shape[0]} samples.")
    print(f"        Feature dimension: {X.shape[1]}")
    print(f"        Images with no hand detected and skipped: {num_no_hand}")

    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(USED_CLASSES, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] X -> {X_PATH}")
    print(f"[SAVED] y -> {Y_PATH}")
    print(f"[SAVED] class names -> {CLASSES_PATH}")


if __name__ == "__main__":
    main()
