# pylint: skip-file
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


HAGRID_ROOT = Path("/Users/yfanw./Downloads/hagrid-classification-512p")

USED_CLASSES = ["palm", "fist", "like", "stop", "ok"]

SAMPLES_PER_CLASS = 20


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


def collect_sample_paths(
    root: Path, classes: List[str], n_per_class: int
) -> List[Tuple[Path, str]]:

    samples = []

    for cls in classes:
        cls_dir = root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")

        img_paths = list(cls_dir.glob("*.jpeg"))

        if not img_paths:
            raise RuntimeError(f"No images found in {cls_dir}")

        random.shuffle(img_paths)
        take = min(n_per_class, len(img_paths))

        for p in img_paths[:take]:
            samples.append((p, cls))

    random.shuffle(samples)
    return samples


def count_extended_fingers(hand_landmarks) -> int:
    lm = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    wrist_y = lm[0][1]
    extended = 0

    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if lm[tip_idx][1] < lm[pip_idx][1] and lm[tip_idx][1] < wrist_y:
            extended += 1

    if lm[FINGER_TIPS[0]][0] < lm[FINGER_PIPS[0]][0]:
        extended += 1

    return extended


def map_count_to_label(count: int) -> str:
    mapping = {
        0: "fist_like",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "palm_like",
    }
    return mapping.get(count, "unknown")


def run_offline_demo():

    print("[INFO] Collecting sample image paths...")
    samples = collect_sample_paths(HAGRID_ROOT, USED_CLASSES, SAMPLES_PER_CLASS)
    print(f"[INFO] Total samples collected: {len(samples)}")

    stats = {"total": 0, "no_hand": 0}

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:

        for img_path, cls in samples:
            stats["total"] += 1

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                stats["no_hand"] += 1
                print(f"[NO HAND] {img_path.name:25s}   true={cls}")
                continue

            hand = results.multi_hand_landmarks[0]
            finger_cnt = count_extended_fingers(hand)
            pseudo_label = map_count_to_label(finger_cnt)

            print(
                f"[OK] {img_path.name:25s}  true={cls:6s}  fingers={finger_cnt}  pseudo={pseudo_label}"
            )

    print("\n==== SUMMARY ====")
    print(f"Total images: {stats['total']}")
    print(f"No hand detected: {stats['no_hand']}")


if __name__ == "__main__":
    run_offline_demo()
