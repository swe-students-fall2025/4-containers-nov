from __future__ import annotations

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


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
        0: "fist",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "palm",
    }
    return mapping.get(count, "unknown")


def run_live_demo():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            label_text = "no hand"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                cnt = count_extended_fingers(hand_landmarks)
                label_text = map_count_to_label(cnt)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

            cv2.putText(
                frame,
                label_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Live Gesture Demo (press q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_demo()
