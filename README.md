![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# Containerized App Exercise

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

# Machine Learning Client – Hand Gesture Recognition

This is the **machine-learning-client** part of our 3-container system:

- **Machine-Learning-Client**: Runs gesture recognition using MediaPipe + PyTorch MLP
- **Web-App**: Flask dashboard (TBD)
- **Mongodb**: Stores gesture events

This README explains how to set up and run the **local ML client** on macOS (M-series),
so that you can:
1. Re-extract features from the HaGRID dataset (offline)
2. Train an MLP classifier in PyTorch
3. Run a live webcam demo that recognizes a set of hand gestures

---

## 1. Project structure (ML client)

```text
machine-learning-client/
  data/
    hagrid_keypoints_X.npy      # extracted keypoint features (N x 63)
    hagrid_keypoints_y.npy      # integer labels (N,)
    hagrid_classes.json         # class name list, e.g. ["palm", "fist", ...]
  models/
    gesture_mlp.pt              # trained PyTorch MLP
  src/
    extract_keypoints_from_hagrid.py   # offline feature extraction from HaGRID
    train_mlp_torch.py                 # trains the MLP on keypoints
    live_mediapipe_mlp.py              # live webcam demo using MediaPipe + MLP
    offline_mediapipe_kaggle_demo.py   # small offline demo (optional)
    ...                                # other helper modules
  requirements-macos.txt
  README.md
