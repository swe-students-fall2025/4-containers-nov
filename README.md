![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# Containerized App Exercise

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

# Teammates

Ivan Wang, [Harrison Gao](https://github.com/HTK-G), [Sina Liu](https://github.com/SinaL0123), Serena, [Hanqi Gui](https://github.com/hanqigui)

# Machine Learning Client — Hand Gesture Recognition

This folder contains the **machine-learning-client** subsystem of our 3-container project:

- **Machine Learning Client** → collects sensor data (webcam), performs gesture recognition with MediaPipe + PyTorch, and later sends results to MongoDB.
- **Web App** (TBD) → visualizes gesture events stored in the database.
- **MongoDB** → central datastore for gesture metadata.

The ML client runs entirely as a _backend service_ (no user-facing UI).  
It processes camera input, performs ML inference, and will later communicate with the database once integrated with the web app.

---

# 1. Project Structure

## Project Structure

```text
├── docker-compose.yml
├── instructions.md
├── LICENSE
├── machine-learning-client
│   ├── data
│   │   ├── hagrid_keypoints_X.npy
│   │   └── hagrid_keypoints_y.npy
│   ├── Dockerfile
│   ├── models
│   │   ├── gesture_mlp.pt
│   │   └── train_mlp.py
│   ├── Pipfile
│   ├── Pipfile.lock
│   ├── src
│   │   ├── __init__.py
│   │   ├── extract_keypoints_from_hagrid.py
│   │   └── live_mediapipe_mlp.py
│   └── tests
│       ├── __init__.py
│       ├── test_extract_keypoints_from_hagrid.py
│       └── test_live_mediapipe_mlp.py
├── README.md
└── web-app
    ├── app.py
    ├── Dockerfile
    ├── Pipfile
    ├── Pipfile.lock
    ├── readme.txt
    ├── static
    │   ├── audios
    │   │   ├── among_us.mp3
    │   │   ├── android_beep.mp3
    │   │   ├── bom.mp3
    │   │   ├── error.mp3
    │   │   ├── playme.mp3
    │   │   ├── rick_roll.mp3
    │   │   ├── rizz.mp3
    │   │   ├── sponge_bob.mp3
    │   │   └── uwu.mp3
    │   ├── hagrid_classes.json
    │   ├── images
    │   │   ├── fist.png
    │   │   ├── like.png
    │   │   ├── ok.png
    │   │   ├── one.png
    │   │   ├── palm.png
    │   │   ├── stop.png
    │   │   ├── thinking.png
    │   │   ├── three.png
    │   │   └── two_up.png
    │   ├── script.js
    │   └── style.css
    ├── templates
    │   └── index.html
    └── tests
        ├──   __init__.py
        ├── conftest.py
        └── test_app.py
```

# 2. Environment Setup (macOS, M-series)

## **1. Install pipenv (if not installed)**

```bash
pip install pipenv
```

## 2. Install all ML client dependencies

From the repository root:

```bash
cd machine-learning-client
pipenv install --dev
```

This installs all dependencies, including:

- mediapipe
- opencv-python
- numpy
- torch (with MPS acceleration for Apple Silicon)
- pylint + black
- pytest (required later for unit testing)

# 3. Run Live Gesture Recognition (MediaPipe + PyTorch)

Make sure your webcam is connected, then run:

```bash
cd machine-learning-client
pipenv run python src/live_mediapipe_mlp.py
```

You should see:

- a live webcam preview window

- detected hand skeletons

- predicted gesture label displayed on the frame

Press **q** to exit.

## Note About Running the ML Client in Docker on macOS

On macOS, Docker containers cannot easily access the host webcam, because the macOS camera is not exposed as a Linux-style /dev/video0 device inside containers.
For this reason, the live gesture recognition demo cannot run inside the Docker container on macOS.

During development and demo, we run the ML client directly on the host machine, where the webcam works normally:

```bash
pipenv run python src/live_mediapipe_mlp.py
```

The Docker image of the ML client is still fully functional for:

- CI / GitHub Actions

- dependency isolation

- database integration tests

- running without a webcam (e.g., headless mode)

This behavior is expected on macOS and does not affect the overall functionality of the 3-container system.

# 4. MongoDB Integration

The ML client is already connected to MongoDB using pymongo.

In src/live_mediapipe_mlp.py, a MongoDB database named handsense is created:

```bash
mongo_client = MongoClient("mongodb://localhost:27017")
mongo_db = mongo_client["handsense"]
gesture_collection = mongo_db["gesture_events"]
```

For every detected hand gesture, an event document is inserted:

```bash
event = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "gesture": pred_label,
    "confidence": confidence,
    "handedness": handedness,
}
gesture_collection.insert_one(event)
```

This allows the Web App subsystem to read and visualize real-time gesture activity from:

```bash
handsense.gesture_events
```
