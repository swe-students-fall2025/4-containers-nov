# ✋ HandSense — Containerized Machine Learning + Web Dashboard System

![ML Client CI](https://github.com/swe-students-fall2025/4-containers-nov/actions/workflows/ml-client-ci.yml/badge.svg)
![Web App CI](https://github.com/swe-students-fall2025/4-containers-nov/actions/workflows/web-app-ci.yml/badge.svg)
![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

A fully containerized, three-service application that performs **real-time hand gesture recognition** using a MediaPipe + PyTorch machine-learning client, stores gesture events inside **MongoDB**, and visualizes them through a **Flask-based web dashboard**.

This project demonstrates how separate services communicate inside a Dockerized micro-service architecture.

---

## 👥 Teammates

- [Ivan Wang](https://github.com/Ivan-Wang-tech)  
- [Harrison Gao](https://github.com/HTK-G)  
- [Sina Liu](https://github.com/SinaL0123)  
- [Serena Wang](https://github.com/serena0615)  
- [Hanqi Gui](https://github.com/hanqigui)

---

## 🧱 System Overview

The system consists of **three Dockerized services**:

```
+------------------------+     +-----------------------+     +------------------------+
|   Machine Learning     |     |       MongoDB         |     |       Web App          |
|       Client           | --> |   handsense database  | --> |   Dashboard (Flask)    |
| (MediaPipe + PyTorch)  |     |     Gesture_events    |     |   Visualize gestures   |
+------------------------+     +-----------------------+     +------------------------+
```

### 🔹 Machine-Learning Client  
Runs locally or inside Docker.  
It uses a webcam → detects hands using MediaPipe → predicts gestures using a PyTorch MLP → inserts events into `handsense.gesture_events` collection.

### 🔹 MongoDB  
Stores gesture logs, statistics, and capture state toggles.

### 🔹 Web App  
Reads gesture events from MongoDB and presents a dashboard showing:

- Live latest gesture  
- Gesture distribution  
- Recent event timeline  
- Toggle capture control (`/api/control`)

After all services run, you can visit:

👉 **http://localhost:5000**

---

## 📁 Project Structure

```text
├── docker-compose.yml
├── instructions.md
├── LICENSE
├── machine-learning-client
│   ├── data
│   │   ├── hagrid_keypoints_X.npy
│   │   └── hagrid_keypoints_y.npy
│   ├── Dockerfile
│   ├── models
│   │   ├── gesture_mlp.pt
│   │   └── train_mlp.py
│   ├── Pipfile
│   ├── Pipfile.lock
│   ├── src
│   │   ├── __init__.py
│   │   ├── extract_keypoints_from_hagrid.py
│   │   └── live_mediapipe_mlp.py
│   └── tests
│       ├── __init__.py
│       ├── test_extract_keypoints_from_hagrid.py
│       └── test_live_mediapipe_mlp.py
├── README.md
└── web-app
    ├── app.py
    ├── Dockerfile
    ├── Pipfile
    ├── Pipfile.lock
    ├── readme.txt
    ├── static
    │   ├── audios
    │   │   ├── among_us.mp3
    │   │   ├── android_beep.mp3
    │   │   ├── bom.mp3
    │   │   ├── error.mp3
    │   │   ├── playme.mp3
    │   │   ├── rick_roll.mp3
    │   │   ├── rizz.mp3
    │   │   ├── sponge_bob.mp3
    │   │   └── uwu.mp3
    │   ├── hagrid_classes.json
    │   ├── images
    │   │   ├── fist.png
    │   │   ├── like.png
    │   │   ├── ok.png
    │   │   ├── one.png
    │   │   ├── palm.png
    │   │   ├── stop.png
    │   │   ├── thinking.png
    │   │   ├── three.png
    │   │   └── two_up.png
    │   ├── script.js
    │   └── style.css
    ├── templates
    │   └── index.html
    └── tests
        ├── __init__.py
        ├── conftest.py
        └── test_app.py
```

---

## ⚙️ 1. Environment Setup (Any Platform)

The recommended workflow uses **pipenv** for dependency management.

### macOS / Linux / Windows (WSL)

#### Install pipenv
```bash
pip install pipenv
```

---

## ⚙️ 2. Running the System (Docker)

From project root:

```bash
docker compose up --build
```

This starts:

| Service | URL | Purpose |
|---------|-----|---------|
| web-app | http://localhost:5000 | Dashboard UI |
| mongodb | localhost:27017 | Database |
| ml-client | headless, no UI | Captures gestures + inserts into DB |

To stop:

```bash
docker compose down
```

---

## 👁️ Running the ML Client With Webcam (macOS/Windows/Linux Host)

Since macOS Docker cannot access `/dev/video0`, we run the ML client on host machine:

```bash
cd machine-learning-client
pipenv install --dev
pipenv run python src/live_mediapipe_mlp.py
```

Features:

- Live webcam feed
- MediaPipe hand-tracking
- PyTorch gesture inference
- Inserts gesture records into `handsense.gesture_events`
- Press `q` to quit

---

## 🗄️ 3. MongoDB Configuration + Starter Data

The database name is:

```
handsense
```

Collections automatically created:

| Collection | Purpose |
|------------|---------|
| gesture_events | ML client inserts gesture data |
| controls | Stores capture toggle state |

At first run the ML client ensures:

```json
{
  "_id": "capture_control",
  "enabled": false
}
```

---

## 🔐 4. Environment Variables

Both ml-client and web-app use these:

| Variable | Description |
|----------|-------------|
| MONGO_URI | Mongo connection string (default: `mongodb://mongodb:27017`) |
| MONGO_DB_NAME | Database name (default: `handsense`) |
| SECRET_KEY | Flask sessions |

See `.env.example` below.

---

## 📄 5. .env.example (Required for TA Submission)

Place this file in project root:

```env
# MongoDB configuration
MONGO_URI=mongodb://mongodb:27017
MONGO_DB_NAME=handsense

# Flask secret
SECRET_KEY=dev-secret
```

Then create an actual `.env`:

```bash
cp .env.example .env
```

---

## 🔍 6. Web App (Flask) — Running Locally

```bash
cd web-app
pipenv install --dev
pipenv run flask run --host=0.0.0.0 --port=5000
```

Navigate to:

👉 **http://localhost:5000**

### Endpoints:

| Route | Description |
|-------|-------------|
| `/` | Dashboard UI |
| `/api/latest` | Latest gesture |
| `/api/latest_full` | Latest gesture (detailed) |
| `/api/control` | POST toggle capture |
| `/api/control/status` | GET capture control |

---

## 🧪 7. Testing + Linting + Coverage

### Run ML Client Tests
```bash
cd machine-learning-client
pipenv run pytest --cov=src
pipenv run pylint src
```

### Run Web App Tests
```bash
cd web-app
pipenv run pytest --cov=.
pipenv run pylint app.py
```

Coverage must be ≥ 80%.

---

## 🧰 8. Docker Compose

```yaml
version: "3.9"

services:
  mongodb:
    image: mongo:6
    container_name: mongodb
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db

  web-app:
    build:
      context: ./web-app
    container_name: web-app
    depends_on:
      - mongodb
    environment:
      MONGO_URI: "mongodb://mongodb:27017"
      MONGO_DB_NAME: "handsense"
      FLASK_APP: "app.py"
      FLASK_RUN_HOST: "0.0.0.0"
    ports:
      - "5000:5000"

  ml-client:
    build:
      context: ./machine-learning-client
    container_name: ml-client
    depends_on:
      - mongodb
    environment:
      MONGO_URI: "mongodb://mongodb:27017"
      MONGO_DB_NAME: "handsense"

volumes:
  mongo-data:
```
