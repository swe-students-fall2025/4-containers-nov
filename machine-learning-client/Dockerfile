# Base Python image (pretty lightweight)
FROM python:3.10-slim

# Prevent Python from writing .pyc files, and make logs show instantly
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Work inside /app
WORKDIR /app

# Install system packages that MediaPipe / OpenCV need
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies using Pipenv
COPY Pipfile Pipfile.lock ./
RUN pip install --no-cache-dir pipenv && \
    pipenv install --system --deploy --ignore-pipfile

# Copy the actual client code + model files
COPY src ./src
COPY models ./models

# MongoDB connection settings
# (Note: `mongodb` is the service name we will use in docker-compose)
ENV MONGO_URI=mongodb://mongodb:27017 \
    MONGO_DB_NAME=handsense

# Run the gesture recognition client by default
CMD ["python", "-u", "src/live_mediapipe_mlp.py"]
