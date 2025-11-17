FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies using Pipenv
COPY Pipfile Pipfile.lock ./
RUN pip install --no-cache-dir pipenv && \
    pipenv install --system --deploy --ignore-pipfile

# Copy Flask code and assets
COPY app.py .
COPY templates ./templates
COPY static ./static

# Default Flask + Mongo config (can be overridden at runtime)
ENV FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    MONGO_URI=mongodb://mongodb:27017 \
    MONGO_DB_NAME=handsense

# Start the Flask dev server when the container runs
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
