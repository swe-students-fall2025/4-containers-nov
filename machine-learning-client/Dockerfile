FROM python:3.10-slim

WORKDIR /app

# Install minimal system libs (but NOT GUI camera libs)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY Pipfile Pipfile.lock ./

RUN pip install --upgrade pip pipenv && \
    pipenv install --deploy --dev --system

COPY src ./src
COPY tests ./tests
COPY models ./models
COPY data ./data

# Default command (run tests)
CMD ["pytest", "-q"]
