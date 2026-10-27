# syntax=docker/dockerfile:1

# ==========================================
# Stage 1: Dependency Builder
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==========================================
# Stage 2: Production Runtime
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=models/agtron_model.cbm \
    PREDICTION_LOG_PATH=logs/predictions.jsonl

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code and model artifact
COPY api/ ./api/
COPY models/ ./models/
COPY app.py ./app.py
COPY verify_model.py ./verify_model.py
COPY pytest.ini ./pytest.ini
COPY tests/ ./tests/

RUN mkdir -p /app/logs

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

