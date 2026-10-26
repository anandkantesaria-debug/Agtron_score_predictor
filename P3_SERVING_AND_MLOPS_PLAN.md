# P3: Serving and MLOps — Execution Plan

## Overview

In [PRODUCTION_READINESS_REVIEW.md](file:///c:/Users/anand/Agtron_score_predictor/PRODUCTION_READINESS_REVIEW.md#L81), **P3 (Serving and MLOps)** transitions the project from a local notebook/Streamlit script into a deployable, observable production service.

### Current State
- Model inference is tightly coupled inside [app.py](file:///c:/Users/anand/Agtron_score_predictor/app.py), meaning other systems cannot call the model via an HTTP API.
- [deploy.yaml](file:///c:/Users/anand/Agtron_score_predictor/.github/workflows/deploy.yaml) only runs tests on push—it does not lint, package a container, or deploy anything.
- Model training in [Catboost_train.py](file:///c:/Users/anand/Agtron_score_predictor/Catboost_train.py) is run manually, and once the model is serving, there is no visibility into prediction logs or whether live roast inputs are drifting away from training data.

---

## Step-by-Step Execution Breakdown

Each step below is scoped so it can be implemented, tested, and committed one at a time.

### Step 1: Build the FastAPI Service (`/health`, `/predict`, `/recommend`)
- **Goal:** Decouple model inference from the Streamlit UI and expose it over REST.
- **What to do:**
  - Create `api/main.py` using FastAPI.
  - Load the CatBoost model once at startup using a FastAPI lifespan context manager.
  - Add `GET /health` returning service status, model version/hash, and whether the model is loaded.
  - Add `POST /predict` and `POST /recommend` endpoints backed by Pydantic request/response schemas (`RoastInput`, `PredictionResponse`, `RecommendationResponse`).
- **Done when:** `uvicorn api.main:app` starts locally and returns valid responses at `/docs`.

---

### Step 2: Add API Tests (`TestClient`)
- **Goal:** Verify endpoint behavior and input validation automatically.
- **What to do:**
  - Create `tests/test_api.py` using FastAPI's `TestClient`.
  - Test `/health` (`200 OK`), `/predict` happy path (`200 OK`), `/recommend` happy path (`200 OK`), and invalid inputs such as out-of-range temperature or unknown blend (`422 Unprocessable Entity`).
- **Done when:** `pytest` passes all API tests.

---

### Step 3: Add Structured Logging and Prediction Logging
- **Goal:** Record every prediction request/response in a machine-readable format for debugging and monitoring.
- **What to do:**
  - Configure structured JSON logging (`structlog` or Python's `logging` with a JSON formatter) in the FastAPI service.
  - Add middleware or endpoint logging that logs request latency, input features (`Blend`, `Temp`, `Quench`, `Consumed`, `Month`), predicted Agtron score, and model version to a log sink or append-only file/table (e.g., `logs/predictions.jsonl` or SQLite).
- **Done when:** Making a `/predict` call writes a structured JSON record with inputs, output, timestamp, and latency.

---

### Step 4: Containerize with `Dockerfile` and `docker-compose.yml`
- **Goal:** Make the API, UI, and MLflow reproducible on any machine or cloud provider.
- **What to do:**
  - Create a multi-stage `Dockerfile` (using `python:3.11-slim`) that installs pinned dependencies, copies `src/`, `api/`, `app/`, and the model artifact, and exposes the FastAPI/Streamlit ports with a Docker `HEALTHCHECK` hitting `/health`.
  - Create a `.dockerignore` to exclude `.git`, raw CSV data, local `mlruns/`, and virtual environments.
  - Create `docker-compose.yml` to run the `api`, `streamlit` UI (configured to call the API), and `mlflow` server together.
- **Done when:** `docker compose up --build` starts the stack and the UI gets predictions through the FastAPI container.

---

### Step 5: Upgrade GitHub Actions CI/CD ([deploy.yaml](file:///c:/Users/anand/Agtron_score_predictor/.github/workflows/deploy.yaml))
- **Goal:** Automate linting, type-checking, testing, Docker builds, and cloud deployment on every push to `main`.
- **What to do:**
  - Update [deploy.yaml](file:///c:/Users/anand/Agtron_score_predictor/.github/workflows/deploy.yaml) into distinct stages:
    1. **Lint & Typecheck:** `ruff check`, `ruff format --check`, and `mypy`.
    2. **Test:** `pytest` (unit + API tests using a lightweight fixture model).
    3. **Build:** Build the Docker image to verify it packages cleanly.
    4. **Deploy:** Trigger deployment to your target host (e.g., Render deploy hook, Fly.io `flyctl deploy`, Hugging Face Spaces push, or Google Cloud Run) on pushes to `main`.
- **Done when:** Pushing to `main` runs the full pipeline and updates the live service automatically.

---

### Step 6: Add Data-Drift Detection (Evidently)
- **Goal:** Detect when incoming production roast inputs or predictions shift away from the training distribution.
- **What to do:**
  - Create a drift check script (`src/agtron/drift.py` or `scripts/check_drift.py`) using `evidently`.
  - Compare a reference dataset (sample of the training split) against the recent logged predictions (`logs/predictions.jsonl`).
  - Output an HTML/JSON drift report checking feature drift (`Temp`, `Quench`, `Consumed`, `Blend` distribution) and prediction drift, and exit with a non-zero status or flag if drift exceeds a threshold.
- **Done when:** Running the script against sample prediction logs generates a drift report and flags synthetic distribution shifts.

---

### Step 7: Add Scheduled Retraining & Champion/Challenger Workflow
- **Goal:** Automate periodic model retraining and only promote a new model if it beats the current champion on a time-based holdout.
- **What to do:**
  - Refactor the promotion logic in [Catboost_train.py](file:///c:/Users/anand/Agtron_score_predictor/Catboost_train.py) so it evaluates both the newly trained model (`challenger`) and the current `champion` on the same holdout window, promoting only if RMSE/MAE improves and passes sanity checks (e.g., monotonicity test).
  - Add `.github/workflows/retrain.yaml` with a `schedule` (`cron`) and `workflow_dispatch` trigger (and/or trigger when drift is detected) that trains, logs metrics, and commits/publishes the promoted model artifact or registers it to remote MLflow.
- **Done when:** Triggering the workflow manually via `workflow_dispatch` trains a challenger, compares it against the champion, and promotes only when metrics improve.

