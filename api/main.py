"""FastAPI service exposing Agtron score prediction and roast settings recommendation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from api.ui import render_mobile_ui

DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/agtron_model.cbm"))
DEFAULT_PREDICTION_LOG_PATH = Path(
    os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl")
)


class JsonFormatter(logging.Formatter):
    """Format standard Python log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_data = getattr(record, "structured_data", None)
        if isinstance(extra_data, dict):
            payload.update(extra_data)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logger() -> logging.Logger:
    """Configure and return the structured JSON logger for the API service."""
    api_logger = logging.getLogger("agtron.api")
    if not api_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        api_logger.addHandler(handler)
        api_logger.setLevel(logging.INFO)
        api_logger.propagate = False
    return api_logger


logger = configure_logger()

# All 249 blend codes observed in training data
KNOWN_BLENDS: frozenset[str] = frozenset(
    {
        "1",
        "102",
        "103",
        "105",
        "106",
        "107",
        "108",
        "112",
        "113",
        "115",
        "125",
        "127",
        "132",
        "133",
        "137",
        "142",
        "144",
        "148",
        "152",
        "156",
        "159",
        "160",
        "190",
        "200",
        "201",
        "202",
        "205",
        "207",
        "208",
        "210",
        "212",
        "214",
        "215",
        "216",
        "217",
        "218",
        "219",
        "220",
        "222",
        "223",
        "224",
        "225",
        "226",
        "227",
        "228",
        "229",
        "230",
        "231",
        "232",
        "234",
        "235",
        "236",
        "240",
        "241",
        "242",
        "245",
        "247",
        "249",
        "250",
        "251",
        "252",
        "253",
        "254",
        "256",
        "257",
        "258",
        "261",
        "267",
        "268",
        "270",
        "271",
        "272",
        "273",
        "274",
        "280",
        "281",
        "282",
        "284",
        "286",
        "287",
        "288",
        "289",
        "291",
        "292",
        "293",
        "294",
        "295",
        "299",
        "302",
        "304",
        "305",
        "306",
        "309",
        "311",
        "312",
        "313",
        "314",
        "318",
        "319",
        "320",
        "331",
        "335",
        "340",
        "341",
        "342",
        "350",
        "379",
        "380",
        "383",
        "396",
        "397",
        "398",
        "399",
        "400",
        "405",
        "407",
        "410",
        "414",
        "434",
        "435",
        "442",
        "444",
        "450",
        "452",
        "454",
        "456",
        "458",
        "484",
        "490",
        "492",
        "494",
        "498",
        "504",
        "507",
        "509",
        "514",
        "516",
        "517",
        "518",
        "521",
        "531",
        "533",
        "534",
        "536",
        "560",
        "581",
        "583",
        "584",
        "600",
        "601",
        "602",
        "603",
        "604",
        "605",
        "607",
        "615",
        "617",
        "620",
        "624",
        "625",
        "626",
        "628",
        "630",
        "632",
        "634",
        "636",
        "650",
        "675",
        "676",
        "680",
        "685",
        "690",
        "693",
        "695",
        "696",
        "697",
        "698",
        "700",
        "702",
        "704",
        "710",
        "712",
        "715",
        "716",
        "717",
        "718",
        "720",
        "722",
        "723",
        "724",
        "725",
        "727",
        "729",
        "731",
        "732",
        "733",
        "734",
        "737",
        "739",
        "741",
        "743",
        "756",
        "763",
        "795",
        "805",
        "815",
        "822",
        "825",
        "826",
        "827",
        "830",
        "831",
        "832",
        "833",
        "834",
        "835",
        "836",
        "837",
        "838",
        "839",
        "841",
        "842",
        "850",
        "851",
        "852",
        "861",
        "862",
        "865",
        "868",
        "870",
        "880",
        "882",
        "884",
        "902",
        "904",
        "910",
        "911",
        "912",
        "925",
        "927",
        "956",
        "960",
        "962",
        "964",
        "966",
        "968",
        "970",
        "974",
        "9010",
    }
)

VALID_MONTHS: frozenset[str] = frozenset(str(m) for m in range(1, 13))


def clean_blend_input(blend_val: Any) -> str:
    """Normalize blend input to clean integer string (e.g. '284.0' or ' 284 ' -> '284')."""
    if isinstance(blend_val, bool) or blend_val is None:
        raise ValueError("Blend must be a valid blend code.")
    try:
        cleaned = str(blend_val).strip()
        numeric = float(cleaned)
        if not numeric.is_integer():
            raise ValueError(f"Blend must be an integer code, got '{blend_val}'.")
        normalized = str(int(numeric))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid blend format: '{blend_val}'.") from exc

    if normalized not in KNOWN_BLENDS:
        raise ValueError(
            f"Unknown blend '{normalized}'. Must be a known training blend."
        )
    return normalized


def compute_file_sha256(path: Path) -> str:
    """Return short 12-char SHA-256 hex digest of a model artifact file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def load_catboost_model(
    model_path: Path | None = None,
) -> tuple[CatBoostRegressor, str]:
    """Load CatBoostRegressor from disk and return (model, version_hash)."""
    resolved_path = (
        model_path
        if model_path is not None
        else Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    )
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found at {resolved_path}")
    model = CatBoostRegressor()
    model.load_model(str(resolved_path))
    model_hash = compute_file_sha256(resolved_path)
    return model, model_hash


def get_prediction_log_path() -> Path:
    """Resolve the prediction JSONL log path from environment or default."""
    return Path(os.getenv("PREDICTION_LOG_PATH", str(DEFAULT_PREDICTION_LOG_PATH)))


def log_prediction_record(
    *,
    endpoint: str,
    payload: RoastInput,
    predicted_agtron: float,
    model_version: str,
    latency_ms: float,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a structured JSON prediction record to logs/predictions.jsonl and the service logger."""
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": "prediction",
        "endpoint": endpoint,
        "Consumed": float(payload.Consumed),
        "Temp": float(payload.Temp),
        "Quench": float(payload.Quench),
        "Month": str(payload.Month),
        "Blend": str(payload.Blend),
        "predicted_agtron": round(float(predicted_agtron), 2),
        "Agtron": round(float(predicted_agtron), 2),
        "model_version": model_version,
        "latency_ms": round(float(latency_ms), 3),
    }
    if extra_fields:
        record.update(extra_fields)

    log_path = get_prediction_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Prediction served", extra={"structured_data": record})
    return record


def _ensure_model_loaded(app_instance: FastAPI) -> None:
    """Ensure model is loaded on app.state (supports both lifespan and direct TestClient calls)."""
    if getattr(app_instance.state, "model", None) is not None:
        return
    try:
        model, model_hash = load_catboost_model()
        app_instance.state.model = model
        app_instance.state.model_version = model_hash
        app_instance.state.model_loaded = True
        logger.info(
            "Model loaded successfully",
            extra={
                "structured_data": {
                    "event": "model_loaded",
                    "model_version": model_hash,
                }
            },
        )
    except (OSError, RuntimeError, ValueError) as exc:
        app_instance.state.model = None
        app_instance.state.model_version = "unavailable"
        app_instance.state.model_loaded = False
        logger.error(
            "Failed to load model",
            extra={"structured_data": {"event": "model_load_error", "error": str(exc)}},
        )


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """Load the CatBoost model once at startup and clean up on shutdown."""
    _ensure_model_loaded(app_instance)
    yield
    app_instance.state.model = None
    app_instance.state.model_loaded = False


class RoastInput(BaseModel):
    """Input features for a coffee roast prediction or recommendation request."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "Consumed": 500.0,
                    "Temp": 440.0,
                    "Quench": 73.0,
                    "Month": "1",
                    "Blend": "284",
                    "target_agtron": 45.0,
                }
            ]
        },
    )

    Consumed: float = Field(
        ...,
        gt=0.0,
        le=1000.0,
        validation_alias=AliasChoices("Consumed", "consumed"),
        description="Consumed green coffee weight in pounds (lb).",
    )
    Temp: float = Field(
        ...,
        ge=300.0,
        le=600.0,
        validation_alias=AliasChoices("Temp", "temp"),
        description="Roast temperature in degrees Fahrenheit (°F).",
    )
    Quench: float = Field(
        ...,
        ge=0.0,
        le=150.0,
        validation_alias=AliasChoices("Quench", "quench"),
        description="Quench water applied in pounds (lb).",
    )
    Month: str = Field(
        ...,
        validation_alias=AliasChoices("Month", "month"),
        description="Roast month ('1' through '12').",
    )
    Blend: str = Field(
        ...,
        validation_alias=AliasChoices("Blend", "blend"),
        description="Coffee blend code (e.g. '284').",
    )
    target_agtron: float = Field(
        default=45.0,
        ge=15.0,
        le=65.0,
        validation_alias=AliasChoices("target_agtron", "Target_Agtron", "TargetAgtron"),
        description="Target Agtron score for recommendations (15.0 to 65.0).",
    )

    @field_validator("Month", mode="before")
    @classmethod
    def validate_month(cls, value: Any) -> str:
        if isinstance(value, bool) or value is None:
            raise ValueError("Month must be between 1 and 12.")
        try:
            cleaned = str(int(float(str(value).strip())))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid month: '{value}'. Must be an integer from 1 to 12."
            ) from exc
        if cleaned not in VALID_MONTHS:
            raise ValueError(f"Month must be between 1 and 12, got '{value}'.")
        return cleaned

    @field_validator("Blend", mode="before")
    @classmethod
    def validate_blend(cls, value: Any) -> str:
        return clean_blend_input(value)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert validated roast inputs into a single-row DataFrame matching model feature order."""
        return pd.DataFrame(
            {
                "Consumed": [float(self.Consumed)],
                "Temp": [float(self.Temp)],
                "Quench": [float(self.Quench)],
                "Month": [str(self.Month)],
                "Blend": [str(self.Blend)],
            }
        )


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = Field(..., description="Service health status ('ok' or 'degraded').")
    model_loaded: bool = Field(
        ..., description="Whether the CatBoost model is loaded in memory."
    )
    model_version: str = Field(
        ..., description="Short SHA-256 hash of the loaded model artifact."
    )
    model_hash: str = Field(
        ..., description="Short SHA-256 hash of the loaded model artifact."
    )


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""

    predicted_agtron: float = Field(..., description="Predicted Agtron color score.")
    model_version: str = Field(
        ..., description="Model version/hash used for inference."
    )


class RecommendationCandidate(BaseModel):
    """A single candidate (Temp, Quench) adjustment evaluated by the recommender."""

    temp: float = Field(..., description="Candidate roast temperature (°F).")
    quench: float = Field(..., description="Candidate quench weight (lb).")
    expected_agtron: float = Field(
        ..., description="Predicted Agtron score at candidate settings."
    )
    error: float = Field(..., description="Absolute distance from target Agtron.")


class RecommendationResponse(BaseModel):
    """Response schema for POST /recommend."""

    current_predicted_agtron: float = Field(
        ..., description="Predicted Agtron for current input settings."
    )
    target_agtron: float = Field(..., description="Requested target Agtron score.")
    gap_to_target: float = Field(
        ..., description="Difference (target_agtron - current_predicted_agtron)."
    )
    recommended_temp: float = Field(..., description="Optimal roast temperature (°F).")
    recommended_quench: float = Field(..., description="Optimal quench weight (lb).")
    expected_agtron: float = Field(
        ..., description="Expected Agtron score at recommended settings."
    )
    distance_from_target: float = Field(
        ..., description="Absolute error between expected and target Agtron."
    )
    top_recommendations: list[RecommendationCandidate] = Field(
        ...,
        description="Top 5 candidate settings ranked by smallest distance from target.",
    )
    model_version: str = Field(
        ..., description="Model version/hash used for inference."
    )


app = FastAPI(
    title="Agtron Score Predictor API",
    description="REST API for coffee roast Agtron color prediction and roast parameter recommendation.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_http_requests(
    request: Request, call_next: Callable[[Request], Any]
) -> Response:
    """Middleware that logs structured JSON metadata and latency for every HTTP request."""
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
    logger.info(
        "HTTP request completed",
        extra={
            "structured_data": {
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            }
        },
    )
    return response


def _get_loaded_model(request: Request) -> tuple[CatBoostRegressor, str]:
    _ensure_model_loaded(request.app)
    model = getattr(request.app.state, "model", None)
    model_version = getattr(request.app.state, "model_version", "unavailable")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return model, model_version


SORTED_KNOWN_BLENDS: list[str] = sorted(KNOWN_BLENDS, key=int)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    """Serve the mobile-responsive web UI at the root URL."""
    return HTMLResponse(content=render_mobile_ui(SORTED_KNOWN_BLENDS))


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(request: Request) -> HealthResponse:
    """Return service health status, whether the model is loaded, and the model version hash."""
    _ensure_model_loaded(request.app)
    loaded = bool(getattr(request.app.state, "model_loaded", False))
    version = str(getattr(request.app.state, "model_version", "unavailable"))
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=version,
        model_hash=version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_agtron(payload: RoastInput, request: Request) -> PredictionResponse:
    """Predict the Agtron color score for a given set of roast inputs."""
    start_time = time.perf_counter()
    model, model_version = _get_loaded_model(request)
    input_df = payload.to_dataframe()
    prediction = round(float(model.predict(input_df)[0]), 2)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    log_prediction_record(
        endpoint="/predict",
        payload=payload,
        predicted_agtron=prediction,
        model_version=model_version,
        latency_ms=latency_ms,
    )

    return PredictionResponse(
        predicted_agtron=prediction,
        model_version=model_version,
    )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Inference"])
def recommend_settings(payload: RoastInput, request: Request) -> RecommendationResponse:
    """Recommend optimal temperature and quench adjustments to hit a target Agtron score."""
    start_time = time.perf_counter()
    model, model_version = _get_loaded_model(request)

    input_df = payload.to_dataframe()
    current_prediction = float(model.predict(input_df)[0])
    target = float(payload.target_agtron)
    gap = target - current_prediction

    temp_changes = list(range(-11, 10))
    quench_changes = list(range(-6, 5))

    grid = [
        (payload.Temp + tc, payload.Quench + qc)
        for tc in temp_changes
        for qc in quench_changes
        if 300.0 <= payload.Temp + tc <= 600.0 and 0.0 <= payload.Quench + qc <= 150.0
    ]
    candidate_temps, candidate_quenches = zip(*grid)
    n_candidates = len(grid)

    candidate_df = pd.DataFrame(
        {
            "Consumed": [float(payload.Consumed)] * n_candidates,
            "Temp": list(candidate_temps),
            "Quench": list(candidate_quenches),
            "Month": [str(payload.Month)] * n_candidates,
            "Blend": [str(payload.Blend)] * n_candidates,
        }
    )

    candidate_predictions = model.predict(candidate_df)
    candidate_errors = [abs(target - float(pred)) for pred in candidate_predictions]

    recommendation_df = (
        pd.DataFrame(
            {
                "Temp": candidate_temps,
                "Quench": candidate_quenches,
                "Expected Agtron": [round(float(p), 2) for p in candidate_predictions],
                "Error": [round(float(e), 2) for e in candidate_errors],
            }
        )
        .sort_values("Error")
        .reset_index(drop=True)
    )

    best_row = recommendation_df.iloc[0]
    top_candidates = [
        RecommendationCandidate(
            temp=float(row["Temp"]),
            quench=float(row["Quench"]),
            expected_agtron=float(row["Expected Agtron"]),
            error=float(row["Error"]),
        )
        for _, row in recommendation_df.head(5).iterrows()
    ]

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    log_prediction_record(
        endpoint="/recommend",
        payload=payload,
        predicted_agtron=current_prediction,
        model_version=model_version,
        latency_ms=latency_ms,
        extra_fields={
            "target_agtron": round(target, 2),
            "recommended_temp": float(best_row["Temp"]),
            "recommended_quench": float(best_row["Quench"]),
            "expected_agtron": float(best_row["Expected Agtron"]),
        },
    )

    return RecommendationResponse(
        current_predicted_agtron=round(current_prediction, 2),
        target_agtron=round(target, 2),
        gap_to_target=round(gap, 2),
        recommended_temp=float(best_row["Temp"]),
        recommended_quench=float(best_row["Quench"]),
        expected_agtron=float(best_row["Expected Agtron"]),
        distance_from_target=float(best_row["Error"]),
        top_recommendations=top_candidates,
        model_version=model_version,
    )
