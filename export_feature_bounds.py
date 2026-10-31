"""Export lightweight training feature boundary statistics (1st/99th percentiles and categorical support) to JSON."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = Path("Data_time_moisture.csv")
DEFAULT_MODEL_PATH = Path("models/agtron_model.cbm")
DEFAULT_BOUNDS_PATH = Path("models/feature_bounds.json")
MIN_CATEGORICAL_SUPPORT = 20


def compute_model_sha256(model_path: Path) -> str:
    """Return short 12-char SHA-256 hex digest of the model file if present."""
    if not model_path.exists():
        return "unknown"
    hasher = hashlib.sha256()
    with model_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def build_feature_bounds_dict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    min_categorical_support: int = MIN_CATEGORICAL_SUPPORT,
) -> dict[str, Any]:
    """Compute 1st/99th percentile bounds for numeric features and low-support counts for categoricals."""
    numeric_cols = ["Consumed", "Temp", "Quench"]
    units = {"Consumed": "lb", "Temp": "°F", "Quench": "lb"}
    quantiles_df = X_train[numeric_cols].quantile([0.01, 0.99])
    raw_quantiles_dict = quantiles_df.to_dict()

    numeric_bounds: dict[str, dict[str, Any]] = {}
    for col in numeric_cols:
        col_series = X_train[col].astype(float)
        numeric_bounds[col] = {
            "p01": round(float(raw_quantiles_dict[col][0.01]), 2),
            "p99": round(float(raw_quantiles_dict[col][0.99]), 2),
            "min": round(float(col_series.min()), 2),
            "max": round(float(col_series.max()), 2),
            "unit": units[col],
        }

    y_series = y_train.astype(float)
    y_quantiles = y_series.quantile([0.01, 0.99]).to_dict()
    numeric_bounds["target_agtron"] = {
        "p01": round(float(y_quantiles[0.01]), 2),
        "p99": round(float(y_quantiles[0.99]), 2),
        "min": round(float(y_series.min()), 2),
        "max": round(float(y_series.max()), 2),
        "unit": "Agtron",
    }

    categorical_support: dict[str, dict[str, Any]] = {}
    for cat_col in ["Month", "Blend"]:
        counts = X_train[cat_col].astype(str).value_counts()
        low_support = {
            str(k): int(v) for k, v in counts[counts < min_categorical_support].items()
        }
        # Sort keys numerically for deterministic JSON diffs
        sorted_low_support = dict(
            sorted(low_support.items(), key=lambda item: int(item[0]))
        )
        categorical_support[cat_col] = {
            "min_support": min_categorical_support,
            "total_unique": int(counts.shape[0]),
            "low_support_counts": sorted_low_support,
        }

    return {
        "model_version": compute_model_sha256(model_path),
        "n_train_rows": len(X_train),
        "quantiles": [0.01, 0.99],
        "min_categorical_support": min_categorical_support,
        "numeric_bounds": numeric_bounds,
        "categorical_support": categorical_support,
    }


def export_feature_bounds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    output_path: Path = DEFAULT_BOUNDS_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Path:
    """Atomically write training feature boundary statistics to models/feature_bounds.json."""
    bounds_payload = build_feature_bounds_dict(X_train, y_train, model_path=model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(bounds_payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, output_path)
    return output_path


def main() -> None:
    """Load local training data using the canonical split and export models/feature_bounds.json."""
    df = pd.read_csv(DEFAULT_DATA_PATH)
    df["Blend"] = df["Blend"].astype(int).astype(str)
    df["Month"] = df["Month"].astype(str)

    X = df[["Consumed", "Temp", "Quench", "Month", "Blend"]]
    y_agtron = df["Agtron"]

    X_train, _, y_train_agtron, _ = train_test_split(
        X, y_agtron, test_size=0.20, random_state=28
    )

    out_path = export_feature_bounds(X_train, y_train_agtron)
    print(f"Exported feature bounds from {len(X_train):,} training rows to {out_path}")


if __name__ == "__main__":
    main()
