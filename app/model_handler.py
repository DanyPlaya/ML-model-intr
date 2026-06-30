"""Model loading, input validation and inference helpers."""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model_v1.pkl"
DEFAULT_MODEL_PATHS = {
    "v1": DEFAULT_MODEL_PATH,
    "v2": Path(__file__).resolve().parents[1] / "models" / "model_v2.pkl",
}


def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()


def load_model(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    selected_path = Path(path).resolve() if path else model_path()
    if not selected_path.exists():
        raise FileNotFoundError(f"Model file not found: {selected_path}")

    with selected_path.open("rb") as file:
        model_bundle = pickle.load(file)

    if not isinstance(model_bundle, dict) or "model" not in model_bundle:
        model_bundle = {
            "model": model_bundle,
            "feature_names": FEATURE_NAMES,
            "model_version": selected_path.stem,
        }

    model_bundle.setdefault("feature_names", FEATURE_NAMES)
    model_bundle.setdefault("model_version", selected_path.stem)
    return model_bundle


def load_models() -> dict[str, dict[str, Any]]:
    configured_paths = {
        "v1": Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve(),
        "v2": Path(os.getenv("MODEL_PATH_V2", str(DEFAULT_MODEL_PATHS["v2"]))).resolve(),
    }

    models: dict[str, dict[str, Any]] = {}
    for alias, path in configured_paths.items():
        if path.exists():
            bundle = load_model(path)
            bundle["model_alias"] = alias
            models[alias] = bundle

    if not models:
        raise FileNotFoundError("No model artifacts were found for v1 or v2.")
    return models


def preprocess_input(data: dict[str, Any], feature_names: list[str] | None = None) -> np.ndarray:
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")

    names = feature_names or FEATURE_NAMES
    missing = [name for name in names if name not in data]
    extra = sorted(set(data) - set(names))

    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unknown features: {', '.join(extra)}")

    values: list[float] = []
    for name in names:
        try:
            value = float(data[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature {name} must be numeric.") from exc
        if not math.isfinite(value):
            raise ValueError(f"Feature {name} must be finite.")
        values.append(value)

    return np.array(values, dtype=float).reshape(1, -1)


def predict_default(model_bundle: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    feature_names = list(model_bundle.get("feature_names", FEATURE_NAMES))
    features = preprocess_input(data, feature_names)
    model = model_bundle["model"]

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    rounded_probability = round(probability, 6)
    artifact_version = str(model_bundle.get("model_version", "unknown"))
    public_version = str(model_bundle.get("model_alias", artifact_version))

    return {
        "prediction": prediction,
        "probability_default": rounded_probability,
        "probability": rounded_probability,
        "model_version": public_version,
        "model_artifact": artifact_version,
    }
