"""Flask API for credit card default prediction."""

from __future__ import annotations

import json
import logging
import os
import time
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request

from app.model_handler import FEATURE_NAMES, load_models, predict_default


def configure_logging() -> None:
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "api.jsonl", encoding="utf-8"),
        ],
    )


configure_logging()
app = Flask(__name__)
MODEL_BUNDLES = load_models()
DEFAULT_MODEL_ALIAS = "v1" if "v1" in MODEL_BUNDLES else sorted(MODEL_BUNDLES)[0]


def log_event(event: str, **fields: object) -> None:
    payload = {"event": event, "timestamp": time.time(), **fields}
    logging.info(json.dumps(payload, ensure_ascii=False))


@app.route("/health", methods=["GET"])
def health():
    """Service health check."""
    return jsonify(
        {
            "status": "ok",
            "message": "Service is running",
            "model_versions": sorted(MODEL_BUNDLES),
        }
    ), 200


@app.route("/metadata", methods=["GET"])
def metadata():
    """Expose model input contract for clients."""
    return jsonify(
        {
            "default_model": DEFAULT_MODEL_ALIAS,
            "model_versions": sorted(MODEL_BUNDLES),
            "features": FEATURE_NAMES,
            "target": "default.payment.next.month",
        }
    ), 200


def normalize_model_alias(value: object | None) -> str | None:
    if value is None:
        return None

    alias = str(value).strip().lower()
    if alias.startswith("model_"):
        alias = alias.replace("model_", "", 1)
    return alias


def choose_model_alias(payload: dict[str, object]) -> tuple[str, str]:
    explicit_alias = normalize_model_alias(request.headers.get("X-Model-Version") or payload.get("model_version"))
    if explicit_alias:
        if explicit_alias not in MODEL_BUNDLES:
            raise ValueError(f"Unknown model_version: {explicit_alias}")
        return explicit_alias, "explicit"

    customer_id = payload.get("customer_id")
    if customer_id is not None and {"v1", "v2"}.issubset(MODEL_BUNDLES):
        bucket = int(sha256(str(customer_id).encode("utf-8")).hexdigest(), 16) % 2
        return ("v1" if bucket == 0 else "v2"), "ab_hash"

    return DEFAULT_MODEL_ALIAS, "default"


def extract_features(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"model_version", "customer_id"}
    }


@app.route("/predict", methods=["POST"])
def predict():
    """Predict credit card default probability."""
    started_at = time.time()
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    data = request.get_json(silent=True)

    if data is None:
        log_event("prediction_error", request_id=request_id, error="Invalid JSON", status_code=400)
        return jsonify({"error": "Request body must be valid JSON object.", "request_id": request_id}), 400
    if not isinstance(data, dict):
        log_event("prediction_error", request_id=request_id, error="JSON is not an object", status_code=400)
        return jsonify({"error": "Request body must be a JSON object.", "request_id": request_id}), 400

    try:
        model_alias, assignment = choose_model_alias(data)
        result = predict_default(MODEL_BUNDLES[model_alias], extract_features(data))
        result["model_alias"] = model_alias
        result["assignment"] = assignment
        result["request_id"] = request_id
    except ValueError as exc:
        log_event("prediction_error", request_id=request_id, error=str(exc), status_code=400)
        return jsonify({"error": str(exc), "request_id": request_id}), 400
    except Exception as exc:
        log_event("prediction_error", request_id=request_id, error=str(exc), status_code=500)
        return jsonify({"error": "Internal prediction error.", "request_id": request_id}), 500

    duration_ms = round((time.time() - started_at) * 1000, 2)
    log_event(
        "prediction",
        request_id=request_id,
        prediction=result["prediction"],
        probability_default=result["probability_default"],
        model_version=result["model_version"],
        model_artifact=result["model_artifact"],
        model_alias=result["model_alias"],
        assignment=result["assignment"],
        duration_ms=duration_ms,
    )
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
