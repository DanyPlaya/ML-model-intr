"""Train a credit card default model on the Kaggle/UCI CSV dataset."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.model_handler import FEATURE_NAMES


TARGET_COLUMN = "default.payment.next.month"


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    if "ID" in data.columns:
        data = data.drop(columns=["ID"])
    return data


def build_classifier(algorithm: str):
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    if algorithm == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def train(csv_path: Path, model_path: Path, algorithm: str) -> dict[str, float | str]:
    data = normalize_columns(pd.read_csv(csv_path))
    missing_columns = [column for column in [*FEATURE_NAMES, TARGET_COLUMN] if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing_columns)}")

    x = data[FEATURE_NAMES].to_numpy()
    y = data[TARGET_COLUMN].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", build_classifier(algorithm)),
        ]
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "algorithm": algorithm,
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "model": pipeline,
                "feature_names": FEATURE_NAMES,
                "model_version": model_path.stem,
                "metrics": metrics,
            },
            file,
        )

    metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/UCI_Credit_Card.csv", help="Path to Kaggle CSV file.")
    parser.add_argument("--model", default="models/model_v1.pkl", help="Output pickle path.")
    parser.add_argument(
        "--algorithm",
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "logistic_regression"],
        help="Sklearn classifier to train.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(Path(args.csv), Path(args.model), args.algorithm)
    print(json.dumps(result, indent=2))
