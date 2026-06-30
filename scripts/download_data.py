"""Download and extract the public Kaggle credit default dataset."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "uciml/default-of-credit-card-clients-dataset"
)
CSV_NAME = "UCI_Credit_Card.csv"


def download_dataset(destination: Path, force: bool = False) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / CSV_NAME
    archive_path = destination / "default-of-credit-card-clients-dataset.zip"

    if csv_path.exists() and not force:
        return csv_path

    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "credit-default-ml/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as archive:
        shutil.copyfileobj(response, archive)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extract(CSV_NAME, destination)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset archive does not contain {CSV_NAME}")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(download_dataset(args.destination, args.force))
