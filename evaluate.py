"""Evaluate trained insurance churn models on a hold‑out dataset.

This script loads all model files from a given directory and
computes the specified metric against a provided dataset.  It
assumes that models were saved using the same format as produced by
``train.py`` (i.e. a dictionary with ``model``, ``preprocessor`` and
``feature_names`` entries).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .preprocess import load_dataset, prepare_features


def evaluate_model(model_obj: Dict, df: pd.DataFrame, metric: str = "roc_auc") -> float:
    """Evaluate a single model object on the dataset."""
    model = model_obj["model"]
    preprocessor = model_obj["preprocessor"]
    X, y = prepare_features(df, preprocessor)
    if metric == "roc_auc":
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X)[:, 1]
        else:
            y_scores = model.decision_function(X)
        return float(roc_auc_score(y, y_scores))
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved churn models.")
    parser.add_argument(
        "--models",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "models"),
        help="Directory containing saved model files (*.joblib).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "customers.csv"),
        help="Path to the CSV dataset to evaluate on.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="roc_auc",
        help="Evaluation metric to report (currently only roc_auc).",
    )
    args = parser.parse_args()
    model_dir = Path(args.models)
    df = load_dataset(args.data)
    for model_file in model_dir.glob("*.joblib"):
        obj = joblib.load(model_file)
        score = evaluate_model(obj, df, metric=args.metric)
        print(f"{model_file.name}: {args.metric}={score:.4f}")


if __name__ == "__main__":
    main()