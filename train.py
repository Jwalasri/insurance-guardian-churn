"""Training script for insurance churn models.

This script trains one or more classification models on a provided
customer churn dataset and writes the resulting model artefacts to a
specified directory.  It supports logistic regression, random
forest and gradient boosting models.  The performance of each model
is printed using a specified metric (currently ROC AUC).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .preprocess import load_dataset, build_preprocessor, prepare_features


MODEL_FACTORY = {
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "rf": lambda: RandomForestClassifier(n_estimators=100),
    "gb": lambda: GradientBoostingClassifier(n_estimators=100),
}


def train_models(
    df: pd.DataFrame,
    model_names: List[str],
    metric: str = "roc_auc",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Dict[str, float | object]]:
    """Train multiple models and return their metrics and artefacts.

    Parameters
    ----------
    df:
        DataFrame containing the customer dataset.
    model_names:
        List of model identifiers to train (e.g. ['logreg','rf']).
    metric:
        Name of the evaluation metric.  Currently only 'roc_auc' is supported.
    test_size:
        Fraction of the data to reserve for testing.
    random_state:
        Random seed for the train/test split.

    Returns
    -------
    dict
        Mapping of model name to a dict containing the trained model and its metric.
    """
    results: Dict[str, Dict[str, float | object]] = {}
    preprocessor, feature_names = build_preprocessor(df)
    X, y = prepare_features(df, preprocessor)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    for name in model_names:
        factory = MODEL_FACTORY.get(name)
        if factory is None:
            raise ValueError(f"Unknown model name: {name}")
        model = factory()
        model.fit(X_train, y_train)
        # Evaluate
        if metric == "roc_auc":
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                # Fallback for models without predict_proba
                y_scores = model.decision_function(X_test)
            score = roc_auc_score(y_test, y_scores)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        results[name] = {
            "model": model,
            "score": float(score),
            "preprocessor": preprocessor,
            "feature_names": feature_names,
        }
    return results


def save_model(model, preprocessor, feature_names: List[str], path: Path) -> None:
    """Save a model, its preprocessing pipeline and feature names to disk."""
    obj = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }
    joblib.dump(obj, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train insurance churn prediction models.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "customers.csv"),
        help="Path to the CSV dataset to use for training.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="logreg,rf,gb",
        help="Comma‑separated list of models to train (choices: logreg, rf, gb).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="roc_auc",
        help="Evaluation metric to report (currently only roc_auc).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "models"),
        help="Directory to write model files to.",
    )
    args = parser.parse_args()
    df = load_dataset(args.data)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    results = train_models(df, model_names, metric=args.metric)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Determine best model based on score
    best_name = None
    best_score = -np.inf
    for name, res in results.items():
        score = res["score"]
        print(f"Model {name}: {args.metric}={score:.4f}")
        model_path = output_dir / f"{name}.joblib"
        save_model(res["model"], res["preprocessor"], res["feature_names"], model_path)
        if score > best_score:
            best_score = score
            best_name = name
    if best_name is not None:
        # Save a copy of the best model as best_model.joblib
        best_model_path = output_dir / "best_model.joblib"
        # Load the saved file again (to include preprocessor & names)
        obj = joblib.load(output_dir / f"{best_name}.joblib")
        joblib.dump(obj, best_model_path)
        print(f"Best model: {best_name} with {args.metric}={best_score:.4f}")


if __name__ == "__main__":
    main()