"""Inference script for customer churn prediction.

This script loads a previously trained churn prediction model and
applies it to a new dataset of customers.  The output is a CSV
containing the original features along with the predicted churn
probability for each customer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .preprocess import prepare_features, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Score new customers for churn risk.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file (.joblib).")
    parser.add_argument("--input", type=str, required=True, help="Path to the CSV file with new customer data.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "scores.csv"),
        help="Path to write the scored output CSV.",
    )
    args = parser.parse_args()
    # Load the model object containing model, preprocessor and feature names
    obj = joblib.load(args.model)
    model = obj["model"]
    preprocessor = obj["preprocessor"]
    # Load input data
    df = load_dataset(args.input)
    X, _ = prepare_features(df, preprocessor)
    # Compute churn probability using predict_proba if available, else decision function
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
        # Convert raw scores to pseudo‑probabilities via logistic function
        probs = 1 / (1 + np.exp(-scores))  # type: ignore[name-defined]
    # Append probability to DataFrame
    df = df.copy()
    df["churn_probability"] = probs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote scores for {len(df)} customers to {output_path}")


if __name__ == "__main__":
    main()