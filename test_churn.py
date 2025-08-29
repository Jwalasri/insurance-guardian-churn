"""Unit tests for the Insurance Guardian churn project."""

from pathlib import Path

import pandas as pd

from data.generate_data import generate
from src.preprocess import build_preprocessor, prepare_features
from src.train import train_models


def test_generate_data() -> None:
    customers = generate(100, random_state=42)
    assert len(customers) == 100
    # Check that required keys are present
    for row in customers[:3]:
        assert all(key in row for key in ["customer_id", "tenure", "premium", "num_claims", "region", "churn"])


def test_training_produces_models(tmp_path: Path) -> None:
    # Create a small DataFrame for training
    customers = generate(200, random_state=1)
    df = pd.DataFrame(customers)
    results = train_models(df, ["logreg"], metric="roc_auc", test_size=0.3, random_state=0)
    assert "logreg" in results
    res = results["logreg"]
    assert res["score"] >= 0.0 and res["score"] <= 1.0
    # Save and reload the model to ensure serialisation works
    from src.train import save_model
    model_path = tmp_path / "model.joblib"
    save_model(res["model"], res["preprocessor"], res["feature_names"], model_path)
    assert model_path.exists()