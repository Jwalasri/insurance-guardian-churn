"""Data loading and preprocessing utilities for insurance churn models."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    return pd.read_csv(path)


def build_preprocessor(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """Construct a preprocessing pipeline for the churn dataset.

    The pipeline imputes missing values, encodes categorical columns and
    leaves numerical columns unchanged.  Returns the pipeline and the
    list of feature names after transformation.
    """
    numeric_features = ["tenure", "premium", "num_claims"]
    categorical_features = ["region"]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Determine feature names after one-hot encoding to use later for interpretability
    # We need to fit the preprocessor on the provided dataframe to get output names.
    preprocessor.fit(df)
    # Generate feature names
    ohe = preprocessor.named_transformers_["cat"]["encoder"]
    cat_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_features + cat_names
    return preprocessor, feature_names


def prepare_features(
    df: pd.DataFrame, preprocessor: ColumnTransformer
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare feature matrix X and label vector y from a DataFrame.

    Parameters
    ----------
    df:
        The input DataFrame containing customer features and the target.
    preprocessor:
        A fitted preprocessing pipeline returned by :func:`build_preprocessor`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of (X, y) where X is a 2D array of features and y is a 1D array of labels.
    """
    X = preprocessor.transform(df)
    y = df["churn"].values
    return X, y