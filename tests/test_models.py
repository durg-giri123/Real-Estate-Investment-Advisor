"""
Unit tests for serialized models, scaler, and inference generation.
"""

import os
import sys
import json
import joblib
import pytest
import pandas as pd
import numpy as np


def test_model_artifacts_exist():
    """Verify all serialized champion models and scalers exist on disk."""
    required_files = [
        "models/best_regression_model.pkl",
        "models/best_classification_model.pkl",
        "models/price_category_classifier.pkl",
        "models/scaler.pkl",
        "models/feature_names.json"
    ]
    for filepath in required_files:
        assert os.path.exists(filepath), f"Missing model artifact: {filepath}"


def test_regression_inference():
    """Verify that regression model generates plausible future property values."""
    reg_model = joblib.load("models/best_regression_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)

    test_df = pd.read_csv("data/processed/test_data.csv")
    X_test = test_df[feature_names]
    X_test_scaled = scaler.transform(X_test)

    preds = reg_model.predict(X_test_scaled)
    assert len(preds) == len(test_df)
    # Predicted future prices should be strictly positive and realistic
    assert (preds > 200000).all()
    assert (preds < 3000000).all()


def test_classification_inference():
    """Verify that classification model generates valid binary labels and probabilities."""
    cls_model = joblib.load("models/best_classification_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)

    test_df = pd.read_csv("data/processed/test_data.csv")
    X_test = test_df[feature_names]
    X_test_scaled = scaler.transform(X_test)

    preds = cls_model.predict(X_test_scaled)
    probs = cls_model.predict_proba(X_test_scaled)

    assert set(np.unique(preds)).issubset({0, 1})
    assert probs.shape == (len(test_df), 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
