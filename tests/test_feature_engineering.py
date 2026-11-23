"""
Unit tests for domain feature engineering and target label creation.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from data_preprocessing import load_raw_data, clean_dataset
from feature_engineering import engineer_features, get_feature_lists, split_and_save_data


@pytest.fixture
def sample_featured_data():
    raw = load_raw_data()
    clean = clean_dataset(raw)
    return engineer_features(clean)


def test_engineered_columns_present(sample_featured_data):
    """Ensure all required domain metrics are engineered."""
    df = sample_featured_data
    expected_cols = [
        "Property_Age", "Price_per_SqFt", "Total_Rooms", "Avg_Room_Size",
        "Bed_to_Bath_Ratio", "Location_School_Index", "Good_Investment",
        "Price_Category_Encoded", "Appreciation_5Yr_Pct"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing engineered column: {col}"


def test_domain_metric_calculations(sample_featured_data):
    """Verify mathematical correctness of engineered metrics."""
    df = sample_featured_data
    # Property age should be positive
    assert (df["Property_Age"] >= 0).all()
    # Price per sqft should equal Current_Price / Area_sqft
    expected_ppsqft = df["Current_Price"] / df["Area_sqft"]
    np.testing.assert_allclose(df["Price_per_SqFt"], expected_ppsqft, rtol=1e-5)
    # Total rooms should equal Bedrooms + Bathrooms
    assert (df["Total_Rooms"] == df["Bedrooms"] + df["Bathrooms"]).all()


def test_good_investment_target_balance(sample_featured_data):
    """Verify that Good_Investment binary target is properly formulated."""
    df = sample_featured_data
    counts = df["Good_Investment"].value_counts()
    assert 0 in counts and 1 in counts
    # Exactly balanced around median
    assert counts[0] == 250
    assert counts[1] == 250


def test_train_test_split_ratio(sample_featured_data):
    """Verify 80/20 train-test split consistency."""
    train_df, test_df = split_and_save_data(sample_featured_data)
    assert len(train_df) == 400
    assert len(test_df) == 100
    assert set(train_df.columns) == set(test_df.columns)
