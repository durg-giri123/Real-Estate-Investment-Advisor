"""
Unit tests for data preprocessing and quality auditing.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from data_preprocessing import load_raw_data, audit_data_quality, clean_dataset


def test_load_raw_data():
    """Verify raw dataset loads correctly with expected row count."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    assert "Current_Price" in df.columns
    assert "Price_After_5_Years" in df.columns


def test_audit_data_quality():
    """Verify that audited dataset has zero missing values and zero duplicates."""
    df = load_raw_data()
    report = audit_data_quality(df)
    assert report["total_records"] == 500
    assert report["duplicate_rows"] == 0
    total_missing = sum(report["missing_values"].values())
    assert total_missing == 0


def test_clean_dataset():
    """Verify that cleaned dataset maintains column integrity and numeric types."""
    df = load_raw_data()
    cleaned = clean_dataset(df)
    assert len(cleaned) == 500
    assert cleaned["Area_sqft"].dtype in ["int64", "float64"]
    assert cleaned["Current_Price"].dtype in ["int64", "float64"]
    assert cleaned["Price_After_5_Years"].dtype in ["int64", "float64"]
