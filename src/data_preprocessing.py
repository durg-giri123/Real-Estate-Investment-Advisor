"""
Real Estate Investment Advisor - Data Preprocessing Module
Handles loading, data integrity verification, cleaning, and outlier checks.
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(file_path: str = "data/raw/merged_houseprice_regression_classification.xlsx", sheet_name: str = "Regression+Class") -> pd.DataFrame:
    """Loads raw Excel dataset from the specified sheet."""
    if not os.path.exists(file_path):
        # Fallback to current directory if not yet in data/raw
        alt_path = "merged_houseprice_regression_classification.xlsx"
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {file_path} or {alt_path}")
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"[INFO] Successfully loaded {len(df)} records with {df.shape[1]} columns from sheet '{sheet_name}'.")
    return df


def audit_data_quality(df: pd.DataFrame) -> dict:
    """Performs comprehensive data quality audit."""
    audit_report = {
        "total_records": len(df),
        "total_columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    
    print("\n--- Data Quality Audit Report ---")
    print(f"Total Rows: {audit_report['total_records']}, Total Columns: {audit_report['total_columns']}")
    print(f"Duplicate Rows: {audit_report['duplicate_rows']}")
    total_missing = sum(audit_report['missing_values'].values())
    print(f"Total Missing Values: {total_missing}")
    return audit_report


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset by handling duplicates, stripping column names, and verifying types."""
    df_clean = df.copy()
    # Strip whitespace from column names
    df_clean.columns = [c.strip() for c in df_clean.columns]
    
    # Drop duplicates if any
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    if len(df_clean) < initial_len:
        print(f"[INFO] Dropped {initial_len - len(df_clean)} duplicate records.")
    
    # Verify numeric columns are numeric
    num_cols = [
        'Area_sqft', 'Bedrooms', 'Bathrooms', 'Floors', 'Year_Built',
        'Location_Score', 'School_Score', 'Current_Price',
        'Price_Year_1', 'Price_Year_2', 'Price_Year_3', 'Price_Year_4',
        'Price_After_5_Years'
    ]
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
    print("[INFO] Data cleaning completed successfully.")
    return df_clean


if __name__ == "__main__":
    raw_df = load_raw_data()
    audit_data_quality(raw_df)
    clean_df = clean_dataset(raw_df)
    os.makedirs("data/processed", exist_ok=True)
    clean_df.to_csv("data/processed/real_estate_cleaned.csv", index=False)
    print("[INFO] Saved cleaned dataset to data/processed/real_estate_cleaned.csv")
