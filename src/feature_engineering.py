"""
Real Estate Investment Advisor - Feature Engineering Module
Transforms cleaned property data into investment metrics, domain features,
and creates target variables for both Regression and Classification models.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def engineer_features(df: pd.DataFrame, reference_year: int = 2026) -> pd.DataFrame:
    """
    Engineers domain-specific real estate metrics:
    - Property_Age: Age in years based on Year_Built
    - Price_per_SqFt: Current price normalized by property square footage
    - Total_Rooms: Combined bedrooms and bathrooms count
    - Avg_Room_Size: Average square footage per functional room
    - Bed_to_Bath_Ratio: Ratio of bedrooms to bathrooms
    - Location_School_Index: Composite infrastructure and amenity quality score
    - Appreciation_5Yr_Amount: Absolute 5-year capital gain ($)
    - Appreciation_5Yr_Pct: 5-year return on investment (ROI %)
    - CAGR_5Yr_Pct: Compound Annual Growth Rate (%)
    - Good_Investment: Binary classification target (1 = High return / Good investment, 0 = Standard/Lower return)
    - Price_Category_Encoded: Ordinal classification target (0=Low, 1=Medium, 2=High)
    """
    data = df.copy()

    # 1. Age and Size metrics
    data['Property_Age'] = reference_year - data['Year_Built']
    data['Price_per_SqFt'] = data['Current_Price'] / data['Area_sqft']
    data['Total_Rooms'] = data['Bedrooms'] + data['Bathrooms']
    data['Avg_Room_Size'] = data['Area_sqft'] / np.maximum(data['Total_Rooms'], 1)
    data['Bed_to_Bath_Ratio'] = data['Bedrooms'] / np.maximum(data['Bathrooms'], 1)

    # 2. Location & Infrastructure Composite Score
    # Combining Location Score (55% weight) and School Rating Score (45% weight)
    data['Location_School_Index'] = (0.55 * data['Location_Score'] + 0.45 * data['School_Score'])

    # 3. Growth & Appreciation Metrics (Ground truth from historical trajectory)
    data['Appreciation_5Yr_Amount'] = data['Price_After_5_Years'] - data['Current_Price']
    data['Appreciation_5Yr_Pct'] = (data['Appreciation_5Yr_Amount'] / data['Current_Price']) * 100
    data['CAGR_5Yr_Pct'] = (((data['Price_After_5_Years'] / data['Current_Price']) ** (1/5)) - 1) * 100

    # 4. Target Variables Formulation
    # Classification Target 1: Binary "Good Investment" (1: Above-median appreciation, 0: Below-median)
    median_appreciation = data['Appreciation_5Yr_Pct'].median()
    data['Good_Investment'] = (data['Appreciation_5Yr_Pct'] >= median_appreciation).astype(int)

    # Classification Target 2: Multi-class "Price_Category" (Low=0, Medium=1, High=2)
    cat_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    data['Price_Category_Encoded'] = data['Price_Category'].map(cat_mapping)

    print(f"[INFO] Feature engineering completed. Output dataset shape: {data.shape}")
    print(f"[INFO] Median 5-Year Appreciation Threshold: {median_appreciation:.2f}%")
    print(f"[INFO] Good Investment Distribution: {dict(data['Good_Investment'].value_counts())}")
    print(f"[INFO] Price Category Distribution: {dict(data['Price_Category'].value_counts())}")

    return data


def get_feature_lists():
    """
    Returns feature column names partitioned by purpose:
    - INFERENCE_FEATURES: Features available at input time (from an investor)
    - REGRESSION_TARGET: 'Price_After_5_Years'
    - CLASSIFICATION_BINARY_TARGET: 'Good_Investment'
    - CLASSIFICATION_MULTICLASS_TARGET: 'Price_Category_Encoded'
    """
    inference_features = [
        'Area_sqft',
        'Bedrooms',
        'Bathrooms',
        'Floors',
        'Property_Age',
        'Location_Score',
        'School_Score',
        'Current_Price',
        'Price_per_SqFt',
        'Total_Rooms',
        'Avg_Room_Size',
        'Bed_to_Bath_Ratio',
        'Location_School_Index'
    ]
    return {
        "features": inference_features,
        "regression_target": "Price_After_5_Years",
        "classification_target": "Good_Investment",
        "multiclass_target": "Price_Category_Encoded"
    }


def split_and_save_data(df: pd.DataFrame, output_dir: str = "data/processed", test_size: float = 0.2, random_state: int = 42):
    """Splits dataset into reproducible train and test sets and saves them as CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full featured dataset
    full_path = os.path.join(output_dir, "real_estate_features.csv")
    df.to_csv(full_path, index=False)
    print(f"[INFO] Full featured dataset saved to {full_path}")

    # Stratified split on Good_Investment
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Good_Investment']
    )

    train_path = os.path.join(output_dir, "train_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[INFO] Train set ({len(train_df)} rows) saved to {train_path}")
    print(f"[INFO] Test set ({len(test_df)} rows) saved to {test_path}")
    return train_df, test_df


if __name__ == "__main__":
    from data_preprocessing import load_raw_data, clean_dataset
    raw = load_raw_data()
    clean = clean_dataset(raw)
    featured = engineer_features(clean)
    split_and_save_data(featured)
