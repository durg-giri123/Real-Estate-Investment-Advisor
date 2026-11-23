"""
Real Estate Investment Advisor - Unified Master Training Script (src/train_models.py)
Executes both Classification and Regression training pipelines with MLflow experiment tracking.
Referenced directly in Project Scorecard and README.
"""

import os
import sys

# Ensure src directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_raw_data, clean_dataset
from feature_engineering import engineer_features, split_and_save_data
from train_classification import train_classification_models
from train_regression import train_regression_models


def run_full_training_pipeline():
    """
    Executes the complete end-to-end model training lifecycle:
    1. Raw Data Loading & Quality Audit
    2. Domain Feature Engineering & Multi-factor Target Formulation
    3. Stratified Train-Test Splitting (80/20)
    4. Classification Models Training & MLflow Logging
    5. Regression Models Training & MLflow Logging
    """
    print("=" * 70)
    print("🚀 STARTING REAL ESTATE INVESTMENT ADVISOR MASTER TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Preprocessing
    print("\n[STEP 1/4] Loading & Cleaning Dataset...")
    raw_df = load_raw_data()
    clean_df = clean_dataset(raw_df)

    # Step 2: Feature Engineering
    print("\n[STEP 2/4] Engineering Domain Features & Multi-factor Target Labels...")
    featured_df = engineer_features(clean_df)
    train_df, test_df = split_and_save_data(featured_df)

    # Step 3: Classification Models
    print("\n[STEP 3/4] Training & Benchmarking Classification Models in MLflow...")
    cls_summary, cls_champion = train_classification_models()

    # Step 4: Regression Models
    print("\n[STEP 4/4] Training & Benchmarking Regression Models in MLflow...")
    reg_summary, reg_champion = train_regression_models()

    print("\n" + "=" * 70)
    print("🏆 MASTER TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"• Champion Classification Model: {cls_champion}")
    print(f"• Champion Regression Model: {reg_champion}")
    print("• MLflow Database: sqlite:///mlflow.db")
    print("• Saved Models: models/best_classification_model.pkl, models/best_regression_model.pkl")
    print("=" * 70)


if __name__ == "__main__":
    run_full_training_pipeline()
