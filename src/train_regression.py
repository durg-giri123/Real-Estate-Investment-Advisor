"""
Real Estate Investment Advisor - Regression Model Training with MLflow
Trains and benchmarks:
1. Ridge / Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor
4. Gradient Boosting Regressor

Predicts continuous 5-year future property valuation (Price_After_5_Years),
evaluates RMSE, MAE, R2, logs runs into MLflow, and saves champion model.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)

from feature_engineering import get_feature_lists


def train_regression_models(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    experiment_name: str = "Real_Estate_Future_Price_Regression"
):
    """Trains regression models, evaluates RMSE, MAE, R2, and logs into MLflow."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # 1. Load data
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"))

    config = get_feature_lists()
    feature_cols = config["features"]
    target_col = config["regression_target"]  # 'Price_After_5_Years'

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"[Regression] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"[Regression] Target mean: ${y_train.mean():,.2f}")

    # Load pre-fitted scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        joblib.dump(scaler, scaler_path)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Setup MLflow Experiment with SQLite backend
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    # 3. Define candidate regression models
    candidate_models = {
        "Ridge_Regression": {
            "model": Ridge(alpha=10.0, random_state=42),
            "use_scaled": True,
            "params": {"alpha": 10.0, "solver": "auto"}
        },
        "Random_Forest": {
            "model": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=3, random_state=42),
            "use_scaled": False,
            "params": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 3}
        },
        "XGBoost": {
            "model": XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.06, subsample=0.85, random_state=42),
            "use_scaled": False,
            "params": {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.06, "subsample": 0.85}
        },
        "Gradient_Boosting": {
            "model": GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42),
            "use_scaled": False,
            "params": {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.08}
        }
    }

    best_r2 = -float("inf")
    best_model_name = None
    best_model_obj = None
    results_summary = []

    for name, spec in candidate_models.items():
        with mlflow.start_run(run_name=name):
            print(f"\n[Regression] Training {name}...")
            reg = spec["model"]
            params = spec["params"]
            use_scaled = spec["use_scaled"]

            # Log params
            mlflow.log_params(params)
            mlflow.log_param("use_scaled_features", use_scaled)

            # Fit
            X_tr = X_train_scaled if use_scaled else X_train
            X_te = X_test_scaled if use_scaled else X_test

            reg.fit(X_tr, y_train)

            # Predictions
            y_pred = reg.predict(X_te)

            # Metrics
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))
            mape = float(mean_absolute_percentage_error(y_test, y_pred)) * 100

            metrics = {
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "mape_pct": mape
            }
            mlflow.log_metrics(metrics)

            print(f"[{name}] R2: {r2:.4f} | RMSE: ${rmse:,.2f} | MAE: ${mae:,.2f} | MAPE: {mape:.2f}%")

            # Actual vs Predicted Plot
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_test / 1e5, y_pred / 1e5, alpha=0.7, color="#1f77b4", edgecolors='k', s=45)
            min_val = min(y_test.min(), y_pred.min()) / 1e5
            max_val = max(y_test.max(), y_pred.max()) / 1e5
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect 1:1 Fit")
            ax.set_title(f"Actual vs Predicted 5-Year Price ({name})", fontweight='bold')
            ax.set_xlabel("Actual Price ($100k)")
            ax.set_ylabel("Predicted Price ($100k)")
            ax.legend()
            fig.tight_layout()
            pred_plot_path = f"reports/figures/reg_pred_{name.lower()}.png"
            fig.savefig(pred_plot_path, dpi=200)
            plt.close(fig)
            mlflow.log_artifact(pred_plot_path)

            # Residuals Distribution Plot
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(residuals / 1e3, kde=True, color="#9467bd", ax=ax, bins=20)
            ax.set_title(f"Residuals Distribution: {name} (Error in $k)", fontweight='bold')
            ax.set_xlabel("Residual (Actual - Predicted) in $k")
            fig.tight_layout()
            res_plot_path = f"reports/figures/reg_resid_{name.lower()}.png"
            fig.savefig(res_plot_path, dpi=200)
            plt.close(fig)
            mlflow.log_artifact(res_plot_path)

            # Feature Importance
            if hasattr(reg, "feature_importances_"):
                feat_imp = pd.Series(reg.feature_importances_, index=feature_cols).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 5))
                feat_imp.head(10).plot(kind='barh', ax=ax, color='#1f77b4')
                ax.invert_yaxis()
                ax.set_title(f"Top 10 Feature Importances ({name})", fontweight='bold')
                ax.set_xlabel("Relative Importance")
                fi_plot_path = f"reports/figures/fi_reg_{name.lower()}.png"
                fig.tight_layout()
                fig.savefig(fi_plot_path, dpi=200)
                plt.close(fig)
                mlflow.log_artifact(fi_plot_path)

            # Log model artifact to MLflow
            try:
                if "xgboost" in reg.__class__.__module__:
                    mlflow.xgboost.log_model(reg, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(reg, artifact_path="model", serialization_format="cloudpickle")
            except Exception as e:
                print(f"[Warning] MLflow model logging notice for {name}: {e}")
                local_temp = f"models/{name.lower()}_temp.pkl"
                joblib.dump(reg, local_temp)
                mlflow.log_artifact(local_temp)

            results_summary.append({
                "Model": name,
                "R2_Score": r2,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE_Pct": mape
            })

            # Check if champion
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model_obj = reg

    # Save champion model locally
    best_path = os.path.join(models_dir, "best_regression_model.pkl")
    joblib.dump(best_model_obj, best_path)
    print(f"\n[Regression] Champion Model: {best_model_name} with R2-Score: {best_r2:.4f}")
    print(f"[Regression] Saved champion model to {best_path}")

    # Summary table
    summary_df = pd.DataFrame(results_summary).sort_values(by="R2_Score", ascending=False)
    summary_csv = "reports/regression_benchmark.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\n--- Regression Models Benchmark ---")
    print(summary_df.to_string(index=False))

    return summary_df, best_model_name


if __name__ == "__main__":
    train_regression_models()
