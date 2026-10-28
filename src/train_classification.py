"""
Real Estate Investment Advisor - Classification Model Training with MLflow
Trains and benchmarks:
1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. Gradient Boosting Classifier

Tracks all experiments in MLflow, logs metrics, confusion matrix, and registers the champion model.
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from feature_engineering import get_feature_lists


def train_classification_models(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    experiment_name: str = "Real_Estate_Investment_Classification"
):
    """Trains multiple classification models, evaluates them, and logs runs in MLflow."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # 1. Load split data
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"))

    config = get_feature_lists()
    feature_cols = config["features"]
    target_col = config["classification_target"]  # 'Good_Investment'

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"[Classification] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"[Classification] Features count: {len(feature_cols)}")

    # 2. Fit Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save feature names and scaler
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f, indent=4)
    print(f"[Classification] Saved scaler and feature_names to {models_dir}/")

    # 3. Setup MLflow Experiment with SQLite backend
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    # 4. Define candidate models
    candidate_models = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "use_scaled": True,
            "params": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}
        },
        "Random_Forest": {
            "model": RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42),
            "use_scaled": False,
            "params": {"n_estimators": 150, "max_depth": 6, "min_samples_split": 4}
        },
        "XGBoost": {
            "model": XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss"),
            "use_scaled": False,
            "params": {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.05}
        },
        "Gradient_Boosting": {
            "model": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42),
            "use_scaled": False,
            "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.08}
        }
    }

    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None
    results_summary = []

    for name, spec in candidate_models.items():
        with mlflow.start_run(run_name=name):
            print(f"\n[Classification] Training {name}...")
            clf = spec["model"]
            params = spec["params"]
            use_scaled = spec["use_scaled"]

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("use_scaled_features", use_scaled)

            # Fit model
            X_tr = X_train_scaled if use_scaled else X_train
            X_te = X_test_scaled if use_scaled else X_test

            clf.fit(X_tr, y_train)

            # Predictions
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else y_pred

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)

            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc
            }
            mlflow.log_metrics(metrics)

            print(f"[{name}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

            # Confusion Matrix Plot
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels=['Moderate (0)', 'Good (1)'],
                        yticklabels=['Moderate (0)', 'Good (1)'])
            ax.set_title(f"Confusion Matrix: {name}", fontweight='bold')
            ax.set_ylabel("Actual Label")
            ax.set_xlabel("Predicted Label")
            cm_plot_path = f"reports/figures/cm_{name.lower()}.png"
            fig.tight_layout()
            fig.savefig(cm_plot_path, dpi=200)
            plt.close(fig)
            mlflow.log_artifact(cm_plot_path)

            # Feature Importance if applicable
            if hasattr(clf, "feature_importances_"):
                feat_imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 5))
                feat_imp.head(10).plot(kind='barh', ax=ax, color='#2ca02c')
                ax.invert_yaxis()
                ax.set_title(f"Top 10 Feature Importances ({name})", fontweight='bold')
                ax.set_xlabel("Relative Importance")
                fi_plot_path = f"reports/figures/fi_cls_{name.lower()}.png"
                fig.tight_layout()
                fig.savefig(fi_plot_path, dpi=200)
                plt.close(fig)
                mlflow.log_artifact(fi_plot_path)

            # Log model artifact to MLflow
            try:
                if "xgboost" in clf.__class__.__module__:
                    mlflow.xgboost.log_model(clf, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(clf, artifact_path="model", serialization_format="cloudpickle")
            except Exception as e:
                print(f"[Warning] MLflow model logging notice for {name}: {e}")
                # Save locally as artifact
                local_temp = f"models/{name.lower()}_temp.pkl"
                joblib.dump(clf, local_temp)
                mlflow.log_artifact(local_temp)

            results_summary.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "ROC_AUC": roc_auc
            })

            # Check if champion
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_model_obj = clf

    # Save champion model locally
    best_path = os.path.join(models_dir, "best_classification_model.pkl")
    joblib.dump(best_model_obj, best_path)
    print(f"\n[Classification] Champion Model: {best_model_name} with F1-Score: {best_f1:.4f}")
    print(f"[Classification] Saved champion model to {best_path}")

    # Summary table
    summary_df = pd.DataFrame(results_summary).sort_values(by="F1_Score", ascending=False)
    summary_csv = "reports/classification_benchmark.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\n--- Classification Models Benchmark ---")
    print(summary_df.to_string(index=False))

    return summary_df, best_model_name


if __name__ == "__main__":
    train_classification_models()
