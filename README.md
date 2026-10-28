# 🏡 Real Estate Investment Advisor: Predicting Property Profitability & Future Value

An end-to-end Machine Learning intelligence application designed to assist real estate investors and homebuyers in evaluating property investment viability and forecasting 5-year capital valuation.

🔗 **Live Streamlit Web Application:** [https://real-estate-investment-advisor-7ruhm7jdwwrsqxmbkwpstq.streamlit.app/](https://real-estate-investment-advisor-7ruhm7jdwwrsqxmbkwpstq.streamlit.app/)  
📂 **GitHub Repository:** [https://github.com/durg-giri123/Real-Estate-Investment-Advisor](https://github.com/durg-giri123/Real-Estate-Investment-Advisor)

---

## 🌟 Key Highlights
* **Dual ML Paradigms:**
  * **Regression ($R^2 = 0.8895$):** Predicts exact 5-year estimated property price, capital gain ($), and annualized CAGR (%).
  * **Classification (77% Tier Accuracy):** Classifies whether a listing is a "Good Investment" and categorizes properties into High, Medium, or Low investment tiers.
* **Domain Feature Engineering:** Computes Price per Sq. Ft., Property Age, Spatial Room Efficiency, and a Composite Location-School Quality Index.
* **MLflow Tracking:** All parameters, evaluation metrics, confusion matrices, and model artifacts are tracked using an SQLite backend (`mlflow.db`).
* **Interactive Streamlit Web Dashboard:** Clean, intuitive UI featuring real-time property forecasting, year-by-year trajectory plots, and a filterable property explorer.

---

## 📂 Project Repository Structure
```
Real Estate Investment Advisor/
├── app/
│   └── app.py                      # Production Streamlit Web Dashboard
├── data/
│   ├── raw/                        # Original Excel dataset
│   └── processed/                  # Cleaned & featured datasets (Train/Test splits)
├── docs/
│   └── PROJECT_REPORT.md           # Comprehensive Capstone Project Report
├── models/
│   ├── best_regression_model.pkl   # Champion Ridge Regression Model (R²=0.8895)
│   ├── best_classification_model.pkl # Logistic Regression Model
│   ├── price_category_classifier.pkl # 3-Class Random Forest Model (77% Acc)
│   ├── scaler.pkl                  # Fitted StandardScaler
│   └── feature_names.json          # 13 Model input features metadata
├── reports/
│   ├── figures/                    # Generated EDA and evaluation plots
│   ├── classification_benchmark.csv # Classification comparison metrics
│   └── regression_benchmark.csv    # Regression comparison metrics
├── src/
│   ├── data_preprocessing.py       # Data loading, cleaning, and quality audit
│   ├── feature_engineering.py      # Domain feature creation and train-test splits
│   ├── eda_analysis.py             # Automated Exploratory Data Analysis
│   ├── train_classification.py     # Classification training with MLflow
│   └── train_regression.py         # Regression training with MLflow
├── mlflow.db                       # MLflow SQLite tracking database
├── requirements.txt                # Python environment dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Access the Live App Online
You can directly use the deployed application without any installation:
👉 **[Open Live Streamlit Application](https://real-estate-investment-advisor-7ruhm7jdwwrsqxmbkwpstq.streamlit.app/)**

### 2. Local Installation & Setup
Launch the interactive web application with a single command:
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501` to test property inputs, view visual EDA insights, and inspect model performance benchmarks.

### 3. Retrain Pipeline (Optional)
To regenerate features and re-run MLflow experiments:
```bash
# Data cleaning & feature engineering
python src/feature_engineering.py

# Exploratory data analysis
python src/eda_analysis.py

# Model training & MLflow tracking
python src/train_classification.py
python src/train_regression.py
```

---

## 📊 Model Evaluation Summary

### Regression Benchmark (`Price_After_5_Years`)
| Model | $R^2$ Score | RMSE ($) | MAE ($) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (Champion)** | **0.8895** | **$76,143.69** | **$62,574.51** | **7.34%** |
| Random Forest Regressor | 0.8704 | $82,457.90 | $67,235.37 | 7.77% |
| Gradient Boosting Regressor | 0.8544 | $87,400.11 | $69,258.09 | 8.04% |
| XGBoost Regressor | 0.8500 | $88,718.13 | $70,837.89 | 8.26% |

### Classification Benchmark
* **Binary "Good Investment" Recommendation:** Logistic Regression achieved $F1 = 0.5600$ and $ROC\text{-}AUC = 0.5796$.
* **3-Class Market Tier Segmentation:** Random Forest achieved **77% Accuracy** ($91\%$ precision on Low-tier, $71\%$ precision on High-tier).

---

## 👥 Contributors & Acknowledgments
Developed as part of the Real Estate Investment Advisor Capstone initiative, following the project specification guidelines for real estate predictive analytics.
