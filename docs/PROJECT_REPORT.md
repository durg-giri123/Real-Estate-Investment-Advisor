# 🏡 Real Estate Investment Advisor: Comprehensive Project Report
**Predicting Property Profitability & 5-Year Capital Valuation**

---

## Executive Summary
This project develops an end-to-end Machine Learning intelligence platform designed to assist retail investors, real estate funds, and homebuyers in evaluating long-term residential property returns. Built upon the verified `merged_houseprice_regression_classification.xlsx` dataset, the platform tackles two core analytical challenges:
1. **Classification:** Classifying whether a residential property qualifies as a **"Good Investment"** (high return potential vs. moderate return) and assigning an investment valuation tier (High, Medium, Low).
2. **Regression:** Accurately forecasting the property's estimated market price in **5 years** (`Price_After_5_Years`), while computing expected total ROI and annualized CAGR.

The end-to-end pipeline features data preprocessing, domain feature engineering, exploratory data analysis, multi-model benchmarking with **MLflow 3.x** experiment tracking, and an interactive **Streamlit** dashboard.

---

## 1. Business Problem & Use Cases

### Problem Statement
Real estate purchases involve substantial capital outlays and multi-year holding periods. Buyers typically face opaque valuation dynamics, emotional biases, and uncertainty regarding future local market appreciation. 

### Key Business Use Cases
* **Data-Driven Investment Screening:** Enable retail and institutional investors to rapidly evaluate prospective property acquisitions using standardized quantitative metrics.
* **Neighborhood Growth Identification:** Quantify the multiplier effect of local infrastructure and educational institutions on future capital appreciation.
* **Automated Real Estate Listing Intelligence:** Allow property portal aggregators to automatically display investment potential scores and 5-year price forecasts on active listings.
* **Capital Risk Mitigation:** Help buyers avoid overvalued properties with low appreciation momentum.

---

## 2. Dataset Architecture & Preprocessing

### Dataset Specifications
* **Source:** `merged_houseprice_regression_classification.xlsx` (`Regression+Class` sheet).
* **Sample Count:** 500 validated residential property records.
* **Data Integrity:** **0 missing values**, **0 duplicate records**.

### Domain Features Engineered
| Feature Name | Formulation | Real Estate Domain Rationale |
| :--- | :--- | :--- |
| `Property_Age` | $2026 - Year\_Built$ | Captures physical wear, structural vintage, and remodeling necessity. |
| `Price_per_SqFt` | $\frac{Current\_Price}{Area\_sqft}$ | Standard unit pricing metric for fair comparison across different sizes. |
| `Total_Rooms` | $Bedrooms + Bathrooms$ | Total functional living spaces within the residence. |
| `Avg_Room_Size` | $\frac{Area\_sqft}{Total\_Rooms}$ | Spatial layout efficiency and room spaciousness indicator. |
| `Bed_to_Bath_Ratio` | $\frac{Bedrooms}{Bathrooms}$ | Utility ratio assessing adequate plumbing and convenience standard. |
| `Location_School_Index` | $0.55 \times Location + 0.45 \times School$ | Composite infrastructure index weighting neighborhood connectivity and school quality. |
| `Appreciation_5Yr_Pct` | $\frac{P_{future} - P_{current}}{P_{current}} \times 100$ | 5-Year total percentage return on property capital. |
| `CAGR_5Yr_Pct` | $\left(\frac{P_{future}}{P_{current}}\right)^{1/5} - 1$ | 5-Year compound annual growth rate. |

### Target Variables
1. **Regression Target (`Price_After_5_Years`):** Continuous market price after 5 years (Mean: $903,526, Range: $396,021 – $1,511,775).
2. **Classification Target (`Good_Investment`):** Binary classification indicating whether a property's 5-year appreciation rate meets or exceeds the market median of **36.21%** (Balanced 50/50 split: 250 Class 0, 250 Class 1).
3. **Multi-Class Tier (`Price_Category`):** Low (165), Medium (165), High (170) market segments.

---

## 3. Exploratory Data Analysis (EDA) Findings

1. **Price & Size Scaling:**
   * Property sizes range from 802 sq ft to 4,973 sq ft (Mean: ~2,980 sq ft).
   * Square footage exhibits a very high correlation with 5-year future price ($r = 0.8729$), demonstrating that physical land footprint is the dominant long-term store of value.
2. **Location & School Quality Correlation:**
   * Properties with `Location_Score` $> 80$ and `School_Score` $> 85$ have an average 5-year appreciation rate of $41.8\%$, compared to $31.2\%$ for properties with scores $< 60$.
   * Top-quartile educational infrastructure exerts a substantial premium on property resale values.
3. **Appreciation Trajectory:**
   * Property price compounding is steady and monotonic across all market segments (Low: $488k \rightarrow $647k, Medium: $672k \rightarrow $894k, High: $815k \rightarrow $1.16M).

---

## 4. Machine Learning Model Development & Benchmarks

### A. Regression Models (`Price_After_5_Years`)
Evaluated across 4 algorithms using an 80/20 stratified train-test split ($N_{train} = 400, N_{test} = 100$):

| Model | $R^2$ Score | RMSE ($) | MAE ($) | MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | **0.8895** | **$76,143.69** | **$62,574.51** | **7.34%** | **🏆 Champion** |
| Random Forest Regressor | 0.8704 | $82,457.90 | $67,235.37 | 7.77% | Strong Baseline |
| Gradient Boosting Regressor | 0.8544 | $87,400.11 | $69,258.09 | 8.04% | Competitive |
| XGBoost Regressor | 0.8500 | $88,718.13 | $70,837.89 | 8.26% | Tree Ensemble |

**Regression Key Takeaway:**
Ridge Regression achieved the lowest error across all metrics ($R^2 = 0.8895$, MAPE = $7.34\%$), accurately generalizing across property sizes without overfitting.

---

### B. Classification Models (`Good_Investment` Binary & Market Tier)

| Task | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Binary Good Investment** | **Logistic Regression** | **0.5600** | **0.5600** | **0.5600** | **0.5600** | **0.5796** |
| Binary Good Investment | Random Forest | 0.5200 | 0.5263 | 0.4000 | 0.4545 | 0.5552 |
| Binary Good Investment | XGBoost | 0.5100 | 0.5122 | 0.4200 | 0.4615 | 0.5260 |
| Binary Good Investment | Gradient Boosting | 0.4900 | 0.4872 | 0.3800 | 0.4270 | 0.5356 |
| **Market Tier (3-Class)** | **Random Forest Classifier** | **0.7700** | **0.7700** | **0.7700** | **0.7600** | **0.8750** |

**Classification Key Takeaway:**
While binary appreciation rate threshold classification reflects real-world market variance around the median line, the **3-Class Market Tier Classifier** achieved **77% Accuracy** (with 91% Precision on Low-tier and 71% Precision on High-tier), providing investors with reliable categorization.

---

## 5. MLflow Experiment Tracking Architecture
All experimentation runs are logged using MLflow 3.x with a SQLite database backend (`sqlite:///mlflow.db`):
* **Parameters Logged:** Model family, hyperparameter configs (`n_estimators`, `max_depth`, `alpha`, `learning_rate`).
* **Metrics Tracked:** $R^2$, RMSE, MAE, MAPE for regression; Accuracy, Precision, Recall, F1, ROC-AUC for classification.
* **Artifacts Persisted:** Confusion matrix heatmaps, actual vs. predicted valuation scatter plots, residual distribution plots, feature importance bar charts, and serialized model binaries.

---

## 6. Streamlit Web Application Features
The web application (`app/app.py`) provides:
1. **Interactive Valuation & Forecast Engine:** User-friendly inputs for property size, BHK, bathrooms, construction year, purchase price, location score, and school rating.
2. **Dynamic Wealth Trajectory:** Interactive Plotly graph projecting property value compounding over a 5-year timeline.
3. **Investment Recommendation Badges:** Clear green/amber badges indicating whether the property is an attractive acquisition.
4. **Market Property Explorer:** Interactive filterable database of 500 properties with dynamic bubble scatter plots and search capabilities.
5. **Model Benchmarking Portal:** Live tables and figures explaining MLflow model evaluation metrics.

---

## 7. Deployment & GitHub Links

* **Live Streamlit Web Application:** [https://real-estate-investment-advisor-7ruhm7jdwwrsqxmbkwpstq.streamlit.app/](https://real-estate-investment-advisor-7ruhm7jdwwrsqxmbkwpstq.streamlit.app/)
* **GitHub Repository:** [https://github.com/durg-giri123/Real-Estate-Investment-Advisor](https://github.com/durg-giri123/Real-Estate-Investment-Advisor)

---

## 8. How to Run Locally

```bash
# 1. Clone repository and navigate to directory
cd "Real Estate Investment Advisor"

# 2. Re-run feature engineering and data preprocessing (optional, pre-executed)
python src/feature_engineering.py

# 3. Re-run model training & MLflow logging (optional, pre-executed)
python src/train_classification.py
python src/train_regression.py

# 4. Launch the Streamlit Web Application
streamlit run app/app.py
```
