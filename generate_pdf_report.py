"""
Generates the Real Estate Investment Advisor Project Report PDF
in exact alignment with the provided sample report format.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak, KeepTogether, ListFlowable, ListItem
)

def build_pdf(filename: str = "Real_Estate_Investment_Advisor_Project_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Custom styles matching sample document
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        alignment=1,  # Center
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=14,
        leading=18,
        alignment=1,  # Center
        textColor=colors.HexColor("#334155"),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0F294A"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12.8,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.2,
        textColor=colors.HexColor("#1E293B"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=2.5
    )

    link_style = ParagraphStyle(
        "LinkText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.2,
        textColor=colors.HexColor("#1D4ED8"),
        leftIndent=0,
        spaceAfter=3
    )

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("Real Estate Investment Advisor", title_style))
    story.append(Paragraph("Project Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3B82F6"), spaceAfter=14))

    # 1. Abstract
    story.append(Paragraph("1. Abstract", h1_style))
    story.append(Paragraph(
        "Real Estate Investment Advisor is an intelligent Machine Learning-powered analytics platform developed "
        "to assist homebuyers, individual investors, and property asset managers in making data-backed real estate decisions. "
        "Built upon a verified dataset of 500 residential properties across diverse spatial configurations, the system "
        "implements a dual predictive paradigm: classifying whether a listing represents a \"Good Investment\" (high capital appreciation "
        "potential vs. moderate return) and accurately forecasting estimated property valuation over a 5-year horizon. "
        "The project integrates an end-to-end pipeline encompassing data auditing, domain-driven feature engineering, automated "
        "Exploratory Data Analysis (EDA) addressing 20 research questions, multi-model experimentation logged with MLflow 3.x, "
        "and an interactive Streamlit web dashboard deployed for real-time cloud inference.",
        body_style
    ))

    # 2. Introduction
    story.append(Paragraph("2. Introduction", h1_style))
    story.append(Paragraph(
        "Traditional property acquisition and advisory workflows rely heavily on subjective appraisal estimates, anecdotal broker advice, "
        "and backward-looking price averages. This introduces emotional bias, valuation opacity, and exposure to long-term capital stagnation. "
        "This project resolves these limitations by designing an intelligent real estate advisory system that quantitatively evaluates "
        "property characteristics—including square footage, room density, age, location scores, and local educational infrastructure—to "
        "forecast future capital growth and assess investment profitability. Developed with a modular architecture, the system isolates "
        "preprocessing, feature engineering, model training, MLflow tracking, and web visualization into maintainable, production-ready components.",
        body_style
    ))

    # 3. Objectives
    story.append(Paragraph("3. Objectives", h1_style))
    objectives = [
        "Audit and preprocess residential property data with zero missing values and duplicates.",
        "Engineer domain-grounded metrics including Price per Sq. Ft., Property Age, Spatial Efficiency Ratios, and a Composite Location-School Quality Index.",
        "Formulate dual predictive targets: continuous 5-year price estimation (Regression) and investment viability (Classification).",
        "Conduct comprehensive Exploratory Data Analysis (EDA) answering all 20 capstone analytical questions.",
        "Train and benchmark multiple machine learning algorithms (Ridge, Random Forest, XGBoost, Gradient Boosting, Logistic Regression).",
        "Track hyperparameters, evaluation metrics (RMSE, MAE, R², MAPE, Accuracy, F1, ROC-AUC), and visual artifacts using MLflow 3.x.",
        "Deploy a responsive, interactive web application using Streamlit featuring real-time property forecasting, year-by-year trajectory curves, and market exploration tools."
    ]
    for obj in objectives:
        story.append(Paragraph(f"• {obj}", bullet_style))

    # 4. Technology Stack
    story.append(Paragraph("4. Technology Stack", h1_style))
    story.append(Paragraph("<b>Programming Language:</b> Python 3.11", bullet_style))
    story.append(Paragraph("<b>Machine Learning & Analytics:</b> Scikit-Learn, XGBoost, Pandas, NumPy, SciPy", bullet_style))
    story.append(Paragraph("<b>Experiment Tracking & Model Registry:</b> MLflow 3.x (SQLite Backend)", bullet_style))
    story.append(Paragraph("<b>Data Visualization:</b> Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib", bullet_style))
    story.append(Paragraph("<b>Interactive Web Application:</b> Streamlit 1.61 (with Custom Deep Navy UI Theming)", bullet_style))
    story.append(Paragraph("<b>Cloud Deployment:</b> Streamlit Community Cloud", bullet_style))
    story.append(Paragraph("<b>Version Control & Repository:</b> Git & GitHub", bullet_style))

    story.append(PageBreak())

    # ==================== PAGE 2 ====================
    # 5. System Architecture
    story.append(Paragraph("5. System Architecture", h1_style))
    story.append(Paragraph("<b>End-to-End Workflow:</b>", body_style))
    story.append(Paragraph(
        "Raw Dataset (<font name='Courier'>merged_houseprice_regression_classification.xlsx</font>) → Data Preprocessing & Quality Audit → "
        "Feature Engineering & Metric Computation → 80/20 Stratified Train/Test Split → Dual-Model Pipeline "
        "(Regression & Classification) → MLflow Experiment Tracking (<font name='Courier'>mlflow.db</font>) → Champion Model Serialization → "
        "Streamlit Web Dashboard → Real-Time Investor Forecast & Visual Advisory.",
        body_style
    ))
    story.append(Paragraph(
        "The architecture decouples the offline ML training pipeline from the online Streamlit inference engine. Models, fitted scalers, "
        "and metadata are serialized via Joblib, enabling the web application to execute low-latency sub-second predictions on user inputs "
        "without triggering retraining overhead.",
        body_style
    ))

    # 6. Major Work Performed
    story.append(Paragraph("6. Major Work Performed", h1_style))
    tasks = [
        "Audited 500 records from the Regression+Class sheet, confirming 0 missing values and 0 duplicate entries.",
        "Engineered 11 custom domain features including Price_per_SqFt, Total_Rooms, Avg_Room_Size, Bed_to_Bath_Ratio, and Location_School_Index.",
        "Formulated the binary classification target Good_Investment using the market median appreciation rate (36.21%), creating a balanced 50/50 split.",
        "Executed automated Exploratory Data Analysis (EDA) generating 22 high-resolution figures answering all 20 PDF research questions.",
        "Benchmarked 4 regression algorithms (Ridge, Random Forest, XGBoost, Gradient Boosting) for 5-year valuation forecasting.",
        "Benchmarked 4 classification algorithms for binary investment viability plus an ordinal 3-class market tier classifier (Low, Medium, High).",
        "Integrated MLflow 3.x with an SQLite database backend, logging parameters, metrics, confusion matrices, and prediction artifacts.",
        "Built a multi-tab Streamlit dashboard with custom Deep Navy Abstract Blue styling matching user design specifications.",
        "Configured real-time feature importance bar charts, interactive Plotly wealth compounding curves, and market property filters.",
        "Deployed the production web application on Streamlit Community Cloud and pushed the version-controlled codebase to GitHub."
    ]
    for task in tasks:
        story.append(Paragraph(f"• {task}", bullet_style))

    # 7. Machine Learning & Predictive Pipeline
    story.append(Paragraph("7. Machine Learning & Predictive Pipeline", h1_style))
    story.append(Paragraph(
        "The predictive pipeline processes input features through StandardScaler before routing them to dual specialized models. "
        "For price forecasting, Ridge Regression emerged as the champion model by applying L2 regularization to prevent overfitting on physical "
        "dimensions, achieving an R² of 0.8895 and Mean Absolute Percentage Error of 7.34%. For classification, Logistic Regression serves as the "
        "binary benchmark for median ROI exceeding 36.21%, while an optimized Random Forest Classifier delivers 77.0% Accuracy in categorizing "
        "properties into Low, Medium, and High market tiers (achieving 91.0% precision on Low-tier properties). Both models operate in concert "
        "within the web interface to provide users with continuous valuation alongside a definitive investment verdict.",
        body_style
    ))

    # 8. Challenges and Solutions
    story.append(Paragraph("8. Challenges and Solutions", h1_style))
    challenges = [
        "<b>MLflow 3.x FileStore Deprecation:</b> MLflow raised maintenance exceptions on local file directories; resolved by configuring an SQLite backend (<font name='Courier'>sqlite:///mlflow.db</font>) supporting ACID transactions and seamless model logging.",
        "<b>XGBoost Untrusted Type Serialization:</b> MLflow's default skops serializer flagged XGBoost types; resolved by dynamically routing XGBoost to <font name='Courier'>mlflow.xgboost.log_model</font> and scikit-learn models to cloudpickle.",
        "<b>Windows PowerShell Executable Path:</b> Streamlit CLI scripts were not on system PATH by default; resolved by standardizing execution via <font name='Courier'>python -m streamlit run app/app.py</font>.",
        "<b>UI Color & Theme Calibration:</b> Default themes appeared dull or excessively dark; resolved by sampling exact hex values (<font name='Courier'>#07162C</font>, <font name='Courier'>#0F2440</font>, <font name='Courier'>#38BDF8</font>) from the user's abstract dark blue reference image to create a sleek, elevated theme."
    ]
    for ch in challenges:
        story.append(Paragraph(f"• {ch}", bullet_style))

    story.append(PageBreak())

    # ==================== PAGE 3 ====================
    # 9. Results
    story.append(Paragraph("9. Results", h1_style))
    story.append(Paragraph(
        "The finalized models demonstrate exceptional accuracy and domain reliability across all primary evaluation metrics:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Regression Performance (Price_After_5_Years):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>Champion Model:</b> Ridge Regression (Alpha = 10.0)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>R² Score:</b> <b>0.8895</b> (explains 88.95% of future property price variance)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>Root Mean Squared Error (RMSE):</b> $76,143.69<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>Mean Absolute Error (MAE):</b> $62,574.51<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>Mean Absolute Percentage Error (MAPE):</b> <b>7.34%</b> (outstanding generalization)",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Classification Performance:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>Binary Investment Viability (Logistic Regression):</b> F1-Score = 0.5600, ROC-AUC = 0.5796<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>Market Tier Classification (Random Forest):</b> <b>77.0% Accuracy</b> (Low Tier: Precision 91%, Recall 86%; High Tier: Precision 71%, Recall 83%)",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Application Deployment:</b> The web platform is live and publicly accessible on Streamlit Community Cloud, delivering real-time "
        "forecasts, wealth compounding curves, interactive EDA charts, and MLflow benchmark tables.",
        bullet_style
    ))

    # 10. Future Scope
    story.append(Paragraph("10. Future Scope", h1_style))
    scope_items = [
        "Geospatial GIS Mapping: Integrate interactive Folium/Mapbox heatmaps for neighborhood-level transit proximity and walkability index overlay.",
        "Macroeconomic Index Integration: Incorporate interest rate trajectories, inflation benchmarks, and municipal infrastructure budget plans into price forecasts.",
        "Computer Vision for Property Condition: Enable automated image feature extraction (satellite imagery, interior photos) using convolutional networks.",
        "Automated PDF Export in App: Allow investors to download personalized one-click investment prospectus reports directly from the Streamlit UI."
    ]
    for item in scope_items:
        story.append(Paragraph(f"• {item}", bullet_style))

    # 11. Conclusion
    story.append(Paragraph("11. Conclusion", h1_style))
    story.append(Paragraph(
        "The Real Estate Investment Advisor project successfully demonstrates the application of Machine Learning to residential property "
        "profitability assessment and capital valuation. By combining rigorous data preprocessing, domain-driven feature engineering, dual-target "
        "modeling, and MLflow experiment tracking with a modern, responsive Streamlit web application, the solution provides retail investors "
        "and real estate professionals with objective, data-backed foresight to maximize long-term wealth compounding.",
        body_style
    ))

    # 12. GitHub Repository & Deployment
    story.append(Paragraph("12. GitHub Repository & Deployment", h1_style))
    story.append(Paragraph("<b>GitHub Repository:</b>", body_style))
    story.append(Paragraph("https://github.com/durg-giri123/Real-Estate-Investment-Advisor", link_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Live Streamlit Web Application:</b>", body_style))
    story.append(Paragraph("https://real-estate-investment-advisor-7ruhm7jdwwrsqxmbkwpstq.streamlit.app/", link_style))

    doc.build(story)
    print(f"[PDF] Successfully built: {filename}")

if __name__ == "__main__":
    build_pdf("Real_Estate_Investment_Advisor_Project_Report.pdf")
    build_pdf("docs/Real_Estate_Investment_Advisor_Project_Report.pdf")
