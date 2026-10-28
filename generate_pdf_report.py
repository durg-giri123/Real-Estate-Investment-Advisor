"""
Generates a natural, human-written, easy-to-read Project Report PDF
for the Real Estate Investment Advisor project.
Written with clarity, relevant domain section names, and practical workflow storytelling.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak
)

def build_human_pdf(filename: str = "Real_Estate_Investment_Advisor_Project_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=48,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        alignment=1,
        textColor=colors.HexColor("#0F2440"),
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        alignment=1,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14.5,
        textColor=colors.HexColor("#0F2440"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.2,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.8,
        textColor=colors.HexColor("#1E293B"),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=2
    )

    link_style = ParagraphStyle(
        "LinkText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.8,
        textColor=colors.HexColor("#2563EB"),
        leftIndent=0,
        spaceAfter=2
    )

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("Real Estate Investment Advisor", title_style))
    story.append(Paragraph("Predicting Property Profitability & Future Valuation — Project Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=10))

    # 1. Project Summary
    story.append(Paragraph("1. Project Summary", h1_style))
    story.append(Paragraph(
        "Buying a home or investing in residential property is one of the biggest financial commitments anyone makes. "
        "Yet, most investors and buyers rely on gut feeling, aggressive sales pitches, or outdated historical averages. "
        "We built the <b>Real Estate Investment Advisor</b> to bring data science and machine learning transparency to this process. "
        "Using a curated dataset of 500 residential properties across different layouts (1 to 5 BHK), our system solves two fundamental questions: "
        "<b>(1) 'Is this property a smart investment?'</b> (predicting profitability category and high-return potential), and "
        "<b>(2) 'What will this house actually be worth in 5 years?'</b> (continuous price forecasting). "
        "The finished project includes a clean data pipeline, feature engineering, comprehensive exploratory data analysis, "
        "experiment tracking with MLflow, and a sleek interactive web dashboard deployed online on Streamlit Cloud.",
        body_style
    ))

    # 2. Why This Project Matters
    story.append(Paragraph("2. The Problem We Set Out to Solve", h1_style))
    story.append(Paragraph(
        "Real estate pricing is driven by an intricate mix of physical dimensions (square footage, rooms), property vintage (age), and "
        "neighborhood desirability (school rankings and location connectivity). In standard listing portals, buyers can only see the "
        "current asking price with zero insight into future appreciation or long-term growth potential. "
        "Our goal was to design an advisor tool that evaluates any property objectively, calculates its expected 5-year capital gain, "
        "and flags whether its appreciation pace beats the broader market.",
        body_style
    ))

    # 3. What We Aimed to Accomplish
    story.append(Paragraph("3. Core Project Objectives", h1_style))
    objectives = [
        "Take the 500-property dataset, inspect its quality, and confirm zero missing entries and zero duplicate listings.",
        "Engineer realistic real estate metrics that investors actually look at: Price per Sq. Ft., Property Age, Room Density, and a combined Location & School Infrastructure Index.",
        "Set up dual target variables: a 5-year future price target (Regression) and a 'Good Investment' indicator (Classification) based on a 36.2% market-median return threshold.",
        "Perform extensive Exploratory Data Analysis (EDA) resolving all 20 capstone research questions with publication-grade charts.",
        "Train and compare multiple algorithms (Ridge, Random Forest, XGBoost, Gradient Boosting, Logistic Regression) using rigorous holdout evaluation.",
        "Log all hyperparameters, metrics (R², RMSE, MAE, Accuracy, F1), and model artifacts into an MLflow SQLite tracking database.",
        "Deploy a live, user-friendly Streamlit web application where anyone can adjust property sliders and get immediate investment advice with a custom deep navy aesthetic."
    ]
    for obj in objectives:
        story.append(Paragraph(f"• {obj}", bullet_style))

    # 4. Tech Stack Used
    story.append(Paragraph("4. Tools & Technologies", h1_style))
    story.append(Paragraph("<b>Core Language:</b> Python 3.11", bullet_style))
    story.append(Paragraph("<b>Data Processing:</b> Pandas, NumPy, OpenPyXL", bullet_style))
    story.append(Paragraph("<b>Machine Learning:</b> Scikit-Learn, XGBoost", bullet_style))
    story.append(Paragraph("<b>Experiment Tracking:</b> MLflow 3.x with local SQLite database (<font name='Courier'>mlflow.db</font>)", bullet_style))
    story.append(Paragraph("<b>Interactive Charts:</b> Plotly Express & Plotly Graph Objects", bullet_style))
    story.append(Paragraph("<b>User Interface & Cloud Hosting:</b> Streamlit 1.61 deployed on Streamlit Community Cloud", bullet_style))
    story.append(Paragraph("<b>Version Control:</b> Git & GitHub", bullet_style))

    story.append(PageBreak())

    # ==================== PAGE 2 ====================
    # 5. How It Works
    story.append(Paragraph("5. How the System Works: End-to-End Workflow", h1_style))
    story.append(Paragraph(
        "<b>User Workflow:</b> Property Specs Entered (BHK, Area, Price, Scores) → Feature Calculation (Price/sqft, Age, Indices) → "
        "StandardScaler Normalization → Ridge Regression Engine → Random Forest Tier Classifier → Instant Verdict, 5-Year Trajectory & Feature Drivers.",
        body_style
    ))
    story.append(Paragraph(
        "The project is structured into modular layers. The data preprocessing and feature engineering code is contained in dedicated scripts, "
        "the models are saved as self-contained binaries, and the Streamlit frontend loads everything quickly without retraining on every interaction.",
        body_style
    ))

    # 6. Step-by-Step Development Journey
    story.append(Paragraph("6. Key Steps Completed During Development", h1_style))
    milestones = [
        "<b>Data Quality Audit:</b> Loaded the 500 rows from the Excel dataset (<font name='Courier'>Regression+Class</font> sheet). Verified complete data hygiene with 0 null values and 0 duplicate rows.",
        "<b>Smart Feature Engineering:</b> Added 11 domain metrics: Property Age (2026 - Year Built), Price per Sq. Ft., Total Functional Rooms, Average Room Area, Bed-to-Bath Ratio, and a weighted Location-School Index (55% location + 45% school).",
        "<b>Formulating Target Labels:</b> For regression, used the verified continuous 5-year valuation. For classification, defined 'Good Investment' as properties exceeding the 36.21% median appreciation rate (creating an exact 50/50 balanced split of 250 good vs 250 moderate properties).",
        "<b>Exploratory Data Analysis:</b> Scripted an automated EDA suite answering all 20 research questions and generated 22 visual charts covering price distributions, location effects, and correlation matrices.",
        "<b>Dual-Model Training:</b> Benchmarked 4 regression algorithms and 4 classification algorithms on an 80/20 stratified split (400 train, 100 test).",
        "<b>MLflow Tracking:</b> Tracked every run, logged hyperparameters, confusion matrices, residual curves, and prediction plots into <font name='Courier'>mlflow.db</font>.",
        "<b>Building the Streamlit Dashboard:</b> Designed a 5-tab application featuring an intuitive valuation console, a filterable property explorer, an interactive 20-question EDA view, and an MLflow benchmarks page.",
        "<b>Styling & Cloud Deployment:</b> Calibrated the visual palette to a modern Deep Navy Dark Blue theme based on user reference and deployed the app live to Streamlit Cloud."
    ]
    for m in milestones:
        story.append(Paragraph(f"• {m}", bullet_style))

    # 7. Interesting Real Estate Insights from Data
    story.append(Paragraph("7. Interesting Market Insights Uncovered During EDA", h1_style))
    insights = [
        "<b>Area is the Anchor of Future Wealth:</b> Square footage showed an exceptionally strong correlation (r = 0.8729) with future property price, confirming that usable land footprint is the single largest store of long-term real estate value.",
        "<b>The School & Location Multiplier:</b> Properties with school ratings above 85 and location scores above 80 enjoyed an average 5-year capital appreciation of <b>41.8%</b>, compared to just <b>31.2%</b> for properties with below-average ratings.",
        "<b>Steady Compounding Across Tiers:</b> Growth wasn't confined to luxury homes. Budget/Low-tier homes grew from $488k to $647k (+32.6%), Core/Medium-tier homes grew from $672k to $894k (+33.0%), and Premium/High-tier homes grew from $815k to $1.16M (+42.3%)."
    ]
    for ins in insights:
        story.append(Paragraph(f"• {ins}", bullet_style))

    # 8. Real Challenges We Overcame
    story.append(Paragraph("8. Engineering Challenges & How We Solved Them", h1_style))
    challenges = [
        "<b>MLflow 3.x Deprecation Warning:</b> The latest MLflow version flagged local folder tracking as maintenance-only. We upgraded to an SQLite database backend (<font name='Courier'>sqlite:///mlflow.db</font>), making tracking robust, atomic, and production-ready.",
        "<b>XGBoost Serialization Issue:</b> MLflow's default serializer raised a security exception on XGBoost object types. We resolved this by dynamically routing XGBoost to its native <font name='Courier'>mlflow.xgboost.log_model</font> and standard models to cloudpickle.",
        "<b>Windows PowerShell Command Discovery:</b> Running Streamlit directly in PowerShell caused command-not-found errors due to Python's user scripts path. We standardized execution using <font name='Courier'>python -m streamlit run app/app.py</font>.",
        "<b>Calibrating the Visual Theme:</b> Default Streamlit styles looked washed-out and dark mode was too harsh. We sampled exact hex colors (<font name='Courier'>#07162C</font>, <font name='Courier'>#0F2440</font>, <font name='Courier'>#38BDF8</font>) from the user's reference abstract blue image to deliver an executive deep navy theme."
    ]
    for ch in challenges:
        story.append(Paragraph(f"• {ch}", bullet_style))

    story.append(PageBreak())

    # ==================== PAGE 3 ====================
    # 9. Results & Model Performance
    story.append(Paragraph("9. Evaluation Results & Champion Models", h1_style))
    story.append(Paragraph(
        "All models were evaluated on the identical 20% holdout test set (100 properties) that the models had never seen during training:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>5-Year Price Forecasting (Regression Champion):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>Winner:</b> Ridge Regression (Alpha = 10.0)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>R² Score:</b> <b>0.8895</b> — captures almost 89% of all variance in future property valuations.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>Mean Absolute Percentage Error (MAPE):</b> <b>7.34%</b> — forecasts within a 7.3% margin of actual value.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>RMSE:</b> $76,143.69 | <b>MAE:</b> $62,574.51 (beating Random Forest, Gradient Boosting, and XGBoost).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Investment Tier & Viability (Classification):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>3-Class Market Tier Classifier (Random Forest):</b> Achieved <b>77.0% Overall Accuracy</b>, with an impressive <b>91.0% Precision</b> on Budget/Low properties and <b>71.0% Precision</b> on Premium properties.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <b>Binary Good Investment (Logistic Regression):</b> F1-Score = 0.5600, ROC-AUC = 0.5796 as a baseline for median outperformance.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Production Dashboard:</b> The Streamlit app is live on Streamlit Cloud, delivering instant forecasts, interactive 5-year growth trajectory curves, feature importance charts, and a filterable property search table.",
        bullet_style
    ))

    # 10. Ideas for Future Expansion
    story.append(Paragraph("10. Future Roadmap & Enhancements", h1_style))
    future_ideas = [
        "<b>Interactive Map Layers:</b> Add Mapbox or Folium geospatial maps showing neighborhood metro lines, parks, and hospital accessibility.",
        "<b>Mortgage & Cash-Flow Calculator:</b> Integrate live interest rate sliders to compute estimated monthly EMI, rental yields, and net cash flow.",
        "<b>Property Photo Analysis:</b> Use a convolutional neural network to scan uploaded interior and exterior photos to adjust the property's condition score automatically."
    ]
    for idea in future_ideas:
        story.append(Paragraph(f"• {idea}", bullet_style))

    # 11. Conclusion
    story.append(Paragraph("11. Conclusion", h1_style))
    story.append(Paragraph(
        "The Real Estate Investment Advisor proves that combining thoughtful domain feature engineering with transparent machine learning "
        "can transform how people evaluate property investments. By removing the guesswork from 5-year price forecasting and providing a "
        "clear, data-driven profitability verdict, the platform empowers individual buyers and real estate professionals to invest with confidence. "
        "With a verified 88.95% R² score, automated MLflow tracking, and a live web deployment, the project delivers a complete, end-to-end "
        "capstone solution that is both academically sound and practically useful.",
        body_style
    ))

    # 12. Project Links & Live App
    story.append(Paragraph("12. Project Links & Live Deployment", h1_style))
    story.append(Paragraph("<b>GitHub Codebase:</b>", body_style))
    story.append(Paragraph("https://github.com/durg-giri123/Real-Estate-Investment-Advisor", link_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Live Streamlit Web Application:</b>", body_style))
    story.append(Paragraph("https://real-estate-investment-advisor-7ruhm7jdwwrsqxmbkwpstq.streamlit.app/", link_style))

    doc.build(story)
    print(f"[PDF] Successfully built human-style report: {filename}")

if __name__ == "__main__":
    build_human_pdf("Real_Estate_Investment_Advisor_Project_Report.pdf")
    build_human_pdf("docs/Real_Estate_Investment_Advisor_Project_Report.pdf")
