"""
Real Estate Investment Advisor - Exploratory Data Analysis (EDA) Module
Generates comprehensive statistical summaries and publication-ready visualizations
addressing price dynamics, location drivers, correlations, and investment viability.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def run_full_eda(data_path: str = "data/processed/real_estate_features.csv", output_dir: str = "reports/figures") -> dict:
    """Executes full automated EDA pipeline and saves analytical figures."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    print(f"[EDA] Running EDA on {len(df)} records...")

    insights = {}

    # 1. Price & Size Distribution Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("1. Real Estate Price & Size Distributions", fontsize=16, fontweight='bold', y=0.98)

    sns.histplot(df['Current_Price'] / 1e5, kde=True, ax=axes[0, 0], color='#1f77b4', bins=25)
    axes[0, 0].set_title("Current Price Distribution ($ in 100k)", fontweight='bold')
    axes[0, 0].set_xlabel("Current Price ($100k)")
    axes[0, 0].set_ylabel("Property Count")

    sns.histplot(df['Price_After_5_Years'] / 1e5, kde=True, ax=axes[0, 1], color='#2ca02c', bins=25)
    axes[0, 1].set_title("Estimated 5-Year Future Price ($ in 100k)", fontweight='bold')
    axes[0, 1].set_xlabel("Future Price ($100k)")
    axes[0, 1].set_ylabel("Property Count")

    sns.histplot(df['Area_sqft'], kde=True, ax=axes[1, 0], color='#ff7f0e', bins=25)
    axes[1, 0].set_title("Property Size Distribution (Sq. Ft.)", fontweight='bold')
    axes[1, 0].set_xlabel("Area (sqft)")
    axes[1, 0].set_ylabel("Property Count")

    sns.histplot(df['Price_per_SqFt'], kde=True, ax=axes[1, 1], color='#9467bd', bins=25)
    axes[1, 1].set_title("Price per Sq. Ft. Distribution", fontweight='bold')
    axes[1, 1].set_xlabel("Price / Sq. Ft. ($)")
    axes[1, 1].set_ylabel("Property Count")

    plt.tight_layout()
    p1 = os.path.join(output_dir, "01_price_and_size_distribution.png")
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close(fig)
    insights['distribution_plot'] = p1

    # 2. Correlation Matrix Heatmap
    corr_cols = [
        'Area_sqft', 'Bedrooms', 'Bathrooms', 'Floors', 'Property_Age',
        'Location_Score', 'School_Score', 'Current_Price', 'Price_per_SqFt',
        'Location_School_Index', 'Appreciation_5Yr_Pct', 'Price_After_5_Years',
        'Good_Investment'
    ]
    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("2. Feature Correlation Matrix with Targets", fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "02_correlation_heatmap.png")
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    insights['correlation_plot'] = p2

    # 3. Location, School Score & Good Investment Relationship
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("3. Impact of Location & School Quality on Investment Viability", fontsize=16, fontweight='bold', y=1.02)

    sns.scatterplot(
        data=df,
        x='Location_Score',
        y='Appreciation_5Yr_Pct',
        hue='Good_Investment',
        palette={0: '#d9534f', 1: '#2ca02c'},
        alpha=0.85,
        s=70,
        ax=axes[0]
    )
    axes[0].axhline(df['Appreciation_5Yr_Pct'].median(), color='black', linestyle='--', label='Median ROI Threshold (36.2%)')
    axes[0].set_title("Appreciation % vs. Location Score", fontweight='bold')
    axes[0].set_xlabel("Location Score (0 - 100)")
    axes[0].set_ylabel("5-Year Appreciation (%)")
    axes[0].legend(title="Good Investment", labels=["Below Median", "Above Median (Good)"])

    sns.boxplot(
        data=df,
        x='Price_Category',
        y='Location_School_Index',
        palette="Blues",
        order=['Low', 'Medium', 'High'],
        ax=axes[1]
    )
    axes[1].set_title("Composite Location-School Index by Price Tier", fontweight='bold')
    axes[1].set_xlabel("Price Category")
    axes[1].set_ylabel("Location-School Composite Index")

    plt.tight_layout()
    p3 = os.path.join(output_dir, "03_location_school_impact.png")
    fig.savefig(p3, dpi=300, bbox_inches='tight')
    plt.close(fig)
    insights['location_school_plot'] = p3

    # 4. Multi-Year Appreciation Trajectory by Investment Category
    yearly_cols = ['Current_Price', 'Price_Year_1', 'Price_Year_2', 'Price_Year_3', 'Price_Year_4', 'Price_After_5_Years']
    tier_trends = df.groupby('Price_Category')[yearly_cols].mean().loc[['Low', 'Medium', 'High']]

    fig, ax = plt.subplots(figsize=(10, 6))
    years = [0, 1, 2, 3, 4, 5]
    colors = {'Low': '#d9534f', 'Medium': '#f0ad4e', 'High': '#2ca02c'}

    for cat in ['Low', 'Medium', 'High']:
        values = tier_trends.loc[cat].values / 1e5
        ax.plot(years, values, marker='o', linewidth=2.5, label=f"{cat} Tier Properties", color=colors[cat])
        for x, y in zip(years, values):
            ax.annotate(f"${y:.1f}k", (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8)

    ax.set_title("4. 5-Year Property Price Growth Trajectory by Category", fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel("Year Timeline (Year 0 to Year 5)")
    ax.set_ylabel("Average Property Price ($100k)")
    ax.set_xticks(years)
    ax.set_xticklabels(['Year 0 (Current)', 'Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5 (Future)'])
    ax.legend(title="Investment Tier", loc="upper left")
    plt.tight_layout()
    p4 = os.path.join(output_dir, "04_price_trajectory_trends.png")
    fig.savefig(p4, dpi=300, bbox_inches='tight')
    plt.close(fig)
    insights['trajectory_plot'] = p4

    # 5. Key Numerical Insights Summary
    summary_stats = {
        "mean_current_price": float(df['Current_Price'].mean()),
        "mean_future_price": float(df['Price_After_5_Years'].mean()),
        "mean_appreciation_pct": float(df['Appreciation_5Yr_Pct'].mean()),
        "median_appreciation_pct": float(df['Appreciation_5Yr_Pct'].median()),
        "max_appreciation_pct": float(df['Appreciation_5Yr_Pct'].max()),
        "min_appreciation_pct": float(df['Appreciation_5Yr_Pct'].min()),
        "mean_price_per_sqft": float(df['Price_per_SqFt'].mean()),
        "correlation_area_future_price": float(df['Area_sqft'].corr(df['Price_After_5_Years'])),
        "correlation_current_future_price": float(df['Current_Price'].corr(df['Price_After_5_Years'])),
    }
    insights['stats'] = summary_stats

    print("[EDA] Visual plots saved successfully to:", output_dir)
    print(f"[EDA] Average Current Price: ${summary_stats['mean_current_price']:,.2f}")
    print(f"[EDA] Average 5-Yr Future Price: ${summary_stats['mean_future_price']:,.2f}")
    print(f"[EDA] Median 5-Yr Appreciation: {summary_stats['median_appreciation_pct']:.2f}%")
    print(f"[EDA] Area to Future Price Correlation: {summary_stats['correlation_area_future_price']:.4f}")

    return insights


if __name__ == "__main__":
    run_full_eda()
