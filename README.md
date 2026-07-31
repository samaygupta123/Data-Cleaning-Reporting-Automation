# 🍫 Chocolate Sales – Complete Data Analytics Platform (Streamlit)
# Thiranex Data Analytics Internship Forth Project

This single‑dashboard application combines **Predictive Analytics** and **Data Cleaning & Reporting Automation** for the Chocolate Sales dataset.

Working Link: http://localhost:8501/

## ✨ Features

### 📈 Predictive Analytics Tab
- **Time Series Forecasting** (Holt‑Winters): daily/weekly/monthly aggregation, train/test split, future forecasts.
- **Regression Modeling**: Linear Regression / Random Forest to predict `Amount` or `Boxes Shipped`.
- **Manual prediction form** for new transactions.
- Interactive performance metrics (MAE, RMSE, R²) and plots.

### 🧹 Data Cleaning & Reporting Tab
- **Data quality summary** (missing values, duplicates, negative amounts, outliers).
- **One‑click Excel report generation** (`Cleaned_Report.xlsx`) with:
  - Cleaned data
  - Data quality metrics
  - Summary statistics
  - Sales by product / country
  - Monthly sales trend
  - Embedded charts (top products, country share, monthly trend, amount distribution)
- Download the Excel report directly from the dashboard.

### 📊 Data Overview & EDA Tab
- Raw data preview, summary statistics.
- Visualisations: top products, country pie chart, monthly trend, boxplots.

## 🚀 How to Run

1. **Place the dataset** `Chocolate Sales.csv` in the same folder as `app.py`.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
