# Retail Demand Forecasting & Inventory Optimization

## Project Overview

This project focuses on analyzing retail sales data, identifying sales patterns, and developing demand forecasting models to support better inventory planning and business decision-making.

The goal is to understand historical sales performance, identify high-performing product categories and stores, and compare forecasting models to estimate future demand.

## Business Problem

Retail businesses need to maintain the right inventory levels to meet customer demand.

Poor demand forecasting can lead to:
- Overstocking and increased storage costs.
- Understocking and lost sales opportunities.
- Inefficient inventory planning.
- Difficulty identifying sales trends and high-performing product categories.

This project uses data analytics and machine learning techniques to explore these challenges.

## Project Objectives

- Analyze historical retail sales data.
- Identify top-performing product families.
- Compare sales performance across stores.
- Understand daily sales trends.
- Explore the impact of promotions on sales.
- Develop and compare demand forecasting models.
- Present business insights through an interactive Power BI dashboard.

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python | Data analysis and forecasting |
| Pandas, NumPy | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning |
| MySQL | Data storage and SQL analysis |
| Power BI | Interactive dashboard |
| VS Code | Development environment |

## Dataset

**Dataset:** Store Sales - Time Series Forecasting

**Source:** Kaggle – Corporación Favorita Grocery Sales Forecasting

The dataset contains historical retail sales information, including:
- Date
- Store number
- Product family
- Sales
- Promotion information

## Project Workflow

1. Data Collection and Preparation
2. Data Cleaning and Exploration
3. SQL-Based Business Analysis
4. Exploratory Data Analysis (EDA)
5. Sales Trend Visualization
6. Demand Forecasting
7. Random Forest Model Development
8. Model Performance Comparison
9. Power BI Dashboard Development
10. Business Insights and Recommendations

## Machine Learning Models

The project includes forecasting scripts for:

- Demand Forecasting
- Random Forest Forecasting

Model comparison and prediction outputs are saved in the project folders.

## Power BI Dashboard

The interactive dashboard provides:

- Top 10 Product Families by Sales
- Total Sales by Store
- Daily Retail Sales Trend (2013–2017)
- Average Sales
- Total Sales
- Store and Product Family Filters

## Project Structure

```text
01_Retail_Demand_Forecasting/
│
├── dataset/
├── models/
├── notebooks/
│   ├── 01_data_exploration.py
│   ├── 01_retail_data_analysis.py
│   ├── 02_prepare_full_dataset.py
│   ├── 03_demand_forecasting.py
│   ├── 04_random_forest_forecasting.py
│   └── create_sample.py
│
├── scripts/
├── sql/
├── visualizations/
│   ├── actual_vs_predicted_sales.png
│   ├── business_insights.txt
│   ├── model_comparison.csv
│   ├── model_performance_comparison.png
│   ├── random_forest_actual_vs_predicted.png
│   ├── random_forest_feature_importance.csv
│   ├── random_forest_predictions.csv
│   └── top_10_store_category_sales.png
│
└── Retail_Demand_Forecasting_Dashboard.pbix
```

## Key Deliverables

- Python data analysis and forecasting scripts.
- SQL business analysis.
- Model comparison results.
- Actual vs predicted sales visualizations.
- Feature importance analysis.
- Interactive Power BI dashboard.

## Business Value

This project demonstrates how retail sales data can be transformed into actionable insights to support:

- Better demand planning.
- Improved inventory decisions.
- Identification of high-performing product categories.
- Store-level sales analysis.
- Data-driven business decisions.

## Future Improvements

- Incorporate holiday and transaction data.
- Improve forecasting accuracy through additional model experiments.
- Develop more advanced time-series forecasting models.
- Build inventory optimization recommendations.
- Deploy the forecasting solution as an interactive application.

## Author

**Abhi**

Data Analytics Portfolio Project

**Skills Demonstrated:** Python | SQL | Power BI | Data Analysis | Machine Learning | Data Visualization