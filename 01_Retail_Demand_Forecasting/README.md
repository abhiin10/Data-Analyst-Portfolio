# Retail Demand Forecasting & Inventory Optimization

## Project Overview

This project analyzes historical retail sales data to identify sales patterns, evaluate store and product-family performance, and develop machine learning models for demand forecasting.

The objective is to transform historical sales data into meaningful business insights and support better demand planning and inventory-related decisions.

## Business Problem

Retail businesses need to estimate customer demand accurately to maintain appropriate inventory levels.

Inaccurate demand forecasting can result in:

- Overstocking and increased storage costs.
- Understocking and missed sales opportunities.
- Inefficient inventory planning.
- Difficulty identifying sales trends and high-performing product categories.

This project explores how data analytics and machine learning can help address these challenges.

## Project Objectives

- Analyze historical retail sales data.
- Identify top-performing product families.
- Compare sales performance across stores.
- Understand daily sales trends.
- Explore promotion-related sales patterns.
- Develop and compare demand forecasting models.
- Present business insights through an interactive Power BI dashboard.

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python | Data analysis and forecasting |
| Pandas, NumPy | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning |
| MySQL | SQL-based business analysis |
| Power BI | Interactive dashboard |
| VS Code | Development environment |
| Git & GitHub | Version control |

## Dataset

**Dataset:** Store Sales - Time Series Forecasting  
**Source:** Kaggle – Corporación Favorita Grocery Sales Forecasting

The dataset includes historical retail sales information such as:

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

## Machine Learning

The project includes forecasting scripts for:

- Demand Forecasting
- Random Forest Forecasting

Model comparison results, predictions, and feature importance outputs are stored in the `visualizations` folder.

## Model Performance Comparison

Two forecasting approaches were evaluated:

1. 7-Day Seasonal Naive
2. Random Forest Regressor

### Evaluation Results

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| 7-Day Seasonal Naive | 241.38 | 381.16 | 12.99% |
| Random Forest | **222.52** | **309.87** | **11.68%** |

### Key Finding

The Random Forest model achieved lower MAE, RMSE, and MAPE than the 7-Day Seasonal Naive baseline in this experiment.

This indicates improved forecasting performance for the evaluated dataset and testing period.

*Note: These results represent the forecasting experiment conducted for Store 1 and the BEVERAGES product family, using a 90-day testing period. They should not be interpreted as performance across all stores and product families.*


## Power BI Dashboard

The interactive dashboard presents:

- Top 10 Product Families by Sales
- Total Sales by Store
- Daily Retail Sales Trend (2013–2017)
- Average Sales
- Total Sales
- Store and Product Family Filters

## Key Deliverables

- Python data analysis and forecasting scripts.
- SQL business analysis queries.
- Model comparison results.
- Actual vs. predicted sales visualizations.
- Random Forest feature importance analysis.
- Interactive Power BI dashboard.

## Business Value

The analysis demonstrates how retail sales data can support:

- Better understanding of demand patterns.
- Store-level sales performance analysis.
- Identification of high-performing product categories.
- Promotion-related sales analysis.
- More informed, data-driven planning decisions.

## Future Improvements

- Incorporate additional holiday and transaction-related analysis.
- Experiment with advanced time-series forecasting models.
- Improve forecasting accuracy through additional feature engineering.
- Develop inventory optimization recommendations.
- Deploy the forecasting solution as an interactive application.

## Project Structure

```text
01_Retail_Demand_Forecasting/
│
├── dataset/
├── models/
├── notebooks/
├── scripts/
├── sql/
├── visualizations/
├── README.md
└── Retail_Demand_Forecasting_Dashboard.pbix
```

## Author

**Abhi**  
Data Analytics Portfolio Project

**Skills Demonstrated:** Python | SQL | Power BI | Data Analysis | Machine Learning | Data Visualization
