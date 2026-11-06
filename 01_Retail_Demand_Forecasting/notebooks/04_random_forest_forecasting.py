
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# --------------------------------------------------
# 1. PROJECT SETUP AND DATA LOADING
# --------------------------------------------------

project_dir = Path(__file__).resolve().parents[1]

file_path = project_dir / "dataset" / "retail_daily_aggregated.csv"

output_dir = project_dir / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(file_path, parse_dates=["date"])

print("--- DATASET LOADED ---")
print("Total records:", len(df))
print("Unique stores:", df["store_nbr"].nunique())
print("Product categories:", df["family"].nunique())
print("Date range:", df["date"].min(), "to", df["date"].max())


# --------------------------------------------------
# 2. SELECT STORE AND PRODUCT CATEGORY
# --------------------------------------------------

store_number = 1
category_name = "BEVERAGES"

sample = df[
    (df["store_nbr"] == store_number) &
    (df["family"] == category_name)
].copy()

sample = sample.sort_values("date").reset_index(drop=True)

if sample.empty:
    raise ValueError("No data found for the selected store and category.")

if sample["date"].duplicated().any():
    raise ValueError("Duplicate dates found in the selected series.")

print("\n--- SELECTED FORECASTING SERIES ---")
print("Store:", store_number)
print("Category:", category_name)
print("Total records:", len(sample))
print("First date:", sample["date"].min())
print("Last date:", sample["date"].max())

print("\nFirst 10 records:")
print(sample[["date", "sales"]].head(10))


# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

sample["day_of_week"] = sample["date"].dt.dayofweek
sample["month"] = sample["date"].dt.month
sample["day_of_month"] = sample["date"].dt.day

sample["is_weekend"] = (
    sample["day_of_week"] >= 5
).astype(int)

# Historical sales features
sample["lag_7"] = sample["sales"].shift(7)
sample["lag_14"] = sample["sales"].shift(14)
sample["lag_28"] = sample["sales"].shift(28)

print("\n--- FEATURE ENGINEERING ---")

print(
    sample[
        [
            "date",
            "sales",
            "day_of_week",
            "month",
            "is_weekend",
            "lag_7",
            "lag_14",
            "lag_28"
        ]
    ].head(35).tail(10)
)


# --------------------------------------------------
# 4. MODEL DATA PREPARATION
# --------------------------------------------------

model_data = sample.dropna(
    subset=["lag_7", "lag_14", "lag_28"]
).copy()

features = [
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "lag_7",
    "lag_14",
    "lag_28"
]

X = model_data[features]
y = model_data["sales"]

print("\n--- MODEL DATA PREPARATION ---")
print("Total model records:", len(model_data))
print("Feature columns:", features)
print("Missing values in X:", X.isnull().sum().sum())
print("Missing values in y:", y.isnull().sum())


# --------------------------------------------------
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------

test_days = 90

if len(model_data) <= test_days:
    raise ValueError("Not enough records for the selected test period.")

X_train = X.iloc[:-test_days]
X_test = X.iloc[-test_days:]

y_train = y.iloc[:-test_days]
y_test = y.iloc[-test_days:]

print("\n--- TRAIN TEST SPLIT ---")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training date range:",
    model_data["date"].iloc[:-test_days].min(),
    "to",
    model_data["date"].iloc[:-test_days].max()
)

print(
    "Testing date range:",
    model_data["date"].iloc[-test_days:].min(),
    "to",
    model_data["date"].iloc[-test_days:].max()
)


# --------------------------------------------------
# 6. RANDOM FOREST MODEL TRAINING
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\n--- RANDOM FOREST MODEL TRAINING ---")

rf_model.fit(X_train, y_train)

print("Random Forest model training completed!")


# --------------------------------------------------
# 7. GENERATE PREDICTIONS
# --------------------------------------------------

y_pred = rf_model.predict(X_test)

results = pd.DataFrame({
    "date": model_data["date"].iloc[-test_days:].values,
    "actual_sales": y_test.values,
    "predicted_sales": y_pred
})

print("\n--- PREDICTION RESULTS ---")
print(results.head(15))


# --------------------------------------------------
# 8. MODEL EVALUATION
# --------------------------------------------------

rf_mae = mean_absolute_error(y_test, y_pred)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

non_zero = y_test > 0

if non_zero.any():
    rf_mape = (
        np.mean(
            np.abs(
                (y_test[non_zero] - y_pred[non_zero])
                / y_test[non_zero]
            )
        ) * 100
    )
else:
    rf_mape = np.nan

print("\n--- RANDOM FOREST MODEL EVALUATION ---")
print("MAE:", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("MAPE:", round(rf_mape, 2), "%")
print("Days excluded from MAPE:", int((~non_zero).sum()))


# --------------------------------------------------
# 9. ACTUAL VS PREDICTED VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    results["date"],
    results["actual_sales"],
    label="Actual Sales"
)

plt.plot(
    results["date"],
    results["predicted_sales"],
    label="Random Forest Predictions"
)

plt.title("Random Forest: Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

chart_path = (
    output_dir / "random_forest_actual_vs_predicted.png"
)

plt.savefig(chart_path, dpi=300, bbox_inches="tight")

print("\nVisualization saved:", chart_path)

plt.show()


# --------------------------------------------------
# 10. MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

# Previously calculated Seasonal Naive baseline metrics
naive_mae = 241.38
naive_rmse = 381.16
naive_mape = 12.99

comparison = pd.DataFrame({
    "Model": [
        "7-Day Seasonal Naive",
        "Random Forest"
    ],
    "MAE": [
        naive_mae,
        rf_mae
    ],
    "RMSE": [
        naive_rmse,
        rf_rmse
    ],
    "MAPE": [
        naive_mape,
        rf_mape
    ]
})

print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(comparison.round(2).to_string(index=False))


# Save comparison report
comparison_path = output_dir / "model_comparison.csv"

comparison.to_csv(comparison_path, index=False)

# Save prediction results
results_path = output_dir / "random_forest_predictions.csv"

results.to_csv(results_path, index=False)

print("\nComparison report saved:", comparison_path)
print("Prediction results saved:", results_path)


# --------------------------------------------------
# 11. FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n--- FEATURE IMPORTANCE ---")
print(importance.to_string(index=False))

importance_path = output_dir / "random_forest_feature_importance.csv"

importance.to_csv(importance_path, index=False)

print("\nFeature importance saved:", importance_path)


importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n--- FEATURE IMPORTANCE ---")
print(importance.to_string(index=False))

# MODEL PERFORMANCE COMPARISON VISUALIZATION

import matplotlib.pyplot as plt

models = comparison["Model"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["MAE", "RMSE", "MAPE"]

for ax, metric in zip(axes, metrics):
    bars = ax.bar(models, comparison[metric])

    ax.set_title(f"{metric} Comparison")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=15)
    ax.bar_label(bars, fmt="%.2f")

fig.suptitle("Forecasting Model Performance Comparison")
plt.tight_layout()

comparison_chart_path = (
    project_dir
    / "visualizations"
    / "model_performance_comparison.png"
)

plt.savefig(
    comparison_chart_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nModel comparison chart saved:", comparison_chart_path)

plt.show()




# BUSINESS INSIGHTS REPORT

# Get model metrics directly from comparison DataFrame
baseline = comparison.iloc[0]
best_model = comparison.iloc[1]

# Load feature importance from saved CSV
feature_path = (
    project_dir
    / "visualizations"
    / "random_forest_feature_importance.csv"
)

feature_df = pd.read_csv(feature_path)

# Prepare feature importance text
feature_text = feature_df.to_string(index=False)

# Create report content
report = f"""
========================================================
       RETAIL DEMAND FORECASTING - BUSINESS INSIGHTS
========================================================

1. PROJECT OBJECTIVE
--------------------
Develop a retail demand forecasting solution to predict
sales and support inventory planning.

2. DATASET OVERVIEW
-------------------
Total records: {len(df)}
Unique stores: {df['store_nbr'].nunique()}
Product categories: {df['family'].nunique()}
Date range: {df['date'].min()} to {df['date'].max()}

Selected store: {store_number}
Selected category: {category_name}

3. FORECASTING METHODOLOGY
--------------------------
Models evaluated:
- 7-Day Seasonal Naive
- Random Forest Regressor

Random Forest features:
- Day of week
- Month
- Day of month
- Weekend indicator
- Lag 7, Lag 14 and Lag 28 sales

Training records: {len(X_train)}
Testing records: {len(X_test)}

4. MODEL PERFORMANCE
--------------------
7-Day Seasonal Naive:
MAE: {baseline['MAE']:.2f}
RMSE: {baseline['RMSE']:.2f}
MAPE: {baseline['MAPE']:.2f}%

Random Forest:
MAE: {best_model['MAE']:.2f}
RMSE: {best_model['RMSE']:.2f}
MAPE: {best_model['MAPE']:.2f}%

Best-performing model: Random Forest

5. FEATURE IMPORTANCE
---------------------
{feature_text}

6. KEY BUSINESS INSIGHTS
------------------------
- Random Forest achieved lower errors than the baseline.
- Lag 7 was the most influential predictive feature.
- Weekly demand patterns can support inventory planning.
- Forecasts can help retailers plan stock replenishment.

7. BUSINESS RECOMMENDATIONS
---------------------------
- Monitor weekly sales patterns.
- Use demand forecasts to support replenishment decisions.
- Review prediction errors regularly.
- Validate the model across more stores and categories.

8. PROJECT LIMITATIONS
----------------------
- Current evaluation focuses on Store 1 and BEVERAGES.
- Testing period contains 90 days.
- Predictions use actual historical sales for rolling
  one-day-ahead forecasting.
- Holidays, promotions and external factors can be
  incorporated in future model improvements.

========================================================
                    END OF REPORT
========================================================
"""

# Save report
report_path = (
    project_dir
    / "visualizations"
    / "business_insights.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

print("\nBusiness insights report saved successfully:")
print(report_path)
