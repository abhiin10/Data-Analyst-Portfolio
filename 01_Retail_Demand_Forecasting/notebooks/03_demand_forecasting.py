
import pandas as pd
from pathlib import Path

# Load the prepared full dataset
project_dir = Path(__file__).resolve().parents[1]
file_path = project_dir / "dataset" / "retail_daily_aggregated.csv"

df = pd.read_csv(file_path, parse_dates=["date"])

print("\n--- FORECASTING DATA VALIDATION ---")

# Basic information
print("Total records:", len(df))
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Unique stores:", df["store_nbr"].nunique())
print("Product categories:", df["family"].nunique())

# Check duplicate date-store-category combinations
duplicates = df.duplicated(
    subset=["date", "store_nbr", "family"]
).sum()

print("Duplicate records:", duplicates)

# Check missing values
print("\nMissing values:")
print(df[["date", "store_nbr", "family", "sales"]].isnull().sum())

# Select one store and category
sample = df[
    (df["store_nbr"] == 1) &
    (df["family"] == "BEVERAGES")
].copy()

sample = sample.sort_values("date")

print("\n--- SELECTED FORECASTING SERIES ---")
print("Store: 1")
print("Category: BEVERAGES")
print("Records:", len(sample))
print("First date:", sample["date"].min())
print("Last date:", sample["date"].max())

print("\nFirst 10 observations:")
print(sample[["date", "sales"]].head(10))

# 1. Prepare time-series data
ts = sample[["date", "sales"]].copy()

ts = ts.sort_values("date")
ts = ts.set_index("date")

# 2. Create a complete daily date range
date_range = pd.date_range(
    start=ts.index.min(),
    end=ts.index.max(),
    freq="D"
)

ts = ts.reindex(date_range)
ts.index.name = "date"

# 3. Fill missing sales with zero
ts["sales"] = ts["sales"].fillna(0)

# 4. Display prepared data
print("\n--- PREPARED FORECASTING DATA ---")
print(ts.head(10))

print("\nTotal observations:", len(ts))
print("Missing values:", ts["sales"].isna().sum())
print("Date frequency:", pd.infer_freq(ts.index))


# 5. Split data into training and testing sets

test_days = 90

train = ts.iloc[:-test_days].copy()
test = ts.iloc[-test_days:].copy()

print("\n--- TRAIN TEST SPLIT ---")

print("Training records:", len(train))
print("Testing records:", len(test))

print("Training date range:",
      train.index.min(), "to", train.index.max())

print("Testing date range:",
      test.index.min(), "to", test.index.max())



# 6. Build a 7-day Seasonal Naive Forecast

predictions = test.copy()

# Use sales from 7 days earlier
predictions["predicted_sales"] = ts["sales"].shift(7).loc[test.index]

print("\n--- FORECAST RESULTS ---")

print(predictions.head(15))

print("\nActual sales total:",
      round(predictions["sales"].sum(), 2))

print("Predicted sales total:",
      round(predictions["predicted_sales"].sum(), 2))


# 7. Evaluate forecasting model

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

actual = predictions["sales"]
forecast = predictions["predicted_sales"]

mae = mean_absolute_error(actual, forecast)

rmse = np.sqrt(mean_squared_error(actual, forecast))

# Calculate MAPE only where actual sales are greater than zero
non_zero = actual > 0

mape = (
    np.mean(
        np.abs(
            (actual[non_zero] - forecast[non_zero])
            / actual[non_zero]
        )
    ) * 100
)

print("\n--- MODEL EVALUATION ---")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape, 2), "%")
print("Days excluded from MAPE:", int((~non_zero).sum()))

# 8. Actual vs Predicted Sales Visualization

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

plt.plot(
    predictions.index,
    predictions["sales"],
    label="Actual Sales"
)

plt.plot(
    predictions.index,
    predictions["predicted_sales"],
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Retail Sales")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()
plt.grid(True)

plt.tight_layout()

# Save chart
output_path = project_dir / "visualizations" / "actual_vs_predicted_sales.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("Chart saved successfully:", output_path)

plt.show()