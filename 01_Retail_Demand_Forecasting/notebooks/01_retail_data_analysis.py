import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)

print("All libraries imported successfully!")

import pandas as pd

df = pd.read_csv(
    "01_Retail_Demand_Forecasting/dataset/retail_daily_aggregated.csv"
)


print(df.head())
print(df.shape)

# 1. Dataset information
print("\n--- DATASET INFORMATION ---")
df.info()

# 2. Statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 3. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 4. Duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(df.duplicated().sum())

# 5. Unique values
print("\n--- UNIQUE VALUES ---")
print("Stores:", df["store_nbr"].nunique())
print("Product Categories:", df["family"].nunique())
print("Unique Dates:", df["date"].nunique())

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

print("\n--- UPDATED DATA TYPES ---")
print(df.dtypes)

# Extract date components
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_name"] = df["date"].dt.day_name()

print("\n--- DATE ANALYSIS ---")
print(df[["date", "year", "month", "day", "day_name"]].head(10))

# Daily sales analysis
daily_sales = df.groupby("date")["sales"].sum().reset_index()

# Sort dates
daily_sales = daily_sales.sort_values("date")

# Display daily sales
print("\n--- DAILY SALES ---")
print(daily_sales.head(10))

# Create sales trend chart
plt.figure(figsize=(14, 6))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"],
    marker="o"
)

plt.title("Daily Retail Sales Trend", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()

# Daily sales analysis
daily_sales = df.groupby("date")["sales"].sum().reset_index()
daily_sales = daily_sales.sort_values("date")

# Create professional chart
plt.figure(figsize=(14, 6))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"],
    color="blue",
    linewidth=2,
    marker="o",
    markersize=4
)

plt.title("Daily Retail Sales Trend", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# Calculate 7-day moving average
daily_sales["7_day_average"] = (
    daily_sales["sales"].rolling(window=7).mean()
)

# Plot daily sales and moving average
plt.figure(figsize=(14, 6))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"],
    label="Daily Sales",
    marker="o",
    alpha=0.5
)

plt.plot(
    daily_sales["date"],
    daily_sales["7_day_average"],
    label="7-Day Moving Average",
    linewidth=3
)

plt.title("Daily Retail Sales with 7-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.legend()
plt.grid(True, alpha=0.4)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# Top 10 product categories by sales

category_sales = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n--- TOP 10 PRODUCT CATEGORIES ---")
print(category_sales)

# Create bar chart
plt.figure(figsize=(12, 6))

category_sales.sort_values().plot(
    kind="barh",
    edgecolor="black"
)

plt.title("Top 10 Product Categories by Total Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product Category")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()

# Store performance analysis

store_sales = (
    df.groupby("store_nbr")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n--- TOP 10 STORES BY SALES ---")
print(store_sales)

# Create bar chart
plt.figure(figsize=(12, 6))

store_sales.sort_values().plot(
    kind="barh",
    edgecolor="black"
)

plt.title("Top 10 Stores by Total Sales")
plt.xlabel("Total Sales")
plt.ylabel("Store Number")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()


# Weekday sales analysis

df["day_name"] = df["date"].dt.day_name()

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_sales = (
    df.groupby("day_name")["sales"]
    .sum()
    .reindex(weekday_order)
)

print("\n--- SALES BY WEEKDAY ---")
print(weekday_sales)

# Create chart
plt.figure(figsize=(12, 6))

weekday_sales.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Total Retail Sales by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()

# Promotion analysis

df["promotion_status"] = df["onpromotion"].apply(
    lambda x: "Promotion" if x > 0 else "No Promotion"
)

promotion_sales = (
    df.groupby("promotion_status")["sales"]
    .agg(["sum", "mean", "count"])
)

print("\n--- PROMOTION ANALYSIS ---")
print(promotion_sales)

# Create bar chart
plt.figure(figsize=(9, 5))

promotion_sales["sum"].plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Total Sales: Promotion vs No Promotion")
plt.xlabel("Promotion Status")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()

print("\n--- PROMOTION DATA CHECK ---")

print("Unique promotion values:")
print(df["onpromotion"].value_counts().sort_index())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("\n--- FULL DATASET VERIFICATION ---")

print("Total records:", len(df))
print("Unique stores:", df["store_nbr"].nunique())
print("Product categories:", df["family"].nunique())
print("Unique dates:", df["date"].nunique())

print("\nPromotion distribution:")
print((df["onpromotion"] > 0).value_counts())

print("\nFirst 5 records:")
print(df.head())