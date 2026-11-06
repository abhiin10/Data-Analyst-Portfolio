
import pandas as pd
from pathlib import Path

# Load retail sales data
file_path = Path(__file__).resolve().parents[1] / "dataset" / "train.csv"

df = pd.read_csv(file_path, nrows=100000)
# Convert date column into datetime format
df["date"] = pd.to_datetime(df["date"])

# Verify the conversion
print("\nUPDATED DATA TYPES:")
print(df.dtypes)

print("\nDATE RANGE:")
print(df["date"].min(), "to", df["date"].max())

# 1. Display first 5 records
print("\nFIRST 5 RECORDS")
print(df.head())

# 2. Dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# 3. Column information
print("\nDATASET INFORMATION")
print(df.info())

# 4. Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 5. Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())


# 6. Sales by product category
#this is the part where we will analyze the sales by product category. We will group the data by the 'family' column and sum the 'sales' for each category. Then, we will sort the results in descending order to see which categories have the highest sales.
print("\nSALES BY PRODUCT CATEGORY")

category_sales = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print(category_sales)

# 7. Top 5 categories

print("\nTOP 5 SELLING CATEGORIES")
print(category_sales.head(5))


import matplotlib.pyplot as plt

# Top 5 product categories
top_5 = category_sales.head(5)

# Create bar chart
plt.figure(figsize=(10, 5))
top_5.plot(kind="bar")

plt.title("Top 5 Product Categories by Sales")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=30)
plt.tight_layout()

plt.show()



# DATA QUALITY CHECK

print("\n========== DATA QUALITY REPORT ==========")

# 1. Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# 2. Check duplicate records
print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

# 3. Check column data types
print("\nDATA TYPES:")
print(df.dtypes)

# 4. Check total records
print("\nTOTAL RECORDS:")
print(len(df))

print("\n========== REPORT COMPLETED ==========")


# SALES DATE ANALYSIS

print("\nEARLIEST SALES DATE:")
print(df["date"].min())

print("\nLATEST SALES DATE:")
print(df["date"].max())

print("\nNUMBER OF UNIQUE DATES:")
print(df["date"].nunique())

# Daily sales analysis
daily_sales = df.groupby("date")["sales"].sum()

print("\nFIRST 10 DAYS OF SALES:")
print(daily_sales.head(10))


# DAILY SALES TREND VISUALIZATION

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

daily_sales.plot(kind="line", marker="o")

plt.title("Daily Retail Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True)

plt.tight_layout()
plt.show()


# WEEKLY SALES ANALYSIS

weekly_sales = (
    df.set_index("date")
    .resample("W")["sales"]
    .sum()
)

print("\nWEEKLY SALES:")
print(weekly_sales)

# Visualize weekly sales

plt.figure(figsize=(12, 5))

weekly_sales.plot(kind="bar")

plt.title("Weekly Retail Sales Analysis")
plt.xlabel("Week")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# MONTHLY SALES ANALYSIS

monthly_sales = (
    df.set_index("date")
    .resample("ME")["sales"]
    .sum()
)

print("\nMONTHLY SALES:")
print(monthly_sales)

# Monthly sales visualization

plt.figure(figsize=(10, 5))

monthly_sales.plot(kind="bar")

plt.title("Monthly Retail Sales Analysis")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



# MONTHLY BUSINESS SUMMARY

monthly_summary = (
    df.groupby(df["date"].dt.to_period("M"))
    .agg(
        total_sales=("sales", "sum"),
        days_in_sample=("date", "nunique")
    )
)

monthly_summary["average_daily_sales"] = (
    monthly_summary["total_sales"] /
    monthly_summary["days_in_sample"]
)

print("\nMONTHLY BUSINESS SUMMARY")
print(monthly_summary)


# TOP 10 PERFORMING STORES

store_sales = (
    df.groupby("store_nbr")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTOP 10 PERFORMING STORES")
print(store_sales.head(10))

# Visualize top 10 stores

plt.figure(figsize=(12, 5))

store_sales.head(10).plot(kind="bar")

plt.title("Top 10 Stores by Total Sales")
plt.xlabel("Store Number")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# TOP 10 PRODUCT CATEGORIES

category_sales = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTOP 10 PRODUCT CATEGORIES")
print(category_sales.head(10))

# Visualization

plt.figure(figsize=(12, 6))

category_sales.head(10).plot(kind="bar")

plt.title("Top 10 Product Categories by Sales")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# STORE-WISE PRODUCT CATEGORY ANALYSIS

store_category_sales = (
    df.groupby(["store_nbr", "family"])["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTOP 10 STORE-CATEGORY COMBINATIONS")
print(store_category_sales.head(10))

# Display as a chart

# PROFESSIONAL HORIZONTAL BAR CHART

top_combinations = store_category_sales.head(10)

labels = [
    f"Store {store} - {category}"
    for store, category in top_combinations.index
]


plt.figure(figsize=(11, 6))

plt.barh(labels[::-1], top_combinations.values[::-1])

plt.title("Top 10 Store-Category Sales Combinations")
plt.xlabel("Total Sales")
plt.ylabel("Store and Product Category")

plt.tight_layout()

from pathlib import Path

output_dir = Path("01_Retail_Demand_Forecasting") / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

# Save chart first
plt.savefig(
    output_dir / "top_10_store_category_sales.png",
    dpi=300,
    bbox_inches="tight"
)

# Display chart last
plt.show()


# ==========================================
# BUSINESS INSIGHTS & FINDINGS
# ==========================================

print("\n========== BUSINESS INSIGHTS ==========")

# 1. Highest-selling product category
category_sales = df.groupby("family")["sales"].sum().sort_values(ascending=False)

print("\n1. TOP 5 PRODUCT CATEGORIES:")
print(category_sales.head(5))

# 2. Highest-performing store
store_sales = df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False)

print("\n2. TOP 5 STORES:")
print(store_sales.head(5))

# 3. Best sales date
daily_sales = df.groupby("date")["sales"].sum().sort_values(ascending=False)

print("\n3. TOP 5 SALES DATES:")
print(daily_sales.head(5))

# 4. Average sales
print("\n4. OVERALL SALES SUMMARY:")
print("Total Sales:", round(df["sales"].sum(), 2))
print("Average Sales:", round(df["sales"].mean(), 2))
print("Maximum Sales:", round(df["sales"].max(), 2))

# 5. Promotion analysis
promotion_sales = df.groupby("onpromotion")["sales"].mean()

print("\n5. AVERAGE SALES BY PROMOTION COUNT:")
print(promotion_sales)

print("\n========== ANALYSIS COMPLETED ==========")



