
import pandas as pd
from pathlib import Path

# Project folders
project_dir = Path(__file__).resolve().parents[1]
dataset_dir = project_dir / "dataset"

input_file = dataset_dir / "train.csv"
output_file = dataset_dir / "retail_daily_aggregated.csv"

print("Starting full dataset preparation...")

if not input_file.exists():
    raise FileNotFoundError(f"Could not find: {input_file}")

# Process the large CSV in manageable chunks
chunk_size = 500_000
partial_results = []

for chunk_number, chunk in enumerate(
    pd.read_csv(input_file, chunksize=chunk_size),
    start=1
):
    print(f"Processing chunk {chunk_number}...")

    chunk["onpromotion"] = chunk["onpromotion"].fillna(0)

    # Aggregate daily sales by store and product category
    daily_data = (
        chunk.groupby(
            ["date", "store_nbr", "family"],
            as_index=False
        )
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum")
        )
    )

    partial_results.append(daily_data)

# Combine and aggregate across all chunks
print("Combining processed data...")

df = pd.concat(partial_results, ignore_index=True)

df = (
    df.groupby(
        ["date", "store_nbr", "family"],
        as_index=False
    )
    .agg(
        sales=("sales", "sum"),
        onpromotion=("onpromotion", "sum")
    )
)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(
    ["date", "store_nbr", "family"]
).reset_index(drop=True)

# Save the prepared dataset
df.to_csv(output_file, index=False)

print("\n--- DATA PREPARATION COMPLETE ---")
print("Total aggregated records:", len(df))
print("Unique stores:", df["store_nbr"].nunique())
print("Product categories:", df["family"].nunique())
print("Unique dates:", df["date"].nunique())
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Promotion records:", (df["onpromotion"] > 0).sum())
print("Saved file:", output_file)
