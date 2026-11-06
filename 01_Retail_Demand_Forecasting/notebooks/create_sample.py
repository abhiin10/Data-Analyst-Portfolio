
import pandas as pd
from pathlib import Path

# Original dataset
source_file = Path(r"C:\MySQLData2026\Uploads\train.csv")

# Output folder
output_folder = Path("01_Retail_Demand_Forecasting") / "dataset"
output_folder.mkdir(parents=True, exist_ok=True)

# Read first 100,000 records
df = pd.read_csv(source_file, nrows=100_000)

# Save sample dataset
output_file = output_folder / "train_sample_100k.csv"
df.to_csv(output_file, index=False)

print("Sample created successfully!")
print("Total rows:", len(df))
print("Saved at:", output_file.resolve())
print(df.head())
