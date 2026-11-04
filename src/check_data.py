import pandas as pd
from config import RAW_DIR

df = pd.concat([pd.read_parquet(f) for f in RAW_DIR.glob("*.parquet")])

print("Total rows:", f"{len(df):,}")
print("Cities:", df["city"].nunique())
print("Date range:", df["time"].min(), "->", df["time"].max())
print("\nMissing values (%):")
print((df.isna().mean() * 100).round(2))
print("\nAverage PM2.5 by city (highest first):")
print(df.groupby("city")["pm2_5"].mean().sort_values(ascending=False).round(1).head(10))