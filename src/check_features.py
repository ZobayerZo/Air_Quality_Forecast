import pandas as pd
from config import PROCESSED_DIR

df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
print("Shape:", df.shape)
print("Rows with targets:", df["target_72h"].notna().sum())
print("Rows to forecast from (no target yet):", df["target_72h"].isna().sum())
print("\nDhaka sample:")
cols = ["time", "pm2_5", "pm2_5_lag_24h", "pm2_5_roll_mean_24h", "target_24h"]
print(df[df["city"] == "Dhaka"][cols].iloc[1000:1005])