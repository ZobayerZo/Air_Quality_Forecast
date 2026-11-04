import json

import pandas as pd

from config import PROCESSED_DIR, ROOT

OUT = ROOT / "data" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    meta = json.load(open(ROOT / "models" / "meta.json"))
    features, cities = meta["features"], meta["cities"]

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df["city_code"] = df["city"].map({c: i for i, c in enumerate(cities)}).astype("int16")

    # 1. Average PM2.5 by city x month x hour (for the heatmap)
    profile = (df.groupby(["city", "month", "hour"])["pm2_5"].mean().round(1)
                 .reset_index().rename(columns={"pm2_5": "avg_pm2_5"}))
    profile.to_parquet(OUT / "hourly_profile.parquet", index=False)

    # 2. Latest feature row per city (for the what-if lab)
    latest = df.sort_values("time").groupby("city").tail(1)
    latest[["city", "time"] + features].to_parquet(OUT / "latest_features.parquet", index=False)

    for f in ["hourly_profile.parquet", "latest_features.parquet"]:
        print(f"  {f}: {(OUT / f).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()