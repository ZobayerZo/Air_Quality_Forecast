import shutil

import pandas as pd

from config import HORIZONS, PROCESSED_DIR, ROOT

OUT = ROOT / "data" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Last 14 days of observed PM2.5 per city (for the forecast chart + map)
    feats = pd.read_parquet(
        PROCESSED_DIR / "features.parquet",
        columns=["city", "time", "latitude", "longitude", "pm2_5"],
    )
    cutoff = feats["time"].max() - pd.Timedelta(days=14)
    feats[feats["time"] > cutoff].to_parquet(OUT / "recent.parquet", index=False)

    # 2. Forecast uncertainty: 10th/90th percentile of test-period errors
    bt = pd.read_parquet(PROCESSED_DIR / "backtest.parquet")
    parts = []
    for h in HORIZONS:
        resid = pd.DataFrame({"city": bt["city"],
                              "res": bt[f"actual_{h}h"] - bt[f"pred_{h}h"]}).dropna()
        q = resid.groupby("city")["res"].quantile([0.1, 0.9]).unstack().reset_index()
        q.columns = ["city", "q10", "q90"]
        q["horizon_h"] = h
        parts.append(q.round(2))
    pd.concat(parts).to_csv(OUT / "intervals.csv", index=False)

    # 3. Last 120 days of backtest (actual vs predicted), kept small
    recent_bt = bt[bt["time"] > bt["time"].max() - pd.Timedelta(days=120)].copy()
    float_cols = recent_bt.select_dtypes("float").columns
    recent_bt[float_cols] = recent_bt[float_cols].round(1)
    recent_bt.to_parquet(OUT / "backtest_recent.parquet", index=False)

    # 4. Small files that already exist
    for name in ["latest_forecast.csv", "shap_importance.csv", "city_monthly_summary.csv"]:
        shutil.copy(PROCESSED_DIR / name, OUT / name)

    print("Dashboard data written to", OUT)
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()