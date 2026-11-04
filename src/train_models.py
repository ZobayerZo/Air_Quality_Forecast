import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import HORIZONS, PROCESSED_DIR, ROOT

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Time-based split (with a gap so targets never overlap between sets)
TRAIN_END = pd.Timestamp("2025-04-30 23:00")
VAL_END = pd.Timestamp("2025-10-31 23:00")
GAP = pd.Timedelta(hours=max(HORIZONS))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    # ---------- Load ----------
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    cities = sorted(df["city"].unique())
    df["city_code"] = df["city"].map({c: i for i, c in enumerate(cities)}).astype("int16")

    exclude = {"city", "time", "latitude", "longitude"} | {f"target_{h}h" for h in HORIZONS}
    features = [c for c in df.columns if c not in exclude]
    print(f"{len(features)} features, {len(cities)} cities")

    # ---------- Split by time ----------
    train = df[df["time"] <= TRAIN_END - GAP]
    val = df[(df["time"] > TRAIN_END) & (df["time"] <= VAL_END - GAP)]
    test = df[df["time"] > VAL_END]
    print(f"Train: {len(train):,} | Validation: {len(val):,} | Test: {len(test):,}")

    results, models, best_iters = [], {}, {}
    backtest = test[["city", "time"]].copy()

    for h in HORIZONS:
        target = f"target_{h}h"
        tr, va, te = (d.dropna(subset=[target]) for d in (train, val, test))
        print(f"\n=== Horizon +{h}h ===")

        model = lgb.LGBMRegressor(
            n_estimators=1500, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(
            tr[features], tr[target],
            eval_set=[(va[features], va[target])],
            eval_metric="l1",
            categorical_feature=["city_code"],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best_iters[h] = int(model.best_iteration_)
        models[h] = model

        # Predictions on the untouched test set
        pred = np.clip(model.predict(te[features]), 0, None)
        base = te["pm2_5"].to_numpy()          # persistence baseline
        y = te[target].to_numpy()

        for name, p in [("Persistence", base), ("LightGBM", pred)]:
            results.append({"horizon_h": h, "model": name,
                            "MAE": round(mean_absolute_error(y, p), 2),
                            "RMSE": round(rmse(y, p), 2)})

        # Keep predictions for the dashboard's backtest chart
        part = te[["city", "time"]].copy()
        part[f"actual_{h}h"] = y.astype("float32")
        part[f"pred_{h}h"] = pred.astype("float32")
        backtest = backtest.merge(part, on=["city", "time"], how="left")

        model.booster_.save_model(str(MODELS_DIR / f"lgbm_{h}h.txt"))
        print(f"best iteration: {best_iters[h]}")

    # ---------- Results table ----------
    res = pd.DataFrame(results)
    print("\n=========== TEST RESULTS (Nov 2025 - Oct 2026) ===========")
    print(res.to_string(index=False))

    summary = []
    for h in HORIZONS:
        a = res[(res.horizon_h == h) & (res.model == "Persistence")].iloc[0]
        b = res[(res.horizon_h == h) & (res.model == "LightGBM")].iloc[0]
        summary.append({"horizon_h": h,
                        "MAE_improvement_%": round((1 - b.MAE / a.MAE) * 100, 1),
                        "RMSE_improvement_%": round((1 - b.RMSE / a.RMSE) * 100, 1)})
    print("\nLightGBM improvement over persistence:")
    print(pd.DataFrame(summary).to_string(index=False))

    # Dhaka only (your motivation city)
    d = backtest[backtest["city"] == "Dhaka"]
    print("\nDhaka test MAE (LightGBM):")
    for h in HORIZONS:
        x = d.dropna(subset=[f"actual_{h}h"])
        print(f"  +{h}h: {mean_absolute_error(x[f'actual_{h}h'], x[f'pred_{h}h']):.2f} ug/m3")

    # ---------- Save everything the dashboard needs ----------
    res.to_csv(MODELS_DIR / "metrics.csv", index=False)
    backtest.to_parquet(PROCESSED_DIR / "backtest.parquet", index=False)
    with open(MODELS_DIR / "meta.json", "w") as f:
        json.dump({"features": features, "cities": cities, "horizons": HORIZONS,
                   "best_iterations": best_iters,
                   "train_end": str(TRAIN_END), "val_end": str(VAL_END)}, f, indent=2)

    # First forecast: from the last available hour of each city
    latest = df.sort_values("time").groupby("city").tail(1)
    rows = []
    for h in HORIZONS:
        preds = np.clip(models[h].predict(latest[features]), 0, None)
        for city, t, p in zip(latest["city"], latest["time"], preds):
            rows.append({"city": city, "base_time": t, "horizon_h": h,
                         "forecast_time": t + pd.Timedelta(hours=h),
                         "pm2_5_pred": round(float(p), 1)})
    pd.DataFrame(rows).to_csv(PROCESSED_DIR / "latest_forecast.csv", index=False)
    print("\nSaved models, metrics, backtest and latest forecast.")


if __name__ == "__main__":
    main()