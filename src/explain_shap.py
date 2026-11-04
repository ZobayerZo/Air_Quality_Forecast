import json

import matplotlib
matplotlib.use("Agg")  # draw to files, no pop-up windows
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from config import PROCESSED_DIR, ROOT

MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

HORIZON = 24
SAMPLE_PER_CITY = 1500   # rows per city used for SHAP (keeps it fast)

WEATHER = {"temperature_2m", "relative_humidity_2m", "wind_speed_10m",
           "wind_direction_10m", "surface_pressure", "precipitation", "cloud_cover",
           "wind_roll_mean_24h", "humidity_roll_mean_24h", "temp_roll_mean_24h",
           "precip_sum_24h", "pressure_change_24h", "wind_dir_sin", "wind_dir_cos"}
OTHER_POLLUTANTS = {"pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "us_aqi"}
CALENDAR = {"hour", "day_of_week", "month", "day_of_year", "is_weekend",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos"}


def group_of(feature):
    if feature.startswith("pm2_5"):
        return "Recent PM2.5 history"
    if feature in WEATHER:
        return "Weather"
    if feature in OTHER_POLLUTANTS:
        return "Other pollutants"
    if feature in CALENDAR:
        return "Time / season"
    return "Location"


def main():
    with open(MODELS_DIR / "meta.json") as f:
        meta = json.load(f)
    features, cities = meta["features"], meta["cities"]

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df["city_code"] = df["city"].map({c: i for i, c in enumerate(cities)}).astype("int16")

    # Use the test period only (data the model never trained on)
    target = f"target_{HORIZON}h"
    test = df[(df["time"] > pd.Timestamp(meta["val_end"]))].dropna(subset=[target])
    sample = (test.groupby("city", group_keys=False)
                  .sample(n=SAMPLE_PER_CITY, random_state=42)
                  .reset_index(drop=True))
    X = sample[features]
    print(f"Explaining {len(X):,} rows from {sample['city'].nunique()} cities")

    # ---------- SHAP values ----------
    booster = lgb.Booster(model_file=str(MODELS_DIR / f"lgbm_{HORIZON}h.txt"))
    contrib = booster.predict(X, pred_contrib=True)   # last column = base value
    sv = contrib[:, :-1]

    # Sanity check: base + sum of contributions must equal the prediction
    gap = np.abs(contrib.sum(axis=1) - booster.predict(X)).max()
    print(f"Additivity check (should be ~0): {gap:.6f}")

    # ---------- Importance table (for the dashboard) ----------
    abs_sv = pd.DataFrame(np.abs(sv), columns=features)
    abs_sv["city"] = sample["city"].values
    per_city = abs_sv.groupby("city").mean().T
    per_city["ALL"] = abs_sv.drop(columns="city").mean()
    long = (per_city.reset_index().rename(columns={"index": "feature"})
            .melt(id_vars="feature", var_name="city", value_name="mean_abs_shap"))
    long["group"] = long["feature"].map(group_of)
    long["mean_abs_shap"] = long["mean_abs_shap"].round(4)
    long.to_csv(PROCESSED_DIR / "shap_importance.csv", index=False)

    # ---------- Console summary ----------
    for name in ["ALL", "Dhaka"]:
        d = long[long["city"] == name].sort_values("mean_abs_shap", ascending=False)
        print(f"\nTop 10 features ({name}):")
        print(d.head(10)[["feature", "mean_abs_shap"]].to_string(index=False))
        g = d.groupby("group")["mean_abs_shap"].sum()
        print(f"Share of influence by group ({name}):")
        print((g / g.sum() * 100).round(1).sort_values(ascending=False).to_string())

    # ---------- Plot 1: global top 15 ----------
    top = (long[long["city"] == "ALL"].sort_values("mean_abs_shap", ascending=False)
           .head(15).iloc[::-1])
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2a7de1")
    plt.xlabel("Mean |SHAP value| (ug/m3)")
    plt.title(f"What drives the +{HORIZON}h PM2.5 forecast (all cities)")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "shap_global_top15.png", dpi=150)
    plt.close()

    # ---------- Plot 2: Dhaka beeswarm ----------
    mask = (sample["city"] == "Dhaka").values
    plt.figure()
    shap.summary_plot(sv[mask], X[mask], max_display=15, show=False)
    plt.title(f"Dhaka: feature effects on the +{HORIZON}h PM2.5 forecast")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "shap_dhaka_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- Plot 3: driver groups, Dhaka vs all ----------
    groups = (long[long["city"].isin(["ALL", "Dhaka"])]
              .groupby(["city", "group"])["mean_abs_shap"].sum().unstack(0))
    groups = groups / groups.sum() * 100
    groups = groups.sort_values("ALL")
    groups.plot(kind="barh", figsize=(8, 4.5), color=["#e0562a", "#2a7de1"])
    plt.xlabel("Share of model influence (%)")
    plt.ylabel("")
    plt.title("Which kinds of information the model relies on")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "shap_driver_groups.png", dpi=150)
    plt.close()

    print("\nSaved 3 plots to /assets and shap_importance.csv")


if __name__ == "__main__":
    main()