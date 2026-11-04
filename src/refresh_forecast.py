import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from collect_data import get_json
from config import AQ_VARIABLES, CITIES, HORIZONS, ROOT, WEATHER_VARIABLES
from features_pandas import build_features

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"
MODELS_DIR = ROOT / "models"
OUT = ROOT / "data" / "dashboard"
PAST_DAYS = 14


def fetch_recent(city, lat, lon, now):
    common = {"latitude": lat, "longitude": lon, "past_days": PAST_DAYS,
              "forecast_days": 1, "timezone": "UTC"}
    aq = get_json(AQ_URL, {**common, "hourly": ",".join(AQ_VARIABLES)})
    wx = get_json(WX_URL, {**common, "hourly": ",".join(WEATHER_VARIABLES)})
    df = pd.DataFrame(aq["hourly"]).merge(pd.DataFrame(wx["hourly"]), on="time")
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"] <= now]                      # drop future hours
    df = df.loc[: df["pm2_5"].last_valid_index()]   # drop trailing empty hours
    df.insert(0, "city", city)
    df.insert(1, "latitude", lat)
    df.insert(2, "longitude", lon)
    return df


def main():
    meta = json.load(open(MODELS_DIR / "meta.json"))
    features, cities = meta["features"], meta["cities"]
    now = pd.Timestamp.now("UTC").floor("h").tz_localize(None)
    print("Refreshing forecasts at", now, "UTC")

    frames = []
    for city, (lat, lon) in CITIES.items():
        frames.append(fetch_recent(city, lat, lon, now))
        print(f"  fetched {city}")
    recent = pd.concat(frames, ignore_index=True)

    feats = build_features(recent, cities)
    latest = feats.sort_values("time").groupby("city").tail(1)
    bad = latest[latest[features].isna().any(axis=1)]["city"].tolist()
    if bad:
        print("WARNING: skipping cities with incomplete features:", bad)
        latest = latest[~latest["city"].isin(bad)]

    rows = []
    for h in HORIZONS:
        booster = lgb.Booster(model_file=str(MODELS_DIR / f"lgbm_{h}h.txt"))
        preds = np.clip(booster.predict(latest[features]), 0, None)
        for city, t, p in zip(latest["city"], latest["time"], preds):
            rows.append({"city": city, "base_time": t, "horizon_h": h,
                         "forecast_time": t + pd.Timedelta(hours=int(h)),
                         "pm2_5_pred": round(float(p), 1)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "latest_forecast.csv", index=False)
    latest[["city", "time"] + features].to_parquet(OUT / "latest_features.parquet", index=False)
    keep = recent[["city", "time", "latitude", "longitude", "pm2_5"]]
    keep.to_parquet(OUT / "recent.parquet", index=False)
    print(f"Saved forecasts for {latest['city'].nunique()} cities")
    print(pd.DataFrame(rows).query("city == 'Dhaka'").to_string(index=False))


if __name__ == "__main__":
    main()