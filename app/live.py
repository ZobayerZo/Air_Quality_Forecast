"""Live data: fetch the latest hours from Open-Meteo and forecast with the trained models.

Called by the dashboard when it loads. Results are cached for 15 minutes and shared by
all visitors, so the free API is called at most once per 15 minutes.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features_pandas import build_features  # same feature code as the pipeline

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"
AQ_VARIABLES = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "us_aqi"]
WX_VARIABLES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "wind_direction_10m", "surface_pressure", "precipitation", "cloud_cover"]
HORIZONS = [24, 48, 72]
PAST_DAYS = 14


def _get(url, params, retries=3):
    last = "unknown error"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except requests.exceptions.RequestException as e:
            last = type(e).__name__
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last)


def _fetch_city(city, lat, lon, now):
    common = {"latitude": lat, "longitude": lon, "past_days": PAST_DAYS,
              "forecast_days": 1, "timezone": "UTC"}
    aq = _get(AQ_URL, {**common, "hourly": ",".join(AQ_VARIABLES)})
    wx = _get(WX_URL, {**common, "hourly": ",".join(WX_VARIABLES)})
    df = pd.DataFrame(aq["hourly"]).merge(pd.DataFrame(wx["hourly"]), on="time")
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"] <= now].reset_index(drop=True)          # drop future hours
    df = df.loc[: df["pm2_5"].last_valid_index()]              # drop trailing empty hours
    df.insert(0, "city", city)
    df.insert(1, "latitude", lat)
    df.insert(2, "longitude", lon)
    return df


@st.cache_data(ttl=900, show_spinner="Fetching the latest air-quality data...")
def fetch_all(coords_items):
    """coords_items: tuple of (city, (lat, lon)). Returns a dict; never raises."""
    try:
        import lightgbm as lgb  # imported here so a problem never breaks the whole app
        coords = dict(coords_items)
        with open(ROOT / "models" / "meta.json") as f:
            meta = json.load(f)
        features, cities = meta["features"], meta["cities"]
        now = pd.Timestamp.now("UTC").tz_localize(None).floor("h")

        results, errors = {}, {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {c: pool.submit(_fetch_city, c, lat, lon, now)
                       for c, (lat, lon) in coords.items()}
            for city, fut in futures.items():
                try:
                    results[city] = fut.result()
                except Exception as e:
                    errors[city] = str(e)
        if not results:
            return {"ok": False, "error": "Open-Meteo did not respond"}

        recent = pd.concat(results.values(), ignore_index=True)
        feats = build_features(recent, cities)
        latest = feats.sort_values("time").groupby("city").tail(1)
        latest = latest[~latest[features].isna().any(axis=1)]

        rows = []
        for h in HORIZONS:
            booster = lgb.Booster(model_file=str(ROOT / "models" / f"lgbm_{h}h.txt"))
            preds = np.clip(booster.predict(latest[features]), 0, None)
            for city, t, p in zip(latest["city"], latest["time"], preds):
                rows.append({"city": city, "base_time": t, "horizon_h": h,
                             "forecast_time": t + pd.Timedelta(hours=h),
                             "pm2_5_pred": round(float(p), 1)})
        return {
            "ok": True,
            "recent": recent[["city", "time", "latitude", "longitude", "pm2_5"]],
            "forecast": pd.DataFrame(rows),
            "latest_feats": latest[["city", "time"] + features].reset_index(drop=True),
            "failed": sorted(errors),
        }
    except Exception as e:  # any problem: let the app fall back to the saved files
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def merge_live(d, res):
    """Replace saved rows with live rows, city by city (saved rows stay for failed cities)."""
    out = dict(d)
    for key in ("recent", "forecast", "latest_feats"):
        old, new = d.get(key), res[key]
        if old is None:
            out[key] = new
        else:
            kept = old[~old["city"].isin(new["city"].unique())]
            out[key] = pd.concat([kept, new], ignore_index=True)
    return out