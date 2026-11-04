import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "us_aqi",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "precipitation", "cloud_cover",
]
PM_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]


def build_features(df, cities):
    """Pandas version of the Spark pipeline: same features, same names."""
    code = {c: i for i, c in enumerate(cities)}
    parts = []
    for city, g in df.groupby("city"):
        g = g.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        g[NUMERIC_COLS] = g[NUMERIC_COLS].ffill()
        g["pm2_5"] = g["pm2_5"].clip(lower=0)
        pm = g["pm2_5"]

        for h in PM_LAGS:
            g[f"pm2_5_lag_{h}h"] = pm.shift(h)
        for n in (6, 24, 168):
            g[f"pm2_5_roll_mean_{n}h"] = pm.rolling(n).mean()
        g["pm2_5_roll_std_24h"] = pm.rolling(24).std()
        g["pm2_5_roll_max_24h"] = pm.rolling(24).max()
        g["pm2_5_roll_std_168h"] = pm.rolling(168).std()

        g["wind_roll_mean_24h"] = g["wind_speed_10m"].rolling(24).mean()
        g["humidity_roll_mean_24h"] = g["relative_humidity_2m"].rolling(24).mean()
        g["temp_roll_mean_24h"] = g["temperature_2m"].rolling(24).mean()
        g["precip_sum_24h"] = g["precipitation"].rolling(24).sum()
        g["pressure_change_24h"] = g["surface_pressure"] - g["surface_pressure"].shift(24)

        t = g["time"].dt
        g["hour"] = t.hour
        g["day_of_week"] = (t.dayofweek + 1) % 7 + 1      # Spark: Sunday=1 ... Saturday=7
        g["month"] = t.month
        g["day_of_year"] = t.dayofyear
        g["is_weekend"] = g["day_of_week"].isin([1, 7]).astype(int)
        two_pi = 2 * np.pi
        g["hour_sin"] = np.sin(g["hour"] * two_pi / 24)
        g["hour_cos"] = np.cos(g["hour"] * two_pi / 24)
        g["doy_sin"] = np.sin(g["day_of_year"] * two_pi / 365.25)
        g["doy_cos"] = np.cos(g["day_of_year"] * two_pi / 365.25)
        g["wind_dir_sin"] = np.sin(np.radians(g["wind_direction_10m"]))
        g["wind_dir_cos"] = np.cos(np.radians(g["wind_direction_10m"]))
        g["city_code"] = code[city]
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out["city_code"] = out["city_code"].astype("int16")
    return out