import time
from datetime import date

import pandas as pd
import requests

from config import (AQ_VARIABLES, CITIES, END_DATE, RAW_DIR, START_DATE,
                    WEATHER_VARIABLES)

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_json(url, params, retries=6):
    """Call an API, retrying on timeouts, connection errors and rate limits."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=120)
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as error:
            wait = 15 * (attempt + 1)
            print(f"   Network problem ({type(error).__name__}). Retrying in {wait}s...")
            time.sleep(wait)
            continue

        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f"   Rate limited. Waiting {wait}s...")
            time.sleep(wait)
        else:
            print(f"   Error {response.status_code}: {response.text[:200]}")
            time.sleep(5)
    raise RuntimeError(f"Failed after {retries} tries: {url}")


def year_chunks(start, end):
    """Split the date range into (start, end) pairs, one per calendar year."""
    start_year, end_year = int(start[:4]), int(end[:4])
    chunks = []
    for year in range(start_year, end_year + 1):
        chunk_start = max(start, f"{year}-01-01")
        chunk_end = min(end, f"{year}-12-31")
        chunks.append((chunk_start, chunk_end))
    return chunks


def fetch_city(city, lat, lon):
    frames = []
    for chunk_start, chunk_end in year_chunks(START_DATE, END_DATE):
        common = {"latitude": lat, "longitude": lon,
                  "start_date": chunk_start, "end_date": chunk_end,
                  "timezone": "UTC"}

        aq = get_json(AQ_URL, {**common, "hourly": ",".join(AQ_VARIABLES)})
        wx = get_json(WEATHER_URL, {**common, "hourly": ",".join(WEATHER_VARIABLES)})

        aq_df = pd.DataFrame(aq["hourly"])
        wx_df = pd.DataFrame(wx["hourly"])
        merged = aq_df.merge(wx_df, on="time", how="inner")
        frames.append(merged)
        time.sleep(1)  # be polite to the free API

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df.insert(0, "city", city)
    df.insert(1, "latitude", lat)
    df.insert(2, "longitude", lon)
    return df


def main():
    print(f"Collecting {START_DATE} -> {END_DATE} for {len(CITIES)} cities")
    for i, (city, (lat, lon)) in enumerate(CITIES.items(), start=1):
        out_file = RAW_DIR / f"{city.replace(' ', '_').lower()}.parquet"
        if out_file.exists():
            print(f"[{i}/{len(CITIES)}] {city}: already downloaded, skipping")
            continue
        print(f"[{i}/{len(CITIES)}] Downloading {city}...")
        df = fetch_city(city, lat, lon)
        df.to_parquet(out_file, index=False)
        print(f"   Saved {len(df):,} rows -> {out_file.name}")
    print("Done!")


if __name__ == "__main__":
    main()