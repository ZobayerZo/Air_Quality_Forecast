from datetime import date, timedelta
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Air-quality archive starts around Oct 2022
START_DATE = "2022-11-01"

# Weather archive (ERA5) lags a few days behind today, so stop a week back.
# The daily GitHub Action in Step 6 will fill the recent days.
END_DATE = (date.today() - timedelta(days=7)).isoformat()

# City name -> (latitude, longitude)
CITIES = {
    "Dhaka": (23.8103, 90.4125),
    "Chittagong": (22.3569, 91.7832),
    "Delhi": (28.6139, 77.2090),
    "Kolkata": (22.5726, 88.3639),
    "Mumbai": (19.0760, 72.8777),
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Kathmandu": (27.7172, 85.3240),
    "Beijing": (39.9042, 116.4074),
    "Shanghai": (31.2304, 121.4737),
    "Bangkok": (13.7563, 100.5018),
    "Hanoi": (21.0278, 105.8342),
    "Jakarta": (-6.2088, 106.8456),
    "Seoul": (37.5665, 126.9780),
    "Tokyo": (35.6762, 139.6503),
    "Tehran": (35.6892, 51.3890),
    "Riyadh": (24.7136, 46.6753),
    "Cairo": (30.0444, 31.2357),
    "Lagos": (6.5244, 3.3792),
    "Nairobi": (-1.2921, 36.8219),
    "Johannesburg": (-26.2041, 28.0473),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Istanbul": (41.0082, 28.9784),
    "Mexico City": (19.4326, -99.1332),
    "Sao Paulo": (-23.5505, -46.6333),
    "Los Angeles": (34.0522, -118.2437),
    "New York": (40.7128, -74.0060),
    "Sydney": (-33.8688, 151.2093),
}

# What we download
AQ_VARIABLES = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "us_aqi"]
WEATHER_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "precipitation", "cloud_cover",
]

# Where processed data goes
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Forecast horizons in hours (we predict PM2.5 at t+24, t+48, t+72)
HORIZONS = [24, 48, 72]