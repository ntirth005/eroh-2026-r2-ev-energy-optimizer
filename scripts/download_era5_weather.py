"""Download hourly ERA5 weather for the five cities from the Open-Meteo archive API (ERA5 reanalysis, local time).
Writes data/era5_<CODE>.csv with: time, temperature_2m, cloud_cover, wind_speed_10m, wind_speed_100m (km/h),
shortwave_radiation (W/m2).

Usage (from the repo root):
    pip install pandas requests
    python scripts/download_era5_weather.py
Attribution: Copernicus Climate Change Service (ERA5), Open-Meteo (CC BY 4.0)."""
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CITIES = {
    "DEL": (28.6139, 77.2090),   # Delhi NCR
    "MUM": (19.0760, 72.8777),   # Mumbai MMR
    "BLR": (12.9716, 77.5946),   # Bengaluru
    "HYD": (17.3850, 78.4867),   # Hyderabad
    "MAA": (13.0827, 80.2707),   # Chennai
}
url = "https://archive-api.open-meteo.com/v1/archive"

for code, (lat, lon) in CITIES.items():
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2021-11-22",
        "end_date": "2026-11-20",
        "hourly": "temperature_2m,cloud_cover,wind_speed_10m,wind_speed_100m,shortwave_radiation",
        "timezone": "Asia/Kolkata",
    }
    data = requests.get(url, params=params, timeout=60).json()["hourly"]
    df = pd.DataFrame(data)
    out = DATA_DIR / f"era5_{code}.csv"
    df.to_csv(out, index=False)
    print(f"{code}: saved {len(df)} rows -> {out}")
