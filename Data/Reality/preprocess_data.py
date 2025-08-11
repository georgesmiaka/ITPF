# preprocess_knn.py
import pandas as pd
import numpy as np

INPUT = "data/vehicle_activities.csv"
OUTPUT = "data/vehicle_activities_processed.csv"

def cyclical_enc(val, period):
    ang = 2 * np.pi * (val % period) / period
    return np.sin(ang), np.cos(ang)

def main():
    df = pd.read_csv(INPUT)

    # Parse times (ISO8601 with offset -> tz-aware)
    df["start_dt"] = pd.to_datetime(df["startTime"], utc=True)
    df["end_dt"]   = pd.to_datetime(df["endTime"],   utc=True)

    # Duration (seconds)
    df["duration_s"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds()

    # Basic sanity filters (optional but useful)
    # keep durations between 1 min and 3 hours
    df = df[(df["duration_s"] >= 60) & (df["duration_s"] <= 3*3600)]
    # keep reasonably certain detections
    if "probability" in df.columns:
        df = df[df["probability"] >= 0.5]

    # Time features from local start time (convert from UTC to Europe/Stockholm if you want)
    # Since your ISO strings include offsets already, we can also just use .dt.tz_convert:
    df["start_local"] = df["start_dt"].dt.tz_convert("Europe/Stockholm")
    df["hour"] = df["start_local"].dt.hour
    df["dow"]  = df["start_local"].dt.dayofweek  # 0=Mon

    df["hour_sin"], df["hour_cos"] = zip(*df["hour"].map(lambda h: cyclical_enc(h, 24)))
    df["dow_sin"],  df["dow_cos"]  = zip(*df["dow"].map(lambda d: cyclical_enc(d, 7)))

    # Keep only columns we’ll use downstream
    keep = [
        "start_lat","start_lon","end_lat","end_lon",
        "distance_m","duration_s","hour_sin","hour_cos",
        "dow_sin","dow_cos"
    ]
    df[keep].to_csv(OUTPUT, index=False)
    #print(f"Wrote {OUTPUT} with {len(df)} rows.")

if __name__ == "__main__":
    main()
