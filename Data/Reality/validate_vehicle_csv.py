#!/usr/bin/env python3
import os
import math
import pandas as pd
from datetime import timezone

# Fixed input path
CSV_PATH = "data/vehicle_activities.csv"

REQUIRED_COLUMNS = [
    "start_lat","start_lon","end_lat","end_lon",
    "startTime","endTime","distance_m","probability"
]

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def fail(msg):
    print(f"ERROR: {msg}")
    exit(1)

def warn(msg):
    print(f"WARNING: {msg}")

def main():
    # --- file check ---
    if not os.path.isfile(CSV_PATH):
        fail(f"File not found: {CSV_PATH}")
    if not CSV_PATH.lower().endswith(".csv"):
        fail("File must have .csv extension")

    try:
        df = pd.read_csv(CSV_PATH, dtype=str)
    except Exception as e:
        fail(f"Could not read CSV: {e}")

    # --- schema check ---
    if list(df.columns) != REQUIRED_COLUMNS:
        fail(f"Header mismatch.\nExpected: {','.join(REQUIRED_COLUMNS)}\nFound:    {','.join(df.columns)}")

    if len(df) == 0:
        fail("CSV has no rows.")

    errors = 0

    # --- numeric type checks ---
    for col in ["start_lat","start_lon","end_lat","end_lon","distance_m","probability"]:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except:
            print(f"ERROR: Column '{col}' contains non-numeric values.")
            errors += 1

    # --- time type checks ---
    for col in ["startTime","endTime"]:
        try:
            ts = pd.to_datetime(df[col], utc=True, errors="raise")
            if ts.isna().any():
                print(f"ERROR: Column '{col}' has unparsable datetimes.")
                errors += 1
            df[col+"_parsed"] = ts
        except:
            print(f"ERROR: Column '{col}' has invalid ISO8601 datetimes.")
            errors += 1

    if errors:
        exit(1)

    # --- nulls ---
    if df.isna().any().any():
        na_counts = df.isna().sum()
        print("ERROR: Missing values detected:\n" + na_counts[na_counts>0].to_string())
        exit(1)

    # --- ranges ---
    def check_range(name, s, lo, hi):
        nonlocal errors
        bad = ~s.between(lo, hi)
        if bad.any():
            print(f"ERROR: {name} out of range [{lo}, {hi}] in {bad.sum()} rows.")
            errors += 1

    check_range("start_lat", df["start_lat"], -90, 90)
    check_range("end_lat",   df["end_lat"],   -90, 90)
    check_range("start_lon", df["start_lon"], -180, 180)
    check_range("end_lon",   df["end_lon"],   -180, 180)
    check_range("distance_m",df["distance_m"],0, float("inf"))
    check_range("probability",df["probability"],0,1)

    # --- duration check ---
    df["duration_s"] = (df["endTime_parsed"] - df["startTime_parsed"]).dt.total_seconds()
    if (df["duration_s"] <= 0).any():
        print("ERROR: endTime must be after startTime.")
        errors += 1

    min_dur, max_dur = 2, 4*3600
    too_short = df["duration_s"] < min_dur
    too_long  = df["duration_s"] > max_dur
    if too_short.any():
        warn(f"{too_short.sum()} trips shorter than {min_dur}s.")
    #if too_long.any():
    #    warn(f"{too_long.sum()} trips longer than {max_dur}s.")

    # --- distance sanity ---
    hv = [
        haversine_m(sl, so, el, eo)
        for sl, so, el, eo in zip(df["start_lat"], df["start_lon"], df["end_lat"], df["end_lon"])
    ]
    df["haversine_m"] = hv

    too_short_dist = df["distance_m"] < 0.8 * df["haversine_m"]
    too_long_dist  = df["distance_m"] > 5.0 * df["haversine_m"]
    #if too_short_dist.any():
    #    print(f"ERROR: {too_short_dist.sum()} rows have distance_m < 80% of straight-line distance.")
    #    errors += 1
    #if too_long_dist.any():
    #    warn(f"{too_long_dist.sum()} rows have distance_m > 5x straight-line distance.")

    # --- duplicates ---
    dup_mask = df.duplicated(subset=REQUIRED_COLUMNS, keep=False)
    if dup_mask.any():
        warn(f"{dup_mask.sum()} duplicate rows found.")

    if errors:
        fail("Validation failed.")
    else:
        print("✅ Validation passed.")
        print(f"Rows: {len(df)}")
        print(f"Duration range: {df['duration_s'].min():.1f}s → {df['duration_s'].max():.1f}s")
        print(f"Median haversine/distance ratio: {(df['distance_m']/df['haversine_m']).median():.2f}")

if __name__ == "__main__":
    main()
