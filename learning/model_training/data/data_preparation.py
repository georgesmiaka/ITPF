import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("Starting data preparation...")
print("")
# Load the CSV file into a DataFrame
df = pd.read_csv('files/data_trips.csv')

# Select only the specified columns
selected_columns = ['x', 'y', 'trip_id', 'town']
df_selected = df[selected_columns]

# Display the first few rows of the resulting DataFrame
print(df_selected.head())
print("")
print("Generating fake dates...")
print("")


# Generate random start dates within the range 2019-2024 for each trip_id
start_dates = {}
for trip_id in df_selected['trip_id'].unique():
    start_year = random.randint(2019, 2024)
    start_month = random.randint(1, 12)
    start_day = random.randint(1, 28)  # To avoid issues with February
    start_hour = random.randint(0, 23)
    start_minute = random.randint(0, 59)
    start_second = random.randint(0, 59)
    start_date = datetime(start_year, start_month, start_day, start_hour, start_minute, start_second)
    start_dates[trip_id] = start_date

# Generate timestamps for each data point within the same trip_id
timestamps = []
for trip_id in df_selected['trip_id']:
    if trip_id not in start_dates:
        continue
    start_date = start_dates[trip_id]
    timestamps.append(start_date)
    start_dates[trip_id] += timedelta(seconds=1)

# Add the timestamps to the DataFrame
df_selected['timestamp'] = timestamps

# Sort the DataFrame by the 'timestamp' column
df_selected = df_selected.sort_values(by='timestamp')

# Save the resulting DataFrame to a new CSV file (optional)
df_selected.to_csv('files/data.csv', index=False)

# Display the first few rows of the resulting DataFrame
print(df_selected.head())