import pandas as pd
import hopsworks
from datetime import datetime
import glob

# Step 1: Connect to Hopsworks
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citybike",
    api_key_value="t4rwmi4VfZBIaFqR.ulzGrQ0eIDKCUKgaPYEpNH4JHWuXbIu3YU8gogK8ldP5tpUpnMVIG4uQX1BFW9Wb"
)
fs = project.get_feature_store()

# Step 2: Load all 2023 Citi Bike CSV files
files = glob.glob("data/2023*-citibike-tripdata.csv")
df = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(files)], ignore_index=True)

# Step 3: Parse and filter datetime
df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")
df = df.dropna(subset=["started_at", "ended_at"])

# ✅ Keep only 2023 rides
df = df[df["started_at"].dt.year == 2023]

# Step 4: Extract date and hour
df["date"] = df["started_at"].dt.date
df["hour"] = df["started_at"].dt.hour

# Step 5: Select top 3 most common stations
top_stations = df["start_station_name"].value_counts().head(3).index.tolist()
df = df[df["start_station_name"].isin(top_stations)]

# Step 6: Aggregate ride counts
grouped = df.groupby(["start_station_name", "date", "hour"]).size().reset_index(name="ride_count")

# Step 7: Save to Hopsworks Feature Store
fg = fs.get_or_create_feature_group(
    name="citibike_hourly_counts",
    version=1,
    primary_key=["start_station_name", "date", "hour"],
    description="Hourly ride count for top 3 stations (2023 only)"
)

fg.insert(grouped)

print("✅ Feature group saved to Hopsworks.")
