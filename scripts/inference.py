import hopsworks
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os

# ✅ Connect to Hopsworks
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citybike",
    api_key_value="t4rwmi4VfZBIaFqR.ulzGrQ0eIDKCUKgaPYEpNH4JHWuXbIu3YU8gogK8ldP5tpUpnMVIG4uQX1BFW9Wb"
)
fs = project.get_feature_store()
fg = fs.get_feature_group("citibike_hourly_counts", version=1)
df = fg.read()

# ✅ Generate lag features
df = df.sort_values(["start_station_name", "date", "hour"])
for lag in range(1, 29):
    df[f"lag_{lag}"] = df.groupby("start_station_name")["ride_count"].shift(lag)
df.dropna(inplace=True)

# ✅ Get latest lag state per station
top_stations = df["start_station_name"].value_counts().head(20).index.tolist()
latest_rows = df[df["start_station_name"].isin(top_stations)].groupby("start_station_name").tail(1).copy()

lag_cols = [f"lag_{i}" for i in range(1, 29)]

# ✅ Load best model
model = joblib.load("model_artifacts/best_lgbm_model.pkl")

# ✅ Predict for next 7 days (168 hours)
predictions = []
for hour_offset in range(1, 24 * 7 + 1):  # 1 to 168
    future_time = datetime.now() + timedelta(hours=hour_offset)
    dayofweek = future_time.weekday()
    is_weekend = int(dayofweek >= 5)

    latest_rows["dayofweek"] = dayofweek
    latest_rows["is_weekend"] = is_weekend
    latest_rows["date"] = future_time.date()
    latest_rows["hour"] = future_time.hour

    features = latest_rows[lag_cols + ["dayofweek", "is_weekend"]]
    y_pred = model.predict(features)
    latest_rows["predicted_ride_count"] = y_pred

    predictions.append(latest_rows[["start_station_name", "date", "hour", "dayofweek", "is_weekend", "predicted_ride_count"]].copy())

    # Shift lag features
    for i in range(28, 1, -1):
        latest_rows[f"lag_{i}"] = latest_rows[f"lag_{i-1}"]
    latest_rows["lag_1"] = y_pred

# ✅ Save final predictions
all_predictions = pd.concat(predictions, ignore_index=True)
all_predictions.to_csv("data/predictions.csv", index=False)

print("✅ Final 7-day predictions saved to data/predictions.csv")
