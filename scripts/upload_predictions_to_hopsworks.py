import pandas as pd
import hopsworks
from datetime import datetime

# Login to Hopsworks
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citybike",
    api_key_value="t4rwmi4VfZBIaFqR.ulzGrQ0eIDKCUKgaPYEpNH4JHWuXbIu3YU8gogK8ldP5tpUpnMVIG4uQX1BFW9Wb"
)
fs = project.get_feature_store()

# Load predictions
df = pd.read_csv("data/predictions.csv")

# OPTIONAL: Add timestamp for versioning
df["prediction_timestamp"] = datetime.utcnow()

# Upload to Hopsworks
fg = fs.get_or_create_feature_group(
    name="citibike_predictions",
    version=1,
    primary_key=["start_station_name", "date", "hour"],
    description="Predicted ride counts for top stations"
)

fg.insert(df)

print("✅ Predictions uploaded to Hopsworks.")
