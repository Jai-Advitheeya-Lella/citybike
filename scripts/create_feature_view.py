import hopsworks

# Connect to Hopsworks
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citybike",
    api_key_value="t4rwmi4VfZBIaFqR.ulzGrQ0eIDKCUKgaPYEpNH4JHWuXbIu3YU8gogK8ldP5tpUpnMVIG4uQX1BFW9Wb"
)

fs = project.get_feature_store()

# Get existing feature group
fg = fs.get_feature_group("citibike_hourly_counts", version=1)

# Define the query
query = fg.select_all()

# Create or get the feature view
fv = fs.get_or_create_feature_view(
    name="citibike_hourly_view",
    version=1,
    description="Feature view with hourly ride counts for top stations",
    labels=["ride_count"],
    query=query
)

# Create training dataset and capture the version
version, job = fv.create_training_data(
    description="Training data for LightGBM ride prediction",
    write_options={"wait_for_job": True}
)

print(f"Created training dataset with version: {version}")

# Get the training data using the feature view method
feature_df, label_df = fv.get_training_data(training_dataset_version=version)

# Compute statistics on the feature group instead
fg.compute_statistics()

print("✅ Feature view, training data, and statistics created successfully.")
