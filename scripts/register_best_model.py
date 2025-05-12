import hopsworks
import mlflow
import joblib
import os

# ✅ Connect to Hopsworks
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citybike",
    api_key_value="t4rwmi4VfZBIaFqR.ulzGrQ0eIDKCUKgaPYEpNH4JHWuXbIu3YU8gogK8ldP5tpUpnMVIG4uQX1BFW9Wb"
)

fs = project.get_feature_store()
mr = project.get_model_registry()

# ✅ Load the feature view and its training dataset schema
fv = fs.get_feature_view("citibike_hourly_view", version=1)
X_train, y_train = fv.get_training_data(training_dataset_version=4)  # make sure version matches your created one

# ✅ Load best model from MLflow
mlflow.set_experiment("citibike_trip_prediction_v2")
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("citibike_trip_prediction_v2")
runs = client.search_runs(experiment.experiment_id, order_by=["metrics.mae ASC"])

best_run = runs[0]
run_id = best_run.info.run_id
model_uri = f"runs:/{run_id}/best_tuned_model"

# ✅ Save the model locally
model = mlflow.sklearn.load_model(model_uri)
model_dir = "model_artifacts"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "best_lgbm_model.pkl")
joblib.dump(model, model_path)

# ✅ Register model with metadata
model_hops = mr.python.create_model(
    name="citibike_best_model",
    metrics={"mae": best_run.data.metrics["mae"]},
    description="Best tuned LightGBM model with lag/time features, linked to Feature View",
    feature_view=fv,
    training_dataset_version=4
)

# ✅ Save the model artifacts to Hopsworks
model_hops.save(model_dir)

print(f"✅ Model registered to Hopsworks with MAE: {best_run.data.metrics['mae']:.2f}")
print(f"✅ Model version: {model_hops.version}")
