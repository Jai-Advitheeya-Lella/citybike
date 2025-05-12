import pandas as pd
import numpy as np
import hopsworks
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ✅ Connect to Hopsworks
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citybike",
    api_key_value="t4rwmi4VfZBIaFqR.ulzGrQ0eIDKCUKgaPYEpNH4JHWuXbIu3YU8gogK8ldP5tpUpnMVIG4uQX1BFW9Wb"
)
fs = project.get_feature_store()
fg = fs.get_feature_group("citibike_hourly_counts", version=1)
df = fg.read()

# ✅ Lag Features
df = df.sort_values(["start_station_name", "date", "hour"])
for lag in range(1, 29):
    df[f"lag_{lag}"] = df.groupby("start_station_name")["ride_count"].shift(lag)

df.dropna(inplace=True)

# ✅ Train/test split
X = df[[f"lag_{i}" for i in range(1, 29)]]
y = df["ride_count"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Start MLflow logging
mlflow.set_experiment("citibike_trip_prediction")

# 🔹 Baseline Model: Predict previous hour
with mlflow.start_run(run_name="baseline_model"):
    y_pred_baseline = X_test["lag_1"]
    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    mlflow.log_param("model", "baseline_lag1")
    mlflow.log_metric("mae", mae_baseline)
    print(f"Baseline MAE: {mae_baseline:.2f}")

# 🔹 LightGBM Model
with mlflow.start_run(run_name="lightgbm_all_lags"):
    lgbm = LGBMRegressor()
    lgbm.fit(X_train, y_train)
    preds = lgbm.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mlflow.log_param("model", "lightgbm_28lags")
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lgbm, "model")
    print(f"LightGBM MAE: {mae:.2f}")

# 🔹 PCA Model with Top 10 Components
with mlflow.start_run(run_name="pca_top10_lightgbm"):
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    pca_model = LGBMRegressor()
    pca_model.fit(X_train_pca, y_train)
    preds_pca = pca_model.predict(X_test_pca)
    mae_pca = mean_absolute_error(y_test, preds_pca)
    mlflow.log_param("model", "lightgbm_pca10")
    mlflow.log_metric("mae", mae_pca)
    mlflow.sklearn.log_model(pca_model, "model_pca")
    print(f"PCA LightGBM MAE: {mae_pca:.2f}")
