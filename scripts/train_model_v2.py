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

# ✅ Feature engineering
df["date"] = pd.to_datetime(df["date"])
df["dayofweek"] = df["date"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# ✅ Lag features
df = df.sort_values(["start_station_name", "date", "hour"])
for lag in range(1, 29):
    df[f"lag_{lag}"] = df.groupby("start_station_name")["ride_count"].shift(lag)

df.dropna(inplace=True)

# ✅ Features & target
lag_features = [f"lag_{i}" for i in range(1, 29)]
extra_features = ["dayofweek", "is_weekend"]
X = df[lag_features + extra_features]
y = df["ride_count"]

# ✅ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ MLflow experiment
mlflow.set_experiment("citibike_trip_prediction_v2")

# 🔹 1. Baseline model (lag_1)
with mlflow.start_run(run_name="baseline_lag1"):
    baseline_pred = X_test["lag_1"]
    mae_baseline = mean_absolute_error(y_test, baseline_pred)
    mlflow.log_param("model", "baseline_lag1")
    mlflow.log_metric("mae", mae_baseline)
    print(f"Baseline MAE: {mae_baseline:.2f}")

# 🔹 2. LightGBM (lags only)
with mlflow.start_run(run_name="lightgbm_lags_only"):
    model_lags = LGBMRegressor()
    model_lags.fit(X_train[lag_features], y_train)
    preds = model_lags.predict(X_test[lag_features])
    mae = mean_absolute_error(y_test, preds)
    mlflow.log_param("model", "lightgbm_lags")
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(model_lags, "model_lags")
    print(f"LightGBM (lags only) MAE: {mae:.2f}")

# 🔹 3. LightGBM (lags + time)
with mlflow.start_run(run_name="lightgbm_lags_timefeatures"):
    model_lags_time = LGBMRegressor()
    model_lags_time.fit(X_train, y_train)
    preds = model_lags_time.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mlflow.log_param("model", "lightgbm_lags+time")
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(model_lags_time, "model_lags_time")
    print(f"LightGBM (lags + time) MAE: {mae:.2f}")

# 🔹 4. LightGBM tuning
best_mae = float("inf")
best_params = {}

learning_rates = [0.05, 0.1]
max_depths = [5, 10]

for lr in learning_rates:
    for depth in max_depths:
        with mlflow.start_run(run_name=f"tuned_lgbm_lr{lr}_depth{depth}"):
            model = LGBMRegressor(learning_rate=lr, max_depth=depth)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)

            mlflow.log_params({
                "model": "lightgbm_tuned",
                "learning_rate": lr,
                "max_depth": depth
            })
            mlflow.log_metric("mae", mae)

            if mae < best_mae:
                best_mae = mae
                best_params = {"learning_rate": lr, "max_depth": depth}
                mlflow.sklearn.log_model(model, "best_tuned_model")

print(f"Best tuned LightGBM MAE: {best_mae:.2f} with params {best_params}")

# 🔹 5. PCA (top 10 components from lag features)
with mlflow.start_run(run_name="lightgbm_pca10"):
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train[lag_features])
    X_test_pca = pca.transform(X_test[lag_features])

    model_pca = LGBMRegressor()
    model_pca.fit(X_train_pca, y_train)
    preds_pca = model_pca.predict(X_test_pca)
    mae_pca = mean_absolute_error(y_test, preds_pca)

    mlflow.log_param("model", "lightgbm_pca10")
    mlflow.log_param("pca_components", 10)
    mlflow.log_metric("mae", mae_pca)
    mlflow.sklearn.log_model(model_pca, "model_pca10")

    print(f"PCA LightGBM MAE: {mae_pca:.2f}")
