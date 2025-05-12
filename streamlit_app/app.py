import streamlit as st
import pandas as pd
import hopsworks
import altair as alt

# --- Connect to Hopsworks ---
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citybike",
    api_key_value="t4rwmi4VfZBIaFqR.ulzGrQ0eIDKCUKgaPYEpNH4JHWuXbIu3YU8gogK8ldP5tpUpnMVIG4uQX1BFW9Wb"
)
fs = project.get_feature_store()
fg = fs.get_feature_group("citibike_predictions", version=1)

# --- Load Data ---
df = fg.read()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["start_station_name", "date", "hour"])
df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["hour"].astype(str) + ":00")

# --- Sidebar Controls ---
st.sidebar.title("🚲 Citi Bike Prediction Dashboard")

station = st.sidebar.selectbox("📍 Choose a station", df["start_station_name"].unique())

available_dates = sorted(df["date"].dt.date.unique())
selected_date = st.sidebar.selectbox("📅 Select date", available_dates)

chart_type = st.sidebar.radio("📈 Chart type", ["Line", "Bar"])

compare_hour = st.sidebar.selectbox("📊 Compare Predictions at Hour", sorted(df["hour"].unique()))

if st.sidebar.button("🔄 Refresh Data"):
    st.experimental_rerun()

# --- Filter Data for Main Plot ---
filtered_df = df[
    (df["start_station_name"] == station) &
    (df["date"].dt.date == selected_date)
].copy()

# --- Main Title ---
st.title(f"📊 Predicted Rides – {station} ({selected_date})")

# --- Primary Chart (Per Day) ---
if chart_type == "Line":
    chart = alt.Chart(filtered_df).mark_line(point=True).encode(
        x=alt.X("datetime:T", title="Hour of Day"),
        y=alt.Y("predicted_ride_count:Q", title="Predicted Ride Count"),
        tooltip=["datetime:T", "predicted_ride_count"]
    )
else:
    chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X("datetime:T", title="Hour of Day"),
        y=alt.Y("predicted_ride_count:Q", title="Predicted Ride Count"),
        tooltip=["datetime:T", "predicted_ride_count"]
    )

st.altair_chart(chart, use_container_width=True)

# --- Comparison Chart Across Days at Selected Hour ---
st.markdown(f"### 📅 Comparison of Ride Predictions at Hour {compare_hour}:00")

hour_comparison = df[
    (df["start_station_name"] == station) &
    (df["hour"] == compare_hour)
][["date", "predicted_ride_count"]].copy()

comparison_chart = alt.Chart(hour_comparison).mark_bar().encode(
    x=alt.X("date:T", title="Date"),
    y=alt.Y("predicted_ride_count:Q", title="Predicted Rides"),
    tooltip=["date:T", "predicted_ride_count"]
).properties(
    width=700,
    height=300
)

st.altair_chart(comparison_chart, use_container_width=True)

# --- Sidebar Model Info ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Model Info")
st.sidebar.markdown("**Model Used:** Tuned LightGBM")
st.sidebar.markdown("**Best MAE:** 2.55")
st.sidebar.markdown("**Data Source:** Full 2023 Citi Bike")
st.sidebar.markdown("**Prediction Range:** 7 Days (hourly)")
st.sidebar.markdown("**Backend:** Hopsworks Feature Store")

# --- Sidebar Stats ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Prediction Stats")
if not filtered_df.empty:
    total = int(filtered_df["predicted_ride_count"].sum())
    peak_hour = int(filtered_df.loc[filtered_df["predicted_ride_count"].idxmax(), "hour"])
    st.sidebar.markdown(f"**Total Rides:** {total}")
    st.sidebar.markdown(f"**Peak Hour:** {peak_hour}:00")
else:
    st.sidebar.warning("No data for this selection.")
