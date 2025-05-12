import streamlit as st
import pandas as pd
import hopsworks
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Citi Bike Predictions",
    page_icon="🚲",
    layout="wide"
)

# --- Connect to Hopsworks ---
try:
    with st.spinner('Connecting to Hopsworks...'):
        # In production, use st.secrets["HOPSWORKS_API_KEY"] instead of hardcoding
        project = hopsworks.login(
            host="c.app.hopsworks.ai",
            project="citybike",
            api_key_value="t4rwmi4VfZBIaFqR.ulzGrQ0eIDKCUKgaPYEpNH4JHWuXbIu3YU8gogK8ldP5tpUpnMVIG4uQX1BFW9Wb"
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group("citibike_predictions", version=1)
except Exception as e:
    st.error(f"Error connecting to Hopsworks: {e}")
    st.stop()

# --- Load Data ---
with st.spinner('Fetching prediction data...'):
    df = fg.read()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["start_station_name", "date", "hour"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["hour"].astype(str) + ":00")

# --- Sidebar Controls ---
st.sidebar.title("🚲 Citi Bike Prediction Dashboard")

station = st.sidebar.selectbox("📍 Choose a station", sorted(df["start_station_name"].unique()))

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
st.markdown("### Hourly Predictions")
if chart_type == "Line":
    chart = alt.Chart(filtered_df).mark_line(point=True).encode(
        x=alt.X("datetime:T", title="Hour of Day"),
        y=alt.Y("predicted_ride_count:Q", title="Predicted Ride Count"),
        tooltip=["datetime:T", "predicted_ride_count"]
    ).properties(height=300)
else:
    chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X("datetime:T", title="Hour of Day"),
        y=alt.Y("predicted_ride_count:Q", title="Predicted Ride Count"),
        tooltip=["datetime:T", "predicted_ride_count"]
    ).properties(height=300)

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
    height=300
)

st.altair_chart(comparison_chart, use_container_width=True)

# --- Trend Analysis Section ---
st.markdown("### 📈 Trend Analysis")

# Get data for the selected station across all available dates
trend_data = df[df["start_station_name"] == station].groupby("date")["predicted_ride_count"].sum().reset_index()
trend_data["day"] = trend_data["date"].dt.day_name()

trend_chart = alt.Chart(trend_data).mark_line(point=True).encode(
    x=alt.X("date:T", title="Date"),
    y=alt.Y("predicted_ride_count:Q", title="Total Daily Predicted Rides"),
    tooltip=["date:T", "predicted_ride_count", "day"]
).properties(height=300)

st.altair_chart(trend_chart, use_container_width=True)

# --- Weekly Pattern Heatmap ---
st.markdown("### 🔥 Weekly Pattern Heatmap")

if len(df["date"].dt.date.unique()) >= 7:  # Only show if we have at least a week of data
    # Create a pivot table for the heatmap
    heatmap_data = df[df["start_station_name"] == station].copy()
    heatmap_data["day_of_week"] = heatmap_data["date"].dt.day_name()
    heatmap_pivot = heatmap_data.pivot_table(
        index="day_of_week", 
        columns="hour", 
        values="predicted_ride_count",
        aggfunc="mean"
    ).reset_index()
    
    # Convert to long format for Altair
    heatmap_long = pd.melt(
        heatmap_pivot, 
        id_vars=["day_of_week"], 
        var_name="hour", 
        value_name="avg_predicted_rides"
    )
    
    # Create heatmap
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap = alt.Chart(heatmap_long).mark_rect().encode(
        x=alt.X("hour:O", title="Hour of Day"),
        y=alt.Y("day_of_week:N", title="Day of Week", sort=days_order),
        color=alt.Color("avg_predicted_rides:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["day_of_week", "hour", "avg_predicted_rides"]
    ).properties(
        height=300
    )
    
    st.altair_chart(heatmap, use_container_width=True)
else:
    st.info("Not enough data for weekly pattern analysis")

# --- Station Map ---
st.markdown("### 🗺️ Station Location")

# Check if location data is available
if "latitude" in df.columns and "longitude" in df.columns:
    station_info = df[df["start_station_name"] == station][["latitude", "longitude"]].iloc[0]
    map_data = pd.DataFrame({
        "lat": [station_info["latitude"]],
        "lon": [station_info["longitude"]],
        "name": [station]
    })
    st.map(map_data)
else:
    st.info("Station location data not available. Consider adding latitude and longitude to your dataset.")

# --- Download Section ---
st.markdown("### 📥 Download Prediction Data")
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"citibike_predictions_{station}_{selected_date}.csv",
    mime="text/csv"
)

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
    avg_rides = round(filtered_df["predicted_ride_count"].mean(), 1)
    st.sidebar.markdown(f"**Total Rides:** {total}")
    st.sidebar.markdown(f"**Peak Hour:** {peak_hour}:00")
    st.sidebar.markdown(f"**Average Hourly Rides:** {avg_rides}")
else:
    st.sidebar.warning("No data for this selection.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Hopsworks, and LightGBM")
