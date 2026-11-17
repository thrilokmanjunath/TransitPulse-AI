"""Page 2: Route Performance."""

import streamlit as st
import pandas as pd
import plotly.express as px

from transitpulse.config import settings
from transitpulse.ingestion.transit_connector import TransitConnector

st.title("📊 Route Corridor Performance Analysis")
st.markdown("Detailed stop-by-stop delay progression, headway variance, and bottleneck detection.")

transit_conn = TransitConnector()
df_routes = transit_conn.get_routes()
df_stops = transit_conn.get_stops()

route_choices = list(df_routes["route_id"])
selected_route = st.selectbox("Select Route Corridor for Deep-Dive:", route_choices, index=1)

route_info = df_routes[df_routes["route_id"] == selected_route].iloc[0]
route_stops = df_stops[df_stops["route_id"] == selected_route].sort_values("stop_sequence")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Origin Terminal", route_info["origin_name"])
with col2:
    st.metric("Destination Terminal", route_info["destination_name"])
with col3:
    st.metric("Total Corridor Distance", f"{route_info['distance_km']} km")
with col4:
    st.metric("Scheduled Run Time", f"{route_info['scheduled_duration_min']} min")

st.markdown("---")

# Load simulated trip records for empirical distribution
sim_file = settings.DATA_SIMULATED / "bmtc_simulated_trips.parquet"
if sim_file.exists():
    df_sim = pd.read_parquet(sim_file)
    df_route_sim = df_sim[df_sim["route_id"] == selected_route]

    st.subheader(f"Stop-by-Stop Delay Propagation on Route {selected_route}")
    fig_box = px.box(
        df_route_sim,
        x="stop_name",
        y="observed_delay_min",
        color="delay_category",
        title=f"Empirical Delay Distribution Across Stop Sequence ({len(df_route_sim):,} Observations)",
        labels={"stop_name": "Bus Stop Name", "observed_delay_min": "Observed Delay (min)"},
        category_orders={"delay_category": ["On-time", "Minor", "Moderate", "Severe"]},
        color_discrete_map={
            "On-time": "#10B981",
            "Minor": "#FBBF24",
            "Moderate": "#F97316",
            "Severe": "#EF4444",
        },
        height=450,
    )
    fig_box.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_box, use_container_width=True)

    # Hourly congestion heatmap
    st.subheader("Diurnal Delay Profile by Hour of Day")
    df_route_sim["hour"] = pd.to_datetime(df_route_sim["scheduled_time"]).dt.hour
    hourly_avg = df_route_sim.groupby(["hour", "stop_name"])["observed_delay_min"].mean().reset_index()
    fig_heat = px.density_heatmap(
        hourly_avg,
        x="hour",
        y="stop_name",
        z="observed_delay_min",
        title="Spatiotemporal Heatmap: Hour of Day vs Bus Stop Delay",
        labels={"hour": "Hour of Day (IST)", "stop_name": "Bus Stop", "observed_delay_min": "Mean Delay (min)"},
        color_continuous_scale="Reds",
        height=480,
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.warning("Simulated trip data not found. Please run the ingestion pipeline first.")
