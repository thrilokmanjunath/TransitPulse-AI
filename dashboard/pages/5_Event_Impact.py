"""Page 5: Urban Event Impact Analysis."""

import pandas as pd
import plotly.express as px
import streamlit as st

from transitpulse.config import settings
from transitpulse.ingestion.event_connector import EventConnector
from transitpulse.ingestion.transit_connector import TransitConnector

st.title("🏟️ Urban Event Shockwaves & Spatial Proximity")
st.markdown(
    "Analysis of mega-event crowd gravity, venue distance decay, and transit corridor disruption."
)

event_conn = EventConnector()
df_events = event_conn.get_events()
transit_conn = TransitConnector()
df_stops = transit_conn.get_stops()

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Event Venues & Transit Network Spatial Overlay")
    # Plot both bus stops and venues
    venues_df = df_events.copy()
    venues_df["Type"] = "Major Event Venue"
    venues_df["Name"] = venues_df["venue_name"]
    venues_df["lat"] = venues_df["venue_lat"]
    venues_df["lon"] = venues_df["venue_lon"]
    venues_df["Size"] = venues_df["expected_attendance"]

    fig_venues = px.scatter_mapbox(
        venues_df,
        lat="lat",
        lon="lon",
        size="Size",
        color="event_category",
        hover_name="Name",
        hover_data=["event_name", "expected_attendance"],
        zoom=10.5,
        center={"lat": settings.CENTROID_LAT, "lon": settings.CENTROID_LON},
        mapbox_style="carto-positron",
        title="Major Bengaluru Venues (Circle size = Capacity)",
        height=450,
    )
    st.plotly_chart(fig_venues, use_container_width=True)

with c2:
    st.subheader("Venue Capacity & Category Directory")
    st.dataframe(
        df_events[
            ["event_name", "venue_name", "event_category", "expected_attendance", "start_time"]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")

# Proximity Distance Decay Analysis
sim_file = settings.DATA_SIMULATED / "bmtc_simulated_trips.parquet"
if sim_file.exists():
    df_sim = pd.read_parquet(sim_file)
    st.subheader("Spatial Distance Decay: Delay vs Proximity to Active Venue")

    # Filter within 10km of events
    df_near = df_sim[df_sim["dist_nearest_event_km"] <= 8.0].copy()
    df_near["Distance Ring"] = pd.cut(
        df_near["dist_nearest_event_km"],
        bins=[0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0],
        labels=["<500m", "500m-1km", "1-2km", "2-3.5km", "3.5-5km", "5-8km"],
    )
    decay_summary = (
        df_near.groupby("Distance Ring", observed=False)["observed_delay_min"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_decay = px.line(
            decay_summary,
            x="Distance Ring",
            y="mean",
            markers=True,
            title="Mean Transit Delay vs Distance from Venue (Spatial Decay)",
            labels={"Distance Ring": "Buffer Distance Ring", "mean": "Mean Delay (min)"},
            height=380,
        )
        st.plotly_chart(fig_decay, use_container_width=True)

    with col_d2:
        fig_event_box = px.box(
            df_near,
            x="Distance Ring",
            y="observed_delay_min",
            color="Distance Ring",
            title="Delay Variance by Event Proximity Band",
            height=380,
        )
        st.plotly_chart(fig_event_box, use_container_width=True)
