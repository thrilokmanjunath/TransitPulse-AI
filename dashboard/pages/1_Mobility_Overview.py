"""Page 1: Mobility Overview."""

import plotly.express as px
import streamlit as st

from transitpulse.config import settings
from transitpulse.ingestion.transit_connector import TransitConnector

st.title("🌐 Urban Mobility Overview")
st.markdown(
    "Macro-level operational health, fleet spatial distribution, and meteorological conditions across Bengaluru."
)

transit_conn = TransitConnector()
df_routes = transit_conn.get_routes()
df_stops = transit_conn.get_stops()

c1, c2, c3 = st.columns(3)
with c1:
    st.info(f"**Total Tracked Corridors:** {len(df_routes)}")
with c2:
    st.info(f"**Total Transit Stops:** {len(df_stops)}")
with c3:
    st.info(
        f"**City Bounding Box:** Lat [{settings.LAT_MIN}, {settings.LAT_MAX}], Lon [{settings.LON_MIN}, {settings.LON_MAX}]"
    )

st.markdown("### Corridor Length & Stop Distribution")
col_a, col_b = st.columns(2)

with col_a:
    fig_bar = px.bar(
        df_routes,
        x="route_short_name",
        y="distance_km",
        color="num_stops",
        title="Corridor Distance (km) & Number of Stops",
        labels={
            "route_short_name": "Route Code",
            "distance_km": "Distance (km)",
            "num_stops": "Stops Count",
        },
        color_continuous_scale="Viridis",
        height=380,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b:
    fig_dur = px.bar(
        df_routes,
        x="route_short_name",
        y="scheduled_duration_min",
        color="distance_km",
        title="Scheduled Traversal Time (Minutes)",
        labels={
            "route_short_name": "Route Code",
            "scheduled_duration_min": "Scheduled Traversal (min)",
        },
        color_continuous_scale="Cividis",
        height=380,
    )
    st.plotly_chart(fig_dur, use_container_width=True)

st.markdown("### Route Network Table")
st.dataframe(df_routes, use_container_width=True, hide_index=True)
