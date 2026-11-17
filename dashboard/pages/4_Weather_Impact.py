"""Page 4: Weather Impact Analysis."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from transitpulse.config import settings

st.title("🌧️ Weather & Rainfall Impact on Urban Transit")
st.markdown(
    "Empirical quantification of meteorological road friction, rainfall intensity thresholds, and monsoon downpour delays."
)

sim_file = settings.DATA_SIMULATED / "bmtc_simulated_trips.parquet"
if not sim_file.exists():
    st.error("Simulated data not found.")
    st.stop()

df = pd.read_parquet(sim_file)

c1, c2, c3 = st.columns(3)
rain_mask = df["precip_mm"] > 0.05
with c1:
    st.metric("Rain-Affected Trips", f"{rain_mask.sum():,} ({rain_mask.mean() * 100:.1f}%)")
with c2:
    mean_rain_delay = df[rain_mask]["observed_delay_min"].mean()
    mean_clear_delay = df[~rain_mask]["observed_delay_min"].mean()
    st.metric(
        "Mean Delay (Raining)",
        f"{mean_rain_delay:.1f} min",
        delta=f"+{mean_rain_delay - mean_clear_delay:.1f} min vs Clear",
        delta_color="inverse",
    )
with c3:
    st.metric(
        "Severe Delay Probability Under Rain",
        f"{(df[rain_mask]['delay_category'] == 'Severe').mean() * 100:.1f}%",
    )

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Precipitation Rate vs Transit Delay")
    sample_df = df.sample(min(2000, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample_df,
        x="precip_mm",
        y="observed_delay_min",
        color="route_id",
        trendline="lowess",
        title="Non-Linear Precipitation Friction Curve (LOWESS Smoothing)",
        labels={
            "precip_mm": "Rainfall Intensity (mm/h)",
            "observed_delay_min": "Observed Delay (min)",
            "route_id": "Route",
        },
        height=420,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("Delay Distribution: Clear vs Rainy Conditions")
    df_box = df.copy()
    df_box["Weather Condition"] = np.where(
        df_box["precip_mm"] > 5.0,
        "Heavy Rain (>5mm)",
        np.where(df_box["precip_mm"] > 0.1, "Light Rain (0-5mm)", "Clear"),
    )
    fig_w_box = px.box(
        df_box,
        x="Weather Condition",
        y="observed_delay_min",
        color="Weather Condition",
        title="Delay Escalation by Meteorological Severity",
        labels={"observed_delay_min": "Delay (min)"},
        category_orders={"Weather Condition": ["Clear", "Light Rain (0-5mm)", "Heavy Rain (>5mm)"]},
        height=420,
    )
    st.plotly_chart(fig_w_box, use_container_width=True)

st.markdown("### Corridor Sensitivity to Precipitation")
route_rain_sens = df.groupby(["route_id", rain_mask])["observed_delay_min"].mean().unstack()
route_rain_sens.columns = ["Clear Delay (min)", "Rainy Delay (min)"]
route_rain_sens["Net Weather Penalty (min)"] = (
    route_rain_sens["Rainy Delay (min)"] - route_rain_sens["Clear Delay (min)"]
)
st.dataframe(route_rain_sens.round(2), use_container_width=True)
