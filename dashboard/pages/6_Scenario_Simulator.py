"""Page 6: Scenario Simulator for Urban What-If Policy Analysis."""

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from transitpulse.config import settings
from transitpulse.features.builder import FeatureBuilder
from transitpulse.ingestion.event_connector import EventConnector
from transitpulse.ingestion.transit_connector import TransitConnector

st.title("🧪 What-If Scenario Simulator")
st.markdown(
    "Interactive policy sandbox for evaluating extreme weather downpours, stadium exits, and peak traffic interventions."
)

transit_conn = TransitConnector()
df_routes = transit_conn.get_routes()
df_stops = transit_conn.get_stops()


# Load Model
@st.cache_resource
def load_model_and_meta():
    model = joblib.load(settings.ARTIFACTS_DIR / "best_model.joblib")
    with open(settings.DATA_PROCESSED / "feature_metadata.json", "r") as f:
        meta = pd.read_json(
            settings.DATA_PROCESSED / "feature_metadata.json", typ="series"
        ).to_dict()
    return model, meta["experiment_feature_sets"]["E4"]


try:
    model, feature_names = load_model_and_meta()
except Exception as e:
    st.error(f"Model not loaded: {e}")
    st.stop()

col_ctrl, col_res = st.columns([1, 1])

with col_ctrl:
    st.subheader("1. Scenario Parameters")
    route_id = st.selectbox("Select Target Corridor:", df_routes["route_id"], index=1)
    scenario_title = st.text_input("Scenario Name:", "Monsoon Downpour + IPL Match at Chinnaswamy")

    st.markdown("**Adverse Conditions:**")
    rain_rate = st.slider(
        "Rainfall Intensity (mm/h):", min_value=0.0, max_value=35.0, value=15.0, step=1.0
    )
    is_peak = st.checkbox("Active Evening Peak Commute (18:30 IST)", value=True)
    is_event = st.checkbox("Major Stadium Event (Gates Open / Match Finish)", value=True)
    attendance = (
        st.slider(
            "Expected Venue Attendance:", min_value=5000, max_value=45000, value=35000, step=5000
        )
        if is_event
        else 0
    )

    run_sim = st.button("⚡ Run Scenario Simulation", type="primary", use_container_width=True)

with col_res:
    st.subheader("2. Projected Scenario Impact")

    # Build Baseline vs Scenario
    matching_stops = df_stops[df_stops["route_id"] == route_id].sort_values("stop_sequence")
    mid_stop = matching_stops.iloc[len(matching_stops) // 2]
    eval_time = pd.Timestamp.now(tz=settings.TIMEZONE).replace(hour=18 if is_peak else 14, minute=0)

    base_rec = pd.DataFrame(
        [
            {
                "trip_id": "SIM-BASE",
                "route_id": route_id,
                "route_short_name": mid_stop["route_short_name"],
                "stop_id": mid_stop["stop_id"],
                "stop_sequence": int(mid_stop["stop_sequence"]),
                "stop_name": mid_stop["stop_name"],
                "stop_lat": float(mid_stop["stop_lat"]),
                "stop_lon": float(mid_stop["stop_lon"]),
                "scheduled_time": eval_time,
                "scheduled_travel_time_sec": float(mid_stop["scheduled_travel_time_sec"]),
                "prev_stop_id": "PREV",
                "prev_stop_delay_min": 1.5,
                "observed_delay_min": 0.0,
                "delay_category": "On-time",
                "precip_mm": 0.0,
                "temperature_c": 26.0,
                "relative_humidity": 60.0,
                "wind_gusts_kmh": 8.0,
                "weather_code": 0,
                "dist_nearest_event_km": 15.0,
                "min_to_nearest_event": 999.0,
                "dataset_mode": "simulated",
            }
        ]
    )

    scen_rec = base_rec.copy()
    scen_rec["precip_mm"] = rain_rate
    scen_rec["weather_code"] = 65 if rain_rate >= 10.0 else (61 if rain_rate > 0 else 0)
    scen_rec["dist_nearest_event_km"] = 1.0 if is_event else 15.0
    scen_rec["min_to_nearest_event"] = 20.0 if is_event else 999.0

    event_conn = EventConnector()
    df_events = event_conn.get_events()
    fb = FeatureBuilder()
    X_base = fb.build_features(base_rec, df_stops, df_events)[feature_names]
    X_scen = fb.build_features(scen_rec, df_stops, df_events)[feature_names]

    base_delay = float(model.predict_reg(X_base)[0])
    scen_delay = float(model.predict_reg(X_scen)[0])
    scen_cls = str(model.predict_cls(X_scen)[0])
    net_delay = max(0.0, scen_delay - base_delay)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Normal Baseline", f"{base_delay:.1f} min")
    with m2:
        st.metric(
            "Scenario Delay",
            f"{scen_delay:.1f} min",
            delta=f"+{net_delay:.1f} min",
            delta_color="inverse",
        )
    with m3:
        st.metric("Severity Category", scen_cls)

    st.markdown("---")
    st.markdown("#### Delay Escalation Comparison")
    cmp_df = pd.DataFrame(
        {
            "Condition": ["Normal Baseline", "Shock Scenario"],
            "Expected Delay (min)": [base_delay, scen_delay],
        }
    )
    fig_cmp = px.bar(
        cmp_df,
        x="Condition",
        y="Expected Delay (min)",
        color="Condition",
        color_discrete_map={"Normal Baseline": "#3B82F6", "Shock Scenario": "#EF4444"},
        text="Expected Delay (min)",
        height=280,
    )
    fig_cmp.update_traces(texttemplate="%{text:.1f}m", textposition="outside")
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Attribution breakdown
    rain_fac = round(rain_rate * 1.2 + (0.2 * rain_rate**1.3), 1) if rain_rate > 0 else 0.0
    event_fac = round((attendance / 10000.0) * 3.2, 1) if is_event else 0.0
    peak_fac = 4.5 if is_peak else 0.0

    breakdown_df = pd.DataFrame(
        [
            {"Exogenous Driver": "Monsoon Rainfall Friction", "Marginal Impact (min)": rain_fac},
            {"Exogenous Driver": "Stadium Crowd Shock", "Marginal Impact (min)": event_fac},
            {"Exogenous Driver": "Commuter Peak Traffic", "Marginal Impact (min)": peak_fac},
        ]
    )
    fig_fac = px.pie(
        breakdown_df,
        names="Exogenous Driver",
        values="Marginal Impact (min)",
        title="Estimated Breakdown of Exogenous Delay Additions",
        hole=0.4,
        height=260,
    )
    st.plotly_chart(fig_fac, use_container_width=True)
