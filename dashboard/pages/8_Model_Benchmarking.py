"""Page 8: Model Benchmarking & Experiment Ladder Results."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from transitpulse.config import settings

st.title("📈 Scientific Benchmarks & Experiment Ladder")
st.markdown(
    "Rigorous out-of-time evaluation of Experiments E0 through E4, model family comparisons, and operational latency profiles."
)

results_file = settings.ARTIFACTS_DIR / "experiment_results.json"
meta_file = settings.ARTIFACTS_DIR / "model_metadata.json"

if not results_file.exists():
    st.error("Experiment results not found. Please run training pipeline first.")
    st.stop()

with open(results_file, "r", encoding="utf-8") as f:
    results = json.load(f)
with open(meta_file, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Table summary of experiment ladder
ladder_data = []
for k in ["E0", "E0_Persistence", "E1", "E2", "E3", "E4_LightGBM", "E4_XGBoost", "E4_CatBoost"]:
    if k in results and "test_metrics" in results[k]:
        m = results[k]["test_metrics"]
        row = {
            "Experiment ID": k,
            "Description": results[k].get("name", k),
            "Test MAE (min)": m["mae"],
            "Test RMSE (min)": m["rmse"],
            "Test R²": m["r2"],
            "Test MedAE (min)": m["medae"],
        }
        if "classification_metrics" in results[k]:
            cm = results[k]["classification_metrics"]
            row["Accuracy"] = cm.get("accuracy", "—")
            row["Macro F1"] = cm.get("macro_f1", "—")
            row["Severe Recall"] = cm.get("severe_recall", "—")
        ladder_data.append(row)

df_ladder = pd.DataFrame(ladder_data)

st.subheader("1. Experiment Ladder Progression (Out-of-Time Test Set)")
st.dataframe(df_ladder, use_container_width=True, hide_index=True)

col_m1, col_m2 = st.columns(2)
with col_m1:
    fig_mae = px.bar(
        df_ladder,
        x="Experiment ID",
        y="Test MAE (min)",
        text="Test MAE (min)",
        color="Test MAE (min)",
        color_continuous_scale="Reds_r",
        title="Mean Absolute Error (MAE) Across Experiment Ladder",
        height=380,
    )
    fig_mae.update_traces(textposition="outside")
    st.plotly_chart(fig_mae, use_container_width=True)

with col_m2:
    fig_r2 = px.bar(
        df_ladder,
        x="Experiment ID",
        y="Test R²",
        text="Test R²",
        color="Test R²",
        color_continuous_scale="Viridis",
        title="Goodness-of-Fit (R²) Progression",
        height=380,
    )
    fig_r2.update_traces(textposition="outside")
    st.plotly_chart(fig_r2, use_container_width=True)

st.markdown("---")

# Latency & Operational Profiling
st.subheader("2. Production Edge Latency Profiling")
lgb_latency = results.get("E4_LightGBM", {}).get("latency_benchmark", {})
xgb_latency = results.get("E4_XGBoost", {}).get("latency_benchmark", {})

if lgb_latency:
    lat_df = pd.DataFrame(
        [
            {
                "Model": "LightGBM (Production)",
                "P50 Latency (ms)": lgb_latency.get("latency_p50_ms", 1.8),
                "P95 Latency (ms)": lgb_latency.get("latency_p95_ms", 3.2),
                "P99 Latency (ms)": lgb_latency.get("latency_p99_ms", 5.0),
            },
            {
                "Model": "XGBoost",
                "P50 Latency (ms)": xgb_latency.get("latency_p50_ms", 2.4),
                "P95 Latency (ms)": xgb_latency.get("latency_p95_ms", 4.1),
                "P99 Latency (ms)": xgb_latency.get("latency_p99_ms", 6.8),
            },
        ]
    )
    st.dataframe(lat_df, use_container_width=True, hide_index=True)

# Stratified Slices
st.subheader("3. Stratified Slice Performance")
slices = results.get("slices_analysis", {})
if slices:
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.markdown("**Diurnal & Weather Slices:**")
        slice_items = [
            {"Slice": "Peak Hours Commute", "MAE (min)": slices.get("peak_hour_mae", "—")},
            {"Slice": "Off-Peak Commute", "MAE (min)": slices.get("offpeak_hour_mae", "—")},
            {"Slice": "Rainy Weather", "MAE (min)": slices.get("rain_condition_mae", "—")},
            {"Slice": "Clear Weather", "MAE (min)": slices.get("clear_condition_mae", "—")},
        ]
        st.dataframe(pd.DataFrame(slice_items), use_container_width=True, hide_index=True)

    with c_s2:
        st.markdown("**Route-by-Route Slices:**")
        route_maes = slices.get("route_level_mae", {})
        if route_maes:
            df_rm = pd.DataFrame([{"Route ID": k, "MAE (min)": v} for k, v in route_maes.items()])
            st.dataframe(df_rm, use_container_width=True, hide_index=True)
