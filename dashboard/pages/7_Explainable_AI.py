"""Page 7: Explainable AI & Feature Attribution."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from transitpulse.config import settings
from transitpulse.explainability.shap_explainer import TransitSHAPExplainer

st.title("🧠 Explainable AI: TreeSHAP Attribution")
st.markdown("Game-theoretic Shapley feature attributions explaining model behavior across urban transit dimensions.")

st.info(
    "ℹ️ **Interpretability Notice:** SHAP values represent localized, additive game-theoretic feature contributions "
    "to the model's statistical prediction. They do NOT represent counterfactual causal interventions."
)

@st.cache_resource
def load_explainer_data():
    model = joblib.load(settings.ARTIFACTS_DIR / "best_model.joblib")
    with open(settings.DATA_PROCESSED / "feature_metadata.json", "r") as f:
        meta = pd.read_json(settings.DATA_PROCESSED / "feature_metadata.json", typ="series").to_dict()
    features = meta["experiment_feature_sets"]["E4"]
    explainer = TransitSHAPExplainer(model, features)
    
    # Load sample test data for global importance
    test_sample = pd.read_parquet(settings.DATA_PROCESSED / "test.parquet").head(250)
    df_global = explainer.compute_global_importance(test_sample)
    return explainer, df_global, features

try:
    explainer, df_global, features = load_explainer_data()
except Exception as e:
    st.error(f"Failed to load explainability data: {e}")
    st.stop()

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Global Feature Importance (Mean |SHAP|)")
    top15 = df_global.head(15).copy()
    fig_imp = px.bar(
        top15,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Top 15 Most Influential Spatiotemporal Features",
        labels={"mean_abs_shap": "Mean |SHAP Value| (Impact in Minutes)", "feature": "Feature Name"},
        color="mean_abs_shap",
        color_continuous_scale="Purples",
        height=520,
    )
    fig_imp.update_layout(yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_imp, use_container_width=True)

with col2:
    st.subheader("Feature Modality Contribution")
    
    # Group features by domain
    def map_domain(f):
        if "precip" in f or "rain" in f or "weather" in f or "temp" in f or "wind" in f:
            return "Meteorological"
        elif "event" in f:
            return "Urban Events"
        elif "peak" in f or "hour" in f or "day" in f or "dow" in f:
            return "Temporal & Commute"
        elif "stop" in f or "route" in f or "prev" in f or "travel" in f:
            return "Transit & Topology"
        return "Geospatial"

    df_global["Domain"] = df_global["feature"].apply(map_domain)
    domain_imp = df_global.groupby("Domain")["mean_abs_shap"].sum().reset_index()

    fig_pie = px.pie(
        domain_imp,
        names="Domain",
        values="mean_abs_shap",
        title="Aggregated Explanatory Power by Modality",
        color_discrete_sequence=px.colors.qualitative.Prism,
        hole=0.35,
        height=400,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
st.subheader("Detailed Feature Attribution Table")
st.dataframe(df_global, use_container_width=True, hide_index=True)
