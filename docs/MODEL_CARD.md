# TransitPulse-AI — Machine Learning Model Card

Adheres to the standardized reporting framework proposed by Mitchell et al. (2019) (*Model Cards for Model Reporting*).

---

## 1. Model Details
- **Model Name:** TransitPulse-AI Multimodal Delay Forecaster
- **Version:** 1.1.0-audited (Independent Scientific Audit Release)
- **Model Architecture:** Gradient Boosted Decision Tree Ensemble (`LightGBM` + `XGBoost` auxiliary)
- **Primary Formulation:** Dual-Objective Spatiotemporal Forecasting:
  1. Continuous delay regression ($\hat{y} \in \mathbb{R}$, in minutes).
  2. Multi-class delay severity categorization ($\hat{c} \in \{\text{On-time}, \text{Minor}, \text{Moderate}, \text{Severe}\}$).
- **Forward Planning Horizons:** 15 minutes, 30 minutes, 60 minutes (spatiotemporally tracked forward stops).
- **Explainability Engine:** TreeSHAP (Game-Theoretic Shapley Additive Explanations).
- **Developers:** Principal Data Scientist & Engineering Team (`thrilokmanjunath`).
- **Independent Audit:** Verified by adversarial validation audit (Phase 2).
- **License:** MIT License.

---

## 2. Intended Use & Target Tasks
### Primary Intended Uses:
- Providing real-time delay predictions to transit commuters and passenger information displays.
- Assisting transit agency dispatchers with headway management and proactive rescheduling during adverse meteorological conditions or stadium events.
- Quantifying the localized impact of monsoon rainfall and high-capacity sporting/entertainment crowds on arterial urban bus corridors.
- Serving as an open research and methodological benchmark for spatiotemporal transit modeling under missing telemetry and environmental shocks.

### Out-of-Scope & Inappropriate Uses:
- Do not use for autonomous vehicle collision avoidance, braking systems, or life-critical safety routing.
- Do not cite simulated delay distributions as measured historical efficiency of the BMTC fleet without explicit qualification.
- Do not interpret SHAP statistical feature attributions as counterfactual causal interventions.
- Do not claim real-world production SLAs on hardware fleets until empirical validation milestones (RW-1 through RW-5) are satisfied.

---

## 3. Training Data & Preprocessing
- **Geographic Coverage:** Greater Bengaluru Metropolitan Region, Karnataka, India (Centroid: 12.9716° N, 77.5946° E).
- **Transit Network:** 5 high-frequency representative BMTC corridors (335-E, 500-D ORR, KIA-9 Airport Express, G-2, 201-R) spanning 50 bus stops.
- **Weather Observations:** Hourly Open-Meteo ERA5 reanalysis and ECMWF forecasts (temperature, precipitation rate, rolling 3h accumulation, relative humidity, wind gusts, WMO codes).
- **Urban Events:** Curated public events across major Bengaluru venues (M. Chinnaswamy Stadium, Palace Grounds, BIEC, Kanteerava, KTPO, Koramangala).
- **Operational Simulation:** Redesigned stochastic transit simulator with segment-heterogeneous autoregressive decay $\rho_s \in [0.55, 0.90]$, lognormal boarding variance, exponential signal queuing, GPS multipath jitter, and 10% telemetry dropout probability.
- **Temporal Alignment:** Strict chronological ordering; 70% Train ($N=16,660$), 15% Validation ($N=3,570$), 15% Test ($N=3,570$).
- **Leakage Safeguards:** Features enforce a strict availability cutoff. Upstream delay is absent on initial stops and during missing vehicle telemetry pings.

---

## 4. Evaluation Methodology & Benchmark Results
Evaluated on out-of-time test cohort (held out chronologically from May 12 to May 14, 2024):

| Experiment | Feature Set | Test MAE (min) | Test RMSE (min) | Test $R^2$ | Macro F1 | Severe Delay Recall |
|---|---|---|---|---|---|---|
| **$E_0$ (Historical)** | Historical Route Mean Baseline | 6.077 | 8.550 | 0.0575 | — | — |
| **$E_{0,\text{pers}}$ (Persistence)** | Upstream Delay Persistence Baseline | 2.329 | 3.082 | 0.8776 | — | — |
| **$E_1$** | Transit + Temporal (LightGBM) | 1.549 | 2.410 | 0.9251 | 0.828 | 0.825 |
| **$E_2$** | Transit + Temporal + Weather | 1.566 | 2.551 | 0.9161 | 0.828 | 0.841 |
| **$E_3$** | Full Multimodal (Weather + Events + Spatial) | 1.455 | 2.153 | 0.9402 | 0.830 | 0.841 |
| **$E_4$ (LightGBM)** | **Optuna-Tuned LightGBM (Production Default)** | **1.463** | **2.220** | **0.9365** | **0.832** | **0.836** |
| **$E_4$ (XGBoost)** | Tuned XGBoost | **1.382** | **1.944** | **0.9513** | — | — |
| **$E_4$ (CatBoost)** | Tuned CatBoost | 1.621 | 3.225 | 0.8659 | — | — |

*Exact runtime metrics recorded dynamically in `models/artifacts/model_metadata.json`.*

### Latency Profile:
- **P50 Latency:** 0.30 ms
- **P95 Latency:** 0.37 ms
- **P99 Latency:** 0.56 ms
- Microsecond edge inference profile suitable for high-concurrency dispatch and consumer apps.

---

## 5. Quantitative Bias & Limitations
- **Spatial Coverage:** Currently trained on high-density arterial corridors; peripheral feeder routes with erratic headways may experience higher prediction variance.
- **Weather Sensitivity:** Test MAE during dry periods is **1.236m**, expanding to **1.806m** during precipitation events due to increased physical stochasticity.
- **Missing Telemetry:** During GPS dropouts ($N=287$ test records), prediction MAE is **1.451m**, effectively stabilized by route prior imputation.
- **Simulation Disclaimer:** All operational delay labels reflect calibrated stochastic simulation (`dataset_mode: simulated`). Refer to `reports/REAL_WORLD_VALIDATION_PLAN.md` for the transition roadmap to real-world AVL feeds.
