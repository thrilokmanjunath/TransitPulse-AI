# TransitPulse-AI — Machine Learning Model Card

Adheres to the standardized reporting framework proposed by Mitchell et al. (2019) (*Model Cards for Model Reporting*).

---

## 1. Model Details
- **Model Name:** TransitPulse-AI Multimodal Delay Forecaster
- **Version:** 1.0.0 (Production Release)
- **Model Architecture:** Gradient Boosted Decision Tree Ensemble (`LightGBM` + `XGBoost` auxiliary)
- **Primary Formulation:** Dual-Objective Spatiotemporal Forecasting:
  1. Continuous delay regression ($\hat{y} \in \mathbb{R}$, in minutes).
  2. Multi-class delay severity categorization ($\hat{c} \in \{\text{On-time}, \text{Minor}, \text{Moderate}, \text{Severe}\}$).
- **Forward Planning Horizons:** 15 minutes, 30 minutes, 60 minutes.
- **Explainability Engine:** TreeSHAP (Game-Theoretic Shapley Additive Explanations).
- **Developers:** Principal Data Scientist & Engineering Team (`thrilokmanjunath`).
- **License:** MIT License.

---

## 2. Intended Use & Target Tasks
### Primary Intended Uses:
- Providing real-time delay predictions to transit commuters and passenger information displays.
- Assisting transit agency dispatchers with headway management and proactive rescheduling during adverse meteorological conditions or stadium events.
- Quantifying the localized impact of monsoon rainfall and high-capacity sporting/entertainment crowds on arterial urban bus corridors.

### Out-of-Scope & Inappropriate Uses:
- Do not use for autonomous vehicle collision avoidance, braking systems, or life-critical safety routing.
- Do not cite simulated delay distributions as measured historical efficiency of the BMTC fleet without explicit qualification.
- Do not interpret SHAP statistical feature attributions as counterfactual causal interventions.

---

## 3. Training Data & Preprocessing
- **Geographic Coverage:** Greater Bengaluru Metropolitan Region, Karnataka, India (Centroid: 12.9716° N, 77.5946° E).
- **Transit Network:** 5 high-frequency representative BMTC corridors (335-E, 500-D ORR, KIA-9 Airport Express, G-2, 201-R) spanning 50 bus stops.
- **Weather Observations:** Hourly Open-Meteo ERA5 reanalysis and ECMWF forecasts (temperature, precipitation rate, rolling 3h accumulation, relative humidity, wind gusts, WMO codes).
- **Urban Events:** Curated public events across major Bengaluru venues (M. Chinnaswamy Stadium, Palace Grounds, BIEC, Kanteerava, KTPO, Koramangala).
- **Temporal Alignment:** Strict chronological ordering; 70% Train, 15% Validation, 15% Test.
- **Leakage Safeguards:** Features enforce a strict $T_{\text{pred}} - 2\text{ minute}$ availability cutoff.

---

## 4. Evaluation Methodology & Benchmark Results
Evaluated on out-of-time test cohort (held out chronologically):

| Experiment | Feature Set | Test MAE (min) | Test RMSE (min) | Test $R^2$ | Macro F1 | Severe Delay Recall |
|---|---|---|---|---|---|---|
| **E0** | Historical Baseline | ~5.82 | ~7.45 | 0.28 | — | — |
| **E1** | Transit + Temporal | ~3.95 | ~5.12 | 0.65 | 0.72 | 0.70 |
| **E2** | Transit + Temporal + Weather | ~2.64 | ~3.45 | 0.81 | 0.84 | 0.82 |
| **E3** | Full Multimodal (Weather + Events + Spatial) | ~1.85 | ~2.52 | 0.89 | 0.91 | 0.89 |
| **E4** | Optuna-Tuned LightGBM (Production) | **~1.45** | **~2.05** | **0.93** | **0.94** | **0.93** |

*Exact runtime metrics recorded dynamically in `models/artifacts/model_metadata.json`.*

### Latency Profile:
- **P50 Latency:** 1.8 ms
- **P95 Latency:** 3.4 ms
- **P99 Latency:** 5.2 ms
- Suitable for high-throughput REST APIs and real-time edge dispatch dashboards.

---

## 5. Quantitative Bias & Limitations
- **Spatial Coverage:** Currently trained on high-density arterial corridors; peripheral feeder routes with erratic headways may experience higher prediction variance.
- **Extreme Weather Shocks:** Rainfall rates exceeding 30 mm/h represent rare tail events in the training data, widening confidence intervals.
- **Simulation Disclaimer:** When running in `dataset_mode: simulated`, operational delay patterns reflect calibrated stochastic physics rather than raw hardware GPS telemetry from BMTC buses.
