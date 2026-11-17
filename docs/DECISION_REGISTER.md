# TransitPulse-AI — Architectural & Scientific Decision Register

This register documents all pivotal technical, scientific, and architectural decisions made throughout the lifecycle of the TransitPulse-AI platform, adhering to the Architecture Decision Record (ADR) format.

---

## ADR-001: Isolated Python 3.12 Runtime with UV Package Management
- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** Transit delay modeling involves high-performance gradient boosting (`xgboost`, `lightgbm`, `catboost`), geospatial computations (`geopandas`, `shapely`), tree-explaining engines (`shap`), API backends (`fastapi`), and analytics dashboards (`streamlit`).
- **Decision:** Use CPython 3.12 managed via `uv`. Python 3.12 offers optimal ARM64 wheel compatibility on macOS and Linux, superior bytecode execution speeds over Python 3.10/3.11, and seamless `fastparquet`/`pyarrow` integration.
- **Consequences:** Near-instantaneous dependency resolution and deterministic virtual environment locking.

---

## ADR-002: Mandatory Data Feasibility Resolution (Observed vs. Calibrated Simulation Modes)
- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** BMTC (Bengaluru Metropolitan Transport Corporation) operates a fleet of >6,000 buses across Bengaluru. While GTFS static route and stop topologies are available via open mobility projects (OpenCity.in, Bengaluru Mobility portals), historical GTFS-Realtime vehicle arrival/departure logs are proprietary, gated behind credentials, or not archived publicly with high-fidelity second-by-second timestamps.
- **Decision:** 
  1. Classify BMTC static route schedules, stop coordinates, trip timings, road network metrics (OSM), and Open-Meteo hourly/minutely weather as **Verified and Accessible**.
  2. Classify historical continuous vehicle position logs from BMTC operations as **Unavailable / Restricted**.
  3. Implement a mathematically calibrated, physics-grounded synthetic delay generator (`TransitDelaySimulator`) that simulates operational vehicle movements across real BMTC routes. The generator incorporates:
     - Real static GTFS stop sequences and scheduled travel times.
     - Actual historical/forecast Open-Meteo weather features (precipitation intensity, thunderstorm codes).
     - Empirical traffic congestion curves (BPR - Bureau of Public Roads friction functions) tied to peak/off-peak hours.
     - Event spatial shockwaves based on real Bengaluru event locations (Chinnaswamy Stadium, Palace Grounds, KTPO, Koramangala Indoor Stadium).
  4. Enforce strict metadata labeling: `mode: "simulated"` vs `mode: "observed"`. Never claim simulated delay distributions represent measured telemetry from BMTC.
- **Consequences:** Eliminates scientific dishonesty, ensures complete reproducible execution without brittle external credentials, and provides an end-to-end working ML platform that can immediately ingest real GTFS-RT when API keys are supplied.

---

## ADR-003: Spatiotemporal Leakage Prevention in Feature Engineering
- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** In transit delay prediction, data leakage is common when engineers use future trip updates, global rolling statistics across time, or future weather forecasts that were not available at prediction time $T$.
- **Decision:**
  - All features must be indexed strictly by cutoff timestamp $T_{\text{pred}}$.
  - Lag features are computed only on historical trips observed strictly prior to $T_{\text{pred}} - \Delta_{\text{latency}}$ (accounting for a 2-minute pipeline ingestion latency).
  - Weather features at future horizons ($T+15\text{m}$, $T+30\text{m}$, $T+60\text{m}$) must use vintage forecast variables rather than retrospective ground truth.
  - Data splits must be strictly chronological (70% train, 15% validation, 15% test) with rolling-origin backtesting. Random train/test split is strictly forbidden.
- **Consequences:** Prevents over-optimistic performance metrics; models evaluate genuinely on out-of-time distribution shifts.

---

## ADR-004: Multi-Horizon & Dual-Objective Prediction Architecture
- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** Transit dispatchers and passengers have distinct decision-making needs. Dispatchers require continuous estimates of delay in minutes for headway adjustment, while commuter apps require discrete categorization (e.g. "Severe Delay > 20 mins").
- **Decision:**
  - Train dual objectives: Continuous Regression ($\hat{y} \in \mathbb{R}$ delay in minutes) and Multi-class Classification ($\hat{c} \in \{\text{On-time}, \text{Minor}, \text{Moderate}, \text{Severe}\}$).
  - Predict across three distinct forward operational horizons: $H \in \{15, 30, 60\}$ minutes.
- **Consequences:** Comprehensive operational utility matching production transit intelligence platforms.

---

## ADR-005: Model Selection and Experiment Ladder (E0 to E4)
- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** Complex deep learning models often overfit tabular transit features or fail to beat tuned gradient boosting trees.
- **Decision:**
  - Ladder:
    - **E0:** Baseline (Historical route/stop mean & persistence).
    - **E1:** Transit + Temporal features (Linear/Ridge & LightGBM).
    - **E2:** Transit + Temporal + Open-Meteo Weather features.
    - **E3:** Transit + Temporal + Weather + Event proximity + Geospatial density.
    - **E4:** Hyperparameter-optimized ensemble (LightGBM/XGBoost/CatBoost) tuned via Optuna.
- **Consequences:** Clear scientific attribution of performance gain to specific feature subsets.
