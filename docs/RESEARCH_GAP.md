# TransitPulse-AI — Research Gap & Scientific Contribution

## 1. Identified Gaps in Urban Transit Delay Forecasting

Despite substantial prior literature in intelligent transportation systems, several major scientific and engineering deficits persist:

### Gap 1: Isolated Univariate or Schedule-Only Predictors
The majority of transit operational prediction systems rely predominantly on single-modality signals: historical scheduled run times and recent trip headway deviations. These models fail catastrophically during exogenous demand shocks (monsoonal downpours, waterlogging, large sporting events, concert exits, or infrastructure bottlenecks). 

### Gap 2: Confounding of Weather Forecasts with Ground-Truth Reanalysis
In academic papers evaluating weather impacts on transit, researchers frequently train models using retrospective ground-truth weather observations (such as ERA5 reanalysis). At actual operational inference time ($T_{\text{pred}}$), ground-truth future precipitation is unavailable; the system must rely on meteorological forecast vintages. Using retrospective weather for future horizons introduces severe optimism bias and leakage.

### Gap 3: Coarse Spatial Event Modeling
Existing event-aware transit studies typically apply a global binary flag (`is_event_day = True/False`). However, urban events (such as cricket matches at Chinnaswamy Stadium or exhibitions at Bangalore International Exhibition Centre) create hyper-localized spatial gravity fields that decay non-linearly with distance and peak at specific temporal countdown offsets.

### Gap 4: The "Black Box" Trust Deficit in Transit Dispatch
High-capacity neural networks (e.g. ST-GNNs, LSTMs) are frequently proposed for transit delay prediction, but their internal attributions are opaque. Transit controllers and operations dispatchers cannot determine whether an anticipated 25-minute delay is caused by severe localized rain, peak-hour bottlenecks, route headway congestion, or a stadium event. Without game-theoretic attribution (e.g., TreeSHAP), operators distrust model outputs and fall back to manual intuition.

### Gap 5: Pervasive Temporal Leakage in Benchmark Literature
Numerous published transit studies inadvertently utilize random train/test splits ($k$-fold CV) across historical time series, allowing future trips from the same route to bleed into training folds. Furthermore, features such as `downstream_stop_delay` or `trip_total_duration` are frequently included in feature sets predicting intermediate stop arrivals.

---

## 2. Scientific Contributions of TransitPulse-AI

TransitPulse-AI directly addresses each of these gaps through four primary contributions:

1. **Unified Spatiotemporal Multi-Modal Architecture:**
   Integrates static GTFS topology, dynamic operational telemetry, multi-ring event proximity calculations (500m, 1km, 3km), and multi-horizon Open-Meteo meteorological vectors into a single feature-engineered representation.

2. **Strict Chronological Evaluation with Zero-Leakage Guarantee:**
   Enforces rolling-origin backtesting and chronological block splitting (70% Train, 15% Validation, 15% Test) with strict availability cutoff $T_{\text{pred}} - \Delta_{\text{latency}}$ ($2$ minutes), ensuring all features reflect only data available prior to the prediction horizon.

3. **Multi-Horizon & Multi-Objective Dual Formulation:**
   Jointly predicts continuous delay in minutes ($\hat{y} \in \mathbb{R}$) and categorical delay severity ($\hat{c} \in \{\text{On-Time}, \text{Minor}, \text{Moderate}, \text{Severe}\}$) at 15-minute, 30-minute, and 60-minute forward horizons to satisfy both passenger information displays and bus dispatch controllers.

4. **Interpretable Operational Attribution via TreeSHAP:**
   Integrates localized TreeSHAP additive decompositions to decompose individual delay predictions into transparent, accountable marginal components: Baseline Route Delay + Weather Penalty + Event Proximity Penalty + Peak Congestion Factor.

5. **Principled Data Feasibility & Dual-Mode Governance:**
   Distinguishes between measured operational telemetry (`mode: observed`) and mathematically calibrated simulation (`mode: simulated`), guaranteeing complete reproducibility and absolute scientific honesty regarding BMTC data accessibility.
