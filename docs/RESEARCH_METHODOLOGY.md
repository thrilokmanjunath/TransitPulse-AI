# TransitPulse-AI — Research Methodology & Experimental Framework

## 1. Research Questions & Hypotheses

### Primary Research Questions (RQ)
- **RQ1 (Multimodal Information Value):** Does the joint inclusion of high-resolution meteorological variables (precipitation intensity, wind gusts) and spatial event proximity buffers significantly improve multi-horizon delay prediction over traditional univariate transit schedule lag baselines?
- **RQ2 (Model Hierarchy):** How do tree-based gradient boosting ensembles (LightGBM, XGBoost, CatBoost) compare with linear baselines and sequential deep learning models across prediction horizons of 15, 30, and 60 minutes in terms of both predictive error (MAE, RMSE) and production inference latency?
- **RQ3 (Interpretability & Attribution):** Can game-theoretic feature attribution (TreeSHAP) distinguish between meteorological delay friction, event-driven shockwaves, and endemic route congestion without violating causal non-identifiability principles?
- **RQ4 (Horizon Degradation):** At what rate does predictive fidelity degrade as the operational forecasting horizon expands from $H=15\text{m}$ to $H=60\text{m}$?

### Hypotheses (H)
- **H1:** Models equipped with non-linear precipitation interaction terms ($\text{rain} \times \text{peak\_hour}$) will achieve at least a 15% reduction in MAE during rainfall events compared to purely temporal models.
- **H2:** Gradient boosting trees (particularly LightGBM) will achieve comparable or superior test MAE to deep sequential networks while maintaining sub-10 millisecond inference latencies.
- **H3:** Spatial event proximity features will exhibit a localized decay effect, with significant explanatory weight within 1 km of venue coordinates that rapidly diminishes beyond 3 km.

---

## 2. Experimental Ladder Design (E0 – E4)

To isolate the marginal value of each data modality and modeling enhancement, we formulate a strict five-stage experimental ladder:

```mermaid
flowchart LR
    E0["E0: Baseline (Historical Mean & Persistence)"] --> E1["E1: Transit + Temporal Features"]
    E1 --> E2["E2: Transit + Temporal + Weather (Open-Meteo)"]
    E2 --> E3["E3: Complete Multimodal (Transit + Weather + Events + Spatial)"]
    E3 --> E4["E4: Hyperparameter Optimized Ensembles (Optuna-Tuned)"]
```

| Experiment ID | Feature Modalities Included | Primary Objective |
|---|---|---|
| **E0 (Historical Baseline)** | Scheduled arrival time, static route mean delay, persistence ($\text{delay}_{s-1}$). | Establish baseline lower bound of operational predictability. |
| **E1 (Transit + Temporal)** | E0 + Cyclical hour/day/month encodings, peak-hour indicators, weekend flags. | Quantify the predictive power of recurring commuter diurnal patterns. |
| **E2 (Transit + Temp + Weather)** | E1 + Open-Meteo precipitation rate, 3h rolling rainfall, temperature, wind gusts, WMO weather codes. | Measure the exogenous impact of meteorological disruption and road friction. |
| **E3 (Full Multimodal)** | E2 + Spatial stop density, road hierarchy, event counts (500m, 1km, 3km), distance to nearest venue, minutes to event. | Quantify spatial shockwaves and venue congestion gravity fields. |
| **E4 (Optimized Ensemble)** | Complete feature set from E3 + Optuna hyperparameter optimization across LightGBM, XGBoost, CatBoost. | Determine maximum achievable predictive accuracy and produce production model artifacts. |

---

## 3. Data Splitting & Leakage Prevention Protocol

### Chronological Block Split
To prevent temporal data leakage, random $k$-fold cross-validation is strictly forbidden for time series evaluation. We enforce a chronological split:
- **Training Set (70%):** Earliest continuous temporal block.
- **Validation Set (15%):** Intermediate temporal block (used for early stopping and Optuna hyperparameter optimization).
- **Test Set (15%):** Most recent out-of-time temporal block (held out entirely until final model evaluation).

### Rolling-Origin Temporal Backtesting
For robustness analysis, we implement rolling-origin backtesting over expanding temporal windows:
$$\mathcal{D}_{\text{train}}^{(k)} = [t_0, t_k], \quad \mathcal{D}_{\text{eval}}^{(k)} = (t_k, t_k + \Delta t]$$
with step size $\Delta t = 24$ hours.

### Ingestion Latency Guard
All feature calculations enforce an availability cutoff of $T_{\text{pred}} - 2\text{ minutes}$. No telemetry or updates timestamped after this cutoff may participate in feature generation for target $T_{\text{pred}}$.

---

## 4. Quantitative Evaluation Metrics

### Continuous Delay Regression Metrics
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$$
- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
- **Median Absolute Error (MedAE):**
  $$\text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, \dots, |y_N - \hat{y}_N|)$$

### Multi-Class Classification Metrics
Delay is discretized into four operationally standard categories:
- **On-time:** $\text{delay} \le 5$ min
- **Minor:** $5 < \text{delay} \le 10$ min
- **Moderate:** $10 < \text{delay} \le 20$ min
- **Severe:** $\text{delay} > 20$ min

Evaluation measures:
- Multi-class Accuracy
- Macro-averaged F1 Score
- Severe Delay Recall ($\frac{\text{TP}_{\text{severe}}}{\text{TP}_{\text{severe}} + \text{FN}_{\text{severe}}}$) — critical for dispatch alerts
- Precision-Recall AUC (PR-AUC)

### Operational & Computational Metrics
- **P95 & P99 Inference Latency (ms):** Benchmarked across 1,000 synthetic test requests.
- **Model Serialization Footprint (MB):** Compact artifact constraints for containerized edge deployment.
