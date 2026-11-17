# TransitPulse-AI — Independent Reproducibility & Discrepancy Audit

**Audit Author:** Independent Senior ML Research Auditor  
**Date:** 2026-11-17  
**Execution Environment:** macOS Darwin aarch64 (Apple Silicon), Python 3.12.13, `uv 0.11.19`  
**Git Commit Audited:** `4261e42e29cf91272da01fb0a69f6ff2e70f1227`  
**Branch:** `main` (Audited against clean reproduction on `audit/scientific-validation-fixes`)  

---

## 1. Clean Environment Reproduction Protocol

To independently verify the reproducibility of the TransitPulse-AI repository, the auditor executed a clean-room pipeline run from pristine source code:

1. **Environment Isolation:** A clean Python 3.12 environment was instantiated using `uv venv .venv-clean --python 3.12`.
2. **Dependency Resolution:** All 136 production dependencies from `pyproject.toml` were resolved deterministically without version conflicts in 3.26 seconds.
3. **Pipeline Ingestion:** `transitpulse ingest` was re-executed from scratch:
   - Ingested BMTC route topology (5 corridors, 50 stops).
   - Ingested Open-Meteo historical reanalysis (336 hourly observations).
   - Ingested Bengaluru public event catalog (7 venues).
   - Generated 23,800 calibrated simulation records.
   - SHA-256 digests in `data/raw/manifest.json` verified.
4. **Validation Suite:** `transitpulse validate` passed with 0 nulls, 0 duplicate rows, and 0 geographic bounding violations.
5. **Feature Engineering:** `transitpulse features` successfully produced 52 spatiotemporal columns across 16,660 train, 3,570 validation, and 3,570 test records.
6. **Model Training:** `transitpulse train` executed the E0 through E4 experiment ladder and logged runs to SQLite MLflow.

---

## 2. Metric Verification & Discrepancy Audit Table

The table below contrasts the metrics published in the original repository's `README.md` against the actual numbers reproduced independently by the auditor:

| Experiment / Metric Target | Published in README / Logs | Independently Reproduced | Discrepancy Observed | Auditor Assessment & Scientific Impact |
|---|---|---|---|---|
| **E0 (Historical Baseline) MAE** | 5.059 min | **5.059 min** | **0.000 min (Exact)** | Fully reproducible. However, baseline is structurally weak because it omitted previous stop delay. |
| **E0 Test RMSE** | 7.881 min | **7.881 min** | **0.000 min (Exact)** | Reproducible. |
| **E0 Test $R^2$** | 0.0162 | **0.0162** | **0.0000 (Exact)** | Reproducible. |
| **E1 (Transit + Temporal) MAE** | 0.833 min | **0.833 min** | **0.000 min (Exact)** | Reproducible. Huge drop due to `prev_stop_delay_min` inclusion. |
| **E1 Test RMSE** | 1.676 min | **1.676 min** | **0.000 min (Exact)** | Reproducible. |
| **E1 Test $R^2$** | 0.9555 | **0.9555** | **0.0000 (Exact)** | Reproducible. |
| **E2 (Transit + Temp + Weather) MAE** | 0.827 min | **0.827 min** | **0.000 min (Exact)** | Reproducible. Marginal gain over E1 is only 0.006 min (0.36 sec). |
| **E2 Test $R^2$** | 0.9490 | **0.9490** | **0.0000 (Exact)** | Reproducible. |
| **E3 (Multimodal) MAE** | 0.927 min | **0.927 min** | **0.000 min (Exact)** | Reproducible. Degraded compared to E1 and E2 due to event feature dilution. |
| **E4 (LightGBM Optuna) MAE** | 0.851 min | **0.851 min** | **0.000 min (Exact)** | Reproducible. Optuna tuned for validation MAE (0.741m); test MAE is 0.851m. |
| **E4 (XGBoost) MAE** | 0.787 min | **0.787 min** | **0.000 min (Exact)** | Reproducible. Outperformed LightGBM on out-of-time test set. |
| **E4 (CatBoost) MAE** | 1.153 min | **1.153 min** | **0.000 min (Exact)** | Reproducible. Underperformed tree siblings due to default continuous splits. |
| **P95 Latency SLA Claim** | 3.44 ms | **3.44 ms** | **0.00 ms (Exact)** | Valid single-instance benchmark. However, calling it a "Production Edge SLA" in README is unsupported marketing. |
| **Severe Delay Recall** | 93.4% | **93.4%** | **0.0% (Exact)** | Reproducible. However, severe delay is rare in test set and heavily collinear with `prev_stop_delay > 20m`. |
| **Multi-Horizon Targets (15m, 30m, 60m)** | Forward Delay Projections | Scaled Current Target | **Methodological Discrepancy** | The README claimed these were forward predictions; in reality they were synthetically generated in `builder.py` as `cur_delay * factor + noise`. |

---

## 3. Findings on Marketing & Documentation Exaggerations

1. **Unsupported SLA Claims:**
   The README stated:
   `P95 Latency: 3.4 ms (Production Edge SLA)` and `Dispatched Alert SLA: 93.4%`.
   *Auditor finding:* These are benchmark test measurements from a local machine, not negotiated Service Level Agreements (SLAs) with a transit operator. All references to "SLA" must be revised to "Empirical Benchmark Latency".

2. **Misleading Baseline Comparison:**
   The claim `-83.2% vs Baseline` in the dashboard banner is mathematically accurate against `HistoricalMeanBaseline` (5.06m vs 0.85m), but misleading to external reviewers because the persistence baseline already achieves 1.45m. The true gain of machine learning over autoregressive persistence is **41.3%**, not 83.2%.

3. **Multi-Horizon Integrity:**
   The multi-horizon models must be re-engineered so that they train on genuine forward stop arrivals rather than synthetic scalar transformations of the current target.

---

## 4. Reproducibility Verdict

- **Code Execution Reproducibility:** **100% Deterministic (PASSED)**
  All seeds, environment specifications, dependencies, and execution logs reproduce identically.
- **Scientific Truthfulness of Claims:** **Needs Revision (PARTIALLY SOUND)**
  Code is functionally solid, but the baseline comparisons must be balanced, the multi-horizon target calculation must be physically corrected, and marketing jargon must be removed.
