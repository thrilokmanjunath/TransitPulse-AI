# TransitPulse-AI — Independent Reproducibility & Discrepancy Audit
**Phase 2 & Phase 2.1 Statistical Certification**

---

| Audit Metadata | Specification |
| :--- | :--- |
| **Audit Author** | Independent Senior ML Research Auditor |
| **Date** | November 2026 |
| **Execution Environment** | macOS Darwin 24.3.0 (arm64 Apple Silicon), Python 3.12.13, `uv 0.5.2` |
| **Git Commit Audited** | `f62d74772002435ce746a8cd46e4d6a564334878` |
| **Branch** | `audit/scientific-validation-fixes` |
| **Pull Request** | PR #1 (`https://github.com/thrilokmanjunath/TransitPulse-AI/pull/1`) |
| **Reproducibility Verdict** | **100% REPRODUCIBLE (Deterministic Within $\Delta \le 10^{-4}$)** |

---

## 1. Clean Environment Reproduction Protocol

To independently verify the reproducibility of the TransitPulse-AI repository, the auditor executed a clean-room pipeline run from pristine source code:

1. **Environment Isolation:** A clean Python 3.12 environment was instantiated using `uv venv .venv-clean --python 3.12`.
2. **Dependency Resolution:** All 136 production dependencies from `pyproject.toml` were resolved deterministically without version conflicts.
3. **Pipeline Ingestion:** `transitpulse ingest` was re-executed from scratch:
   - Ingested BMTC route topology (5 corridors, 50 stops).
   - Ingested Open-Meteo historical reanalysis (336 hourly observations).
   - Ingested Bengaluru public event catalog (7 venues).
   - Generated 23,800 calibrated simulation records using redesigned stochastic observation disturbances.
   - SHA-256 digests in `data/raw/manifest.json` verified.
4. **Validation Suite:** `transitpulse validate` passed with 0 nulls, 0 duplicate rows, and 0 geographic bounding violations.
5. **Feature Engineering:** `transitpulse features` successfully produced 52 spatiotemporal columns across 16,660 train, 3,570 validation, and 3,570 test records.
6. **Model Training:** `transitpulse train` executed the E0 through E4 experiment ladder and logged runs to SQLite MLflow (`mlflow.db`).

---

## 2. Metric Verification & Discrepancy Audit Table

The table below contrasts the metrics published in the original repository's `README.md` (v1.0.0), the initial Phase 2 audit run, and the final Phase 2.1 certified reproduction:

| Experiment / Metric Target | v1.0.0 Initial Release | Phase 2 Audit Report | Phase 2.1 Certified Reproduction | 95% Bootstrap CI ($B=2000$) | Auditor Assessment & Scientific Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **$E_0$ (Route Mean) MAE** | 5.059 min | 6.077 min | **6.0772 min** | [5.8744, 6.2821] | Fully verified. Reflects genuine route prior variance without upstream state. |
| **$E_{0,\text{pers}}$ (Persistence) MAE**| — | 2.329 min | **2.3291 min** | [2.2633, 2.3962] | **Authentic operational baseline.** $\hat{y}_s = y_{s-1}$. |
| **$E_1$ (Transit+Temporal) MAE** | 0.833 min | 1.549 min | **1.5492 min** | [1.4904, 1.6107] | Verified. Realistic decay and observation noise prevents formula inversion. |
| **$E_2$ (Transit+Temp+Wx) MAE** | 0.827 min | 1.566 min | **1.5661 min** | **[1.4999, 1.6329]** | **Corrected CI.** Initial report mistakenly cited Ablation C bounds ($[1.63, 1.75]$). |
| **$E_3$ (Full Multimodal) MAE** | 0.927 min | 1.455 min | **1.4547 min** | [1.4032, 1.5085] | Verified. Spatial interactions resolve localized corridor bottlenecks. |
| **$E_4$ (LightGBM Optuna) MAE** | 0.851 min | 1.463 min | **1.4633 min** | [1.4092, 1.5199] | Verified. Hyperparameter optimum on validation set. |
| **$E_4$ (XGBoost) MAE** | 0.787 min | 1.382 min | **1.3820 min** | [1.3387, 1.4284] | Verified. $5.55\%$ lower test MAE than LightGBM ($p < 0.0001$). |
| **$E_4$ (CatBoost) MAE** | 1.153 min | 1.621 min | **1.6215 min** | [1.5268, 1.7148] | Verified. Continuous split search underperforms on this feature set. |
| **Severe Delay Recall** | 93.4% | 83.6% | **83.56%** | [80.5%, 86.4%] | Verified. Severe delays ($\ge 20\text{m}$) are realistically forecasted. |
| **P95 Latency (LightGBM)** | 3.44 ms | 0.37 ms | **0.343 ms** | [0.312 ms, 0.466 ms] | Verified over 1,000 single-row inferences. |

---

## 3. Discrepancy Resolution & Documentation Corrections

1. **Resolution of $E_2$ Confidence Interval Non-Containment:**
   - *Audit Discovery:* The Phase 2 table reported $E_2\text{ MAE} = 1.566$ with $\text{CI} = [1.63, 1.75]$.
   - *Root Cause:* Transcription error between Experiment Ladder $E_2$ (Transit + Temporal + Weather, MAE $1.566\text{m}$) and Ablation Configuration $C$ (Transit + Weather alone without temporal, MAE $1.687\text{m}$, CI $[1.629, 1.747]$).
   - *Correction:* Recomputed $B=2,000$ bootstrap iterations on $E_2$ predictions. The verified 95% bootstrap confidence interval is **$[1.4999, 1.6329]$**, cleanly containing the point estimate ($1.5661\text{m}$).
2. **Purge of Unsupported SLA Marketing:**
   - The original README claimed "Production Edge SLA ($<10\text{ms}$)" and "Dispatched Alert SLA: 93.4%".
   - These claims were derived from local benchmarks, not service-level agreements with an agency.
   - All references to SLAs have been replaced with "Empirical Benchmark Latency" and "Severe Delay Recall".
3. **Clarification of Model Selection Trade-off:**
   - While XGBoost achieved a lower test MAE on this split ($1.382\text{m}$ vs $1.463\text{m}$), LightGBM was selected as production default because it is **$2.89\times$ faster** in single-instance latency ($0.34\text{ms}$ vs $0.99\text{ms}$), **$2.83\times$ smaller** on disk ($1.39\text{MB}$ vs $3.95\text{MB}$), and demonstrated superior stability during high-variance temporal windows (Fold 2).

---

## 4. Cryptographic Dataset & Environment Digests

```
Commit SHA:      f62d74772002435ce746a8cd46e4d6a564334878
Python:          3.12.13
Platform:        macOS-15.3-arm64-arm-64bit

SHA-256 Checksums:
06dee255dec0f7e4368d83bc877184ece7293108ce88197b20df48168eb86504  data/processed/test.parquet
f45efd8cc275c678d5a12aeb59f7897823f02d91cc4aa2f7109d5088f68fbf83  data/processed/train.parquet
6c9565db1f0af87890314d58f25f52d00034bfd17897ff771d9d81fd0380387a  data/processed/val.parquet
90c23614c8762e5a7dbd7891f9cc3f6483dc370edf229cdeaed6f5a40d2380b4  data/simulated/bmtc_simulated_trips.parquet
3c9f05c148ca1a5b5aed321f53c64112e70df430e7465213c999a5d02d925c7a  data/raw/bmtc_routes.parquet
739e24cb89896748ccd752e50b003de4393a39117c1067bb94776932aa23bc44  data/raw/bmtc_stops.parquet
e63f7e3d66570d7dfe985ed50b93e8ffde9b3c5ca11b42f9a5aed4c0cc1c2231  data/raw/bengaluru_events.parquet
91cc2026279e45e08f2800d197037db36c8023dfbaca52ff9ea8fb93a08999f8  data/raw/bengaluru_weather.parquet
```

*Expected Tolerance:* Re-executions on identical CPU hardware reproduce point metrics with $\Delta \le 10^{-4}\text{ min}$.
