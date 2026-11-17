# TransitPulse-AI: Final Reproducibility Checklist & Developer Guide

**Document ID:** `CHK-REPRO-RELEASE-20261117`  
**Target Release:** `v1.1.0` (Audited Scientific & Engineering Portfolio Release)  
**Primary Architecture:** macOS Apple Silicon (arm64 / M1–M4) & Linux (x86_64 / arm64)  
**Python Runtime:** Python `3.12.13` (Compatible with Python `3.10`, `3.11`, `3.12`)  
**Status:** **100% REPRODUCIBLE (ZERO ENVIRONMENT/SEED DRIFT)**

---

## 1. Quickstart: Clean-Clone Setup (Under 60 Seconds)

### Step 1: Clone Repository
```bash
git clone https://github.com/thrilokmanjunath/TransitPulse-AI.git
cd TransitPulse-AI
```

### Step 2: Hermetic Environment Provisioning
TransitPulse-AI supports both modern fast package management via `uv` (recommended) or standard Python `venv`.

```bash
# Option A: Using uv (Fastest — under 10 seconds)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Option B: Standard Python venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### Step 3: Run Automated Verification Suite
```bash
# Execute unit, integration, schema validation, and leakage tests
pytest tests/ -v

# Execute static analysis & SAST security scanning
ruff check src/ tests/
ruff format --check src/ tests/
bandit -c pyproject.toml -r src/
```
*Expected test runtime: ~1.0 second. 22/22 tests passing, 0 linter violations, 0 SAST alerts.*

---

## 2. End-to-End Pipeline Execution Commands & Runtimes

All stages of the data science and machine learning lifecycle can be independently executed via standardized module entry points:

| Stage | Command | Target Artifacts Produced | Runtime (Apple Silicon M4) |
|---|---|---|---|
| **1. Ingestion & Validation** | `python -m transitpulse.ingestion` | `data/raw/*.parquet`, `manifest.json` | 2.8 s |
| **2. Feature Engineering** | `python -m transitpulse.features` | `data/processed/{train,val,test}.parquet`, `feature_metadata.json` | 3.4 s |
| **3. Model Training & Optimization** | `python -m transitpulse.models.trainer` | `models/artifacts/best_model.joblib`, `experiment_results.json` | 18.2 s |
| **4. Multi-Horizon Training** | `python -m transitpulse.models.multi_horizon` | `models/artifacts/multi_horizon/model_h{15,30,60}.joblib` | 5.9 s |
| **5. Statistical Integrity & Paired Bootstrap** | `python -m transitpulse.audit.bootstrap` | `reports/PHASE_2_1_STATISTICAL_INTEGRITY.md` | 11.5 s |
| **6. Microservice Serving** | `uvicorn transitpulse.api.main:app --port 8000` | Real-time REST endpoints (`http://localhost:8000`) | Background |
| **7. Visual Analytics Dashboard** | `streamlit run dashboard/app.py` | Multi-page Streamlit portal (`http://localhost:8501`) | Background |

*Total cold-start pipeline execution: under 45 seconds.*

---

## 3. Cryptographic Provenance & Checksum Register (SHA-256)

To guarantee that no data corruption, silent model alteration, or seed variance occurs across clones, verify the following cryptographic hashes:

### 3.1. Raw Ingestion Datasets (`data/raw/`)
| File | Size (Bytes) | SHA-256 Checksum | Provenance Description |
|---|---|---|---|
| `bmtc_routes.parquet` | 5,713 | `3c9f05c148ca1a5b5aed321f53c64112e70df430e7465213c999a5d02d925c7a` | BMTC static GTFS arterial routes (335-E, 500-D, KIA-9, G-2, 201-R) |
| `bmtc_stops.parquet` | 7,900 | `739e24cb89896748ccd752e50b003de4393a39117c1067bb94776932aa23bc44` | WGS-84 physical bus stop coordinates along routes |
| `bengaluru_weather.parquet`| 12,430 | `91cc2026279e45e08f2800d197037db36c8023dfbaca52ff9ea8fb93a08999f8` | Open-Meteo ERA5 hourly reanalysis weather observations |
| `bengaluru_events.parquet` | 7,495 | `e63f7e3d66570d7dfe985ed50b93e8ffde9b3c5ca11b42f9a5aed4c0cc1c2231` | High-impact sports & convention event schedules & coordinates |
| `manifest.json` | 1,361 | `f0111339c19598067bb31a8ba13141b54afd805b39878f38c363b4e8c44fe612` | Ingestion metadata, timestamps, and schema definitions |

### 3.2. Processed Chronological Splits (`data/processed/`)
| File | Records | SHA-256 Checksum | Split Methodology |
|---|---|---|---|
| `train.parquet` | 16,660 | `f45efd8cc275c678d5a12aeb59f7897823f02d91cc4aa2f7109d5088f68fbf83` | First 70% chronological partition ($t \le t_1$) |
| `val.parquet` | 3,570 | `6c9565db1f0af87890314d58f25f52d00034bfd17897ff771d9d81fd0380387a` | Next 15% chronological partition ($t_1 < t \le t_2$) |
| `test.parquet` | 3,570 | `06dee255dec0f7e4368d83bc877184ece7293108ce88197b20df48168eb86504` | Final 15% out-of-time holdout ($t > t_2$) |

### 3.3. Trained Production Artifacts (`models/artifacts/`)
| File | Size (Bytes) | SHA-256 Checksum | Description |
|---|---|---|---|
| `best_model.joblib` | 1,427,768 | `0291f2a498e81b552efb67b6867b76822c6712e76b3f19c1ea0d1102d824d721` | Production LightGBM dual regression/classification model |
| `xgboost_model.joblib` | 4,045,295 | `73f49ea759c8b6bea0798734201d3cc2ea03cbd12dddf32c23cccf24be2cd20d` | Holdout benchmark XGBoost regressor |
| `experiment_results.json` | 5,278 | `c40357e23a0e61cb3131e3af743c6069a4bf7e031fe4c39c25c6f5426d0e366d` | Complete verified out-of-time benchmark metrics for $E_0 \to E_4$ |
| `model_metadata.json` | 1,732 | `6d4f9a4f20d504b2bbce3b30603a35b040b76c305a93d4d62c79cdced3e25c13` | Architecture configuration, hyperparameter dictionary, features |

---

## 4. Hardware Independence & Tolerance Thresholds

The benchmark numbers published across the documentation reflect seed-deterministic model training (`random_state=42`). Slight floating-point variance across BLAS/LAPACK or CPU architectures should remain strictly within the following tolerance bounds:

| Metric | Certified Benchmark | Acceptable Tolerance Window |
|---|---|---|
| **E4 LightGBM Holdout Test MAE** | `1.463 min` | $[1.450, 1.480]\text{ min}$ |
| **E4 XGBoost Holdout Test MAE** | `1.382 min` | $[1.370, 1.400]\text{ min}$ |
| **E0 Historical Route Mean MAE** | `6.077 min` | $[6.070, 6.085]\text{ min}$ |
| **E0 Upstream Persistence MAE** | `2.329 min` | $[2.320, 2.340]\text{ min}$ |
| **Holdout Test $R^2$** | `0.9365` | $[0.9300, 0.9420]$ |
| **Severe Delay Recall ($\ge 20\text{ min}$)**| `0.8356` | $[0.8250, 0.8500]$ |
| **LightGBM P95 Single-Row Latency** | `0.37 ms` | $< 2.0\text{ ms}$ on modern hardware |

---

## 5. Hermetic CI/CD Self-Healing Guarantee

Fresh checkout environments (such as ephemeral GitHub Actions runners) do not track large binary parquet or joblib files in git. 

To ensure developer ergonomics and 100% CI reproducibility:
1. `tests/conftest.py` implements an automatic session-scoped test harness (`ensure_test_artifacts()`).
2. If `models/artifacts/best_model.joblib` or `data/processed/` are not present on disk upon test discovery, minimal synthetic test data and lightweight surrogate model artifacts are synthesized in $<0.5\text{s}$.
3. All 22 tests execute deterministically on clean checkouts with zero external API calls or network dependencies.
4. When full datasets are generated via `python -m transitpulse.ingestion && python -m transitpulse.features && python -m transitpulse.models.trainer`, the test harness automatically defers to the genuine production artifacts.

---

## 6. Real-World Grounding & Synthetic Data Boundary Declaration

- **Real Reference Geometry:** The bus routes (335-E, 500-D, KIA-9, G-2, 201-R) and stop coordinates correspond to actual physical BMTC transit infrastructure in Bengaluru.
- **Real Weather Context:** Meteorological readings (precipitation, temperature, humidity, wind) are derived from genuine ERA5 reanalysis and Open-Meteo forecasts for the Bengaluru region.
- **Calibrated Simulated Delays:** Operational stop arrival delays are synthesized using `TransitDelaySimulator` incorporating Bureau of Public Roads (BPR) volume-delay curves, heterogeneous segment decay ($\rho_s \in [0.55, 0.90]$), passenger dwell variance, and traffic signal queuing.
- **No Direct Telemetry Claim:** Never present these metrics as real-world BMTC telemetry accuracy. The repository provides a production-grade algorithmic and engineering framework ready for hardware AVL ingestion.
