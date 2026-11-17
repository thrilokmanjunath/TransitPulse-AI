# TransitPulse-AI — Automated Data Quality Audit Report

**Audit Execution Timestamp:** 2026-11-17T19:57:15.948906+05:30
**Geographic Bounding Target:** Bengaluru Metropolitan Area [12.75, 77.4] to [13.25, 77.85]

---

## 1. Executive Summary Table

| Dataset Entity | Total Records | Null Fields | Duplicates | Geo Violations | Outliers | Quality Status |
|---|---|---|---|---|---|---|
| **BMTC Stops** | 50 | 0 | 0 | 0 | 0 | **PASSED** |
| **Open-Meteo Weather** | 336 | 0 | 0 | 0 | 0 | **PASSED** |
| **Transit Trips** | 23,800 | 2380 | 0 | 0 | 0 | **PASSED** |

---

## 2. Detailed Entity Diagnostics

### BMTC Stops
- **Validation Passed:** `True`
- **Duplicate Records:** `0`
- **Geographic Violations:** `0`
- **Null Breakdown:**
  - *Zero null fields detected across all columns.*
- **Outlier Diagnostic Breakdown:**
  - `negative_sequence`: 0 outlier records
- **Auditor Diagnostic Messages:**
  - All BMTC stop schemas, coordinate bounds, and sequence orderings passed.

### Open-Meteo Weather
- **Validation Passed:** `True`
- **Duplicate Records:** `0`
- **Geographic Violations:** `0`
- **Null Breakdown:**
  - *Zero null fields detected across all columns.*
- **Outlier Diagnostic Breakdown:**
  - `extreme_temp`: 0 outlier records
  - `negative_precipitation`: 0 outlier records
  - `invalid_humidity`: 0 outlier records
- **Auditor Diagnostic Messages:**
  - All meteorological ranges, humidity limits, and timestamps valid.

### Transit Trips
- **Validation Passed:** `True`
- **Duplicate Records:** `0`
- **Geographic Violations:** `0`
- **Null Breakdown:**
  - `prev_stop_id`: 2380 nulls
- **Outlier Diagnostic Breakdown:**
  - `extreme_delay`: 0 outlier records
- **Auditor Diagnostic Messages:**
  - All transit operational delay records conform to schema specifications.

---

## 3. Data Integrity & Governance Certification
> [!NOTE]
> All coordinate pairs conform strictly to WGS-84 geometry for Greater Bengaluru.
> Timestamps have been validated for strict monotonic progression with timezone `Asia/Kolkata`.
> No synthetic records are labeled as `observed` BMTC telemetry.