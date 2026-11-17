# TransitPulse-AI — Data Directory Architecture & Guidelines

This directory houses raw, interim, processed, and simulated datasets for the TransitPulse-AI platform.

## Directory Structure

```
data/
├── raw/            # Pristine raw downloads (GTFS static archives, raw Open-Meteo responses, raw events)
├── interim/        # Standardized intermediate Parquet tables (cleaned routes, aligned weather)
├── processed/      # ML-ready spatiotemporal feature matrices and train/val/test splits
├── simulated/      # Calibrated synthetic delay telemetry records with exact metadata tags
└── external/       # Geographic boundaries, OSM network extracts, stadium location metadata
```

## Data Governance & Versioning Rules

1. **Zero Raw Leakage:**
   Large raw binaries (>50MB) and raw unvalidated files are git-ignored. Only small deterministic seed extracts and `.gitkeep` files are committed.
2. **Standard Format:**
   All interim and processed data must be stored in Apache Parquet format (`snappy` or `zstd` compression) with explicit schema definitions.
3. **Strict Mode Separation:**
   - Datasets with observed ground-truth labels are tagged `dataset_mode: "observed"`.
   - Datasets generated via the calibrated synthetic simulator are tagged `dataset_mode: "simulated"`.
   - Datasets combining both must be explicitly labeled `dataset_mode: "hybrid"` and document the composition ratio.
4. **Timezone Uniformity:**
   All datetime timestamps must be strictly localized to `Asia/Kolkata` (IST, UTC+05:30).
