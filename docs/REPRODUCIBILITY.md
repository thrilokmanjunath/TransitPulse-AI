# TransitPulse-AI — Reproducibility & Execution Guide

This document guarantees 100% deterministic reproduction of the TransitPulse-AI data science pipelines, model benchmarks, REST APIs, and dashboard deployments.

---

## 1. Prerequisites & Environment Setup

- **Operating System:** Linux (Ubuntu 22.04+), macOS (Apple Silicon or Intel), or Windows (WSL2).
- **Python:** Python 3.12 (recommended) or Python 3.11+.
- **Package Manager:** `uv` (recommended) or standard `pip`.

### Setup Instructions with `uv`:
```bash
# Clone the repository
git clone https://github.com/thrilokmanjunath/TransitPulse-AI.git
cd TransitPulse-AI

# Create isolated Python 3.12 virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install the transitpulse package in editable mode with development dependencies
uv pip install -e ".[dev]"
```

---

## 2. End-to-End Pipeline Execution via CLI

The platform provides a unified CLI entry point:

```bash
# Step 1: Ingest BMTC transit topology, Open-Meteo weather, events, and simulated delay data
transitpulse ingest --start-date 2024-05-01 --end-date 2024-05-14

# Step 2: Validate dataset schemas, geographic coordinates, and range constraints
transitpulse validate

# Step 3: Extract spatiotemporal features and perform chronological train/val/test splitting
transitpulse features

# Step 4: Run experiment ladder E0-E4, Optuna hyperparameter tuning, and model evaluation
transitpulse train
```

---

## 3. Running Services Locally

### Launch FastAPI Server:
```bash
transitpulse api --host 127.0.0.1 --port 8000
# Interactive Swagger Documentation available at: http://localhost:8000/docs
```

### Launch Streamlit Analytics Dashboard:
```bash
transitpulse dashboard --host 127.0.0.1 --port 8501
# Dashboard available in your browser at: http://localhost:8501
```

---

## 4. Docker Deployment

Deploy the entire platform (FastAPI + Streamlit Dashboard) with Docker Compose:

```bash
docker-compose up --build
```
- API Endpoint: `http://localhost:8000`
- Interactive Dashboard: `http://localhost:8501`

---

## 5. Verification & Testing

Execute the automated test and security verification suite:

```bash
# Run test suite with code coverage
pytest --cov=transitpulse --cov=api tests/ -v

# Run static application security testing (SAST)
bandit -r src api

# Run dependency vulnerability audit
pip-audit
```
