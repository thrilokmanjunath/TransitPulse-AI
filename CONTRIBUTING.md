# Contributing to TransitPulse-AI

Thank you for your interest in contributing to **TransitPulse-AI**! We welcome contributions from data scientists, transport economists, software engineers, and urban mobility researchers.

---

## 1. Development Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/thrilokmanjunath/TransitPulse-AI.git
   cd TransitPulse-AI
   ```

2. **Initialize Python 3.12 Virtual Environment using `uv`:**
   ```bash
   uv venv .venv --python 3.12
   source .venv/bin/activate
   ```

3. **Install Dependencies in Editable Mode:**
   ```bash
   uv pip install -e ".[dev]"
   ```

---

## 2. Invariants & Contribution Rules

1. **Zero Temporal Leakage:**
   - Features must strictly enforce an availability timestamp cutoff $T_{\text{pred}} - 2\text{ minutes}$.
   - Never use future stop observations or global time-series shuffles in cross-validation.
2. **Absolute Scientific Honesty:**
   - Never label synthetic or calibrated simulation observations as measured BMTC hardware telemetry.
   - Maintain `dataset_mode: "simulated"` tags on all synthetic experiments.
3. **Security Standards:**
   - Run Bandit and pip-audit prior to submitting PRs:
     ```bash
     bandit -r src api
     pip-audit
     ```
   - Never commit API keys, `.env` files, or binary datasets over 50MB.
4. **Test Suite Requirements:**
   - Ensure all unit, integration, and leakage tests pass:
     ```bash
     pytest --cov=transitpulse --cov=api tests/ -v
     ```

---

## 3. Pull Request Guidelines

- Use conventional commit messages: `feat:`, `fix:`, `docs:`, `test:`, `ml:`, `refactor:`.
- Update relevant documentation in `docs/` for any modified feature transformations or modeling architectures.
