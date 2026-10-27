# Bio-Aware-Learning

## Overview

TA-BC-BFL: 
This repository contains:
- A **Python simulation** of the TA-BC-BFL framework with Streamlit dashboard
- **CI** (GitHub Actions), **Docker** support, and a comprehensive **test suite** (112 tests)

## Stack
- Python 3.12+
- Streamlit (dashboard)
- NumPy, Pandas, Plotly (computation & visualization)
- Scikit-learn, SciPy (ML utilities)
- Pytest (testing)

## Quick Start

```bash
# Install dependencies (from the repo root)
pip install -e ".[dev]"
# Run Streamlit simulation dashboard
cd simulation && streamlit run app.py --server.port 5000

# Run tests
cd simulation && PYTHONPATH=. pytest tests/ -v

# Docker build
docker build -t bio-aware-learning .
```
## TA-BC-BFL Simulation

## Structure

```text
Bio-Aware-Learning/
├── .github/workflows/ci.yml   # CI: pytest + Docker build
├── simulation/
│   ├── app.py                 # Streamlit dashboard (port 5000)
│   ├── experiments.py         #  experiment suite (E1–E5), CLI
│   ├── experiment_results.json/md  # Reference results (committed)
│   ├── framework/             # Core TA-BC-BFL modules
│   └── tests/                 # Pytest test suite (112 tests)
├── Dockerfile                 # Containerized simulation
├── pyproject.toml             # Python project config
└── uv.lock                    # Locked Python dependencies
```
## Testing
```bash
# unit tests for all framework modules
cd simulation && PYTHONPATH=. pytest tests/ -v --cov=framework
```

## Reproducibility (Paper Experiments)

All quantitative results in the paper are produced by `simulation/experiments.py`
from fully synthetic, seeded data — no external datasets or downloads required.
### Environment
- Python 3.12 (reference environment: 3.12.10, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5)
- Install from the repo root:
```bash
pip install -e ".[dev]"
```
### Exact commands
```bash
cd simulation

# Smoke test (<1 min): 10 rounds, 2 seeds
python experiments.py --quick --quiet --out /tmp/smoke.json
```
### Verify

```bash
cd simulation && PYTHONPATH=. pytest tests/ -v   # 112 tests, incl. seed-reproducibility tests (with and without DP)
```
## CI
GitHub Actions workflow (`.github/workflows/ci.yml`):
1. **Python Tests** — pytest with coverage + experiment smoke run
2. **Docker Build** — image verification (main branch only)
## Docker
```bash
docker build -t bio-aware-learning .
docker run -p 5000:5000 bio-aware-learning
```
