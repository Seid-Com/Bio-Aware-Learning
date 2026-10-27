# Bio-Aware-Learning

## Overview

TA-BC-BFL: 
This repository contains:
- A **Python simulation** of the TA-BC-BFL framework with Streamlit dashboard
- The **paper experiment suite** (`experiments.py`) and its committed reference results
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

Full Python simulation in `simulation/` with interactive Streamlit dashboard.

### Framework Modules (`simulation/framework/`)

| Module | Description |
|---|---|
| `clients.py` | Federated client simulation — synthetic multi-omics healthcare data (genomic, proteomic, clinical, epigenomic, transcriptomic), local SGD training |
| `bio_aware.py` | Bio-aware aggregation — cosine pathway similarity, omics compatibility matrix (5x5 domain knowledge), temperature-softmax weighted aggregation |
| `blockchain.py` | Blockchain trust manager — SHA-256 proof-of-work, trust scoring with historical consistency, MAD-based anomaly detection |
| `privacy.py` | Differential Privacy (Gaussian mechanism with advanced composition budget accountant) + Homomorphic Encryption simulation |
| `attacks.py` | Adversarial attack simulation — model poisoning (3 strategies), Sybil, gradient inversion, backdoor |
| `federated_learning.py` | Core FL loop: TA-BC-BFL + baselines (FedAvg, Per-FedAvg, Blockchain-FL, Trust-FL), configuration validation |


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

# Paper results (Table/Figure source): 100 rounds, 10 clients, 5 seeds
# Runtime ~21 min on a laptop CPU. Deterministic given the same versions.
python experiments.py --rounds 100 --seeds 5 --clients 10 \
    --out experiment_results.json --md experiment_results.md --quiet

# Smoke test (<1 min): 10 rounds, 2 seeds
python experiments.py --quick --quiet --out /tmp/smoke.json
```

Committed outputs `simulation/experiment_results.md` and
`simulation/experiment_results.json` are the reference results; regenerating
them reproduces every reported number (minor floating-point drift across
BLAS/numpy versions is expected; statistical conclusions are stable).

### Fixed configuration

| Parameter | Value |
|---|---|
| Seeds (paired across all methods) | 42, 123, 456, 789, 1024 |
| Clients / rounds / local epochs | 10 / 100 / 5 |
| Learning rate | 0.001 |
| Data | Synthetic 5-omics non-IID (mild), fixed per-client seeds; run seed controls sample counts + model init |
| Adversarial fraction (E2) | 0.2 (model poisoning, label flipping, Sybil) |
| DP | Gaussian mechanism, δ=1e-5, ε ∈ {0.1, 0.5, 1, 2, 5, 10} |
| Epsilon for Full method in E1/E2 | 1.0 |

Every method in a comparison receives the identical seed, so client data,
selection, and initialization are paired (enables the paired t-test).

### Statistical methodology
- 5 independent runs per condition; mean, sample std, two-sided 95% t-interval
- Paired t-test on per-seed final accuracies (scipy `ttest_rel`); significance at α=0.05
- Reported in E5 of `experiment_results.md`

Honest-finding note: on this synthetic benchmark the framework's benefit is
**attack resistance under poisoning/Sybil**, not a clean-accuracy gain;
bio-aware weighting is accuracy-neutral without real multi-omics signal.
These are the claims the experiment suite supports.

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
