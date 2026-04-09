# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SyMANTIC is a PyTorch-based symbolic regression library that discovers interpretable, parsimonious equations from data. It combines mutual information-based feature selection, adaptive feature expansion with mathematical operators, L0-sparse regression with quantile-based partitioning, and Pareto-optimal solution extraction.

Install: `pip install symantic`

## Build & Test Commands

```bash
pip install -e ".[dev]"    # Install in editable mode with test dependencies
pytest                     # Run all tests
pytest tests/test_model.py # Run a specific test file
pytest -k "test_auto"      # Run tests matching a pattern
```

## Package Structure

```text
symantic/                               # Main package (renamed from src/)
├── model.py                            # SymanticModel — main entry point
├── results.py                          # FitResult dataclass (unified return type)
├── validation.py                       # Input validation functions
├── exceptions.py                       # FeatureSpaceLimitError, ValidationError
├── feature_expansion/
│   ├── nondimensional.py               # Non-dimensional feature expansion
│   └── dimensional.py                  # Dimensional feature expansion (sympy units)
├── regression/
│   ├── l0_greedy.py                    # Greedy forward selection + OLS
│   ├── l0_greedy_dimensional.py        # Dimensional-aware regression
│   └── screening.py                    # Dimensional screening helpers
├── pareto.py                           # Pareto frontier identification
├── losses/                             # Loss functions (Phase 4 — planned)
└── dynamics/                           # Dynamic problem support (Phase 5 — planned)
tests/                                  # pytest test suite
docs/scaling.md                         # Computational scaling guide
examples/notebooks/                     # Jupyter notebook examples
```

The old `src/` directory provides a backward-compatible shim with deprecation warnings.

## Architecture

Pipeline: **Feature Space Construction → Sparse Regression → Pareto Analysis**

Two parallel pathways — non-dimensional and dimensional (unit-aware via sympy):

- `symantic/model.py` — **SymanticModel**: orchestrates the pipeline. Routes to dimensional or non-dimensional pathway based on whether `dimensionality` parameter is provided. Imports feature expansion modules as `fcc` (non-dimensional) and `dfcc` (dimensional).
- `symantic/feature_expansion/nondimensional.py` — Feature expansion using operators (+, -, *, /, exp, sin, cos). When `n_expansion=None`, runs adaptive auto-depth mode that iterates until metrics thresholds are met.
- `symantic/feature_expansion/dimensional.py` — Dimensional feature expansion with sympy-based unit validation.
- `symantic/regression/l0_greedy.py` — Greedy forward selection + OLS via `torch.linalg.lstsq`. Uses SIS (Sure Independence Screening) with quantile-based partitioning by complexity.
- `symantic/regression/l0_greedy_dimensional.py` — Dimensional-aware regression that filters features for unit consistency.
- `symantic/pareto.py` — Pareto front identification with configurable utopia point.

**Namespace**: The `__init__.py` exports qualified names (`NonDimensionalRegressor`, `DimensionalRegressor`, etc.) to avoid collisions. Default unqualified `Regressor` maps to non-dimensional.

## Key Dependencies

torch, numpy, pandas, scipy (spearmanr), sklearn (mutual_info_regression), sympy, matplotlib

## Key API

```python
from symantic import SymanticModel

model = SymanticModel(
    df,                          # DataFrame: column 0 = target, columns 1+ = features
    operators=['+','-','*','/'], # Operators for feature expansion
    n_expansion=None,            # None = adaptive auto-depth; integer = fixed depth (uses range(1, n))
    n_term=3,                    # Max terms in equation (sparsity)
    sis_features=20,             # Features to screen per iteration
    metrics=[0.06, 0.995],       # [RMSE_threshold, R2_threshold] for auto-depth stopping
    max_features=2000,           # Max features before stopping expansion (None = default per mode)
    device='cpu',                # 'cpu' or 'cuda'
)
result = model.fit()             # Returns FitResult
# Access via attributes:
result.rmse, result.equation, result.r2, result.pareto_front
# Or backward-compatible unpacking:
res, full_pareto = result        # Auto-depth mode
rmse, equation, r2 = result      # Fixed-depth mode
```

**Note on n_expansion**: When set to an integer, the loop is `range(1, n_expansion)`, so `n_expansion=2` gives 1 expansion level. `None` enables auto-depth.

## Planned Improvements (Multi-Phase)

This project is undergoing restructuring per the plan in `.claude/plans/`:

- **Phase 2**: UX/docs — `FitResult` dataclass, configurable `max_features`, input validation, scaling docs (COMPLETE)
- **Phase 3**: Performance — vectorized operators, generate-and-screen, batched combinations, parallel quantile bins
- **Phase 4**: L1/L2/ElasticNet regularization + classification losses (cross-entropy, hinge, focal)
- **Phase 5**: Dynamic problem placeholders (finite difference, Neural ODE gradient estimation)
