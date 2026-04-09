# Implemented Changes

Tracks all improvements made to SyMANTIC across restructuring phases.

---

## Phase 1: Project Restructure

**Goal**: Proper Python package layout, fix namespace collisions, add build tooling, basic tests.

### Package layout
- Renamed `src/` to `symantic/` with proper subpackage structure (`feature_expansion/`, `regression/`)
- Added `pyproject.toml` with setuptools build config, dependency declarations (torch, numpy, pandas, scipy, scikit-learn, sympy, matplotlib), Python >=3.9
- Added `.github/workflows/ci.yml` for pytest on Python 3.9-3.12
- Added backward-compatible `src/__init__.py` shim that re-exports from `symantic/` with `DeprecationWarning`

### Bug fixes
- **Namespace collision in `__init__.py`**: Both dimensional and non-dimensional `Regressor` and `feature_space_construction` were imported under the same name (dimensional overwrote non-dimensional). Fixed with qualified export names: `NonDimensionalRegressor`, `DimensionalRegressor`, `NonDimensionalFeatureExpander`, `DimensionalFeatureExpander`
- **Hardcoded utopia point in `pareto.py`**: Line 58 hardcoded `utopia_point = torch.tensor([0.0, 0.0])`, overriding the computed utopia. Made it a configurable parameter defaulting to computed `(min_complexity, min_rmse)`
- **Missing `import pandas as pd`** in `l0_greedy_dimensional.py`: Used `pd.Series` but relied on implicit import from old namespace
- **`import pdb`** removed from all 7 files (never used)
- **Duplicate `import time`** removed from `model.py`
- **Inhomogeneous array crash** in `nondimensional.py`: `np.array(n)[s].tolist()` failed when `n` contained mixed types. Replaced all 4 occurrences with list comprehension `[n[i] for i in s]`

### Tests added
- `tests/conftest.py` — Shared fixtures: `simple_linear_df` (y=2x1+3x2), `simple_product_df` (y=x1*x2), `small_df` (y=x1+x2*x3)
- `tests/test_model.py` — Import tests (4), init tests (2), fit smoke tests (2)
- `tests/test_pareto.py` — Pareto front tests (4)
- `tests/test_backwards_compat.py` — Backward compatibility tests (3)

---

## Phase 2: UX & Documentation

**Goal**: Consistent API, configurable limits, input validation, scaling documentation.

### FitResult dataclass (`symantic/results.py`)
- Unified return type from `fit()` replacing 3 different tuple shapes
- Attributes: `rmse`, `equation`, `r2`, `complexity`, `pareto_front`, `all_equations`
- `__iter__` / `__len__` for backward-compatible tuple unpacking:
  - Auto-depth mode: `res, pareto_df = model.fit()` (returns utopia dict + DataFrame)
  - Fixed-depth mode: `rmse, equation, r2 = model.fit()`
  - Multi-task mode: `rmse, equation, r2, equations = model.fit()`
- All 6 return paths in `model.py` `fit()` now wrapped with `FitResult`

### Configurable `max_features` parameter
- Removed `input()` blocking calls in both `nondimensional.py` (line 1511) and `dimensional.py` (line 3119) that blocked non-interactive use
- New `max_features` parameter on `SymanticModel.__init__()` — stops auto-depth expansion with a warning instead of blocking
- Defaults: 2000 (non-dimensional), 10000 (dimensional)
- Threaded through to all 8 `feature_space_construction` calls in `model.py`

### Input validation (`symantic/validation.py`)
- `validate_dataframe()` — checks for empty DataFrame, NaN values, single-column, non-numeric columns
- `validate_operators()` — validates against supported operator set
- `validate_dimensions()` — checks dimensionality list length matches feature count
- All validators run at `SymanticModel.__init__()` time with clear error messages

### Custom exceptions (`symantic/exceptions.py`)
- `FeatureSpaceLimitError(RuntimeError)` — raised when feature space exceeds `max_features`
- `ValidationError(ValueError)` — raised on invalid input

### Scaling documentation (`docs/scaling.md`)
- Feature space growth formula: `k * (b * k)^d` where k=features, b=binary operators, d=depth
- SIS screening cost: O(n_features * n_samples)
- Combination enumeration table: C(sis_features, n_term) across parameter ranges
- Memory formula: `total_features * n_samples * 4 / 1e9` GB (float32)
- Practical recommendations by problem size (<20, 20-50, >50 features)

### Tests added
- `tests/test_results.py` — 8 tests for FitResult unpacking in all 3 modes
- `tests/test_validation.py` — 11 tests for input validation
- Updated `tests/test_model.py` — FitResult assertions, max_features checks, validation error tests

---

## Phase 3: Performance & Parallelization

**Goal**: Reduce memory usage, speed up bottlenecks.

### Pareto front: O(n^2) -> O(n log n) (`symantic/pareto.py`)
- Replaced pairwise `is_pareto_efficient()` loop (compared every point against every other point) with a sort-and-scan algorithm:
  1. Sort by first objective (complexity) ascending
  2. Scan left-to-right tracking minimum second objective (RMSE)
  3. Points with RMSE <= running minimum are Pareto-optimal
- Vectorized euclidean distance computation: replaced `[euclidean_distance(p, utopia) for p in front]` list comprehension with `torch.sqrt(torch.sum((front - utopia.unsqueeze(0)) ** 2, dim=1))`
- Added empty-input handling (previously crashed on empty tensors)

### Memory: `repeat()` -> `expand()` (`symantic/regression/l0_greedy.py`, `l0_greedy_dimensional.py`)
- Replaced `y_centered.unsqueeze(1).repeat(n_combs, 1, 1)` with `y_centered.unsqueeze(0).unsqueeze(2).expand(n_combs, -1, -1)`
- `repeat()` allocates N full copies of the target vector; `expand()` creates a zero-copy view with stride 0 on the broadcast dimension
- Applied to both the higher-dimension path (line 240) and the 1D quantile path (line 496) in both regressors
- Eliminates the single largest memory allocation in the regression pipeline

### Batched combination enumeration (`symantic/regression/l0_greedy.py`, `l0_greedy_dimensional.py`)
- `higher_dimension()` method: OLS solve now processes combinations in batches of 10,000 instead of all at once
- Each batch: fetch feature matrix, solve `torch.linalg.lstsq`, compute RMSE/R2, accumulate results
- After all batches: concatenate and run Pareto selection across combined results
- Bounds peak memory to O(batch_size * n_samples * n_terms) instead of O(C(k,n) * n_samples * n_terms)

### Pre-computed pair indices in feature expansion (`symantic/feature_expansion/nondimensional.py`)
- Hoisted `torch.combinations(arange(n), 2)`, the feature pair data fetch (`df_feature_values.T[combinations2,:]`), and the name-pair list outside the per-operator loop
- Previously: each of the 4 operators ('+', '-', '*', '/') independently computed the same O(n^2) pair indices and fetched the same feature data
- Now: computed once, reused via `.clone()` for each operator (clones needed because some operators mutate `combinations2` in-place)
- Also replaced 4 redundant `torch.combinations()` calls inside operator reference-tracking blocks with `_combinations2.clone()`

### Tests added
- `tests/test_pareto.py` — 3 new tests: correct front identification, 1000-point non-domination verification, empty input handling

---

## Bug Fix: `^N` operator crash in `single_variable()`

**Issue**: Using power operators like `'^2'` in the operators list caused a `RuntimeError: Sizes of tensors must match except in dimension 0` during the second expansion in auto-depth mode. Reported via `local_test/SyMANTIC-DT5.ipynb`.

### Root cause
Two bugs in `symantic/feature_expansion/nondimensional.py`:

1. **Missing `^N` handler in `single_variable()`**: The method had handlers for `'^-1'` (reciprocal) and `"pow(N)"` format, but no handler for `'^N'` format (e.g., `'^2'`, `'^3'`, `'^0.5'`). When `'^2'` was passed, it fell through all `if`/`elif` branches without being processed, leaving `operators_reference` as an empty 1D tensor.

2. **Wrong-direction padding in operator concat logic**: The `else` branch at the end of both `single_variable()` and `combinations()` always padded `self.operators_final` wider (adding NaN columns), but should pad whichever tensor is narrower. When `operators_reference` was narrower (1 column) than `self.operators_final` (5+ columns from binary operators), padding `self.operators_final` made the mismatch worse, causing `torch.cat` to fail.

### Fixes applied (`symantic/feature_expansion/nondimensional.py`)
- Added `^N` operator handler (after `'^-1'`): matches `op.startswith('^') and op != '^-1'`, parses exponent via `float(op[1:])`, applies `torch.pow(self.df_feature_values, exponent)` with proper operator tracking for both `i==1` and `i>=2` expansion levels
- Fixed padding logic in `single_variable()` (line ~738): now pads the narrower tensor instead of always padding `self.operators_final`
- Fixed same padding logic in `combinations()` (line ~1166): same bidirectional padding fix

### Tests added
- `tests/test_model.py` — 2 new tests:
  - `test_power_operator_fit`: `'^2'` with fixed-depth mode
  - `test_power_operator_auto_depth`: `'^2'` with auto-depth mode (reproduces the DT5 notebook crash)

---

## Test Summary

| Phase | Tests Added | Total |
|-------|------------|-------|
| Phase 1 | 15 | 15 |
| Phase 2 | 23 | 38 |
| Phase 3 | 3 | 44* |
| Bug fix (^N) | 2 | 46 |

*3 existing tests were also updated in Phase 2 with additional assertions.
