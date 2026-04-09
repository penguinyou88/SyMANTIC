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

## Phase 4: L1/L2/ElasticNet Regularization

**Goal**: Replace the combinatorial L0 search with penalized regression for faster fitting, especially at higher n_term values.

### Motivation
The L0 greedy regressor enumerates all C(k, n_terms) feature combinations, solving a separate OLS for each. For 4 terms with sis_features=20, this produces C(20,4)*10 bins = ~48,450 OLS solves. L1/ElasticNet solve ONE penalized regression and sweep the alpha path for a Pareto front — ~100 coordinate descent iterations regardless of n_terms.

### New files

#### `symantic/regression/penalized.py` — Non-dimensional penalized regressor
- `PenalizedRegressor` class handling L1 (Lasso), L2 (Ridge), and ElasticNet via sklearn
- Algorithm: SIS screening -> numpy conversion -> regularization path sweep -> Pareto front extraction -> coefficient denormalization -> 9-tuple return
- Key methods: `_sis_screening()`, `_solve_path()`, `_path_to_pareto()`, `_format_equation()`, `regressor_fit()`
- L1/ElasticNet: `sklearn.linear_model.lasso_path()` / `enet_path()` with warm starts
- L2: `sklearn.linear_model.Ridge` over logspace alphas with coefficient thresholding
- Deduplicates solutions with same sparsity pattern (keeps lowest RMSE)
- Returns same 9-tuple as L0 regressor for full API compatibility

#### `symantic/regression/penalized_dimensional.py` — Dimensional penalized regressor
- Mirrors `penalized.py` but adds dimensional filtering before regression
- Filters features to those matching `output_dim` (replicates `l0_greedy_dimensional.get_dimensions_list()` logic)

#### `symantic/regression/factory.py` — Regressor routing
- `get_regressor(regularization='l0', dimensional=False)` returns the appropriate Regressor class
- L0 -> existing `Regressor` classes; L1/L2/ElasticNet -> `PenalizedRegressor` variants

### Files modified

#### `symantic/model.py`
- New parameters: `regularization='l0'`, `reg_alpha=None`, `l1_ratio=0.5`, `reg_threshold=1e-4`, `n_alphas=100`
- GPU auto-detection: `device=None` auto-selects 'cuda' when available, otherwise 'cpu'
- All 4 fixed-depth Regressor call sites updated to use factory pattern
- All 4 auto-depth `feature_space_construction()` calls thread `**self._reg_kwargs`
- `validate_regularization()` called at init time

#### `symantic/feature_expansion/nondimensional.py`
- Added regularization params to `feature_space_construction.__init__()`
- Replaced 2 internal `Regressor(...)` calls with factory-based routing via `get_regressor()`
- Changed import from `l0_greedy.Regressor` to `factory.get_regressor`

#### `symantic/feature_expansion/dimensional.py`
- Same pattern: added reg params to `__init__()`, replaced 2 `Regressor(...)` calls with factory

#### `symantic/regression/l0_greedy.py` + `l0_greedy_dimensional.py`
- Added `**kwargs` to `__init__` signatures to accept and ignore new regularization params when selected via factory

#### `symantic/validation.py`
- Added `validate_regularization()`: validates regularization type, reg_alpha >= 0, l1_ratio in (0, 1]
- Added `SUPPORTED_REGULARIZATIONS` frozenset

#### `symantic/regression/__init__.py` + `symantic/__init__.py`
- Added exports: `PenalizedRegressor`, `DimensionalPenalizedRegressor`, `get_regressor`

### New API usage
```python
from symantic import SymanticModel

# L1 (Lasso) — fast even at n_term=4+
model = SymanticModel(df, operators=['+','-','*','/'],
                      n_expansion=2, n_term=4,
                      regularization='l1')
result = model.fit()

# ElasticNet with custom mixing
model = SymanticModel(df, operators=['+','-','*','/'],
                      regularization='elastic_net', l1_ratio=0.7)

# L2 (Ridge) with coefficient thresholding
model = SymanticModel(df, operators=['+','-','*','/'],
                      regularization='l2', reg_threshold=1e-3)

# Default (L0) unchanged
model = SymanticModel(df, operators=['+','-','*','/'])  # regularization='l0'
```

### Complexity comparison

| n_term | L0 (sis=20) | L1/ElasticNet |
|--------|-------------|---------------|
| 2 | 1,900 OLS | ~100 coord descent |
| 3 | 11,400 OLS | ~100 coord descent |
| 4 | 48,450 OLS | ~100 coord descent |
| 5 | 155,040 OLS | ~100 coord descent |

### Tests added
- `tests/test_penalized_regression.py` — 28 tests:
  - L1/L2/ElasticNet unit tests: return signature, fit quality, equation content
  - Edge cases: single feature, high alpha (all-zero coefficients), Pareto front populated
  - Factory routing: 6 tests for all regularization/dimensional combinations
  - Validation: 7 tests for `validate_regularization()`
  - Integration: 5 tests for `SymanticModel.fit()` with L1/L2/ElasticNet + L0 unchanged + invalid rejected

---

## Test Summary

| Phase | Tests Added | Total |
|-------|------------|-------|
| Phase 1 | 15 | 15 |
| Phase 2 | 23 | 38 |
| Phase 3 | 3 | 44* |
| Bug fix (^N) | 2 | 46 |
| Phase 4 | 28 | 77 |

*3 existing tests were also updated in Phase 2 with additional assertions.
