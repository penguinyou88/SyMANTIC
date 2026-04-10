# Computational Scaling Guide

This document describes how SyMANTIC's search space and runtime scale with its key parameters, helping you choose settings for your problem size.

## Pipeline Overview

SyMANTIC's symbolic regression pipeline has four stages:

1. **Feature Expansion** — combinatorially generates candidate features from input variables
2. **SIS Screening** — ranks features by correlation with the target, keeps top-k
3. **Combination Enumeration** — enumerates all C(k, n_terms) subsets of screened features
4. **OLS Solve** — fits each combination via least-squares, evaluates loss

In **auto-depth mode** (`n_expansion=None`), stages 1-4 repeat with increasing expansion depth until RMSE/R2 thresholds are met or `max_features` is reached.

---

## Feature Space Growth

Given `k` initial features and `b` binary operators (`+`, `-`, `*`, `/`):

| Expansion depth | Approximate feature count |
|-----------------|--------------------------|
| 0 (original)    | k                        |
| 1               | k + b × C(k, 2) ≈ k + b×k²/2 |
| 2               | ~k × (b×k)²             |
| d               | ~k × (b×k)^d            |

**Example**: 10 features, 4 operators → depth 1 yields ~10 + 4×45 = 190 features. Depth 2 yields ~10 × (40)² = 16,000 features.

Unary operators (`exp`, `log`, `sqrt`, `sin`, etc.) add k new features per operator per expansion level.

### Controlling feature growth

| Parameter | Effect |
|-----------|--------|
| `n_expansion` | Fixed depth. `n_expansion=2` → 1 level, `n_expansion=3` → 2 levels. `None` → auto-depth. |
| `max_features` | Stops auto-depth expansion when feature count exceeds this limit. Default: 2000 (non-dimensional), 10000 (dimensional). |
| `initial_screening` | `(n_features, quantile)` — pre-screens input features before expansion. Dramatically reduces combinatorial blowup. Recommended for >50 features. |
| `operators` | Fewer operators → slower growth. Start with `['+', '*']`, add more if needed. |

---

## SIS Screening

**Cost**: O(n_features × n_samples) — computes correlation between each candidate feature and target.

| Parameter | Effect |
|-----------|--------|
| `sis_features` | Number of features retained after screening (default 20). Higher values explore more but increase combination enumeration cost. |

**Guideline**: `sis_features` should be at least 2× `n_term` to give the enumerator enough candidates.

---

## Combination Enumeration

After SIS selects `k = sis_features` features, the regressor enumerates all C(k, n_terms) subsets.

| sis_features | n_term=2 | n_term=3 | n_term=4 |
|-------------|----------|----------|----------|
| 10          | 45       | 120      | 210      |
| 20          | 190      | 1,140    | 4,845    |
| 50          | 1,225    | 19,600   | 230,300  |
| 100         | 4,950    | 161,700  | 3,921,225|

The regressor partitions features into 10 quantile bins by complexity, so the effective enumeration per bin is C(k/10, n_terms).

**Cost per combination**: O(n_samples × n_terms²) for the OLS solve via `torch.linalg.lstsq`.

**Total regression cost**: ~10 × C(k/10, n_terms) × O(n_samples × n_terms²)

---

## Memory

The main memory consumers are:

1. **Feature matrix**: O(n_features × n_samples) — the expanded feature tensor
2. **Combination batch**: O(batch_size × n_samples × n_terms) — the regression batch

The `max_features` parameter directly bounds (1). For very large problems, reduce `sis_features` to control (2).

---

## Practical Recommendations

| Problem size | Recommendations |
|-------------|----------------|
| <20 features, <1000 samples | Default settings work well. `n_expansion=3`, `sis_features=20` |
| 20-50 features | Use `initial_screening=(20, 0.5)` to pre-screen. Consider `max_features=5000` |
| >50 features | `initial_screening` is essential. Start with `operators=['+', '*']`, add more if R2 is poor |
| >3 expansion levels | Increase `max_features` (e.g. 50000) and ensure sufficient RAM (>16GB). Use GPU (`device='cuda'`) if available |
| >10000 samples | SIS screening and OLS are linear in n_samples, so this scales well. Main bottleneck is feature count |

### Quick scaling formula

```
Total features ≈ k × (b × k)^d
Total combinations ≈ 10 × C(sis_features/10, n_term)
Memory (GB) ≈ total_features × n_samples × 4 / 1e9  (float32)
```

Where `k` = initial features, `b` = number of binary operators, `d` = expansion depth.
