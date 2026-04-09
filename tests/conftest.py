"""Shared test fixtures for SyMANTIC tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_linear_df():
    """DataFrame where y = 2*x1 + 3*x2 (easy to discover)."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.01
    data = np.column_stack((y, x1, x2))
    return pd.DataFrame(data, columns=["y", "x1", "x2"])


@pytest.fixture
def simple_product_df():
    """DataFrame where y = x1 * x2 (requires feature expansion)."""
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(0.5, 3.0, n)
    x2 = np.random.uniform(0.5, 3.0, n)
    y = x1 * x2 + np.random.randn(n) * 0.01
    data = np.column_stack((y, x1, x2))
    return pd.DataFrame(data, columns=["y", "x1", "x2"])


@pytest.fixture
def small_df():
    """Minimal DataFrame for fast smoke tests."""
    np.random.seed(42)
    n = 30
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    y = x1 + x2 * x3
    data = np.column_stack((y, x1, x2, x3))
    return pd.DataFrame(data, columns=["y", "x1", "x2", "x3"])
