"""Tests for the SymanticModel class."""

import pytest
import numpy as np
import pandas as pd


class TestSymanticModelImport:
    """Test that the package can be imported correctly."""

    def test_import_symantic_model(self):
        from symantic import SymanticModel
        assert SymanticModel is not None

    def test_import_pareto(self):
        from symantic import pareto
        assert pareto is not None

    def test_import_qualified_names(self):
        from symantic import NonDimensionalFeatureExpander
        from symantic import DimensionalFeatureExpander
        from symantic import NonDimensionalRegressor
        from symantic import DimensionalRegressor
        assert NonDimensionalFeatureExpander is not None
        assert DimensionalFeatureExpander is not None
        assert NonDimensionalRegressor is not None
        assert DimensionalRegressor is not None

    def test_no_namespace_collision(self):
        """Verify that qualified imports are distinct classes."""
        from symantic import NonDimensionalRegressor, DimensionalRegressor
        # These should be different classes from different modules
        assert NonDimensionalRegressor is not DimensionalRegressor


class TestSymanticModelInit:
    """Test SymanticModel initialization."""

    def test_basic_init(self, small_df):
        from symantic import SymanticModel
        model = SymanticModel(
            df=small_df,
            operators=['+', '*'],
            n_expansion=1,
            n_term=2,
            sis_features=5,
        )
        assert model is not None
        assert model.dimension == 2
        assert model.sis_features == 5

    def test_default_params(self, small_df):
        from symantic import SymanticModel
        model = SymanticModel(
            df=small_df,
            operators=['+', '*'],
        )
        # Default n_term=None means dimension=3
        assert model.dimension == 3
        assert model.sis_features == 20


class TestSymanticModelFit:
    """Smoke tests for model fitting."""

    def test_fixed_depth_fit(self, small_df):
        """Test that fit() with fixed n_expansion runs without error.

        Note: n_expansion uses range(1, n_expansion), so n_expansion=2
        means 1 level of expansion.
        """
        from symantic import SymanticModel
        model = SymanticModel(
            df=small_df,
            operators=['+', '*'],
            n_expansion=2,
            n_term=2,
            sis_features=5,
        )
        result = model.fit()
        assert result is not None
        # Fixed depth returns (rmse, equation, r2)
        rmse, equation, r2 = result
        assert isinstance(rmse, float)
        assert isinstance(equation, str)
        assert isinstance(r2, float)
        assert rmse >= 0
        assert r2 <= 1.0

    def test_auto_depth_fit(self, simple_linear_df):
        """Test auto-depth mode (n_expansion=None) with easy linear problem.

        Uses a simple y = 2*x1 + 3*x2 problem with relaxed metrics
        so it converges quickly in the first expansion.
        """
        from symantic import SymanticModel
        model = SymanticModel(
            df=simple_linear_df,
            operators=['+', '-'],
            n_expansion=None,
            n_term=2,
            sis_features=5,
            metrics=[0.5, 0.9],  # Relaxed thresholds for fast convergence
        )
        res, full_pareto = model.fit()
        assert 'utopia' in res
        assert 'expression' in res['utopia']
        assert 'rmse' in res['utopia']
        assert 'r2' in res['utopia']
        assert isinstance(full_pareto, pd.DataFrame)
        assert 'Equation' in full_pareto.columns
