"""Tests for the SymanticModel class."""

import pytest
import numpy as np
import pandas as pd

from symantic.results import FitResult
from symantic.exceptions import ValidationError


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

    def test_import_new_classes(self):
        from symantic import FitResult, FeatureSpaceLimitError, ValidationError
        assert FitResult is not None
        assert FeatureSpaceLimitError is not None
        assert ValidationError is not None

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

    def test_max_features_default_nondim(self, small_df):
        from symantic import SymanticModel
        model = SymanticModel(df=small_df, operators=['+', '*'])
        assert model.max_features == 2000

    def test_max_features_custom(self, small_df):
        from symantic import SymanticModel
        model = SymanticModel(df=small_df, operators=['+', '*'], max_features=5000)
        assert model.max_features == 5000

    def test_validation_empty_df(self):
        from symantic import SymanticModel
        with pytest.raises(ValidationError, match="empty"):
            SymanticModel(df=pd.DataFrame(), operators=['+'])

    def test_validation_bad_operators(self, small_df):
        from symantic import SymanticModel
        with pytest.raises(ValidationError, match="Unsupported"):
            SymanticModel(df=small_df, operators=['modulo'])


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
        assert isinstance(result, FitResult)
        # Backward-compatible unpacking
        rmse, equation, r2 = result
        assert isinstance(rmse, float)
        assert isinstance(equation, str)
        assert isinstance(r2, float)
        assert rmse >= 0
        assert r2 <= 1.0
        # Also accessible via attributes
        assert result.rmse == rmse
        assert result.equation == equation
        assert result.r2 == r2
        assert result.pareto_front is None

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
        result = model.fit()
        assert isinstance(result, FitResult)
        # Backward-compatible unpacking
        res, full_pareto = result
        assert 'utopia' in res
        assert 'expression' in res['utopia']
        assert 'rmse' in res['utopia']
        assert 'r2' in res['utopia']
        assert isinstance(full_pareto, pd.DataFrame)
        assert 'Equation' in full_pareto.columns
        # Also accessible via attributes
        assert result.pareto_front is not None
        assert result.complexity is not None
