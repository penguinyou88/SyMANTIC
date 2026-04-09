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

    def test_power_operator_fit(self, small_df):
        """Test that ^2 unary operator works in fixed-depth mode.

        Regression test for tensor shape mismatch when using ^N operators.
        """
        from symantic import SymanticModel
        model = SymanticModel(
            df=small_df,
            operators=['+', '*', '^2'],
            n_expansion=2,
            n_term=2,
            sis_features=5,
        )
        result = model.fit()
        assert result is not None
        assert isinstance(result, FitResult)
        rmse, equation, r2 = result
        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_power_operator_auto_depth(self, simple_linear_df):
        """Test that ^2 unary operator works in auto-depth mode.

        Regression test for the DT5 notebook crash.
        """
        from symantic import SymanticModel
        model = SymanticModel(
            df=simple_linear_df,
            operators=['+', '-', '*', '/', '^2'],
            n_expansion=None,
            n_term=2,
            sis_features=5,
            metrics=[0.5, 0.9],
        )
        result = model.fit()
        assert isinstance(result, FitResult)
        res, full_pareto = result
        assert 'utopia' in res


class TestLevelPruning:
    """Tests for inter-level feature pruning in auto-depth mode."""

    def test_level_pruning_auto_depth(self, simple_linear_df):
        """level_pruning=True should complete without error in auto-depth."""
        from symantic import SymanticModel
        model = SymanticModel(
            df=simple_linear_df,
            operators=['+', '-', '*', '/'],
            n_expansion=None,
            n_term=2,
            sis_features=5,
            metrics=[0.5, 0.9],
            level_pruning=True,
        )
        result = model.fit()
        assert isinstance(result, FitResult)
        assert result.r2 > 0.5

    def test_level_pruning_with_l1(self, simple_linear_df):
        """level_pruning + L1 regularization in auto-depth mode."""
        from symantic import SymanticModel
        model = SymanticModel(
            df=simple_linear_df,
            operators=['+', '-', '*', '/'],
            n_expansion=None,
            n_term=2,
            sis_features=5,
            metrics=[0.5, 0.9],
            regularization='l1',
            level_pruning=True,
        )
        result = model.fit()
        assert isinstance(result, FitResult)
        assert result.r2 > 0.5

    def test_level_pruning_default_off(self, small_df):
        """level_pruning defaults to False."""
        from symantic import SymanticModel
        model = SymanticModel(df=small_df, operators=['+', '*'])
        assert model.level_pruning is False

    def test_level_pruning_fixed_depth_ignored(self, small_df):
        """level_pruning has no effect in fixed-depth mode (no crash)."""
        from symantic import SymanticModel
        model = SymanticModel(
            df=small_df,
            operators=['+', '*'],
            n_expansion=2,
            n_term=2,
            sis_features=5,
            level_pruning=True,
        )
        result = model.fit()
        assert isinstance(result, FitResult)


class TestEvaluate:
    """Tests for the evaluate() method."""

    def test_evaluate_simple_equation(self, simple_linear_df):
        """Evaluate a simple linear equation on test data."""
        from symantic import SymanticModel
        model = SymanticModel(df=simple_linear_df, operators=['+', '-'])
        df_test = simple_linear_df.copy()
        # y = 2*x1 + 3*x2
        preds, eq = model.evaluate("2*x1 + 3*x2", df_test)
        assert preds is not None
        expected = 2 * df_test['x1'] + 3 * df_test['x2']
        np.testing.assert_allclose(preds, expected, rtol=1e-10)

    def test_evaluate_with_power(self, simple_linear_df):
        """Evaluate equation with ^ (caret) exponentiation."""
        from symantic import SymanticModel
        model = SymanticModel(df=simple_linear_df, operators=['+', '-'])
        df_test = simple_linear_df.copy()
        preds, eq = model.evaluate("x1^2 + x2", df_test)
        assert preds is not None
        expected = df_test['x1'] ** 2 + df_test['x2']
        np.testing.assert_allclose(preds, expected, rtol=1e-10)
        assert '**' in eq  # ^ should be converted to **

    def test_evaluate_with_numpy_functions(self):
        """Evaluate equation containing math functions (exp, sin, log)."""
        from symantic import SymanticModel
        df = pd.DataFrame({'y': [1, 2, 3], 'x': [0.1, 0.5, 1.0]})
        model = SymanticModel(df=df, operators=['+', 'exp'])
        preds, eq = model.evaluate("exp(x) + sin(x)", df)
        assert preds is not None
        expected = np.exp(df['x']) + np.sin(df['x'])
        np.testing.assert_allclose(preds, expected, rtol=1e-10)

    def test_evaluate_invalid_equation(self, simple_linear_df):
        """Invalid equation returns None without crashing."""
        from symantic import SymanticModel
        model = SymanticModel(df=simple_linear_df, operators=['+'])
        preds, eq = model.evaluate("undefined_var + 1", simple_linear_df)
        assert preds is None

    def test_evaluate_custom_functions(self):
        """Evaluate with user-provided custom functions."""
        from symantic import SymanticModel
        df = pd.DataFrame({'y': [1, 4, 9], 'x': [1, 2, 3]})
        model = SymanticModel(df=df, operators=['+'])
        preds, eq = model.evaluate(
            "my_square(x)",
            df,
            custom_functions={'my_square': lambda v: v ** 2},
        )
        assert preds is not None
        np.testing.assert_allclose(preds, df['x'] ** 2, rtol=1e-10)
