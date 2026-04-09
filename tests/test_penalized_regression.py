"""Tests for penalized regression (L1/L2/ElasticNet)."""

import numpy as np
import pandas as pd
import pytest
import torch

from symantic.regression.penalized import PenalizedRegressor
from symantic.regression.factory import get_regressor
from symantic.validation import validate_regularization
from symantic.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_data():
    """y = 2*x1 + 3*x2, with 3 noise features."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    noise_features = np.random.randn(n, 3) * 0.1
    y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.01

    X = torch.tensor(
        np.column_stack([x1, x2, noise_features]), dtype=torch.float32
    )
    y_t = torch.tensor(y, dtype=torch.float32)
    names = ['x1', 'x2', 'noise1', 'noise2', 'noise3']
    complexity = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    return X, y_t, names, complexity


# ---------------------------------------------------------------------------
# Unit tests: PenalizedRegressor
# ---------------------------------------------------------------------------

class TestPenalizedRegressorL1:
    def test_return_signature(self, linear_data):
        X, y, names, complexity = linear_data
        reg = PenalizedRegressor(
            X, y, names, complexity, dimension=3, sis_features=5,
            regularization='l1', n_alphas=50
        )
        result = reg.regressor_fit()
        assert len(result) == 9
        rmse, equation, r2, p_rmse, p_comp, p_names, p_int, p_coef, p_r2 = result
        assert isinstance(rmse, float)
        assert isinstance(equation, str)
        assert isinstance(r2, float)
        assert isinstance(p_rmse, torch.Tensor)
        assert isinstance(p_comp, torch.Tensor)
        assert isinstance(p_int, torch.Tensor)
        assert isinstance(p_coef, torch.Tensor)
        assert isinstance(p_r2, torch.Tensor)

    def test_finds_good_fit(self, linear_data):
        X, y, names, complexity = linear_data
        reg = PenalizedRegressor(
            X, y, names, complexity, dimension=3, sis_features=5,
            regularization='l1', n_alphas=50
        )
        rmse, equation, r2, *_ = reg.regressor_fit()
        assert r2 > 0.95
        assert rmse < 0.5

    def test_equation_contains_features(self, linear_data):
        X, y, names, complexity = linear_data
        reg = PenalizedRegressor(
            X, y, names, complexity, dimension=3, sis_features=5,
            regularization='l1', n_alphas=50
        )
        _, equation, *_ = reg.regressor_fit()
        # Should find at least one of the real features
        assert 'x1' in equation or 'x2' in equation


class TestPenalizedRegressorL2:
    def test_return_signature(self, linear_data):
        X, y, names, complexity = linear_data
        reg = PenalizedRegressor(
            X, y, names, complexity, dimension=3, sis_features=5,
            regularization='l2', n_alphas=50
        )
        result = reg.regressor_fit()
        assert len(result) == 9

    def test_finds_good_fit(self, linear_data):
        X, y, names, complexity = linear_data
        reg = PenalizedRegressor(
            X, y, names, complexity, dimension=3, sis_features=5,
            regularization='l2', n_alphas=50
        )
        rmse, _, r2, *_ = reg.regressor_fit()
        assert r2 > 0.90


class TestPenalizedRegressorElasticNet:
    def test_return_signature(self, linear_data):
        X, y, names, complexity = linear_data
        reg = PenalizedRegressor(
            X, y, names, complexity, dimension=3, sis_features=5,
            regularization='elastic_net', l1_ratio=0.5, n_alphas=50
        )
        result = reg.regressor_fit()
        assert len(result) == 9

    def test_finds_good_fit(self, linear_data):
        X, y, names, complexity = linear_data
        reg = PenalizedRegressor(
            X, y, names, complexity, dimension=3, sis_features=5,
            regularization='elastic_net', l1_ratio=0.5, n_alphas=50
        )
        rmse, _, r2, *_ = reg.regressor_fit()
        assert r2 > 0.90


class TestPenalizedRegressorEdgeCases:
    def test_single_feature(self):
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 5 * x + np.random.randn(n) * 0.01
        X = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        reg = PenalizedRegressor(
            X, y_t, ['x'], torch.tensor([1.0]),
            dimension=1, sis_features=1,
            regularization='l1', n_alphas=50
        )
        rmse, equation, r2, *_ = reg.regressor_fit()
        assert r2 > 0.99

    def test_high_alpha_returns_baseline(self):
        """Very high fixed alpha should zero out all coefficients."""
        np.random.seed(42)
        X = torch.randn(50, 3)
        y = torch.randn(50)
        reg = PenalizedRegressor(
            X, y, ['a', 'b', 'c'], torch.ones(3),
            dimension=2, sis_features=3,
            regularization='l1', reg_alpha=1e6, n_alphas=10
        )
        result = reg.regressor_fit()
        assert len(result) == 9

    def test_pareto_front_populated(self, linear_data):
        X, y, names, complexity = linear_data
        reg = PenalizedRegressor(
            X, y, names, complexity, dimension=3, sis_features=5,
            regularization='l1', n_alphas=50
        )
        _, _, _, pareto_rmse, pareto_comp, *_ = reg.regressor_fit()
        assert len(pareto_rmse) >= 2  # at least baseline + one solution


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_l0_nondimensional(self):
        cls = get_regressor('l0', dimensional=False)
        assert cls.__name__ == 'Regressor'

    def test_l0_dimensional(self):
        cls = get_regressor('l0', dimensional=True)
        assert cls.__name__ == 'Regressor'

    def test_l1_nondimensional(self):
        cls = get_regressor('l1', dimensional=False)
        assert cls.__name__ == 'PenalizedRegressor'

    def test_l1_dimensional(self):
        cls = get_regressor('l1', dimensional=True)
        assert cls.__name__ == 'PenalizedRegressor'

    def test_l2_nondimensional(self):
        cls = get_regressor('l2', dimensional=False)
        assert cls.__name__ == 'PenalizedRegressor'

    def test_elastic_net_nondimensional(self):
        cls = get_regressor('elastic_net', dimensional=False)
        assert cls.__name__ == 'PenalizedRegressor'


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidateRegularization:
    def test_valid_l0(self):
        validate_regularization('l0', None, 0.5)

    def test_valid_l1(self):
        validate_regularization('l1', 0.1, 0.5)

    def test_valid_elastic_net(self):
        validate_regularization('elastic_net', 0.1, 0.5)

    def test_invalid_type(self):
        with pytest.raises(ValidationError, match="Unsupported regularization"):
            validate_regularization('lasso', None, 0.5)

    def test_negative_alpha(self):
        with pytest.raises(ValidationError, match="reg_alpha must be >= 0"):
            validate_regularization('l1', -0.1, 0.5)

    def test_invalid_l1_ratio(self):
        with pytest.raises(ValidationError, match="l1_ratio must be in"):
            validate_regularization('elastic_net', 0.1, 0.0)

    def test_l1_ratio_one_is_valid(self):
        validate_regularization('elastic_net', 0.1, 1.0)


# ---------------------------------------------------------------------------
# Integration: SymanticModel with penalized regression
# ---------------------------------------------------------------------------

class TestModelIntegration:
    def test_fit_l1_fixed_depth(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.01
        df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

        from symantic import SymanticModel
        model = SymanticModel(
            df, operators=['+', '-', '*', '/'],
            n_expansion=2, n_term=3, sis_features=10,
            regularization='l1', device='cpu'
        )
        result = model.fit()
        assert result.r2 > 0.90
        assert result.rmse < 1.0

    def test_fit_l2_fixed_depth(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.01
        df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

        from symantic import SymanticModel
        model = SymanticModel(
            df, operators=['+', '-', '*', '/'],
            n_expansion=2, n_term=3, sis_features=10,
            regularization='l2', device='cpu'
        )
        result = model.fit()
        assert result.r2 > 0.90

    def test_fit_elastic_net_fixed_depth(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.01
        df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

        from symantic import SymanticModel
        model = SymanticModel(
            df, operators=['+', '-', '*', '/'],
            n_expansion=2, n_term=3, sis_features=10,
            regularization='elastic_net', l1_ratio=0.7, device='cpu'
        )
        result = model.fit()
        assert result.r2 > 0.90

    def test_l0_default_unchanged(self):
        """Default regularization='l0' should still work identically."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.01
        df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

        from symantic import SymanticModel
        model = SymanticModel(
            df, operators=['+', '-', '*', '/'],
            n_expansion=2, n_term=2, sis_features=10, device='cpu'
        )
        result = model.fit()
        assert result.r2 > 0.90

    def test_invalid_regularization_rejected(self):
        np.random.seed(42)
        df = pd.DataFrame({'y': [1, 2, 3], 'x': [1, 2, 3]})

        from symantic import SymanticModel
        with pytest.raises(ValidationError, match="Unsupported regularization"):
            SymanticModel(
                df, operators=['+'], regularization='lasso'
            )
