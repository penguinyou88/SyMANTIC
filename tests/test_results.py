"""Tests for FitResult dataclass."""

import pytest
import pandas as pd

from symantic.results import FitResult


class TestFitResultFixedDepth:
    """Test FitResult in fixed-depth mode (no pareto_front)."""

    def test_unpacking_three(self):
        result = FitResult(rmse=0.1, equation="x1 + x2", r2=0.95)
        rmse, equation, r2 = result
        assert rmse == 0.1
        assert equation == "x1 + x2"
        assert r2 == 0.95

    def test_len_three(self):
        result = FitResult(rmse=0.1, equation="x1 + x2", r2=0.95)
        assert len(result) == 3

    def test_attributes(self):
        result = FitResult(rmse=0.1, equation="x1 + x2", r2=0.95)
        assert result.rmse == 0.1
        assert result.equation == "x1 + x2"
        assert result.r2 == 0.95
        assert result.pareto_front is None
        assert result.all_equations is None


class TestFitResultAutoDepth:
    """Test FitResult in auto-depth mode (pareto_front is set)."""

    @pytest.fixture
    def auto_depth_result(self):
        pareto_df = pd.DataFrame({
            'Loss': [0.1, 0.2],
            'Complexity': [3.0, 1.0],
            'R2': [0.95, 0.80],
            'Equation': ["x1 + x2", "x1"],
        })
        return FitResult(
            rmse=0.1,
            equation="x1 + x2",
            r2=0.95,
            complexity=3.0,
            pareto_front=pareto_df,
        )

    def test_unpacking_two(self, auto_depth_result):
        res, pareto_df = auto_depth_result
        assert 'utopia' in res
        assert 'expression' in res['utopia']
        assert 'rmse' in res['utopia']
        assert 'r2' in res['utopia']
        assert 'complexity' in res['utopia']
        assert isinstance(pareto_df, pd.DataFrame)

    def test_len_two(self, auto_depth_result):
        assert len(auto_depth_result) == 2

    def test_utopia_values(self, auto_depth_result):
        res, _ = auto_depth_result
        assert res['utopia']['expression'] == "x1 + x2"
        assert res['utopia']['rmse'] == 0.1
        assert res['utopia']['r2'] == 0.95
        assert res['utopia']['complexity'] == 3.0


class TestFitResultMultiTask:
    """Test FitResult in multi-task mode (all_equations is set)."""

    def test_unpacking_four(self):
        result = FitResult(
            rmse=0.1, equation="x1 + x2", r2=0.95,
            all_equations=["x1 + x2", "x3 * x4"],
        )
        rmse, equation, r2, equations = result
        assert rmse == 0.1
        assert equation == "x1 + x2"
        assert r2 == 0.95
        assert equations == ["x1 + x2", "x3 * x4"]

    def test_len_four(self):
        result = FitResult(
            rmse=0.1, equation="x1 + x2", r2=0.95,
            all_equations=["x1 + x2", "x3 * x4"],
        )
        assert len(result) == 4
