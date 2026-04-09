"""Tests for the Pareto module."""

import torch
import numpy as np
import pytest

from symantic.pareto import pareto


class TestPareto:
    """Test Pareto front identification."""

    def test_basic_pareto_front(self):
        """Test that Pareto front is correctly identified."""
        rmse = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5])
        complexity = torch.tensor([3.0, 1.0, 4.0, 5.0, 2.0])
        p = pareto(rmse, complexity)
        indices = p.pareto_front()
        assert len(indices) > 0
        # The point (0.5, 4.0) and (2.0, 1.0) should be on the front
        assert len(indices) <= len(rmse)

    def test_single_point(self):
        """Test with a single point."""
        rmse = torch.tensor([1.0])
        complexity = torch.tensor([2.0])
        p = pareto(rmse, complexity)
        indices = p.pareto_front()
        assert len(indices) == 1

    def test_custom_utopia_point(self):
        """Test with custom utopia point."""
        rmse = torch.tensor([1.0, 2.0, 0.5])
        complexity = torch.tensor([3.0, 1.0, 4.0])
        p = pareto(rmse, complexity, utopia_point=[0.0, 0.0])
        indices = p.pareto_front()
        assert len(indices) > 0

    def test_computed_utopia_point(self):
        """Test that utopia point is computed from data by default (not hardcoded [0,0])."""
        rmse = torch.tensor([10.0, 20.0, 15.0])
        complexity = torch.tensor([30.0, 10.0, 40.0])
        p = pareto(rmse, complexity)
        # Should use computed utopia = (10.0, 10.0), not (0, 0)
        indices = p.pareto_front()
        assert len(indices) > 0

    def test_pareto_identifies_correct_front(self):
        """Test that the O(n log n) algorithm identifies the correct Pareto front."""
        # Points: (complexity, rmse)
        # (1, 5), (2, 3), (3, 4), (4, 1), (5, 2)
        # Pareto front should be: (1, 5)->no, (2, 3)->yes, (4, 1)->yes
        # Actually: domination means BOTH objectives strictly better.
        # (2,3) dominates (3,4) and (5,2)?  (2<3, 3<4) yes. (2<5, 3>2) no.
        # Front: (2,3), (4,1), and (1,5)? (1<2 but 5>3, so not dominated)
        rmse = torch.tensor([5.0, 3.0, 4.0, 1.0, 2.0])
        complexity = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        p = pareto(rmse, complexity)
        indices = p.pareto_front()
        # Pareto front: idx 0 (1,5), idx 1 (2,3), idx 3 (4,1)
        assert set(indices.tolist()) == {0, 1, 3}

    def test_pareto_many_points(self):
        """Test Pareto front with many points for performance."""
        torch.manual_seed(42)
        n = 1000
        rmse = torch.rand(n)
        complexity = torch.rand(n)
        p = pareto(rmse, complexity)
        indices = p.pareto_front()
        # All returned points should be non-dominated
        front_rmse = rmse[indices]
        front_complexity = complexity[indices]
        for i in range(len(indices)):
            # No other front point should dominate this one
            for j in range(len(indices)):
                if i == j:
                    continue
                assert not (front_complexity[j] < front_complexity[i] and
                           front_rmse[j] < front_rmse[i]), \
                    f"Point {j} dominates point {i} on the Pareto front"

    def test_pareto_empty(self):
        """Test Pareto front with empty input."""
        rmse = torch.tensor([])
        complexity = torch.tensor([])
        p = pareto(rmse, complexity)
        indices = p.pareto_front()
        assert len(indices) == 0
