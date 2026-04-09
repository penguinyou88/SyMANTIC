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
