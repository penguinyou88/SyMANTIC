"""Tests for input validation."""

import pytest
import numpy as np
import pandas as pd

from symantic.validation import (
    validate_dataframe,
    validate_operators,
    validate_dimensions,
)
from symantic.exceptions import ValidationError


class TestValidateDataframe:
    """Test validate_dataframe()."""

    def test_valid_df(self, small_df):
        validate_dataframe(small_df)  # should not raise

    def test_not_a_dataframe(self):
        with pytest.raises(ValidationError, match="Expected a pandas DataFrame"):
            validate_dataframe([[1, 2], [3, 4]])

    def test_empty_df(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_dataframe(pd.DataFrame())

    def test_single_column(self):
        df = pd.DataFrame({"y": [1, 2, 3]})
        with pytest.raises(ValidationError, match="at least 2 columns"):
            validate_dataframe(df)

    def test_nan_values(self):
        df = pd.DataFrame({"y": [1, np.nan, 3], "x": [4, 5, 6]})
        with pytest.raises(ValidationError, match="NaN"):
            validate_dataframe(df)

    def test_non_numeric_column(self):
        df = pd.DataFrame({"y": [1, 2, 3], "x": ["a", "b", "c"]})
        with pytest.raises(ValidationError, match="Non-numeric"):
            validate_dataframe(df)


class TestValidateOperators:
    """Test validate_operators()."""

    def test_valid_operators(self):
        validate_operators(['+', '-', '*', '/'])  # should not raise

    def test_empty_operators(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_operators([])

    def test_not_a_list(self):
        with pytest.raises(ValidationError, match="Expected a list"):
            validate_operators("+")

    def test_unsupported_operator(self):
        with pytest.raises(ValidationError, match="Unsupported"):
            validate_operators(['+', 'modulo'])


class TestValidateDimensions:
    """Test validate_dimensions()."""

    def test_none_dimensionality(self, small_df):
        validate_dimensions(None, small_df)  # should not raise

    def test_wrong_length(self, small_df):
        # small_df has 3 feature columns, so dimensionality should have 3 entries
        with pytest.raises(ValidationError, match="must match"):
            validate_dimensions(["dim1", "dim2"], small_df)

    def test_correct_length(self, small_df):
        validate_dimensions(["dim1", "dim2", "dim3"], small_df)  # should not raise
