"""Input validation for SyMANTIC."""

import numpy as np
import pandas as pd

from .exceptions import ValidationError

# Operators supported by the feature expansion modules.
# Static operators matched exactly.
SUPPORTED_OPERATORS = frozenset([
    '+', '-', '*', '/',
    'exp', 'exp(-1)', 'ln', 'log', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'tanh',
    '^-1', '+1', '-1', '/2',
])


def _is_dynamic_operator(op: str) -> bool:
    """Check if op matches a dynamic pattern like pow(N) or ^N."""
    import re
    # ^N pattern: e.g. ^2, ^3, ^0.5, ^-2
    if re.fullmatch(r'\^-?[\d.]+', op):
        return True
    # pow(N) pattern: e.g. pow(2), pow(1/3), pow(0.5)
    if re.fullmatch(r'pow\([^)]+\)', op):
        return True
    return False


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have at least 2 columns (1 target + 1 feature) and no NaN values.

    Raises
    ------
    ValidationError
        If validation fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )
    if df.empty:
        raise ValidationError("DataFrame is empty.")
    if df.shape[1] < 2:
        raise ValidationError(
            f"DataFrame must have at least 2 columns (1 target + 1 feature), "
            f"got {df.shape[1]}."
        )
    if df.isnull().any().any():
        cols_with_nan = df.columns[df.isnull().any()].tolist()
        raise ValidationError(
            f"DataFrame contains NaN values in columns: {cols_with_nan}. "
            f"Please handle missing values before fitting."
        )
    # Check that all columns are numeric
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValidationError(
            f"All columns must be numeric. Non-numeric columns: {non_numeric}."
        )


def validate_operators(operators: list) -> None:
    """Validate the operators list.

    Parameters
    ----------
    operators : list of str
        Operators to use in feature expansion.

    Raises
    ------
    ValidationError
        If any operator is not in the supported set.
    """
    if not operators:
        raise ValidationError("Operators list cannot be empty.")
    if not isinstance(operators, (list, tuple)):
        raise ValidationError(
            f"Expected a list of operators, got {type(operators).__name__}."
        )
    unsupported = {
        op for op in operators
        if op not in SUPPORTED_OPERATORS and not _is_dynamic_operator(op)
    }
    if unsupported:
        raise ValidationError(
            f"Unsupported operators: {unsupported}. "
            f"Supported operators: {sorted(SUPPORTED_OPERATORS)}, "
            f"plus pow(N) and ^N patterns (e.g. pow(2), ^0.5)."
        )


def validate_dimensions(dimensionality, df: pd.DataFrame) -> None:
    """Validate dimensionality specification for dimensional regression.

    Parameters
    ----------
    dimensionality : list
        List of sympy dimension expressions, one per feature column
        (excluding the target column).
    df : pd.DataFrame
        The input DataFrame (target is first column).

    Raises
    ------
    ValidationError
        If dimensionality list length doesn't match feature count.
    """
    if dimensionality is None:
        return
    n_features = df.shape[1] - 1  # exclude target column
    if len(dimensionality) != n_features:
        raise ValidationError(
            f"dimensionality list length ({len(dimensionality)}) must match "
            f"number of feature columns ({n_features})."
        )
