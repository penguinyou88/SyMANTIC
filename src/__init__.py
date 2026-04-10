"""
Backward-compatibility shim for the old src/ import path.

The package has been restructured. Please use:
    from symantic import SymanticModel

instead of:
    from src import SymanticModel
"""
import warnings

warnings.warn(
    "Importing from 'src' is deprecated. Use 'from symantic import SymanticModel' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from symantic import SymanticModel
from symantic import pareto
from symantic import feature_space_construction
from symantic import Regressor
