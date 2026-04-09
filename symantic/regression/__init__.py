"""Regression modules for SyMANTIC."""

from .l0_greedy import Regressor as NonDimensionalRegressor
from .l0_greedy_dimensional import Regressor as DimensionalRegressor
from .screening import Regressor as DimensionalScreeningRegressor
