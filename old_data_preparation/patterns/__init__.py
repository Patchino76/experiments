"""
Pattern discovery implementations.

This package contains various pattern discovery algorithms that can be used
with the data preparation pipeline.

Available patterns:
- density: Discovers patterns with stable WaterZumpf and varying Ore/WaterMill
- inverse: Discovers patterns with stable Ore/WaterMill and varying WaterZumpf
- dynamic: Discovers patterns where all variables are changing
- pressure: Discovers patterns with stable PressureHC and varying other variables
"""
from typing import Dict, Type
from ..core.pattern_registry import PatternRegistry

# Import pattern implementations to register them
from .density import DensityPattern
from .inverse import InversePattern
from .dynamic import DynamicPattern
from .pressure import PressurePattern

__all__ = [
    'DensityPattern',
    'InversePattern',
    'DynamicPattern',
    'PressurePattern'
]
from .density import DensityPattern

__all__ = ['DensityPattern']
