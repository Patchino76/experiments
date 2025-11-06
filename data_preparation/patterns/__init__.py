"""
Pattern discovery implementations.

Importing these modules triggers pattern registration.
"""

# Import to trigger registration
from . import mv_pattern
from . import constraint_pattern

from .mv_pattern import MVPattern
from .constraint_pattern import ConstraintPattern

__all__ = ['MVPattern', 'ConstraintPattern']
