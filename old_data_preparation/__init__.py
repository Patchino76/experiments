"""
Data Preparation Pipeline for Pattern Discovery in Time Series Data.

This package provides a modular framework for discovering patterns in time series data,
with a focus on industrial process data analysis.

Key features:
- Multiple pattern discovery algorithms
- Extensible architecture for adding new patterns
- Standardized output format
- Visualization tools

Example usage:
    >>> from data_preparation.core.pipeline import Pipeline
    >>> from data_preparation.config.defaults import get_default_config
    >>> 
    >>> # Get default configuration
    >>> config = get_default_config()
    >>> 
    >>> # Initialize and run pipeline
    >>> pipeline = Pipeline(config, output_dir='output/pattern_discovery')
    >>> results = pipeline.run(your_dataframe)
"""

from .core.pipeline import Pipeline
from .config.defaults import get_default_config
from .motif import Motif, MotifInstance

__version__ = '0.1.0'
__all__ = ['Pipeline', 'get_default_config', 'Motif', 'MotifInstance']
