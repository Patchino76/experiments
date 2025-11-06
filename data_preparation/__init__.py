"""
Data Preparation Package

A modular, extensible system for discovering motifs and preparing
segmented datasets for ball mill modeling.

Key Features:
- Universal pattern system with configurable constraints
- Pattern registry for easy extension
- Generic analysis and visualization
- Clean separation of concerns
"""

__version__ = "2.0.0"

from .pipeline import DataPreparationPipeline
from .config.defaults import PipelineConfig, PatternConfig

__all__ = ['DataPreparationPipeline', 'PipelineConfig', 'PatternConfig']
