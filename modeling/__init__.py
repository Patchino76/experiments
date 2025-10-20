"""
Mill Modeling Pipeline

A modular solution for mill process optimization using motif discovery
and cascade XGBoost models.
"""

__version__ = "1.0.0"

from .config import PipelineConfig, DataConfig, MotifConfig, ModelConfig, PathConfig
from .database import DataLoader, filter_data, validate_required_columns
from .motif_discovery import MotifDiscovery, CorrelationFilter, Motif, MotifInstance
from .density_analysis import DensityMotifDiscovery, analyze_density_behavior
from .segmentation import (
    create_segmented_dataset,
    merge_motif_collections,
    extract_motif_summary,
    create_instance_catalog
)

__all__ = [
    'PipelineConfig',
    'DataConfig',
    'MotifConfig',
    'ModelConfig',
    'PathConfig',
    'DataLoader',
    'MotifDiscovery',
    'CorrelationFilter',
    'DensityMotifDiscovery',
    'Motif',
    'MotifInstance',
]
