"""
Default configuration for pattern discovery.

This module contains default configuration values for all pattern types.
"""
from pathlib import Path

# Base output directory
DEFAULT_OUTPUT_DIR = Path("output/pattern_discovery")

# Common pattern parameters
COMMON_PARAMS = {
    'window_size': 60,  # minutes
    'max_motifs': 10,
    'radius': 4.5,
    'enabled': True,
    'output': {
        'save_plots': True,
        'save_csv': True,
        'save_json': True
    }
}

# Pattern-specific defaults
PATTERN_DEFAULTS = {
    'density': {
        'waterzumpf_max_cv': 0.01,  # Maximum CV for WaterZumpf
        'ore_min_cv': 0.0008,       # Minimum CV for Ore
        'watermill_min_cv': 0.0015, # Minimum CV for WaterMill
    },
    'inverse': {
        'ore_max_cv': 0.01,
        'watermill_max_cv': 0.01,
        'waterzumpf_min_cv': 0.0008,
    },
    'dynamic': {
        'ore_min_cv': 0.0008,
        'watermill_min_cv': 0.0015,
        'waterzumpf_min_cv': 0.0008,
    },
    'pressure': {
        'pressure_max_cv': 0.01,
        'ore_min_cv': 0.0008,
        'watermill_min_cv': 0.0015,
        'waterzumpf_min_cv': 0.0008,
    }
}

def get_default_config() -> dict:
    """
    Get the default configuration for all patterns.
    
    Returns:
        Dictionary with default configuration
    """
    config = {
        'output_dir': str(DEFAULT_OUTPUT_DIR),
        'patterns': {}
    }
    
    # Apply common params to all patterns
    for pattern_name in PATTERN_DEFAULTS:
        config['patterns'][pattern_name] = {
            **COMMON_PARAMS.copy(),
            **PATTERN_DEFAULTS[pattern_name]
        }
    
    return config
