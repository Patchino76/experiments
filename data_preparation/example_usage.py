"""
Example usage of the data preparation pipeline.

Demonstrates various configuration options and customizations.
"""

import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent))

from pipeline import DataPreparationPipeline
from config.defaults import PipelineConfig, PatternConfig, DataConfig
from config.pattern_configs import (
    create_mv_pattern,
    create_density_pattern,
    create_inverse_pattern,
    create_dynamic_pattern,
    create_custom_pattern
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_default_configuration():
    """Example 1: Use default configuration."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Default Configuration")
    print("=" * 80)
    
    config = PipelineConfig.create_default(
        mill_number=8,
        start_date="2025-01-01",
        end_date="2025-11-03"
    )
    
    pipeline = DataPreparationPipeline(config)
    pipeline.run()


def example_2_custom_patterns():
    """Example 2: Custom pattern selection."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Pattern Selection")
    print("=" * 80)
    
    # Create custom pattern list
    patterns = [
        create_mv_pattern(enabled=True, max_motifs=25),
        create_density_pattern(enabled=True, max_motifs=20),
        create_inverse_pattern(enabled=False),  # Disabled
        create_dynamic_pattern(enabled=True, max_motifs=10)
    ]
    
    config = PipelineConfig.create_default(
        mill_number=8,
        start_date="2025-01-01",
        end_date="2025-11-03",
        patterns=patterns
    )
    
    pipeline = DataPreparationPipeline(config)
    pipeline.run()


def example_3_custom_constraint_pattern():
    """Example 3: Create a completely custom constraint pattern."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Constraint Pattern")
    print("=" * 80)
    
    # Define a custom pattern: Stable Ore, varying WaterMill and WaterZumpf
    custom_pattern = create_custom_pattern(
        name='stable_ore',
        constraints={
            'Ore': {
                'type': 'stable',
                'max_cv': 0.005  # Very stable
            },
            'WaterMill': {
                'type': 'varying',
                'min_cv': 0.001
            },
            'WaterZumpf': {
                'type': 'varying',
                'min_cv': 0.001
            }
        },
        window_size=60,
        max_motifs=10
    )
    
    patterns = [
        create_mv_pattern(),
        custom_pattern
    ]
    
    config = PipelineConfig.create_default(
        mill_number=8,
        start_date="2025-01-01",
        end_date="2025-11-03",
        patterns=patterns
    )
    
    pipeline = DataPreparationPipeline(config)
    pipeline.run()


def example_4_manual_configuration():
    """Example 4: Fully manual configuration."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Fully Manual Configuration")
    print("=" * 80)
    
    # Create data config
    data_config = DataConfig(
        mill_number=8,
        start_date="2025-01-01",
        end_date="2025-11-03",
        resample_freq='1min',
        mv_features=['Ore', 'WaterMill', 'WaterZumpf'],
        cv_features=['DensityHC', 'PulpHC', 'PressureHC', 'CirculativeLoad'],
        dv_features=['Class_15', 'Daiki', 'FE'],
        target='PSI200',
        filter_thresholds={
            'Ore': (100, 220),
            'PulpHC': (400, 600),
            'DensityHC': (1600, 1800),
        }
    )
    
    # Create pattern configs
    patterns = [
        PatternConfig(
            name='mv',
            type='mv',
            enabled=True,
            window_size=60,
            max_motifs=20,
            radius=4.5,
            features=['Ore', 'WaterMill', 'WaterZumpf']
        ),
        PatternConfig(
            name='density',
            type='constraint',
            enabled=True,
            window_size=60,
            max_motifs=15,
            radius=4.5,
            constraints={
                'WaterZumpf': {'type': 'stable', 'max_cv': 0.01},
                'Ore': {'type': 'varying', 'min_cv': 0.0008},
                'WaterMill': {'type': 'varying', 'min_cv': 0.0015}
            }
        )
    ]
    
    # Create pipeline config
    config = PipelineConfig(
        data=data_config,
        patterns=patterns,
        use_database=True,
        save_mv_only=True,
        save_combined=True,
        save_to_database=False
    )
    
    pipeline = DataPreparationPipeline(config)
    pipeline.run()


def example_5_toggle_patterns_at_runtime():
    """Example 5: Toggle patterns on/off at runtime."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Toggle Patterns at Runtime")
    print("=" * 80)
    
    # Start with default config
    config = PipelineConfig.create_default(
        mill_number=8,
        start_date="2025-01-01",
        end_date="2025-11-03"
    )
    
    # Disable specific patterns
    for pattern in config.patterns:
        if pattern.name in ['inverse', 'pressure']:
            pattern.enabled = False
            logger.info(f"Disabled pattern: {pattern.name}")
    
    # Modify parameters for enabled patterns
    for pattern in config.patterns:
        if pattern.name == 'mv':
            pattern.max_motifs = 30
            logger.info(f"Increased MV max_motifs to 30")
        elif pattern.name == 'density':
            pattern.window_size = 90
            logger.info(f"Increased density window_size to 90")
    
    pipeline = DataPreparationPipeline(config)
    pipeline.run()


def example_6_analysis_only():
    """Example 6: Run only specific patterns for analysis."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Analysis-Focused Configuration")
    print("=" * 80)
    
    # Focus on constraint patterns for analysis
    patterns = [
        create_density_pattern(enabled=True, max_motifs=20),
        create_inverse_pattern(enabled=True, max_motifs=20),
        create_dynamic_pattern(enabled=True, max_motifs=20)
    ]
    
    config = PipelineConfig.create_default(
        mill_number=8,
        start_date="2025-01-01",
        end_date="2025-11-03",
        patterns=patterns
    )
    
    # Don't save MV-only dataset
    config.save_mv_only = False
    
    pipeline = DataPreparationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    # Run one of the examples
    print("\nAvailable examples:")
    print("1. Default configuration")
    print("2. Custom pattern selection")
    print("3. Custom constraint pattern")
    print("4. Fully manual configuration")
    print("5. Toggle patterns at runtime")
    print("6. Analysis-focused configuration")
    
    choice = input("\nSelect example (1-6) or press Enter for default: ").strip()
    
    if choice == '1' or choice == '':
        example_1_default_configuration()
    elif choice == '2':
        example_2_custom_patterns()
    elif choice == '3':
        example_3_custom_constraint_pattern()
    elif choice == '4':
        example_4_manual_configuration()
    elif choice == '5':
        example_5_toggle_patterns_at_runtime()
    elif choice == '6':
        example_6_analysis_only()
    else:
        print("Invalid choice. Running default example.")
        example_1_default_configuration()
