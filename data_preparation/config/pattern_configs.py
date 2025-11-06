"""
Pre-defined pattern configurations.

Provides factory functions for creating common pattern configurations.
"""

from typing import List
from .defaults import PatternConfig


def get_default_patterns() -> List[PatternConfig]:
    """
    Get default set of patterns for ball mill modeling.
    
    Returns:
        List of pattern configurations
    """
    return [
        create_mv_pattern(),
        create_density_pattern(),
        create_inverse_pattern(),
        create_dynamic_pattern(),
        create_pressure_pattern(enabled=False)  # Optional, disabled by default
    ]


def create_mv_pattern(
    enabled: bool = True,
    window_size: int = 60,
    max_motifs: int = 20,
    radius: float = 4.5
) -> PatternConfig:
    """
    Create standard MV motif pattern.
    
    Discovers repeating patterns in manipulated variables without constraints.
    
    Args:
        enabled: Enable this pattern
        window_size: Window size in minutes
        max_motifs: Maximum number of motifs
        radius: Distance threshold
        
    Returns:
        PatternConfig for MV pattern
    """
    return PatternConfig(
        name='mv',
        type='mv',
        enabled=enabled,
        window_size=window_size,
        max_motifs=max_motifs,
        radius=radius,
        features=['Ore', 'WaterMill', 'WaterZumpf'],
        save_analysis=True,
        save_plots=True
    )


def create_density_pattern(
    enabled: bool = True,
    window_size: int = 60,
    max_motifs: int = 15,
    radius: float = 4.5
) -> PatternConfig:
    """
    Create density constraint pattern.
    
    Finds patterns where WaterZumpf is stable but Ore and WaterMill vary.
    This captures steady sump water with varying feed conditions.
    
    Args:
        enabled: Enable this pattern
        window_size: Window size in minutes
        max_motifs: Maximum number of motifs
        radius: Distance threshold
        
    Returns:
        PatternConfig for density pattern
    """
    return PatternConfig(
        name='density',
        type='constraint',
        enabled=enabled,
        window_size=window_size,
        max_motifs=max_motifs,
        radius=radius,
        constraints={
            'WaterZumpf': {
                'type': 'stable',
                'max_cv': 0.01
            },
            'Ore': {
                'type': 'varying',
                'min_cv': 0.0008
            },
            'WaterMill': {
                'type': 'varying',
                'min_cv': 0.0015
            }
        },
        save_analysis=True,
        save_plots=True
    )


def create_inverse_pattern(
    enabled: bool = True,
    window_size: int = 60,
    max_motifs: int = 10,
    radius: float = 4.5
) -> PatternConfig:
    """
    Create inverse constraint pattern.
    
    Finds patterns where Ore and WaterMill are stable but WaterZumpf varies.
    This captures steady feed with sump water adjustments.
    
    Args:
        enabled: Enable this pattern
        window_size: Window size in minutes
        max_motifs: Maximum number of motifs
        radius: Distance threshold
        
    Returns:
        PatternConfig for inverse pattern
    """
    return PatternConfig(
        name='inverse',
        type='constraint',
        enabled=enabled,
        window_size=window_size,
        max_motifs=max_motifs,
        radius=radius,
        constraints={
            'Ore': {
                'type': 'stable',
                'max_cv': 0.01
            },
            'WaterMill': {
                'type': 'stable',
                'max_cv': 0.01
            },
            'WaterZumpf': {
                'type': 'varying',
                'min_cv': 0.0008
            }
        },
        save_analysis=True,
        save_plots=True
    )


def create_dynamic_pattern(
    enabled: bool = True,
    window_size: int = 60,
    max_motifs: int = 10,
    radius: float = 4.5
) -> PatternConfig:
    """
    Create dynamic pattern.
    
    Finds patterns where all MVs vary together.
    This captures transient/dynamic operations with coordinated adjustments.
    
    Args:
        enabled: Enable this pattern
        window_size: Window size in minutes
        max_motifs: Maximum number of motifs
        radius: Distance threshold
        
    Returns:
        PatternConfig for dynamic pattern
    """
    return PatternConfig(
        name='dynamic',
        type='constraint',
        enabled=enabled,
        window_size=window_size,
        max_motifs=max_motifs,
        radius=radius,
        constraints={
            'Ore': {
                'type': 'varying',
                'min_cv': 0.0008
            },
            'WaterMill': {
                'type': 'varying',
                'min_cv': 0.0015
            },
            'WaterZumpf': {
                'type': 'varying',
                'min_cv': 0.0008
            }
        },
        save_analysis=True,
        save_plots=True
    )


def create_pressure_pattern(
    enabled: bool = False,
    window_size: int = 60,
    max_motifs: int = 10,
    radius: float = 4.5
) -> PatternConfig:
    """
    Create pressure constraint pattern.
    
    Finds patterns where PressureHC is stable but MVs vary.
    This indicates good process control and potentially optimal operating regions.
    
    Args:
        enabled: Enable this pattern (disabled by default)
        window_size: Window size in minutes
        max_motifs: Maximum number of motifs
        radius: Distance threshold
        
    Returns:
        PatternConfig for pressure pattern
    """
    return PatternConfig(
        name='pressure',
        type='constraint',
        enabled=enabled,
        window_size=window_size,
        max_motifs=max_motifs,
        radius=radius,
        constraints={
            'PressureHC': {
                'type': 'stable',
                'max_cv': 0.01
            },
            'Ore': {
                'type': 'varying',
                'min_cv': 0.0008
            },
            'WaterMill': {
                'type': 'varying',
                'min_cv': 0.0015
            },
            'WaterZumpf': {
                'type': 'varying',
                'min_cv': 0.0008
            }
        },
        save_analysis=True,
        save_plots=True
    )


def create_custom_pattern(
    name: str,
    constraints: dict,
    enabled: bool = True,
    window_size: int = 60,
    max_motifs: int = 15,
    radius: float = 4.5
) -> PatternConfig:
    """
    Create a custom constraint pattern.
    
    Example:
        create_custom_pattern(
            name='my_pattern',
            constraints={
                'Ore': {'type': 'stable', 'max_cv': 0.01},
                'WaterMill': {'type': 'varying', 'min_cv': 0.001}
            }
        )
    
    Args:
        name: Pattern name
        constraints: Dictionary of feature constraints
        enabled: Enable this pattern
        window_size: Window size in minutes
        max_motifs: Maximum number of motifs
        radius: Distance threshold
        
    Returns:
        PatternConfig for custom pattern
    """
    return PatternConfig(
        name=name,
        type='constraint',
        enabled=enabled,
        window_size=window_size,
        max_motifs=max_motifs,
        radius=radius,
        constraints=constraints,
        save_analysis=True,
        save_plots=True
    )
