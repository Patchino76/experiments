"""
Circulative Load Calculation Module

This module calculates the circulative load for ball mill operations in a closed-circuit
grinding system with hydrocyclone classification.

Formula: CL = (M_solid_to_cyclone - Fresh_Feed) / Fresh_Feed

Typical values: 1.5 to 3.0 (150-300%) for closed-circuit grinding with hydrocyclone.
"""

import pandas as pd
import numpy as np


def calculate_circulative_load(df, rho_solid=2900):
    """
    Calculate circulative load based on mill operation parameters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing mill operation data
    rho_solid : float, optional
        Density of solid ore in kg/m³ (default: 2900 for copper ore)
    
    Required columns:
    -----------------
    - Ore: Fresh feed rate (t/h)
    - PulpHC: Pulp flow rate to hydrocyclone (m³/h)
    - DensityHC: Pulp density to hydrocyclone (kg/m³)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added columns:
        - C_v: Volumetric concentration
        - C_m: Mass concentration
        - M_solid_to_cyclone: Mass flow of solids to cyclone (t/h)
        - CirculativeLoad: Circulative load ratio
    
    Calculation Steps:
    ------------------
    1. Calculate volumetric concentration (C_v) from pulp density:
       C_v = (rho_pulp - rho_water) / (rho_solid - rho_water)
       
    2. Calculate mass concentration (C_m):
       C_m = C_v * rho_solid / rho_pulp
       
    3. Calculate mass flow of solids to cyclone (t/h):
       M_solid = PulpHC * DensityHC * C_m / 1000
       
    4. Calculate circulative load ratio:
       CL = (M_solid_to_cyclone - Fresh_Feed) / Fresh_Feed
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Constants
    rho_water = 1000  # kg/m³
    
    # Step 1: Calculate volumetric concentration (C_v)
    df['C_v'] = (df['DensityHC'] - rho_water) / (rho_solid - rho_water)
    
    # Step 2: Calculate mass concentration (C_m)
    df['C_m'] = df['C_v'] * rho_solid / df['DensityHC']
    
    # Step 3: Calculate mass flow of solids to cyclone (t/h)
    df['M_solid_to_cyclone'] = df['PulpHC'] * df['DensityHC'] * df['C_m'] / 1000
    
    # Step 4: Calculate circulative load ratio
    df['CirculativeLoad'] = (df['M_solid_to_cyclone'] - df['Ore']) / df['Ore']
    
    return df


def validate_circulative_load(df, min_cl=0.5, max_cl=5.0):
    """
    Validate circulative load values and identify potential issues.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with CirculativeLoad column
    min_cl : float, optional
        Minimum acceptable circulative load (default: 0.5)
    max_cl : float, optional
        Maximum acceptable circulative load (default: 5.0)
    
    Returns:
    --------
    dict
        Dictionary containing validation statistics:
        - mean: Mean circulative load
        - median: Median circulative load
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - out_of_range_count: Number of values outside acceptable range
        - out_of_range_pct: Percentage of values outside acceptable range
    """
    cl_values = df['CirculativeLoad'].dropna()
    
    out_of_range = ((cl_values < min_cl) | (cl_values > max_cl)).sum()
    
    stats = {
        'mean': cl_values.mean(),
        'median': cl_values.median(),
        'std': cl_values.std(),
        'min': cl_values.min(),
        'max': cl_values.max(),
        'out_of_range_count': out_of_range,
        'out_of_range_pct': (out_of_range / len(cl_values)) * 100 if len(cl_values) > 0 else 0
    }
    
    return stats
