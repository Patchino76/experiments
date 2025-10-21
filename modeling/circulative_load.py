"""
Circulative Load Calculation for Ball Mills

Calculates circulative load based on mill operation parameters.
Formula: CL = (M_solid_to_cyclone - Fresh_Feed) / Fresh_Feed

Typical values: Circulative load typically ranges from 1.5 to 3.0 (150-300%)
for closed-circuit grinding with hydrocyclone.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_circulative_load(df: pd.DataFrame, rho_solid: float = 2900) -> pd.DataFrame:
    """
    Calculate circulative load for ball mill operations.
    
    The circulative load represents the ratio of material recirculated back to the mill
    from the cyclone compared to the fresh feed entering the system.
    
    Calculation steps:
        1. Calculate volumetric concentration (C_v) from pulp density
        2. Calculate mass concentration (C_m) from C_v
        3. Calculate mass flow of solids to cyclone (M_solid_to_cyclone) in t/h
        4. Calculate circulative load ratio: CL = (M_solid_to_cyclone - Fresh_Feed) / Fresh_Feed
    
    Args:
        df: DataFrame containing mill operation data
        rho_solid: Density of solid particles in kg/m³ (default: 2900 for copper ore)
    
    Required columns:
        - Ore: Fresh feed ore flow rate (t/h)
        - PulpHC: Pulp flow to hydrocyclone (m³/h)
        - DensityHC: Pulp density at hydrocyclone (kg/m³)
    
    Returns:
        DataFrame with added columns:
            - C_v: Volumetric concentration (fraction)
            - C_m: Mass concentration (fraction)
            - M_solid_to_cyclone: Mass flow of solids to cyclone (t/h)
            - CirculativeLoad: Circulative load ratio (dimensionless)
    
    Raises:
        ValueError: If required columns are missing
    """
    # Validate required columns
    required_cols = ['Ore', 'PulpHC', 'DensityHC']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for circulative load calculation: {missing_cols}")
    
    logger.info("Calculating circulative load...")
    logger.info(f"  Using rho_solid = {rho_solid} kg/m³")
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Constants
    rho_water = 1000  # kg/m³
    
    # Step 1: Calculate volumetric concentration (C_v) from pulp density
    # Formula: C_v = (rho_pulp - rho_water) / (rho_solid - rho_water)
    df['C_v'] = (df['DensityHC'] - rho_water) / (rho_solid - rho_water)
    
    # Clip C_v to valid range [0, 1]
    df['C_v'] = df['C_v'].clip(lower=0, upper=1)
    
    # Step 2: Calculate mass concentration (C_m)
    # Formula: C_m = (C_v * rho_solid) / (C_v * rho_solid + (1 - C_v) * rho_water)
    numerator = df['C_v'] * rho_solid
    denominator = df['C_v'] * rho_solid + (1 - df['C_v']) * rho_water
    df['C_m'] = numerator / denominator
    
    # Step 3: Calculate mass flow of solids to cyclone (t/h)
    # Formula: M_solid = PulpHC * DensityHC * C_m / 1000
    # PulpHC is in m³/h, DensityHC in kg/m³, divide by 1000 to get t/h
    df['M_solid_to_cyclone'] = (df['PulpHC'] * df['DensityHC'] * df['C_m']) / 1000
    
    # Step 4: Calculate circulative load ratio
    # Formula: CL = (M_solid_to_cyclone - Fresh_Feed) / Fresh_Feed
    # Avoid division by zero
    df['CirculativeLoad'] = np.where(
        df['Ore'] > 0,
        (df['M_solid_to_cyclone'] - df['Ore']) / df['Ore'],
        np.nan
    )
    
    # Log statistics
    valid_cl = df['CirculativeLoad'].dropna()
    if len(valid_cl) > 0:
        logger.info(f"  ✓ Circulative load calculated for {len(valid_cl)} rows")
        logger.info(f"    Mean: {valid_cl.mean():.3f}")
        logger.info(f"    Median: {valid_cl.median():.3f}")
        logger.info(f"    Std: {valid_cl.std():.3f}")
        logger.info(f"    Min: {valid_cl.min():.3f}")
        logger.info(f"    Max: {valid_cl.max():.3f}")
        
        # Check if values are in typical range
        in_range = ((valid_cl >= 1.5) & (valid_cl <= 3.0)).sum()
        pct_in_range = (in_range / len(valid_cl)) * 100
        logger.info(f"    Values in typical range [1.5, 3.0]: {in_range}/{len(valid_cl)} ({pct_in_range:.1f}%)")
        
        # Warn if many values are outside typical range
        if pct_in_range < 50:
            logger.warning(
                f"  ⚠ Only {pct_in_range:.1f}% of circulative load values are in the typical range [1.5, 3.0]. "
                "This may indicate unusual operating conditions or data quality issues."
            )
    else:
        logger.warning("  ⚠ No valid circulative load values calculated")
    
    return df


def validate_circulative_load(df: pd.DataFrame, 
                              min_valid: float = 0.5, 
                              max_valid: float = 5.0) -> pd.DataFrame:
    """
    Validate and optionally filter circulative load values.
    
    Args:
        df: DataFrame with CirculativeLoad column
        min_valid: Minimum valid circulative load value
        max_valid: Maximum valid circulative load value
    
    Returns:
        DataFrame with validation info logged
    """
    if 'CirculativeLoad' not in df.columns:
        logger.warning("CirculativeLoad column not found in DataFrame")
        return df
    
    logger.info("Validating circulative load values...")
    
    total = len(df)
    valid = df['CirculativeLoad'].notna().sum()
    invalid = total - valid
    
    logger.info(f"  Total rows: {total}")
    logger.info(f"  Valid values: {valid} ({valid/total*100:.1f}%)")
    logger.info(f"  Invalid/NaN values: {invalid} ({invalid/total*100:.1f}%)")
    
    # Check range
    if valid > 0:
        out_of_range = ((df['CirculativeLoad'] < min_valid) | 
                       (df['CirculativeLoad'] > max_valid)).sum()
        logger.info(f"  Out of range [{min_valid}, {max_valid}]: {out_of_range} ({out_of_range/total*100:.1f}%)")
    
    return df
