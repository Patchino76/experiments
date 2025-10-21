"""
Test script for circulative load calculation.

This script demonstrates the circulative load calculation with sample data.
"""

import pandas as pd
import numpy as np
from circulative_load import calculate_circulative_load, validate_circulative_load

# Create sample data
np.random.seed(42)
n_samples = 100

# Typical operating ranges for a ball mill
sample_data = pd.DataFrame({
    'TimeStamp': pd.date_range('2025-01-01', periods=n_samples, freq='1min'),
    'Ore': np.random.uniform(150, 200, n_samples),  # t/h
    'PulpHC': np.random.uniform(450, 550, n_samples),  # m³/h
    'DensityHC': np.random.uniform(1650, 1750, n_samples),  # kg/m³
    'WaterMill': np.random.uniform(80, 120, n_samples),
    'MotorAmp': np.random.uniform(200, 250, n_samples),
})

print("=" * 80)
print("CIRCULATIVE LOAD CALCULATION TEST")
print("=" * 80)
print("\nSample input data (first 5 rows):")
print(sample_data[['Ore', 'PulpHC', 'DensityHC']].head())

# Calculate circulative load
result_df = calculate_circulative_load(sample_data, rho_solid=2900)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print("\nCalculated columns (first 5 rows):")
print(result_df[['Ore', 'PulpHC', 'DensityHC', 'C_v', 'C_m', 'M_solid_to_cyclone', 'CirculativeLoad']].head())

# Validate
print("\n" + "=" * 80)
validate_circulative_load(result_df)

print("\n" + "=" * 80)
print("DETAILED STATISTICS")
print("=" * 80)
print("\nCirculative Load Statistics:")
print(result_df['CirculativeLoad'].describe())

print("\n✓ Test completed successfully!")
print("\nThe circulative load column has been added to the DataFrame.")
print("When running prepare_data.py, this column will be included in initial_data.csv")
