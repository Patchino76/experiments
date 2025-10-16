"""
Quick script to check the actual variability (CV) in the data
to help set appropriate thresholds.
"""

import pandas as pd
import numpy as np

def calculate_variability(data):
    """Calculate coefficient of variation (CV) for a time series."""
    std = np.std(data)
    mean = np.mean(data)
    if mean == 0:
        return 0
    return std / abs(mean)

# Load data
df = pd.read_csv('output/initial_data.csv', parse_dates=['TimeStamp'])
df = df.iloc[:50000]

# Check variability for different window sizes
window_size = 60

features = ['WaterZumpf', 'Ore', 'WaterMill']
print(f"Checking CV for {len(features)} features with window size {window_size}:\n")

for feature in features:
    cvs = []
    for i in range(0, len(df) - window_size, window_size):
        window_data = df[feature].iloc[i:i + window_size].values
        cv = calculate_variability(window_data)
        cvs.append(cv)
    
    cvs = np.array(cvs)
    print(f"{feature}:")
    print(f"  Min CV:    {np.min(cvs):.4f}")
    print(f"  25th %ile: {np.percentile(cvs, 25):.4f}")
    print(f"  Median CV: {np.median(cvs):.4f}")
    print(f"  75th %ile: {np.percentile(cvs, 75):.4f}")
    print(f"  Max CV:    {np.max(cvs):.4f}")
    print(f"  Mean CV:   {np.mean(cvs):.4f}")
    print()

# Check how many windows would pass different thresholds
print("\nTesting different threshold combinations:")
print("-" * 60)

test_configs = [
    {"name": "Current (strict)", "wz_max": 0.02, "ore_min": 0.03, "wm_min": 0.05},
    {"name": "Relaxed 1", "wz_max": 0.05, "ore_min": 0.02, "wm_min": 0.03},
    {"name": "Relaxed 2", "wz_max": 0.10, "ore_min": 0.01, "wm_min": 0.02},
    {"name": "Very relaxed", "wz_max": 0.15, "ore_min": 0.005, "wm_min": 0.01},
]

for config in test_configs:
    count = 0
    for i in range(0, len(df) - window_size, window_size):
        wz_cv = calculate_variability(df['WaterZumpf'].iloc[i:i + window_size].values)
        ore_cv = calculate_variability(df['Ore'].iloc[i:i + window_size].values)
        wm_cv = calculate_variability(df['WaterMill'].iloc[i:i + window_size].values)
        
        if (wz_cv <= config['wz_max'] and 
            ore_cv >= config['ore_min'] and 
            wm_cv >= config['wm_min']):
            count += 1
    
    print(f"{config['name']:20s}: {count:4d} windows pass "
          f"(WZ≤{config['wz_max']:.3f}, Ore≥{config['ore_min']:.3f}, WM≥{config['wm_min']:.3f})")
