"""
Debug script to understand why no motifs are being found.
"""

import pandas as pd
import numpy as np
import stumpy

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

# Configuration
window_size = 60
radius = 3.5

# Prepare multivariate time series
features = ['WaterZumpf', 'Ore', 'WaterMill']
ts_list = []
for col in features:
    ts = np.array(df[col])
    ts = (ts - np.mean(ts)) / np.std(ts)
    ts_list.append(ts)

T = np.array(ts_list)
print(f"Computing multivariate matrix profile (shape: {T.shape})...")

# Compute multivariate matrix profile
matrix_profile, profile_indices = stumpy.mstump(T, m=window_size)
mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))

print(f"\nMatrix profile statistics:")
print(f"  Min distance: {np.nanmin(mp_distances):.4f}")
print(f"  25th %ile:    {np.nanpercentile(mp_distances, 25):.4f}")
print(f"  Median:       {np.nanmedian(mp_distances):.4f}")
print(f"  75th %ile:    {np.nanpercentile(mp_distances, 75):.4f}")
print(f"  Max distance: {np.nanmax(mp_distances):.4f}")
print(f"  Radius:       {radius:.4f}")

# Count how many are below radius
below_radius = np.sum(mp_distances < radius)
print(f"\n  Windows below radius: {below_radius} / {len(mp_distances)} ({100*below_radius/len(mp_distances):.1f}%)")

# Check top 20 candidates
print(f"\nTop 20 candidates (by distance):")
sorted_indices = np.argsort(mp_distances)

WATERZUMPF_MAX_CV = 0.01
ORE_MIN_CV = 0.0008
WATERMILL_MIN_CV = 0.0015
RELATIVE_VARIABILITY_FACTOR = 1.2

for rank, idx in enumerate(sorted_indices[:20], 1):
    dist = mp_distances[idx]
    
    # Check variability
    waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + window_size].values
    ore_data = df['Ore'].iloc[idx:idx + window_size].values
    watermill_data = df['WaterMill'].iloc[idx:idx + window_size].values
    
    waterzumpf_cv = calculate_variability(waterzumpf_data)
    ore_cv = calculate_variability(ore_data)
    watermill_cv = calculate_variability(watermill_data)
    
    # Check constraints
    pass_wz = waterzumpf_cv <= WATERZUMPF_MAX_CV
    pass_ore = ore_cv >= ORE_MIN_CV
    pass_wm = watermill_cv >= WATERMILL_MIN_CV
    pass_ore_rel = ore_cv >= waterzumpf_cv * RELATIVE_VARIABILITY_FACTOR
    pass_wm_rel = watermill_cv >= waterzumpf_cv * RELATIVE_VARIABILITY_FACTOR
    
    all_pass = pass_wz and pass_ore and pass_wm and pass_ore_rel and pass_wm_rel
    
    status = "✓ PASS" if all_pass else "✗ FAIL"
    
    print(f"\n  Rank {rank:2d} | idx={idx:5d} | dist={dist:.3f} | {status}")
    print(f"    WZ_CV={waterzumpf_cv:.5f} {'✓' if pass_wz else '✗'} (≤{WATERZUMPF_MAX_CV})")
    print(f"    Ore_CV={ore_cv:.5f} {'✓' if pass_ore else '✗'} (≥{ORE_MIN_CV})")
    print(f"    WM_CV={watermill_cv:.5f} {'✓' if pass_wm else '✗'} (≥{WATERMILL_MIN_CV})")
    print(f"    Ore/WZ ratio={ore_cv/waterzumpf_cv if waterzumpf_cv > 0 else float('inf'):.2f} {'✓' if pass_ore_rel else '✗'} (≥{RELATIVE_VARIABILITY_FACTOR})")
    print(f"    WM/WZ ratio={watermill_cv/waterzumpf_cv if waterzumpf_cv > 0 else float('inf'):.2f} {'✓' if pass_wm_rel else '✗'} (≥{RELATIVE_VARIABILITY_FACTOR})")

print("\n" + "="*60)
