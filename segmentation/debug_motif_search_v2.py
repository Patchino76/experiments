"""
Debug script to understand the motif discovery process step by step.
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
n_windows = len(mp_distances)

# Thresholds
WATERZUMPF_MAX_CV = 0.01
ORE_MIN_CV = 0.0008
WATERMILL_MIN_CV = 0.0015
RELATIVE_VARIABILITY_FACTOR = 1.2

print(f"\n{'='*60}")
print("SIMULATING MOTIF DISCOVERY")
print(f"{'='*60}\n")

# Find first valid seed
print("Step 1: Finding first valid seed...")
seed_idx = None
seed_distance = float('inf')
checked = 0

for i in range(n_windows):
    checked += 1
    dist = mp_distances[i]
    if np.isnan(dist) or np.isinf(dist):
        continue
    
    # Check variability
    waterzumpf_data = df['WaterZumpf'].iloc[i:i + window_size].values
    ore_data = df['Ore'].iloc[i:i + window_size].values
    watermill_data = df['WaterMill'].iloc[i:i + window_size].values
    
    waterzumpf_cv = calculate_variability(waterzumpf_data)
    ore_cv = calculate_variability(ore_data)
    watermill_cv = calculate_variability(watermill_data)
    
    # Apply constraints
    passes = (waterzumpf_cv <= WATERZUMPF_MAX_CV and 
              ore_cv >= ORE_MIN_CV and 
              watermill_cv >= WATERMILL_MIN_CV and
              ore_cv >= waterzumpf_cv * RELATIVE_VARIABILITY_FACTOR and
              watermill_cv >= waterzumpf_cv * RELATIVE_VARIABILITY_FACTOR)
    
    if passes and dist < seed_distance:
        seed_distance = dist
        seed_idx = i
        print(f"  Found candidate seed at idx={i}, dist={dist:.3f}")
        if checked >= 1000:  # Stop after checking first 1000
            break

if seed_idx is None:
    print("  ✗ No valid seed found!")
else:
    print(f"\n  ✓ Selected seed: idx={seed_idx}, dist={seed_distance:.3f}")
    
    # Step 2: Compute distance profile for this seed
    print(f"\nStep 2: Computing distance profile for seed {seed_idx}...")
    distance_components = []
    for dim in range(T.shape[0]):
        query = T[dim, seed_idx:seed_idx + window_size]
        distance_profile = stumpy.mass(query, T[dim])
        distance_components.append(distance_profile[:n_windows])
    
    distance_components = np.array(distance_components)
    aggregated_profile = np.sqrt(np.mean(distance_components**2, axis=0))
    
    print(f"  Distance profile stats:")
    print(f"    Min: {np.nanmin(aggregated_profile):.3f}")
    print(f"    Median: {np.nanmedian(aggregated_profile):.3f}")
    print(f"    Max: {np.nanmax(aggregated_profile):.3f}")
    print(f"    Below radius ({radius}): {np.sum(aggregated_profile < radius)}")
    
    # Step 3: Find matching instances
    print(f"\nStep 3: Finding matching instances...")
    sorted_candidates = np.argsort(aggregated_profile)
    valid_instances = []
    
    for rank, idx in enumerate(sorted_candidates[:50], 1):  # Check top 50
        if len(valid_instances) >= 20:
            break
        
        dist = aggregated_profile[idx]
        if np.isnan(dist) or np.isinf(dist) or dist > radius:
            continue
        
        # Check variability constraints
        waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + window_size].values
        ore_data = df['Ore'].iloc[idx:idx + window_size].values
        watermill_data = df['WaterMill'].iloc[idx:idx + window_size].values
        
        waterzumpf_cv = calculate_variability(waterzumpf_data)
        ore_cv = calculate_variability(ore_data)
        watermill_cv = calculate_variability(watermill_data)
        
        # Apply constraints
        passes = (waterzumpf_cv <= WATERZUMPF_MAX_CV and 
                  ore_cv >= ORE_MIN_CV and 
                  watermill_cv >= WATERMILL_MIN_CV and
                  ore_cv >= waterzumpf_cv * RELATIVE_VARIABILITY_FACTOR and
                  watermill_cv >= waterzumpf_cv * RELATIVE_VARIABILITY_FACTOR)
        
        # Check overlap
        has_overlap = any(abs(idx - vi) < window_size for vi in valid_instances)
        
        if passes and not has_overlap:
            valid_instances.append(idx)
            print(f"  Rank {rank:2d}: idx={idx:5d}, dist={dist:.3f} ✓ VALID")
        elif rank <= 10:  # Show first 10 regardless
            reason = "overlap" if has_overlap else "constraints"
            print(f"  Rank {rank:2d}: idx={idx:5d}, dist={dist:.3f} ✗ ({reason})")
    
    print(f"\n  ✓ Found {len(valid_instances)} valid instances")
    
    if len(valid_instances) >= 2:
        print(f"  ✓ This would form a motif!")
    else:
        print(f"  ✗ Not enough instances for a motif (need ≥2)")

print(f"\n{'='*60}\n")
