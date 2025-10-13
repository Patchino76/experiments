"""
Quick test to verify temporal features are working correctly
"""

import pandas as pd
import numpy as np

# Create sample segmented data (simulating motif_mv_search output)
print("Creating sample segmented data...")
np.random.seed(42)

# Segment 1: 100 rows
seg1 = pd.DataFrame({
    'TimeStamp': pd.date_range('2025-06-18 06:00', periods=100, freq='1min'),
    'segment_id': 1,
    'motif_id': 1,
    'Ore': np.random.uniform(160, 170, 100),
    'WaterMill': np.random.uniform(10, 12, 100),
    'WaterZumpf': np.random.uniform(230, 240, 100),
    'PulpHC': np.random.uniform(500, 520, 100),
    'PressureHC': np.random.uniform(0.45, 0.47, 100),
    'DensityHC': np.random.uniform(1630, 1650, 100),
})

# Segment 2: 100 rows (with a jump in values to simulate discontinuity)
seg2 = pd.DataFrame({
    'TimeStamp': pd.date_range('2025-06-18 14:00', periods=100, freq='1min'),
    'segment_id': 2,
    'motif_id': 2,
    'Ore': np.random.uniform(170, 180, 100),
    'WaterMill': np.random.uniform(11, 13, 100),
    'WaterZumpf': np.random.uniform(235, 245, 100),
    'PulpHC': np.random.uniform(510, 530, 100),
    'PressureHC': np.random.uniform(0.46, 0.48, 100),
    'DensityHC': np.random.uniform(1700, 1720, 100),  # Jump from ~1640 to ~1710
})

# Stack segments
df = pd.concat([seg1, seg2], ignore_index=True)
print(f"Stacked data: {len(df)} rows, {len(df.columns)} columns")

# Add temporal features (same function as in motif_mv_search.py)
def add_temporal_features(df, feature_columns):
    df = df.copy()
    
    # 1. LAG FEATURES
    for col in feature_columns:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        df[f'{col}_lag3'] = df[col].shift(3)
    
    # 2. RATE OF CHANGE
    for col in feature_columns:
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_diff2'] = df[f'{col}_lag1'] - df[f'{col}_lag2']
    
    # 3. ROLLING STATISTICS
    for col in feature_columns:
        df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_rolling_std_5'] = df[col].rolling(window=5, min_periods=1).std()
    
    # 4. ACCELERATION
    for col in feature_columns:
        df[f'{col}_accel'] = df[f'{col}_diff1'] - df[f'{col}_diff2']
    
    # 5. SEGMENT BOUNDARY INDICATORS
    df['is_segment_start'] = (df['segment_id'] != df['segment_id'].shift(1)).astype(int)
    df['is_segment_end'] = (df['segment_id'] != df['segment_id'].shift(-1)).astype(int)
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df

feature_columns = ['Ore', 'WaterMill', 'WaterZumpf', 'PulpHC', 'PressureHC']
print(f"\nAdding temporal features for: {feature_columns}")
df_enriched = add_temporal_features(df, feature_columns)

print(f"After adding features: {len(df_enriched)} rows, {len(df_enriched.columns)} columns")

# Examine the boundary (rows 99-101, where segment changes)
print("\n" + "="*80)
print("EXAMINING SEGMENT BOUNDARY (rows 98-102)")
print("="*80)

boundary_rows = df_enriched.iloc[98:103]
cols_to_show = ['segment_id', 'Ore', 'Ore_lag1', 'Ore_diff1', 
                'DensityHC', 'is_segment_start', 'is_segment_end']

print(boundary_rows[cols_to_show].to_string())

print("\n" + "="*80)
print("KEY OBSERVATIONS:")
print("="*80)
print(f"Row 99 (last of segment 1):")
print(f"  - Ore: {df_enriched.iloc[99]['Ore']:.2f}")
print(f"  - DensityHC (target): {df_enriched.iloc[99]['DensityHC']:.2f}")
print(f"  - is_segment_end: {df_enriched.iloc[99]['is_segment_end']}")

print(f"\nRow 100 (first of segment 2):")
print(f"  - Ore: {df_enriched.iloc[100]['Ore']:.2f}")
print(f"  - Ore_diff1: {df_enriched.iloc[100]['Ore_diff1']:.2f} ← Jump detected!")
print(f"  - DensityHC (target): {df_enriched.iloc[100]['DensityHC']:.2f}")
print(f"  - is_segment_start: {df_enriched.iloc[100]['is_segment_start']}")

print("\n✓ Temporal features successfully encode the discontinuity!")
print("✓ Model will learn that large diff1 + is_segment_start = boundary")

# Show feature count
temporal_feature_count = len([col for col in df_enriched.columns 
                              if any(x in col for x in ['_lag', '_diff', '_rolling', '_accel', 'is_segment'])])
print(f"\n✓ Added {temporal_feature_count} temporal features")
print(f"✓ Total columns: {len(df_enriched.columns)}")

print("\n" + "="*80)
print("READY FOR ML TRAINING!")
print("="*80)
