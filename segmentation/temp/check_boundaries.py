import pandas as pd

df = pd.read_csv('output/segmented_motifs.csv')

print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"\nColumns: {df.columns.tolist()}")

# Find segment boundaries
boundaries = df[df['is_segment_start'] == 1].head(5)

print(f"\n{'='*80}")
print("EXAMINING SEGMENT BOUNDARIES")
print(f"{'='*80}")

for idx in boundaries.index:
    start_idx = max(0, idx-2)
    end_idx = min(len(df), idx+3)
    
    print(f"\n=== Boundary at row {idx} (segment {df.iloc[idx]['segment_id']}) ===")
    print(df[['segment_id', 'DensityHC', 'discontinuity_score', 'is_segment_start', 'is_segment_end']].iloc[start_idx:end_idx].to_string())

print(f"\n{'='*80}")
print("DISCONTINUITY SCORE STATISTICS")
print(f"{'='*80}")
print(df['discontinuity_score'].describe())

print(f"\nHigh discontinuity rows (score > 2.0): {(df['discontinuity_score'] > 2.0).sum()}")
print(f"Segment boundaries: {df['is_segment_start'].sum()}")
