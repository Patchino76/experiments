import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load data
def load_data(filepath):
    """Load CSV data into pandas DataFrame"""
    df = pd.read_csv(filepath, parse_dates=['TimeStamp'])
    return df

# Segment time series using STUMPY
def segment_time_series(data, column_name, window_size=10, n_regimes=5):
    """
    Segment time series using STUMPY's FLUSS algorithm
    
    Parameters:
    - data: pandas Series or numpy array
    - column_name: name of the column being processed
    - window_size: subsequence length for matrix profile
    - n_regimes: number of regimes/segments to identify
    
    Returns:
    - segments: list of (start, end) tuples
    - cac: corrected arc curve (regime change scores)
    """
    # Normalize the data
    ts = np.array(data)
    ts = (ts - np.mean(ts)) / np.std(ts)
    
    # Compute matrix profile
    mp = stumpy.stump(ts, m=window_size)
    
    # Compute the corrected arc curve (CAC) for regime identification
    cac, regime_locations = stumpy.fluss(mp[:, 1], L=window_size, 
                                         n_regimes=n_regimes, 
                                         excl_factor=1)
    
    # Convert regime locations to segments
    regime_locations = sorted(regime_locations)
    segments = []
    start = 0
    for loc in regime_locations:
        if loc > start:
            segments.append((start, loc))
            start = loc
    segments.append((start, len(ts)))
    
    return segments, cac

# Plot segmentation results
def plot_segmentation(df, segments_dict, matrix_profile_col='Ore'):
    """
    Plot segmentation results similar to the example image
    
    Parameters:
    - df: DataFrame with time series data
    - segments_dict: dictionary with column names as keys and segments as values
    - matrix_profile_col: column to show matrix profile for
    """
    n_cols = len(segments_dict)
    fig, axes = plt.subplots(n_cols + 1, 1, figsize=(15, 3 * (n_cols + 1)), 
                             sharex=True)
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    
    # Plot each sensor with segments
    for idx, (col_name, segments) in enumerate(segments_dict.items()):
        ax = axes[idx]
        
        # Plot raw data
        ax.plot(df.index, df[col_name], 'k-', alpha=0.3, linewidth=0.5)
        
        # Highlight segments with different colors
        for seg_idx, (start, end) in enumerate(segments):
            color = colors[seg_idx % len(colors)]
            ax.plot(df.index[start:end], df[col_name].iloc[start:end], 
                   color=color, linewidth=2, label=f'motif {seg_idx + 1}' if idx == 0 else '')
            
            # Add shaded regions
            ax.axvspan(df.index[start], df.index[end], alpha=0.1, color=color)
        
        ax.set_ylabel(col_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([df[col_name].min() - 0.1 * df[col_name].std(), 
                     df[col_name].max() + 0.1 * df[col_name].std()])
        
        if idx == 0:
            ax.legend(loc='upper right', ncol=4)
    
    # Plot matrix profile curve at bottom
    ax_mp = axes[-1]
    
    # Compute matrix profile for visualization
    ts = np.array(df[matrix_profile_col])
    ts = (ts - np.mean(ts)) / np.std(ts)
    mp = stumpy.stump(ts, m=10)
    cac, _ = stumpy.fluss(mp[:, 1], L=10, n_regimes=4, excl_factor=1)
    
    ax_mp.plot(df.index[:len(cac)], cac, 'k-', linewidth=2)
    ax_mp.set_ylabel('matrix profile', fontsize=10)
    ax_mp.set_xlabel('time', fontsize=12)
    ax_mp.grid(True, alpha=0.3)
    ax_mp.invert_yaxis()
    
    plt.suptitle('Semantic Segmentation', fontsize=16, fontweight='bold', x=0.1, y=0.98)
    plt.tight_layout()
    plt.savefig('segmentation_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# Extract and stack segments
def extract_segments(df, segments_dict, pad_value=np.nan):
    """
    Extract segments and stack them chronologically
    
    Parameters:
    - df: DataFrame with time series data
    - segments_dict: dictionary with column names as keys and segments as values
    - pad_value: value to use for padding shorter segments
    
    Returns:
    - stacked_df: DataFrame with stacked segments
    """
    all_segments = []
    
    # Get all unique segment boundaries across all sensors
    all_boundaries = set()
    for segments in segments_dict.values():
        for start, end in segments:
            all_boundaries.add(start)
            all_boundaries.add(end)
    
    boundaries = sorted(list(all_boundaries))
    
    # Extract segments
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        segment_data = df.iloc[start:end].copy()
        segment_data['segment_id'] = i + 1
        segment_data['segment_start'] = start
        segment_data['segment_end'] = end
        segment_data['time_in_segment'] = range(len(segment_data))
        
        all_segments.append(segment_data)
    
    # Stack all segments
    stacked_df = pd.concat(all_segments, ignore_index=True)
    
    return stacked_df

# Main execution
if __name__ == "__main__":
    # Configuration
    SEGMENTATION_FEATURES = ['WaterZumpf', 'DensityHC']
    
    # Load data
    df = load_data('data_initial.csv')
    
    print(f"Loaded data: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Verify segmentation features exist
    missing_features = [f for f in SEGMENTATION_FEATURES if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features {missing_features}")
        SEGMENTATION_FEATURES = [f for f in SEGMENTATION_FEATURES if f in df.columns]
    
    print(f"\nUsing features for segmentation: {SEGMENTATION_FEATURES}")
    
    # Select sensors to segment
    sensor_columns = SEGMENTATION_FEATURES
    
    # Segment each sensor
    segments_dict = {}
    window_size = 10
    n_regimes = 4
    
    print(f"\nSegmenting time series with window_size={window_size}, n_regimes={n_regimes}")
    
    for col in sensor_columns:
        print(f"Processing {col}...")
        segments, cac = segment_time_series(df[col], col, 
                                           window_size=window_size, 
                                           n_regimes=n_regimes)
        segments_dict[col] = segments
        print(f"  Found {len(segments)} segments: {segments}")
    
    # Plot results
    print("\nPlotting segmentation results...")
    plot_segmentation(df, segments_dict)
    
    # Extract and stack segments
    print("\nExtracting and stacking segments...")
    stacked_df = extract_segments(df, segments_dict)
    
    # Ensure all original features are in the output
    print(f"\nFinal output includes all features: {stacked_df.columns.tolist()}")
    
    # Save to CSV
    output_file = 'segmented_data.csv'
    stacked_df.to_csv(output_file, index=False)
    print(f"\nSegmented data saved to {output_file}")
    print(f"Total segments: {stacked_df['segment_id'].nunique()}")
    print(f"Total rows: {len(stacked_df)}")
    
    # Print segment summary
    print("\nSegment Summary:")
    segment_summary = stacked_df.groupby('segment_id').agg({
        'TimeStamp': ['min', 'max', 'count']
    })
    print(segment_summary)