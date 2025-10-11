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

# Discover motifs (similar patterns) using STUMPY
def discover_motifs(data, column_name, window_size=240, max_motifs=50, radius=2.0):
    """
    Discover repeating motifs/patterns in time series using STUMPY
    
    Parameters:
    - data: pandas Series or numpy array
    - column_name: name of the column being processed
    - window_size: length of the pattern to search (e.g., 240 minutes)
    - max_motifs: maximum number of motif groups to find
    - radius: distance threshold for motif matching (lower = more strict)
    
    Returns:
    - motif_segments: list of (start, end, motif_id) tuples
    - motif_distances: distances for each motif pair
    """
    # Normalize the data
    ts = np.array(data)
    ts = (ts - np.mean(ts)) / np.std(ts)
    
    print(f"  Computing matrix profile for {column_name}...")
    # Compute matrix profile
    mp = stumpy.stump(ts, m=window_size)
    
    print(f"  Discovering motifs...")
    motif_segments = []
    motif_distances = []
    used_indices = set()
    
    # Find top motifs
    for motif_idx in range(max_motifs):
        # Find the best motif pair that hasn't been used
        motif_pair = None
        min_dist = float('inf')
        
        for i in range(len(mp)):
            if i in used_indices:
                continue
            if mp[i, 0] < min_dist:
                nn_idx = int(mp[i, 1])
                # Check if neighbor is far enough away and not used
                if abs(i - nn_idx) >= window_size and nn_idx not in used_indices:
                    min_dist = mp[i, 0]
                    motif_pair = (i, nn_idx)
        
        if motif_pair is None or min_dist > radius:
            break
        
        # Add motif instances
        for idx in motif_pair:
            motif_segments.append((idx, idx + window_size, motif_idx + 1))
            used_indices.add(idx)
            # Mark nearby indices as used to avoid overlap
            for offset in range(-window_size, window_size):
                used_indices.add(idx + offset)
        
        motif_distances.append(min_dist)
    
    # Sort by start position
    motif_segments.sort(key=lambda x: x[0])
    
    print(f"  Found {len(motif_segments)} motif instances across {len(set(m[2] for m in motif_segments))} motif groups")
    print(f"  Average distance: {np.mean(motif_distances):.3f}")
    
    return motif_segments, motif_distances

# Combine motifs from multiple features
def combine_motifs(motif_dict, df):
    """
    Combine motifs from multiple features into unified segments
    
    Parameters:
    - motif_dict: dictionary with column names as keys and motif_segments as values
    - df: original dataframe
    
    Returns:
    - combined_segments: list of (start, end, motif_label) tuples
    """
    # Collect all motif windows
    all_windows = []
    
    for col_name, motifs in motif_dict.items():
        for start, end, motif_id in motifs:
            all_windows.append({
                'start': start,
                'end': end,
                'feature': col_name,
                'motif_id': motif_id
            })
    
    # Sort by start position
    all_windows.sort(key=lambda x: x['start'])
    
    # Create combined segments with labels
    combined_segments = []
    for i, window in enumerate(all_windows):
        label = f"{window['feature']}_M{window['motif_id']}"
        combined_segments.append((window['start'], window['end'], label))
    
    return combined_segments

# Plot motif discovery results
def plot_motifs(df, motif_dict, window_size=240):
    """
    Plot discovered motifs
    
    Parameters:
    - df: DataFrame with time series data
    - motif_dict: dictionary with column names as keys and motif segments as values
    - window_size: length of motif windows
    """
    n_cols = len(motif_dict)
    fig, axes = plt.subplots(n_cols + 1, 1, figsize=(15, 3 * (n_cols + 1)), 
                             sharex=True)
    
    if n_cols == 1:
        axes = [axes]
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot each sensor with motifs
    for idx, (col_name, motifs) in enumerate(motif_dict.items()):
        ax = axes[idx]
        
        # Plot raw data
        ax.plot(df.index, df[col_name], 'k-', alpha=0.3, linewidth=0.5)
        
        # Group motifs by motif_id
        motif_groups = {}
        for start, end, motif_id in motifs:
            if motif_id not in motif_groups:
                motif_groups[motif_id] = []
            motif_groups[motif_id].append((start, end))
        
        # Plot each motif group with same color
        for motif_id, instances in motif_groups.items():
            color = colors[(motif_id - 1) % len(colors)]
            for start, end in instances:
                end = min(end, len(df))
                ax.plot(df.index[start:end], df[col_name].iloc[start:end], 
                       color=color, linewidth=2, 
                       label=f'motif {motif_id}' if idx == 0 and instances.index((start, end)) == 0 else '')
                
                # Add shaded regions
                ax.axvspan(df.index[start], df.index[end-1], alpha=0.1, color=color)
        
        ax.set_ylabel(col_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([df[col_name].min() - 0.1 * df[col_name].std(), 
                     df[col_name].max() + 0.1 * df[col_name].std()])
        
        if idx == 0:
            # Only show first few motifs in legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:min(4, len(handles))], labels[:min(4, len(labels))], 
                     loc='upper right', ncol=4)
    
    # Plot matrix profile for first feature
    ax_mp = axes[-1]
    first_col = list(motif_dict.keys())[0]
    ts = np.array(df[first_col])
    ts = (ts - np.mean(ts)) / np.std(ts)
    mp = stumpy.stump(ts, m=window_size)
    
    ax_mp.plot(df.index[:len(mp)], mp[:, 0], 'k-', linewidth=1)
    ax_mp.set_ylabel('matrix profile', fontsize=10)
    ax_mp.set_xlabel('time', fontsize=12)
    ax_mp.grid(True, alpha=0.3)
    
    plt.suptitle('Semantic Segmentation - Motif Discovery', fontsize=16, fontweight='bold', x=0.12, y=0.98)
    plt.tight_layout()
    plt.savefig('motif_discovery_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# Extract motif segments
def extract_motif_segments(df, motif_dict):
    """
    Extract motif segments and stack them
    
    Parameters:
    - df: DataFrame with time series data
    - motif_dict: dictionary with column names as keys and motif segments as values
    
    Returns:
    - stacked_df: DataFrame with stacked motif segments
    """
    all_segments = []
    
    # Collect all unique motif windows
    all_windows = set()
    for col_name, motifs in motif_dict.items():
        for start, end, motif_id in motifs:
            all_windows.add((start, end))
    
    all_windows = sorted(list(all_windows))
    
    # Extract each motif window
    for seg_idx, (start, end) in enumerate(all_windows):
        segment_data = df.iloc[start:end].copy()
        segment_data['segment_id'] = seg_idx + 1
        segment_data['segment_start'] = start
        segment_data['segment_end'] = end
        segment_data['segment_length'] = end - start
        segment_data['time_in_segment'] = range(len(segment_data))
        
        # Add motif labels from each feature
        for col_name, motifs in motif_dict.items():
            matching_motifs = [m[2] for m in motifs if m[0] == start and m[1] == end]
            if matching_motifs:
                segment_data[f'{col_name}_motif_id'] = matching_motifs[0]
            else:
                segment_data[f'{col_name}_motif_id'] = 0
        
        all_segments.append(segment_data)
    
    # Stack all segments
    stacked_df = pd.concat(all_segments, ignore_index=True)
    
    return stacked_df

# Main execution
if __name__ == "__main__":
    # Configuration
    SEGMENTATION_FEATURES = ['WaterZumpf', 'DensityHC']
    WINDOW_SIZE = 240  # Fixed window length in minutes
    MAX_MOTIFS = 5    # Maximum number of motif groups to discover
    RADIUS = 7       # Distance threshold (lower = more strict matching)
    
    # Load data
    df = load_data('data_initial.csv')
    
    print(f"Loaded data: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Verify segmentation features exist
    missing_features = [f for f in SEGMENTATION_FEATURES if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features {missing_features}")
        SEGMENTATION_FEATURES = [f for f in SEGMENTATION_FEATURES if f in df.columns]
    
    print(f"\nUsing features for motif discovery: {SEGMENTATION_FEATURES}")
    print(f"Window size: {WINDOW_SIZE} minutes")
    print(f"Max motifs per feature: {MAX_MOTIFS}")
    print(f"Distance threshold: {RADIUS}")
    
    # Discover motifs in each feature
    motif_dict = {}
    
    for col in SEGMENTATION_FEATURES:
        print(f"\nProcessing {col}...")
        motifs, distances = discover_motifs(df[col], col, 
                                           window_size=WINDOW_SIZE,
                                           max_motifs=MAX_MOTIFS,
                                           radius=RADIUS)
        motif_dict[col] = motifs
        
        # Show statistics
        motif_groups = len(set(m[2] for m in motifs))
        print(f"  Motif groups: {motif_groups}")
        print(f"  Total instances: {len(motifs)}")
        if distances:
            print(f"  Distance range: [{min(distances):.3f}, {max(distances):.3f}]")
    
    # Plot results
    print("\nPlotting motif discovery results...")
    plot_motifs(df, motif_dict, window_size=WINDOW_SIZE)
    
    # Extract and stack segments
    print("\nExtracting and stacking motif segments...")
    stacked_df = extract_motif_segments(df, motif_dict)
    
    # Ensure all original features are in the output
    print(f"\nFinal output includes all features: {stacked_df.columns.tolist()}")
    
    # Save to CSV
    output_file = 'segmented_motifs.csv'
    stacked_df.to_csv(output_file, index=False)
    print(f"\nSegmented motifs saved to {output_file}")
    print(f"Total segments: {stacked_df['segment_id'].nunique()}")
    print(f"Total rows: {len(stacked_df)}")
    
    # Print segment summary
    print("\nSegment Summary:")
    segment_summary = stacked_df.groupby('segment_id').agg({
        'TimeStamp': ['min', 'max', 'count'],
        'segment_length': 'first'
    })
    print(segment_summary.head(10))
    
    # Print motif distribution
    print("\nMotif Distribution:")
    for col in SEGMENTATION_FEATURES:
        motif_col = f'{col}_motif_id'
        if motif_col in stacked_df.columns:
            motif_counts = stacked_df[stacked_df[motif_col] > 0].groupby('segment_id')[motif_col].first().value_counts()
            print(f"\n{col}:")
            print(motif_counts.head(10))