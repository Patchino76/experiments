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

# Discover multivariate motifs using STUMPY
def discover_multivariate_motifs(df, feature_columns, window_size=240, max_motifs=50, radius=3.0):
    """
    Discover repeating motifs/patterns in multivariate time series using STUMPY
    
    Parameters:
    - df: DataFrame with time series data
    - feature_columns: list of column names to use for motif discovery
    - window_size: length of the pattern to search (e.g., 240 minutes)
    - max_motifs: maximum number of motif groups to find
    - radius: distance threshold for motif matching (lower = more strict)
    
    Returns:
    - motif_segments: list of dictionaries with motif information
    - mps: multivariate matrix profile
    """
    # Prepare multivariate time series
    print(f"  Preparing multivariate time series with {len(feature_columns)} features...")
    ts_list = []
    for col in feature_columns:
        ts = np.array(df[col])
        # Normalize each feature
        ts = (ts - np.mean(ts)) / np.std(ts)
        ts_list.append(ts)
    
    # Stack into multivariate array (features x time)
    T = np.array(ts_list)
    
    print(f"  Computing multivariate matrix profile (shape: {T.shape})...")
    # Compute multivariate matrix profile
    mps = stumpy.mstump(T, m=window_size)
    
    # Extract the distance profile (first column contains distances)
    mp_distances = mps[:, 0]
    
    print(f"  Discovering multivariate motifs...")
    motif_segments = []
    used_indices = set()
    
    # Find top motifs
    for motif_idx in range(max_motifs):
        # Find the best motif pair that hasn't been used
        motif_pair = None
        min_dist = float('inf')
        
        for i in range(len(mp_distances)):
            if i in used_indices:
                continue
            if mp_distances[i] < min_dist:
                nn_idx = int(mps[i, 1])  # Nearest neighbor index
                # Check if neighbor is far enough away and not used
                if abs(i - nn_idx) >= window_size and nn_idx not in used_indices:
                    min_dist = mp_distances[i]
                    motif_pair = (i, nn_idx)
        
        if motif_pair is None or min_dist > radius:
            break
        
        # Store motif instances
        motif_info = {
            'motif_id': motif_idx + 1,
            'instances': [],
            'distance': min_dist
        }
        
        for idx in motif_pair:
            instance = {
                'start': idx,
                'end': idx + window_size,
                'data': {col: df[col].iloc[idx:idx + window_size].values 
                        for col in feature_columns}
            }
            motif_info['instances'].append(instance)
            motif_segments.append((idx, idx + window_size, motif_idx + 1))
            
            used_indices.add(idx)
            # Mark nearby indices as used to avoid overlap
            for offset in range(-window_size, window_size):
                if 0 <= idx + offset < len(df):
                    used_indices.add(idx + offset)
        
        motif_segments.append(motif_info)
    
    # Separate motif_info from segments
    motif_info_list = [m for m in motif_segments if isinstance(m, dict)]
    segment_tuples = [m for m in motif_segments if isinstance(m, tuple)]
    
    print(f"  Found {len(motif_info_list)} motif groups with {len(segment_tuples)} total instances")
    print(f"  Average distance: {np.mean([m['distance'] for m in motif_info_list]):.3f}")
    
    return motif_info_list, segment_tuples, mps

# Plot individual motifs with all instances
def plot_individual_motifs(df, motif_info_list, feature_columns, top_n=10):
    """
    Plot each motif separately showing all instances
    
    Parameters:
    - df: DataFrame with time series data
    - motif_info_list: list of motif information dictionaries
    - feature_columns: list of features to plot
    - top_n: number of top motifs to plot
    """
    n_features = len(feature_columns)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for motif in motif_info_list[:top_n]:
        motif_id = motif['motif_id']
        instances = motif['instances']
        distance = motif['distance']
        
        print(f"Plotting Motif {motif_id} ({len(instances)} instances, distance: {distance:.3f})...")
        
        # Create subplots for each feature
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 2.5 * n_features), sharex=True)
        if n_features == 1:
            axes = [axes]
        
        # Plot each feature
        for feat_idx, feature in enumerate(feature_columns):
            ax = axes[feat_idx]
            
            # Plot all instances of this motif
            for inst_idx, instance in enumerate(instances):
                data = instance['data'][feature]
                time_steps = range(len(data))
                
                color = colors[inst_idx % len(colors)]
                ax.plot(time_steps, data, color=color, linewidth=2, 
                       label=f'Instance {inst_idx + 1}', alpha=0.8)
            
            ax.set_ylabel(feature, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            
            if feat_idx == 0:
                ax.set_title(f'Motif {motif_id} - Distance: {distance:.3f}', 
                           fontsize=13, fontweight='bold', pad=10)
        
        axes[-1].set_xlabel('Time (minutes within motif)', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'motif_{motif_id}_instances.png', dpi=150, bbox_inches='tight')
        plt.close()

# Plot overview of all motifs
def plot_motif_overview(df, segment_tuples, feature_columns):
    """
    Plot overview showing all discovered motifs on the timeline
    
    Parameters:
    - df: DataFrame with time series data
    - segment_tuples: list of (start, end, motif_id) tuples
    - feature_columns: list of features to plot
    """
    n_features = len(feature_columns)
    fig, axes = plt.subplots(n_features + 1, 1, figsize=(15, 3 * (n_features + 1)), 
                             sharex=True)
    
    if n_features == 0:
        axes = [axes]
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Group segments by motif_id
    motif_groups = {}
    for start, end, motif_id in segment_tuples:
        if motif_id not in motif_groups:
            motif_groups[motif_id] = []
        motif_groups[motif_id].append((start, end))
    
    # Plot each feature
    for feat_idx, feature in enumerate(feature_columns):
        ax = axes[feat_idx]
        
        # Plot raw data
        ax.plot(df.index, df[feature], 'k-', alpha=0.3, linewidth=0.5)
        
        # Plot motifs
        for motif_id, instances in sorted(motif_groups.items()):
            color = colors[(motif_id - 1) % len(colors)]
            for inst_idx, (start, end) in enumerate(instances):
                end = min(end, len(df))
                ax.plot(df.index[start:end], df[feature].iloc[start:end], 
                       color=color, linewidth=2, 
                       label=f'motif {motif_id}' if feat_idx == 0 and inst_idx == 0 else '')
                
                # Add shaded regions
                ax.axvspan(df.index[start], df.index[end-1], alpha=0.1, color=color)
        
        ax.set_ylabel(feature, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if feat_idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:min(10, len(handles))], labels[:min(10, len(labels))], 
                     loc='upper right', ncol=5, fontsize=8)
    
    # Plot matrix profile
    ax_mp = axes[-1]
    # Use first feature for matrix profile visualization
    ts = np.array(df[feature_columns[0]])
    ts = (ts - np.mean(ts)) / np.std(ts)
    
    ax_mp.plot(df.index, [0] * len(df), 'k-', alpha=0)  # Placeholder
    ax_mp.set_ylabel('timeline', fontsize=10)
    ax_mp.set_xlabel('time', fontsize=12)
    ax_mp.grid(True, alpha=0.3)
    
    # Mark motif locations
    for motif_id, instances in sorted(motif_groups.items()):
        color = colors[(motif_id - 1) % len(colors)]
        for start, end in instances:
            ax_mp.axvspan(df.index[start], df.index[min(end-1, len(df)-1)], 
                         alpha=0.3, color=color)
    
    plt.suptitle('Multivariate Motif Discovery', fontsize=16, fontweight='bold', x=0.12, y=0.98)
    plt.tight_layout()
    plt.savefig('motif_overview.png', dpi=150, bbox_inches='tight')
    plt.show()

# Extract motif segments
def extract_motif_segments(df, segment_tuples):
    """
    Extract motif segments and stack them
    
    Parameters:
    - df: DataFrame with time series data
    - segment_tuples: list of (start, end, motif_id) tuples
    
    Returns:
    - stacked_df: DataFrame with stacked motif segments
    """
    all_segments = []
    
    # Get unique windows
    windows = {}
    for start, end, motif_id in segment_tuples:
        key = (start, end)
        if key not in windows:
            windows[key] = []
        windows[key].append(motif_id)
    
    # Extract each window
    for seg_idx, ((start, end), motif_ids) in enumerate(sorted(windows.items())):
        segment_data = df.iloc[start:end].copy()
        segment_data['segment_id'] = seg_idx + 1
        segment_data['motif_id'] = motif_ids[0]  # Primary motif ID
        segment_data['segment_start'] = start
        segment_data['segment_end'] = end
        segment_data['segment_length'] = end - start
        segment_data['time_in_segment'] = range(len(segment_data))
        
        all_segments.append(segment_data)
    
    # Stack all segments
    stacked_df = pd.concat(all_segments, ignore_index=True)
    
    return stacked_df

# Main execution
if __name__ == "__main__":
    # Configuration
    SEGMENTATION_FEATURES = ['WaterZumpf', 'DensityHC']
    WINDOW_SIZE = 240  # Fixed window length in minutes
    MAX_MOTIFS = 50    # Maximum number of motif groups to discover
    RADIUS = 3.0       # Distance threshold (lower = more strict matching)
    TOP_MOTIFS_TO_PLOT = 10  # Number of top motifs to plot individually
    
    # Load data
    df = load_data('data_initial.csv')
    
    print(f"Loaded data: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Verify segmentation features exist
    missing_features = [f for f in SEGMENTATION_FEATURES if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features {missing_features}")
        SEGMENTATION_FEATURES = [f for f in SEGMENTATION_FEATURES if f in df.columns]
    
    print(f"\n{'='*60}")
    print(f"MULTIVARIATE MOTIF DISCOVERY")
    print(f"{'='*60}")
    print(f"Features: {SEGMENTATION_FEATURES}")
    print(f"Window size: {WINDOW_SIZE} minutes")
    print(f"Max motifs: {MAX_MOTIFS}")
    print(f"Distance threshold: {RADIUS}")
    print(f"{'='*60}\n")
    
    # Discover multivariate motifs
    print("Starting multivariate motif discovery...")
    motif_info_list, segment_tuples, mps = discover_multivariate_motifs(
        df, SEGMENTATION_FEATURES, 
        window_size=WINDOW_SIZE,
        max_motifs=MAX_MOTIFS,
        radius=RADIUS
    )
    
    # Show statistics
    print(f"\n{'='*60}")
    print(f"DISCOVERY RESULTS")
    print(f"{'='*60}")
    print(f"Total motif groups: {len(motif_info_list)}")
    print(f"Total motif instances: {len(segment_tuples)}")
    
    for motif in motif_info_list[:TOP_MOTIFS_TO_PLOT]:
        print(f"\nMotif {motif['motif_id']}:")
        print(f"  Instances: {len(motif['instances'])}")
        print(f"  Distance: {motif['distance']:.4f}")
        print(f"  Positions: {[inst['start'] for inst in motif['instances']]}")
    
    # Plot individual motifs
    print(f"\n{'='*60}")
    print(f"PLOTTING INDIVIDUAL MOTIFS (Top {TOP_MOTIFS_TO_PLOT})")
    print(f"{'='*60}")
    plot_individual_motifs(df, motif_info_list, SEGMENTATION_FEATURES, top_n=TOP_MOTIFS_TO_PLOT)
    
    # Plot overview
    print(f"\nPlotting overview...")
    plot_motif_overview(df, segment_tuples, SEGMENTATION_FEATURES)
    
    # Extract and stack segments
    print(f"\n{'='*60}")
    print(f"EXTRACTING SEGMENTS")
    print(f"{'='*60}")
    stacked_df = extract_motif_segments(df, segment_tuples)
    
    print(f"Final output columns: {stacked_df.columns.tolist()}")
    
    # Save to CSV
    output_file = 'segmented_motifs.csv'
    stacked_df.to_csv(output_file, index=False)
    print(f"\nSegmented motifs saved to {output_file}")
    print(f"Total segments: {stacked_df['segment_id'].nunique()}")
    print(f"Total rows: {len(stacked_df)}")
    
    # Print segment summary
    print(f"\n{'='*60}")
    print(f"SEGMENT SUMMARY")
    print(f"{'='*60}")
    segment_summary = stacked_df.groupby('motif_id').agg({
        'segment_id': 'count',
        'segment_length': 'first'
    }).rename(columns={'segment_id': 'num_instances'})
    print(segment_summary)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Generated files:")
    print(f"  - motif_overview.png (timeline with all motifs)")
    print(f"  - motif_1_instances.png through motif_{TOP_MOTIFS_TO_PLOT}_instances.png")
    print(f"  - {output_file} (segmented data for ML training)")