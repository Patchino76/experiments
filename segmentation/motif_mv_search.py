import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
import warnings
import os
warnings.filterwarnings('ignore')

# Load data
def load_data(filepath):
    """Load CSV data into pandas DataFrame"""
    df = pd.read_csv(filepath, parse_dates=['TimeStamp'])
    return df

# Discover multivariate motifs using STUMPY
def discover_multivariate_motifs(df, feature_columns, window_size=240, max_motifs=50, radius=3.0, max_instances_per_motif=10):
    """
    Discover repeating motifs/patterns in multivariate time series using STUMPY
    
    Parameters:
    - df: DataFrame with time series data
    - feature_columns: list of column names to use for motif discovery
    - window_size: length of the pattern to search (e.g., 240 minutes)
    - max_motifs: maximum number of motif groups to find
    - radius: distance threshold for motif matching (lower = more strict)
    - max_instances_per_motif: maximum number of windows to include per motif group
    
    Returns:
    - motif_info_list: list of dictionaries with motif information
    - segment_tuples: list of (start, end, motif_id) tuples
    - mps: multivariate matrix profile summary
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
    matrix_profile, profile_indices = stumpy.mstump(T, m=window_size)
    mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))
    mps = np.column_stack((mp_distances, profile_indices[0]))
    
    print(f"  Discovering multivariate motifs...")
    motif_info_list = []
    segment_tuples = []
    used_indices = set()
    n_windows = matrix_profile.shape[1]

    for motif_idx in range(max_motifs):
        seed_idx = None
        seed_distance = float('inf')

        for i in range(n_windows):
            if i in used_indices:
                continue
            dist = mp_distances[i]
            if np.isnan(dist) or np.isinf(dist):
                continue
            if dist < seed_distance:
                seed_distance = dist
                seed_idx = i

        if seed_idx is None:
            break
        if radius is not None and seed_distance > radius:
            break

        distance_components = []
        for dim in range(T.shape[0]):
            query = T[dim, seed_idx:seed_idx + window_size]
            if len(query) < window_size:
                continue
            distance_profile = stumpy.mass(query, T[dim])
            distance_components.append(distance_profile[:n_windows])

        if not distance_components:
            continue

        distance_components = np.array(distance_components)
        aggregated_profile = np.sqrt(np.mean(distance_components**2, axis=0))

        sorted_candidates = np.argsort(aggregated_profile)
        motif_indices = []
        for idx in sorted_candidates:
            if len(motif_indices) >= max_instances_per_motif:
                break
            if idx >= n_windows:
                continue
            dist = aggregated_profile[idx]
            if np.isnan(dist) or np.isinf(dist):
                continue
            if radius is not None and dist > radius:
                break
            if idx in used_indices:
                continue
            if any(abs(idx - existing_idx) < window_size for existing_idx in motif_indices):
                continue
            motif_indices.append(idx)

        if seed_idx not in motif_indices:
            motif_indices.insert(0, seed_idx)

        motif_indices.sort()
        motif_distances = [aggregated_profile[idx] for idx in motif_indices]
        motif_distance_value = (float(np.mean([dist for idx, dist in zip(motif_indices, motif_distances) if idx != seed_idx]))
                                if len(motif_indices) > 1 else float(seed_distance))

        motif_info = {
            'motif_id': motif_idx + 1,
            'instances': [],
            'distance': motif_distance_value
        }

        for idx in motif_indices:
            instance = {
                'start': idx,
                'end': idx + window_size,
                'data': {col: df[col].iloc[idx:idx + window_size].values 
                        for col in feature_columns}
            }
            motif_info['instances'].append(instance)
            segment_tuples.append((idx, idx + window_size, motif_idx + 1))

            used_indices.add(idx)
            for offset in range(-window_size, window_size):
                neighbor = idx + offset
                if 0 <= neighbor < n_windows:
                    used_indices.add(neighbor)

        motif_info_list.append(motif_info)

    print(f"  Found {len(motif_info_list)} motif groups with {len(segment_tuples)} total instances")
    if motif_info_list:
        print(f"  Average distance: {np.mean([m['distance'] for m in motif_info_list]):.3f}")
    else:
        print("  Average distance: N/A")

    return motif_info_list, segment_tuples, mps

# Filter motifs based on cross-correlation constraints
def filter_motifs_by_correlation(motif_info_list, correlation_rules, min_correlation_strength=0.3, filter_level='instance'):
    """
    Filter motifs based on cross-correlation constraints between feature pairs.
    
    Parameters:
    - motif_info_list: list of motif dictionaries from discover_multivariate_motifs
    - correlation_rules: dict mapping feature pairs to expected correlation sign
      Example: {('WaterZumpf', 'DensityHC'): 'neg', ('WaterZumpf', 'PulpHC'): 'pos'}
    - min_correlation_strength: minimum absolute correlation value to consider (default 0.3)
    - filter_level: 'instance' (filter individual instances) or 'motif' (filter entire motif groups)
    
    Returns:
    - filtered_motif_info_list: list of filtered motif dictionaries
    - filtered_segment_tuples: list of (start, end, motif_id) tuples for valid instances
    - correlation_stats: dict with correlation statistics for each motif
    """
    print(f"\n  Filtering motifs by correlation constraints...")
    print(f"  Filter level: {filter_level}")
    print(f"  Min correlation strength: {min_correlation_strength}")
    print(f"  Rules:")
    for (feat1, feat2), sign in correlation_rules.items():
        print(f"    {feat1} vs {feat2}: {sign}")
    
    filtered_motif_info_list = []
    filtered_segment_tuples = []
    correlation_stats = {}
    
    for motif in motif_info_list:
        motif_id = motif['motif_id']
        valid_instances = []
        instance_correlations = []
        
        for instance in motif['instances']:
            # Compute correlations for this instance
            instance_corrs = {}
            all_rules_satisfied = True
            
            for (feat1, feat2), expected_sign in correlation_rules.items():
                if feat1 not in instance['data'] or feat2 not in instance['data']:
                    continue
                    
                data1 = instance['data'][feat1]
                data2 = instance['data'][feat2]
                
                # Compute Pearson correlation
                if len(data1) > 1 and len(data2) > 1:
                    corr, p_value = pearsonr(data1, data2)
                    instance_corrs[(feat1, feat2)] = corr
                    
                    # Check if correlation meets the constraint
                    if abs(corr) < min_correlation_strength:
                        # Correlation too weak, skip this check
                        continue
                    
                    if expected_sign == 'pos' and corr < 0:
                        all_rules_satisfied = False
                        break
                    elif expected_sign == 'neg' and corr > 0:
                        all_rules_satisfied = False
                        break
            
            instance_correlations.append(instance_corrs)
            
            if all_rules_satisfied:
                valid_instances.append(instance)
        
        # Store correlation stats
        correlation_stats[motif_id] = {
            'total_instances': len(motif['instances']),
            'valid_instances': len(valid_instances),
            'avg_correlations': {}
        }
        
        # Compute average correlations across valid instances
        if valid_instances:
            for (feat1, feat2) in correlation_rules.keys():
                corrs = [ic.get((feat1, feat2), np.nan) for ic in instance_correlations 
                        if (feat1, feat2) in ic]
                if corrs:
                    correlation_stats[motif_id]['avg_correlations'][(feat1, feat2)] = np.mean(corrs)
        
        # Apply filter level
        if filter_level == 'instance':
            # Keep motif if at least one instance is valid
            if valid_instances:
                filtered_motif = {
                    'motif_id': motif_id,
                    'instances': valid_instances,
                    'distance': motif['distance']
                }
                filtered_motif_info_list.append(filtered_motif)
                
                # Add to segment tuples
                for instance in valid_instances:
                    filtered_segment_tuples.append((instance['start'], instance['end'], motif_id))
        
        elif filter_level == 'motif':
            # Keep motif only if ALL instances are valid
            if len(valid_instances) == len(motif['instances']) and valid_instances:
                filtered_motif = {
                    'motif_id': motif_id,
                    'instances': valid_instances,
                    'distance': motif['distance']
                }
                filtered_motif_info_list.append(filtered_motif)
                
                # Add to segment tuples
                for instance in valid_instances:
                    filtered_segment_tuples.append((instance['start'], instance['end'], motif_id))
    
    print(f"  Filtered: {len(motif_info_list)} motifs -> {len(filtered_motif_info_list)} motifs")
    print(f"  Total instances: {sum(len(m['instances']) for m in motif_info_list)} -> {len(filtered_segment_tuples)}")
    
    return filtered_motif_info_list, filtered_segment_tuples, correlation_stats

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
        plt.savefig(f'output/motif_{motif_id}_instances.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('output/motif_overview.png', dpi=150, bbox_inches='tight')
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
        
        # Get timestamp boundaries
        segment_start_ts = df['TimeStamp'].iloc[start]
        segment_end_ts = df['TimeStamp'].iloc[end - 1]
        
        # Add metadata columns
        segment_data['segment_id'] = seg_idx + 1
        segment_data['motif_id'] = motif_ids[0]  # Primary motif ID
        segment_data['segment_start'] = segment_start_ts
        segment_data['segment_end'] = segment_end_ts
        
        all_segments.append(segment_data)
    
    # Stack all segments
    stacked_df = pd.concat(all_segments, ignore_index=True)
    
    # Reorder columns: TimeStamp first, then segment_id, motif_id, segment_start, segment_end, then all other columns
    cols = stacked_df.columns.tolist()
    
    # Define the desired order for metadata columns
    priority_cols = ['TimeStamp', 'segment_id', 'motif_id', 'segment_start', 'segment_end']
    
    # Get remaining columns (excluding priority columns)
    other_cols = [col for col in cols if col not in priority_cols]
    
    # Reorder: priority columns first, then the rest
    new_col_order = priority_cols + other_cols
    stacked_df = stacked_df[new_col_order]
    
    return stacked_df

# Create motif summary with correlation analysis
def create_motif_summary(motif_info_list, correlation_rules, output_path='output/motif_summary.csv'):
    """
    Create a summary CSV file with motif statistics and average correlations.
    
    Parameters:
    - motif_info_list: list of motif dictionaries
    - correlation_rules: dict of correlation rules to analyze
    - output_path: path to save the summary CSV
    
    Returns:
    - summary_df: DataFrame with motif summary statistics
    """
    summary_data = []
    
    for motif in motif_info_list:
        motif_id = motif['motif_id']
        num_instances = len(motif['instances'])
        distance = motif['distance']
        
        # Compute average correlations across all instances
        correlation_values = {}
        for (feat1, feat2) in correlation_rules.keys():
            corrs = []
            for instance in motif['instances']:
                if feat1 in instance['data'] and feat2 in instance['data']:
                    data1 = instance['data'][feat1]
                    data2 = instance['data'][feat2]
                    if len(data1) > 1 and len(data2) > 1:
                        corr, _ = pearsonr(data1, data2)
                        corrs.append(corr)
            
            if corrs:
                correlation_values[f'{feat1}_vs_{feat2}'] = np.mean(corrs)
            else:
                correlation_values[f'{feat1}_vs_{feat2}'] = np.nan
        
        # Build summary row
        summary_row = {
            'motif_id': motif_id,
            'num_instances': num_instances,
            'avg_distance': distance,
            **correlation_values
        }
        summary_data.append(summary_row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_df.to_csv(output_path, index=False)
    print(f"\nMotif summary saved to {output_path}")
    
    return summary_df

# Main execution
if __name__ == "__main__":
    # Configuration
    SEGMENTATION_FEATURES = ['WaterZumpf', 'DensityHC', 'PulpHC', 'PressureHC']
    WINDOW_SIZE = 60  # Fixed window length in minutes
    MAX_MOTIFS = 20    # Maximum number of motif groups to discover
    MAX_INSTANCES_PER_MOTIF = 1000  # Maximum windows per motif group
    RADIUS = 5    # Distance threshold (lower = more strict matching)
    TOP_MOTIFS_TO_PLOT = 10  # Number of top motifs to plot individually
    
    # Cross-correlation filtering configuration
    APPLY_CORRELATION_FILTER = True  # Set to False to disable filtering
    CORRELATION_RULES = {
        ('WaterZumpf', 'DensityHC'): 'neg',
        ('WaterZumpf', 'PulpHC'): 'pos',
        ('WaterZumpf', 'PressureHC'): 'pos',
        # ('Ore', 'DensityHC'): 'pos',  # Uncomment if 'Ore' is in your features
    }
    MIN_CORRELATION_STRENGTH = 0.1  # Minimum |correlation| to enforce the rule
    FILTER_LEVEL = 'instance'  # 'instance' or 'motif'
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Load data
    df = load_data('data_initial.csv')
    df = df.iloc[:150000,:]
    
    print(f"Loaded data: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
   
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
        radius=RADIUS,
        max_instances_per_motif=MAX_INSTANCES_PER_MOTIF
    )
    
    # Apply correlation filtering if enabled
    if APPLY_CORRELATION_FILTER:
        print(f"\n{'='*60}")
        print(f"CORRELATION FILTERING")
        print(f"{'='*60}")
        motif_info_list, segment_tuples, correlation_stats = filter_motifs_by_correlation(
            motif_info_list,
            CORRELATION_RULES,
            min_correlation_strength=MIN_CORRELATION_STRENGTH,
            filter_level=FILTER_LEVEL
        )
        
        # Print correlation statistics
        print(f"\n{'='*60}")
        print(f"CORRELATION STATISTICS")
        print(f"{'='*60}")
        for motif_id, stats in correlation_stats.items():
            if stats['valid_instances'] > 0:
                print(f"\nMotif {motif_id}:")
                print(f"  Valid instances: {stats['valid_instances']}/{stats['total_instances']}")
                if stats['avg_correlations']:
                    print(f"  Average correlations:")
                    for (feat1, feat2), corr in stats['avg_correlations'].items():
                        print(f"    {feat1} vs {feat2}: {corr:+.3f}")
    
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
    output_file = 'output/segmented_motifs.csv'
    stacked_df.to_csv(output_file, index=False)
    print(f"\nSegmented motifs saved to {output_file}")
    print(f"Total segments: {stacked_df['segment_id'].nunique()}")
    print(f"Total rows: {len(stacked_df)}")
    
    # Create motif summary with correlation analysis
    print(f"\n{'='*60}")
    print(f"CREATING MOTIF SUMMARY")
    print(f"{'='*60}")
    motif_summary_df = create_motif_summary(motif_info_list, CORRELATION_RULES)
    print("\nMotif Summary:")
    print(motif_summary_df.to_string(index=False))
    
    # Print segment summary
    print(f"\n{'='*60}")
    print(f"SEGMENT SUMMARY")
    print(f"{'='*60}")
    segment_summary = stacked_df.groupby('motif_id').agg({
        'segment_id': pd.Series.nunique
    }).rename(columns={'segment_id': 'num_windows'})
    print(segment_summary)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Generated files in 'output/' folder:")
    print(f"  - motif_overview.png (timeline with all motifs)")
    print(f"  - motif_1_instances.png through motif_{TOP_MOTIFS_TO_PLOT}_instances.png")
    print(f"  - segmented_motifs.csv (segmented data for ML training)")
    print(f"  - motif_summary.csv (motif statistics and correlations)")