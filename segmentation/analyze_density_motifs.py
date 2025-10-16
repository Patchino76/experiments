"""
Analyze ball-milling process: Find motifs where WaterZumpf is constant 
but Ore and WaterMill are changing, then study DensityHC behavior.
"""

import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import correlate
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Configuration
DATA_FILE = 'output/initial_data.csv'
WINDOW_SIZE = 60  # minutes
MAX_MOTIFS = 15
RADIUS = 3.5
OUTPUT_DIR = Path('output')

def load_data(filepath):
    """Load the initial data CSV file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['TimeStamp'])
    df.sort_values('TimeStamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  ✓ Loaded {len(df)} rows")
    print(f"  ✓ Columns: {df.columns.tolist()}")
    df = df.iloc[:50000]
    return df

def calculate_variability(data):
    """Calculate coefficient of variation (CV) for a time series."""
    std = np.std(data)
    mean = np.mean(data)
    if mean == 0:
        return 0
    return std / abs(mean)

def find_constrained_motifs(df, window_size=60, max_motifs=15, radius=3.5):
    """
    Find motifs where WaterZumpf is constant but Ore and WaterMill are changing.
    
    Strategy: Use multivariate motif discovery on all three features, then filter
    instances based on variability constraints.
    """
    print(f"\n{'='*60}")
    print("SEARCHING FOR CONSTRAINED MOTIFS")
    print(f"{'='*60}")
    print(f"Constraint: WaterZumpf stable, Ore & WaterMill more variable")
    print(f"Window size: {window_size} minutes")
    print(f"Max motifs: {max_motifs}")
    print(f"Distance threshold: {radius}")
    print(f"Strategy: Relative variability comparison (relaxed thresholds)")
    
    # Prepare multivariate time series
    features = ['WaterZumpf', 'Ore', 'WaterMill']
    ts_list = []
    for col in features:
        ts = np.array(df[col])
        # Normalize
        ts = (ts - np.mean(ts)) / np.std(ts)
        ts_list.append(ts)
    
    T = np.array(ts_list)
    print(f"\nComputing multivariate matrix profile (shape: {T.shape})...")
    
    # Compute multivariate matrix profile
    matrix_profile, profile_indices = stumpy.mstump(T, m=window_size)
    mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))
    
    print("Discovering motifs and filtering by variability constraints...")
    
    motif_info_list = []
    used_indices = set()
    n_windows = matrix_profile.shape[1]
    
    # Variability thresholds - adjusted based on actual data distribution
    # Strategy: Look for relative differences, not absolute thresholds
    WATERZUMPF_MAX_CV = 0.01   # WaterZumpf should be relatively stable (below median)
    ORE_MIN_CV = 0.0008        # Ore should show some variation (above 25th percentile)
    WATERMILL_MIN_CV = 0.0015  # WaterMill should show some variation (above 25th percentile)
    
    # Additional constraint: Ore and WaterMill should be MORE variable than WaterZumpf
    RELATIVE_VARIABILITY_FACTOR = 1.2  # They should be at least 20% more variable
    
    for motif_idx in range(max_motifs):
        # Find seed with lowest distance that also passes variability constraints
        seed_idx = None
        seed_distance = float('inf')
        
        for i in range(n_windows):
            if i in used_indices:
                continue
            dist = mp_distances[i]
            if np.isnan(dist) or np.isinf(dist):
                continue
            
            # Check if this candidate passes variability constraints
            waterzumpf_data = df['WaterZumpf'].iloc[i:i + window_size].values
            ore_data = df['Ore'].iloc[i:i + window_size].values
            watermill_data = df['WaterMill'].iloc[i:i + window_size].values
            
            waterzumpf_cv = calculate_variability(waterzumpf_data)
            ore_cv = calculate_variability(ore_data)
            watermill_cv = calculate_variability(watermill_data)
            
            # Apply constraints
            if not (waterzumpf_cv <= WATERZUMPF_MAX_CV and 
                    ore_cv >= ORE_MIN_CV and 
                    watermill_cv >= WATERMILL_MIN_CV and
                    ore_cv >= waterzumpf_cv * RELATIVE_VARIABILITY_FACTOR and
                    watermill_cv >= waterzumpf_cv * RELATIVE_VARIABILITY_FACTOR):
                continue
            
            if dist < seed_distance:
                seed_distance = dist
                seed_idx = i
        
        if seed_idx is None or seed_distance > radius:
            break
        
        # Compute distance profile for this seed
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
        
        # Find candidate instances
        sorted_candidates = np.argsort(aggregated_profile)
        valid_instances = []
        
        for idx in sorted_candidates:
            if len(valid_instances) >= 20:  # Max instances per motif
                break
            
            if idx >= n_windows or idx in used_indices:
                continue
            
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
            
            # Apply constraints: absolute thresholds + relative comparison
            if (waterzumpf_cv <= WATERZUMPF_MAX_CV and 
                ore_cv >= ORE_MIN_CV and 
                watermill_cv >= WATERMILL_MIN_CV and
                ore_cv >= waterzumpf_cv * RELATIVE_VARIABILITY_FACTOR and
                watermill_cv >= waterzumpf_cv * RELATIVE_VARIABILITY_FACTOR):
                
                # Avoid overlapping instances
                if any(abs(idx - vi['start']) < window_size for vi in valid_instances):
                    continue
                
                instance = {
                    'start': idx,
                    'end': idx + window_size,
                    'distance': dist,
                    'waterzumpf_cv': waterzumpf_cv,
                    'ore_cv': ore_cv,
                    'watermill_cv': watermill_cv,
                    'data': {
                        'WaterZumpf': waterzumpf_data,
                        'Ore': ore_data,
                        'WaterMill': watermill_data,
                        'DensityHC': df['DensityHC'].iloc[idx:idx + window_size].values,
                        'TimeStamp': df['TimeStamp'].iloc[idx:idx + window_size].values
                    }
                }
                valid_instances.append(instance)
        
        if len(valid_instances) >= 2:  # Need at least 2 instances for a motif
            motif_info = {
                'motif_id': len(motif_info_list) + 1,
                'instances': valid_instances,
                'avg_distance': np.mean([inst['distance'] for inst in valid_instances])
            }
            motif_info_list.append(motif_info)
            
            # Mark as used
            for inst in valid_instances:
                for offset in range(-window_size, window_size):
                    neighbor = inst['start'] + offset
                    if 0 <= neighbor < n_windows:
                        used_indices.add(neighbor)
        else:
            # Even if motif failed, mark the seed as used to avoid infinite loop
            for offset in range(-window_size, window_size):
                neighbor = seed_idx + offset
                if 0 <= neighbor < n_windows:
                    used_indices.add(neighbor)
    
    print(f"\n  ✓ Found {len(motif_info_list)} motif groups")
    print(f"  ✓ Total instances: {sum(len(m['instances']) for m in motif_info_list)}")
    
    return motif_info_list

def analyze_density_behavior(motif_info_list):
    """
    Analyze how DensityHC behaves in the discovered motifs.
    Calculate correlations, lags, and statistics.
    """
    print(f"\n{'='*60}")
    print("ANALYZING DENSITY BEHAVIOR")
    print(f"{'='*60}")
    
    analysis_results = []
    
    for motif in motif_info_list:
        motif_id = motif['motif_id']
        instances = motif['instances']
        
        print(f"\nMotif {motif_id} ({len(instances)} instances):")
        
        # Aggregate statistics across instances
        density_changes = []
        ore_density_corrs = []
        watermill_density_corrs = []
        ore_density_lags = []
        watermill_density_lags = []
        
        for inst in instances:
            density = inst['data']['DensityHC']
            ore = inst['data']['Ore']
            watermill = inst['data']['WaterMill']
            
            # Calculate change in density
            density_change = density[-1] - density[0]
            density_changes.append(density_change)
            
            # Correlations
            if len(density) > 1:
                ore_corr, _ = pearsonr(ore, density)
                watermill_corr, _ = pearsonr(watermill, density)
                ore_density_corrs.append(ore_corr)
                watermill_density_corrs.append(watermill_corr)
            
            # Cross-correlation for lag analysis
            ore_lag = find_optimal_lag(ore, density)
            watermill_lag = find_optimal_lag(watermill, density)
            ore_density_lags.append(ore_lag)
            watermill_density_lags.append(watermill_lag)
        
        # Summary statistics
        avg_density_change = np.mean(density_changes)
        avg_ore_corr = np.mean(ore_density_corrs)
        avg_watermill_corr = np.mean(watermill_density_corrs)
        avg_ore_lag = np.median(ore_density_lags)
        avg_watermill_lag = np.median(watermill_density_lags)
        
        print(f"  Avg DensityHC change: {avg_density_change:+.2f}")
        print(f"  Ore-Density correlation: {avg_ore_corr:+.3f} (lag: {avg_ore_lag:.0f} min)")
        print(f"  WaterMill-Density correlation: {avg_watermill_corr:+.3f} (lag: {avg_watermill_lag:.0f} min)")
        
        analysis_results.append({
            'motif_id': motif_id,
            'num_instances': len(instances),
            'avg_density_change': avg_density_change,
            'avg_ore_density_corr': avg_ore_corr,
            'avg_watermill_density_corr': avg_watermill_corr,
            'avg_ore_lag': avg_ore_lag,
            'avg_watermill_lag': avg_watermill_lag,
            'density_changes': density_changes,
            'ore_density_corrs': ore_density_corrs,
            'watermill_density_corrs': watermill_density_corrs
        })
    
    return analysis_results

def find_optimal_lag(x, y, max_lag=20):
    """Find the lag that maximizes correlation between x and y."""
    if len(x) != len(y) or len(x) < 2:
        return 0
    
    # Normalize
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    
    # Cross-correlation
    correlation = correlate(y, x, mode='same')
    lags = np.arange(-len(x)//2, len(x)//2)
    
    # Limit to max_lag
    valid_indices = np.where(np.abs(lags) <= max_lag)[0]
    valid_lags = lags[valid_indices]
    valid_corr = correlation[valid_indices]
    
    # Find lag with maximum correlation
    max_idx = np.argmax(np.abs(valid_corr))
    optimal_lag = valid_lags[max_idx]
    
    return optimal_lag

def plot_motif_instances(motif_info_list, top_n=5):
    """Plot individual motif instances showing all features."""
    print(f"\n{'='*60}")
    print(f"PLOTTING MOTIF INSTANCES (Top {top_n})")
    print(f"{'='*60}")
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for motif in motif_info_list[:top_n]:
        motif_id = motif['motif_id']
        instances = motif['instances']
        
        print(f"  Plotting Motif {motif_id}...")
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        features = ['WaterZumpf', 'Ore', 'WaterMill', 'DensityHC']
        
        for feat_idx, feature in enumerate(features):
            ax = axes[feat_idx]
            
            for inst_idx, instance in enumerate(instances):
                data = instance['data'][feature]
                time_steps = range(len(data))
                color = colors[inst_idx % len(colors)]
                
                ax.plot(time_steps, data, color=color, linewidth=2, 
                       label=f'Instance {inst_idx + 1}', alpha=0.7)
            
            ax.set_ylabel(feature, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if feat_idx == 0:
                ax.set_title(f'Motif {motif_id} - {len(instances)} Instances', 
                           fontsize=14, fontweight='bold')
            
            if feat_idx == 3:  # DensityHC
                ax.legend(loc='upper right', ncol=3, fontsize=8)
        
        axes[-1].set_xlabel('Time (minutes)', fontsize=11)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'density_motif_{motif_id}_instances.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()

def plot_correlation_analysis(analysis_results):
    """Plot correlation and lag analysis across motifs."""
    print(f"\n  Plotting correlation analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    motif_ids = [r['motif_id'] for r in analysis_results]
    ore_corrs = [r['avg_ore_density_corr'] for r in analysis_results]
    watermill_corrs = [r['avg_watermill_density_corr'] for r in analysis_results]
    ore_lags = [r['avg_ore_lag'] for r in analysis_results]
    watermill_lags = [r['avg_watermill_lag'] for r in analysis_results]
    
    # Plot 1: Ore-Density Correlation
    ax1 = axes[0, 0]
    ax1.bar(motif_ids, ore_corrs, color='steelblue', alpha=0.7)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_xlabel('Motif ID', fontweight='bold')
    ax1.set_ylabel('Correlation', fontweight='bold')
    ax1.set_title('Ore vs DensityHC Correlation', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: WaterMill-Density Correlation
    ax2 = axes[0, 1]
    ax2.bar(motif_ids, watermill_corrs, color='coral', alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xlabel('Motif ID', fontweight='bold')
    ax2.set_ylabel('Correlation', fontweight='bold')
    ax2.set_title('WaterMill vs DensityHC Correlation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ore-Density Lag
    ax3 = axes[1, 0]
    ax3.bar(motif_ids, ore_lags, color='steelblue', alpha=0.7)
    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax3.set_xlabel('Motif ID', fontweight='bold')
    ax3.set_ylabel('Lag (minutes)', fontweight='bold')
    ax3.set_title('Ore → DensityHC Optimal Lag', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: WaterMill-Density Lag
    ax4 = axes[1, 1]
    ax4.bar(motif_ids, watermill_lags, color='coral', alpha=0.7)
    ax4.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax4.set_xlabel('Motif ID', fontweight='bold')
    ax4.set_ylabel('Lag (minutes)', fontweight='bold')
    ax4.set_title('WaterMill → DensityHC Optimal Lag', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'density_correlation_lag_analysis.png', 
               dpi=150, bbox_inches='tight')
    plt.close()

def plot_density_change_distribution(analysis_results):
    """Plot distribution of density changes across all motif instances."""
    print(f"\n  Plotting density change distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collect all density changes
    all_density_changes = []
    for result in analysis_results:
        all_density_changes.extend(result['density_changes'])
    
    # Plot 1: Histogram
    ax1 = axes[0]
    ax1.hist(all_density_changes, bins=30, color='seagreen', alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='red', linewidth=2, linestyle='--', label='No change')
    ax1.axvline(np.mean(all_density_changes), color='blue', linewidth=2, 
               linestyle='--', label=f'Mean: {np.mean(all_density_changes):.2f}')
    ax1.set_xlabel('DensityHC Change', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax1.set_title('Distribution of DensityHC Changes', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot by motif
    ax2 = axes[1]
    density_by_motif = [result['density_changes'] for result in analysis_results]
    motif_labels = [f"M{result['motif_id']}" for result in analysis_results]
    
    bp = ax2.boxplot(density_by_motif, labels=motif_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.axhline(0, color='red', linewidth=1, linestyle='--')
    ax2.set_xlabel('Motif ID', fontweight='bold', fontsize=11)
    ax2.set_ylabel('DensityHC Change', fontweight='bold', fontsize=11)
    ax2.set_title('DensityHC Changes by Motif', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'density_change_distribution.png', 
               dpi=150, bbox_inches='tight')
    plt.close()

def plot_scatter_relationships(motif_info_list):
    """Create scatter plots showing relationships between variables."""
    print(f"\n  Plotting scatter relationships...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Collect data from all instances
    ore_values = []
    watermill_values = []
    density_values = []
    density_changes = []
    
    for motif in motif_info_list:
        for inst in motif['instances']:
            ore_values.extend(inst['data']['Ore'])
            watermill_values.extend(inst['data']['WaterMill'])
            density_values.extend(inst['data']['DensityHC'])
            density_changes.append(inst['data']['DensityHC'][-1] - inst['data']['DensityHC'][0])
    
    # Plot 1: Ore vs DensityHC
    ax1 = axes[0, 0]
    ax1.scatter(ore_values, density_values, alpha=0.3, s=10, color='steelblue')
    ax1.set_xlabel('Ore', fontweight='bold')
    ax1.set_ylabel('DensityHC', fontweight='bold')
    ax1.set_title('Ore vs DensityHC', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(ore_values, density_values, 1)
    p = np.poly1d(z)
    ax1.plot(sorted(ore_values), p(sorted(ore_values)), "r--", linewidth=2, alpha=0.8)
    
    # Plot 2: WaterMill vs DensityHC
    ax2 = axes[0, 1]
    ax2.scatter(watermill_values, density_values, alpha=0.3, s=10, color='coral')
    ax2.set_xlabel('WaterMill', fontweight='bold')
    ax2.set_ylabel('DensityHC', fontweight='bold')
    ax2.set_title('WaterMill vs DensityHC', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(watermill_values, density_values, 1)
    p = np.poly1d(z)
    ax2.plot(sorted(watermill_values), p(sorted(watermill_values)), "r--", linewidth=2, alpha=0.8)
    
    # Plot 3: Ore vs WaterMill
    ax3 = axes[1, 0]
    ax3.scatter(ore_values, watermill_values, alpha=0.3, s=10, color='seagreen')
    ax3.set_xlabel('Ore', fontweight='bold')
    ax3.set_ylabel('WaterMill', fontweight='bold')
    ax3.set_title('Ore vs WaterMill', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Instance-level density change
    ax4 = axes[1, 1]
    instance_indices = range(len(density_changes))
    ax4.scatter(instance_indices, density_changes, alpha=0.6, s=30, color='purple')
    ax4.axhline(0, color='red', linewidth=1, linestyle='--')
    ax4.set_xlabel('Instance Index', fontweight='bold')
    ax4.set_ylabel('DensityHC Change', fontweight='bold')
    ax4.set_title('DensityHC Change per Instance', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'density_scatter_relationships.png', 
               dpi=150, bbox_inches='tight')
    plt.close()

def save_analysis_summary(analysis_results):
    """Save analysis summary to CSV."""
    print(f"\n  Saving analysis summary...")
    
    summary_data = []
    for result in analysis_results:
        summary_data.append({
            'motif_id': result['motif_id'],
            'num_instances': result['num_instances'],
            'avg_density_change': result['avg_density_change'],
            'avg_ore_density_corr': result['avg_ore_density_corr'],
            'avg_watermill_density_corr': result['avg_watermill_density_corr'],
            'avg_ore_lag_min': result['avg_ore_lag'],
            'avg_watermill_lag_min': result['avg_watermill_lag']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / 'density_analysis_summary.csv', index=False)
    print(f"  ✓ Saved to {OUTPUT_DIR / 'density_analysis_summary.csv'}")
    
    return summary_df

def main():
    """Main execution flow."""
    print(f"\n{'='*60}")
    print("BALL-MILLING PROCESS ANALYSIS")
    print("Focus: DensityHC behavior when WaterZumpf is constant")
    print(f"{'='*60}\n")
    
    # Load data
    df = load_data(DATA_FILE)
    
    # Find constrained motifs
    motif_info_list = find_constrained_motifs(
        df, 
        window_size=WINDOW_SIZE,
        max_motifs=MAX_MOTIFS,
        radius=RADIUS
    )
    
    if not motif_info_list:
        print("\n⚠ No motifs found matching the constraints!")
        return
    
    # Analyze density behavior
    analysis_results = analyze_density_behavior(motif_info_list)
    
    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    plot_motif_instances(motif_info_list, top_n=5)
    plot_correlation_analysis(analysis_results)
    plot_density_change_distribution(analysis_results)
    plot_scatter_relationships(motif_info_list)
    
    # Save summary
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    summary_df = save_analysis_summary(analysis_results)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print("\nKey Findings:")
    print(f"  • Found {len(motif_info_list)} motif groups")
    print(f"  • Total instances: {sum(len(m['instances']) for m in motif_info_list)}")
    print(f"  • Avg density change: {np.mean([r['avg_density_change'] for r in analysis_results]):.2f}")
    print(f"  • Avg Ore-Density correlation: {np.mean([r['avg_ore_density_corr'] for r in analysis_results]):.3f}")
    print(f"  • Avg WaterMill-Density correlation: {np.mean([r['avg_watermill_density_corr'] for r in analysis_results]):.3f}")
    
    print(f"\nGenerated files in '{OUTPUT_DIR}/':")
    print(f"  • density_motif_X_instances.png (motif visualizations)")
    print(f"  • density_correlation_lag_analysis.png")
    print(f"  • density_change_distribution.png")
    print(f"  • density_scatter_relationships.png")
    print(f"  • density_analysis_summary.csv")
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
