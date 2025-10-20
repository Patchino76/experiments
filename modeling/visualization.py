"""
Visualization module for motif analysis.

Provides plotting functions for motifs, analysis results, and model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import logging

from motif_discovery import Motif

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


def plot_motif_instances(
    motif: Motif,
    feature_columns: List[str],
    output_path: Path,
    max_instances: int = 10
):
    """
    Plot instances of a single motif.
    
    Args:
        motif: Motif to plot
        feature_columns: Features to plot
        output_path: Path to save plot
        max_instances: Maximum instances to plot
    """
    n_features = len(feature_columns)
    n_instances = min(len(motif.instances), max_instances)
    
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]
    
    for feat_idx, feature in enumerate(feature_columns):
        ax = axes[feat_idx]
        
        for inst_idx, instance in enumerate(motif.instances[:n_instances]):
            if feature in instance.data:
                data = instance.data[feature]
                x = np.arange(len(data))
                ax.plot(x, data, alpha=0.6, label=f'Instance {inst_idx + 1}')
        
        ax.set_ylabel(feature, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if feat_idx == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=min(5, n_instances))
    
    axes[-1].set_xlabel('Time (minutes)', fontsize=11)
    
    fig.suptitle(
        f'Motif {motif.motif_id} - {len(motif.instances)} instances (showing {n_instances})',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved motif {motif.motif_id} plot to {output_path.name}")


def plot_all_motifs(
    motifs: List[Motif],
    feature_columns: List[str],
    output_dir: Path,
    top_n: int = 10
):
    """
    Plot top N motifs individually.
    
    Args:
        motifs: List of motifs
        feature_columns: Features to plot
        output_dir: Directory to save plots
        top_n: Number of top motifs to plot
    """
    logger.info(f"Plotting top {top_n} motifs...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, motif in enumerate(motifs[:top_n]):
        output_path = output_dir / f'motif_{motif.motif_id:02d}.png'
        plot_motif_instances(motif, feature_columns, output_path)


def plot_motif_overview(
    motifs: List[Motif],
    output_path: Path
):
    """
    Create overview plot showing motif statistics.
    
    Args:
        motifs: List of motifs
        output_path: Path to save plot
    """
    logger.info("Creating motif overview plot...")
    
    # Extract statistics
    motif_ids = [m.motif_id for m in motifs]
    num_instances = [len(m.instances) for m in motifs]
    avg_distances = [m.avg_distance for m in motifs]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Number of instances per motif
    ax1 = axes[0]
    bars1 = ax1.bar(motif_ids, num_instances, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Motif ID', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Instances', fontsize=11, fontweight='bold')
    ax1.set_title('Instances per Motif', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Average distance per motif
    ax2 = axes[1]
    bars2 = ax2.bar(motif_ids, avg_distances, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Motif ID', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Distance', fontsize=11, fontweight='bold')
    ax2.set_title('Average Distance per Motif', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved overview plot to {output_path.name}")


def plot_density_analysis(
    analysis_results: List[dict],
    output_path: Path
):
    """
    Plot density behavior analysis results.
    
    Args:
        analysis_results: List of analysis result dictionaries
        output_path: Path to save plot
    """
    logger.info("Creating density analysis plot...")
    
    if not analysis_results:
        logger.warning("  ⚠ No analysis results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    motif_ids = [r['motif_id'] for r in analysis_results]
    
    # Plot 1: Density change
    ax1 = axes[0, 0]
    density_changes = [r['avg_density_change'] for r in analysis_results]
    bars = ax1.bar(motif_ids, density_changes, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Motif ID', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Avg Density Change', fontsize=11, fontweight='bold')
    ax1.set_title('Average DensityHC Change per Motif', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Ore-Density correlation
    ax2 = axes[0, 1]
    ore_corrs = [r['avg_ore_density_corr'] for r in analysis_results]
    bars = ax2.bar(motif_ids, ore_corrs, color='coral', alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Motif ID', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Correlation', fontsize=11, fontweight='bold')
    ax2.set_title('Ore-Density Correlation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: WaterMill-Density correlation
    ax3 = axes[1, 0]
    watermill_corrs = [r['avg_watermill_density_corr'] for r in analysis_results]
    bars = ax3.bar(motif_ids, watermill_corrs, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Motif ID', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Correlation', fontsize=11, fontweight='bold')
    ax3.set_title('WaterMill-Density Correlation', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Number of instances
    ax4 = axes[1, 1]
    num_instances = [r['num_instances'] for r in analysis_results]
    bars = ax4.bar(motif_ids, num_instances, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Motif ID', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Number of Instances', fontsize=11, fontweight='bold')
    ax4.set_title('Instances per Motif', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved density analysis plot to {output_path.name}")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_columns: List[str],
    output_path: Path
):
    """
    Plot correlation heatmap for features.
    
    Args:
        df: DataFrame with features
        feature_columns: Features to include
        output_path: Path to save plot
    """
    logger.info("Creating correlation heatmap...")
    
    # Calculate correlation matrix
    corr_matrix = df[feature_columns].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved correlation heatmap to {output_path.name}")


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_columns: List[str],
    output_path: Path
):
    """
    Plot distributions of features.
    
    Args:
        df: DataFrame with features
        feature_columns: Features to plot
        output_path: Path to save plot
    """
    logger.info("Creating feature distribution plots...")
    
    n_features = len(feature_columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]
        
        if feature in df.columns:
            data = df[feature].dropna()
            
            ax.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel(feature, fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            mean_val = data.mean()
            std_val = data.std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved feature distributions to {output_path.name}")


def save_summary_report(
    motifs: List[Motif],
    analysis_results: Optional[List[dict]],
    output_path: Path
):
    """
    Save text summary report.
    
    Args:
        motifs: List of motifs
        analysis_results: Optional density analysis results
        output_path: Path to save report
    """
    logger.info("Creating summary report...")
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MOTIF DISCOVERY SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Motifs Discovered: {len(motifs)}\n")
        total_instances = sum(len(m.instances) for m in motifs)
        f.write(f"Total Instances: {total_instances}\n")
        
        if motifs:
            avg_distance = np.mean([m.avg_distance for m in motifs])
            f.write(f"Average Distance: {avg_distance:.3f}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("MOTIF DETAILS\n")
        f.write("-" * 80 + "\n\n")
        
        for motif in motifs:
            f.write(f"Motif {motif.motif_id}:\n")
            f.write(f"  Instances: {len(motif.instances)}\n")
            f.write(f"  Avg Distance: {motif.avg_distance:.3f}\n")
            
            if motif.metadata:
                f.write(f"  Metadata: {motif.metadata}\n")
            
            f.write("\n")
        
        if analysis_results:
            f.write("\n" + "-" * 80 + "\n")
            f.write("DENSITY ANALYSIS RESULTS\n")
            f.write("-" * 80 + "\n\n")
            
            for result in analysis_results:
                f.write(f"Motif {result['motif_id']}:\n")
                f.write(f"  Instances: {result['num_instances']}\n")
                f.write(f"  Avg Density Change: {result['avg_density_change']:+.2f}\n")
                f.write(f"  Ore-Density Corr: {result['avg_ore_density_corr']:+.3f}\n")
                f.write(f"  WaterMill-Density Corr: {result['avg_watermill_density_corr']:+.3f}\n")
                f.write(f"  Ore Lag: {result['avg_ore_lag']:.0f} min\n")
                f.write(f"  WaterMill Lag: {result['avg_watermill_lag']:.0f} min\n")
                f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"  ✓ Saved summary report to {output_path.name}")
