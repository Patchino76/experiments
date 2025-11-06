"""
Generic visualization functions for discovered motifs.

Provides visualization capabilities that work with any pattern type.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from pathlib import Path
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.base_pattern import Motif

logger = logging.getLogger(__name__)


class PatternVisualizer:
    """
    Creates visualizations for discovered motifs.
    """
    
    def __init__(self):
        """Initialize visualizer."""
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_motif_overview(
        self,
        motifs: List[Motif],
        output_path: Path
    ):
        """
        Create overview plot of all motifs.
        
        Args:
            motifs: List of motifs
            output_path: Output file path
        """
        logger.info("Creating motif overview plot...")
        
        if not motifs:
            logger.warning("  No motifs to plot")
            return
        
        # Prepare data
        data = []
        for motif in motifs:
            pattern_type = motif.metadata.get('pattern_type', 'unknown')
            data.append({
                'motif_id': motif.motif_id,
                'pattern_type': pattern_type,
                'num_instances': len(motif.instances),
                'avg_distance': motif.avg_distance
            })
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Instances per motif
        ax = axes[0, 0]
        df.plot(x='motif_id', y='num_instances', kind='bar', ax=ax, color='steelblue')
        ax.set_title('Instances per Motif')
        ax.set_xlabel('Motif ID')
        ax.set_ylabel('Number of Instances')
        ax.legend().remove()
        
        # 2. Average distance per motif
        ax = axes[0, 1]
        df.plot(x='motif_id', y='avg_distance', kind='bar', ax=ax, color='coral')
        ax.set_title('Average Distance per Motif')
        ax.set_xlabel('Motif ID')
        ax.set_ylabel('Average Distance')
        ax.legend().remove()
        
        # 3. Pattern type distribution
        ax = axes[1, 0]
        pattern_counts = df['pattern_type'].value_counts()
        pattern_counts.plot(kind='bar', ax=ax, color='seagreen')
        ax.set_title('Motifs by Pattern Type')
        ax.set_xlabel('Pattern Type')
        ax.set_ylabel('Number of Motifs')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Instances by pattern type
        ax = axes[1, 1]
        instance_counts = df.groupby('pattern_type')['num_instances'].sum()
        instance_counts.plot(kind='bar', ax=ax, color='mediumpurple')
        ax.set_title('Total Instances by Pattern Type')
        ax.set_xlabel('Pattern Type')
        ax.set_ylabel('Total Instances')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved overview plot to {output_path.name}")
    
    def plot_density_analysis(
        self,
        analyses: List[Dict[str, Any]],
        output_path: Path,
        pattern_name: str = None
    ):
        """
        Create density analysis visualization.
        
        Args:
            analyses: List of analysis dictionaries
            output_path: Output file path
            pattern_name: Optional pattern name for title
        """
        if not analyses:
            logger.warning(f"  No analyses to plot for {pattern_name}")
            return
        
        logger.info(f"Creating density analysis plot for {pattern_name}...")
        
        df = pd.DataFrame(analyses)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        title_prefix = f"{pattern_name.title()} - " if pattern_name else ""
        
        # 1. Density change distribution
        ax = axes[0, 0]
        ax.hist(df['density_change'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(df['density_change'].mean(), color='red', linestyle='--', label='Mean')
        ax.set_title(f'{title_prefix}Density Change Distribution')
        ax.set_xlabel('Density Change')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # 2. Correlations
        ax = axes[0, 1]
        correlations = df[['ore_correlation', 'watermill_correlation', 'waterzumpf_correlation']].mean()
        correlations.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title(f'{title_prefix}Average Correlations with Density')
        ax.set_xlabel('Variable')
        ax.set_ylabel('Correlation')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Density range vs instances
        ax = axes[1, 0]
        ax.scatter(df.index, df['density_range'], alpha=0.6, color='coral')
        ax.set_title(f'{title_prefix}Density Range per Instance')
        ax.set_xlabel('Instance Index')
        ax.set_ylabel('Density Range')
        
        # 4. Lag analysis
        ax = axes[1, 1]
        lags = df[['ore_lag', 'watermill_lag', 'waterzumpf_lag']].mean()
        lags.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title(f'{title_prefix}Average Lags with Density')
        ax.set_xlabel('Variable')
        ax.set_ylabel('Lag (minutes)')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved density analysis plot to {output_path.name}")
    
    def plot_individual_motifs(
        self,
        motifs: List[Motif],
        features: List[str],
        output_dir: Path,
        top_n: int = 10
    ):
        """
        Plot individual motif instances.
        
        Args:
            motifs: List of motifs
            features: Features to plot
            output_dir: Output directory
            top_n: Number of top motifs to plot
        """
        logger.info(f"Plotting top {top_n} individual motifs...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sort by number of instances
        sorted_motifs = sorted(motifs, key=lambda m: len(m.instances), reverse=True)[:top_n]
        
        for motif in sorted_motifs:
            pattern_type = motif.metadata.get('pattern_type', 'unknown')
            
            fig, axes = plt.subplots(len(features), 1, figsize=(12, 3 * len(features)))
            if len(features) == 1:
                axes = [axes]
            
            fig.suptitle(f'Motif {motif.motif_id} ({pattern_type}) - {len(motif.instances)} instances')
            
            for feat_idx, feature in enumerate(features):
                ax = axes[feat_idx]
                
                for inst_idx, instance in enumerate(motif.instances):
                    data = instance.get_feature(feature)
                    if data is not None:
                        ax.plot(data, alpha=0.6, linewidth=1)
                
                ax.set_ylabel(feature)
                ax.set_xlabel('Time (minutes)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            file_path = output_dir / f'motif_{motif.motif_id:03d}_{pattern_type}.png'
            plt.savefig(file_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        logger.info(f"  ✓ Saved {len(sorted_motifs)} motif plots to {output_dir.name}/")
    
    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        features: List[str],
        output_path: Path
    ):
        """
        Create correlation heatmap.
        
        Args:
            df: DataFrame with features
            features: Features to include
            output_path: Output file path
        """
        logger.info("Creating correlation heatmap...")
        
        # Filter features that exist in df
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 2:
            logger.warning("  Not enough features for correlation heatmap")
            return
        
        # Calculate correlation
        corr_matrix = df[available_features].corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation'}
        )
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved correlation heatmap to {output_path.name}")
    
    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        features: List[str],
        output_path: Path
    ):
        """
        Create feature distribution plots.
        
        Args:
            df: DataFrame with features
            features: Features to plot
            output_path: Output file path
        """
        logger.info("Creating feature distribution plots...")
        
        # Filter features that exist in df
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            logger.warning("  No features available for distribution plots")
            return
        
        # Calculate grid size
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(available_features):
            ax = axes[idx]
            df[feature].hist(bins=50, ax=ax, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(available_features), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved feature distributions to {output_path.name}")
