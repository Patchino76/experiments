"""
Generic analysis functions for discovered motifs.

Provides analysis capabilities that work with any pattern type.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import correlate
from typing import List, Dict, Any
from pathlib import Path
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.base_pattern import Motif

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """
    Analyzes discovered motifs and generates insights.
    """
    
    def __init__(self):
        """Initialize analyzer."""
        pass
    
    def analyze_density_behavior(
        self,
        motifs: List[Motif],
        density_column: str = 'DensityHC'
    ) -> List[Dict[str, Any]]:
        """
        Analyze density behavior in motifs.
        
        Works for any pattern type that includes density data.
        
        Args:
            motifs: List of motifs
            density_column: Name of density column
            
        Returns:
            List of analysis dictionaries
        """
        logger.info(f"Analyzing density behavior for {len(motifs)} motifs...")
        
        analyses = []
        
        for motif in motifs:
            pattern_type = motif.metadata.get('pattern_type', 'unknown')
            
            for inst_idx, instance in enumerate(motif.instances):
                # Get density data
                density_data = instance.get_feature(density_column)
                if density_data is None or len(density_data) == 0:
                    continue
                
                # Basic statistics
                density_mean = np.mean(density_data)
                density_std = np.std(density_data)
                density_change = density_data[-1] - density_data[0]
                density_range = np.max(density_data) - np.min(density_data)
                
                # Correlation analysis with MVs
                correlations = {}
                lags = {}
                
                for mv_name in ['Ore', 'WaterMill', 'WaterZumpf']:
                    mv_data = instance.get_feature(mv_name)
                    if mv_data is not None and len(mv_data) == len(density_data):
                        # Pearson correlation
                        try:
                            corr, _ = pearsonr(mv_data, density_data)
                            correlations[mv_name] = corr
                        except:
                            correlations[mv_name] = 0.0
                        
                        # Lag analysis
                        try:
                            cross_corr = correlate(mv_data, density_data, mode='full')
                            lag_idx = np.argmax(np.abs(cross_corr))
                            lag = lag_idx - (len(mv_data) - 1)
                            lags[mv_name] = int(lag)
                        except:
                            lags[mv_name] = 0
                
                analysis = {
                    'motif_id': motif.motif_id,
                    'pattern_type': pattern_type,
                    'instance_idx': inst_idx,
                    'density_mean': density_mean,
                    'density_std': density_std,
                    'density_change': density_change,
                    'density_range': density_range,
                    'ore_correlation': correlations.get('Ore', 0.0),
                    'watermill_correlation': correlations.get('WaterMill', 0.0),
                    'waterzumpf_correlation': correlations.get('WaterZumpf', 0.0),
                    'ore_lag': lags.get('Ore', 0),
                    'watermill_lag': lags.get('WaterMill', 0),
                    'waterzumpf_lag': lags.get('WaterZumpf', 0),
                }
                
                # Add CV metadata if available
                for key, value in instance.metadata.items():
                    if key.endswith('_cv'):
                        analysis[key] = value
                
                analyses.append(analysis)
        
        logger.info(f"  ✓ Analyzed {len(analyses)} instances")
        
        return analyses
    
    def save_analysis(
        self,
        analyses: List[Dict[str, Any]],
        output_path: Path,
        pattern_name: str
    ):
        """
        Save analysis results to CSV.
        
        Args:
            analyses: List of analysis dictionaries
            output_path: Output directory path
            pattern_name: Name of the pattern
        """
        if not analyses:
            logger.warning(f"No analyses to save for pattern '{pattern_name}'")
            return
        
        df = pd.DataFrame(analyses)
        
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f'{pattern_name}_analysis.csv'
        
        df.to_csv(file_path, index=False)
        logger.info(f"  ✓ Saved analysis to {file_path.name}")
    
    def generate_summary_report(
        self,
        motifs: List[Motif],
        analyses: List[Dict[str, Any]],
        output_path: Path
    ):
        """
        Generate text summary report.
        
        Args:
            motifs: List of motifs
            analyses: List of analysis dictionaries
            output_path: Output file path
        """
        logger.info("Generating summary report...")
        
        lines = [
            "=" * 80,
            "MOTIF DISCOVERY SUMMARY REPORT",
            "=" * 80,
            f"\nTotal Motifs: {len(motifs)}",
            f"Total Instances: {sum(len(m) for m in motifs)}",
            "\n" + "-" * 80,
            "MOTIF DETAILS",
            "-" * 80
        ]
        
        for motif in motifs:
            pattern_type = motif.metadata.get('pattern_type', 'unknown')
            lines.extend([
                f"\nMotif {motif.motif_id} ({pattern_type}):",
                f"  Instances: {len(motif.instances)}",
                f"  Avg Distance: {motif.avg_distance:.4f}"
            ])
        
        if analyses:
            lines.extend([
                "\n" + "-" * 80,
                "DENSITY BEHAVIOR ANALYSIS",
                "-" * 80
            ])
            
            df = pd.DataFrame(analyses)
            
            # Group by pattern type
            for pattern_type in df['pattern_type'].unique():
                pattern_data = df[df['pattern_type'] == pattern_type]
                
                lines.extend([
                    f"\nPattern: {pattern_type}",
                    f"  Instances: {len(pattern_data)}",
                    f"  Avg Density Change: {pattern_data['density_change'].mean():.2f}",
                    f"  Avg Ore Correlation: {pattern_data['ore_correlation'].mean():.3f}",
                    f"  Avg WaterMill Correlation: {pattern_data['watermill_correlation'].mean():.3f}",
                    f"  Avg WaterZumpf Correlation: {pattern_data['waterzumpf_correlation'].mean():.3f}"
                ])
        
        lines.append("\n" + "=" * 80)
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"  ✓ Summary report saved to {output_path.name}")
