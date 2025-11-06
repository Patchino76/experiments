"""
Density Constraint Pattern Discovery.

This module implements pattern discovery for stable WaterZumpf with varying
Ore and WaterMill, which is useful for analyzing density behavior.
"""
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import stumpy
from pathlib import Path
import matplotlib.pyplot as plt

from modeling.motif_discovery import Motif, MotifInstance
from ..core.base_pattern import BasePattern
from ..core.pattern_registry import PatternRegistry

logger = logging.getLogger(__name__)

def calculate_variability(series: pd.Series, window: int = 60) -> np.ndarray:
    """Calculate coefficient of variation for rolling windows."""
    rolling_std = series.rolling(window=window, min_periods=1).std()
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    return (rolling_std / rolling_mean).fillna(0).values

@PatternRegistry.register('density')
class DensityPattern(BasePattern):
    """
    Discovers patterns where WaterZumpf is stable while Ore and WaterMill vary.
    
    This pattern is useful for analyzing how density responds to changes in
    feed rate and mill water while sump water remains stable.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the density pattern discovery.
        
        Args:
            config: Configuration dictionary with pattern parameters
        """
        super().__init__(config)
        self.window_size = config.get('window_size', 60)
        self.max_motifs = config.get('max_motifs', 10)
        self.radius = config.get('radius', 4.5)
        self.waterzumpf_max_cv = config.get('waterzumpf_max_cv', 0.01)
        self.ore_min_cv = config.get('ore_min_cv', 0.0008)
        self.watermill_min_cv = config.get('watermill_min_cv', 0.0015)
    
    def discover(self, data: pd.DataFrame) -> List[Motif]:
        """
        Discover density-constrained patterns in the data.
        
        Args:
            data: Input time series data as a DataFrame
            
        Returns:
            List of discovered Motif objects
        """
        logger.info("Discovering density-constrained patterns...")
        
        # Prepare time series data
        features = ['Ore', 'WaterMill', 'WaterZumpf']
        T = self._prepare_time_series(data, features)
        
        # Calculate matrix profile
        logger.info("  Computing multivariate matrix profile...")
        matrix_profile, profile_indices = stumpy.mstump(T, m=self.window_size)
        mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))
        
        # Find constrained patterns
        motifs = self._find_constrained_patterns(
            data, T, matrix_profile, profile_indices, mp_distances
        )
        
        self.motifs = motifs
        logger.info(f"Discovered {len(motifs)} density-constrained motifs")
        return motifs
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the discovered density patterns.
        
        Args:
            data: Original input data
            
        Returns:
            Dictionary of analysis results
        """
        if not self.motifs:
            return {}
        
        analysis = {
            'num_motifs': len(self.motifs),
            'total_instances': sum(len(motif.instances) for motif in self.motifs),
            'avg_instances_per_motif': np.mean([len(m.instances) for m in self.motifs]),
            'avg_distance': np.mean([m.avg_distance for m in self.motifs])
        }
        
        self.analysis_results = analysis
        return analysis
    
    def visualize(self, output_dir: Path) -> None:
        """
        Generate visualizations for the discovered patterns.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if not self.motifs:
            return
            
        try:
            # Create a figure with subplots for each motif
            for i, motif in enumerate(self.motifs):
                self._plot_motif(motif, output_dir / f'motif_{i+1}.png')
                
            logger.info(f"Saved visualizations to {output_dir}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _prepare_time_series(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """Prepare normalized time series data for analysis."""
        # Normalize each feature
        normalized = (df[features] - df[features].mean()) / df[features].std()
        return normalized.values.T
    
    def _find_constrained_patterns(
        self,
        df: pd.DataFrame,
        T: np.ndarray,
        matrix_profile: np.ndarray,
        profile_indices: np.ndarray,
        mp_distances: np.ndarray
    ) -> List[Motif]:
        """Find patterns that match the density constraints."""
        n_windows = matrix_profile.shape[1]
        used_indices = set()
        motifs = []
        
        for _ in range(self.max_motifs):
            # Find the best seed that meets our constraints
            seed_idx = self._find_constrained_seed(df, mp_distances, used_indices, n_windows)
            if seed_idx is None or mp_distances[seed_idx] > self.radius:
                break
                
            # Find similar instances
            instances = self._find_constrained_instances(
                df, T, seed_idx, n_windows, used_indices, mp_distances
            )
            
            if len(instances) >= 2:  # Require at least 2 instances for a motif
                motif = self._create_motif(instances)
                motifs.append(motif)
                
                # Mark these indices as used
                for inst in instances:
                    for offset in range(-self.window_size, self.window_size):
                        used_indices.add(inst['start'] + offset)
        
        return motifs
    
    def _find_constrained_seed(
        self,
        df: pd.DataFrame,
        mp_distances: np.ndarray,
        used_indices: set,
        n_windows: int
    ) -> Optional[int]:
        """Find the best seed index that meets the constraints."""
        best_idx = None
        best_distance = float('inf')
        
        for i in range(n_windows):
            if i in used_indices:
                continue
                
            # Check variability constraints
            if not self._check_variability_constraints(df, i):
                continue
                
            # Track the best valid seed
            if mp_distances[i] < best_distance:
                best_distance = mp_distances[i]
                best_idx = i
        
        return best_idx
    
    def _check_variability_constraints(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if the window at idx meets the variability constraints."""
        window = df.iloc[idx:idx + self.window_size]
        
        # Calculate CV for each feature in the window
        waterzumpf_cv = window['WaterZumpf'].std() / window['WaterZumpf'].mean()
        ore_cv = window['Ore'].std() / window['Ore'].mean()
        watermill_cv = window['WaterMill'].std() / window['WaterMill'].mean()
        
        # Check constraints
        return (waterzumpf_cv <= self.waterzumpf_max_cv and
                ore_cv >= self.ore_min_cv and
                watermill_cv >= self.watermill_min_cv)
    
    def _find_constrained_instances(
        self,
        df: pd.DataFrame,
        T: np.ndarray,
        seed_idx: int,
        n_windows: int,
        used_indices: set,
        mp_distances: np.ndarray
    ) -> List[Dict]:
        """Find all instances of the pattern starting at seed_idx."""
        instances = []
        
        # Add the seed instance
        seed_data = {
            'start': seed_idx,
            'end': seed_idx + self.window_size,
            'distance': mp_distances[seed_idx],
            'data': df.iloc[seed_idx:seed_idx + self.window_size].to_dict('list')
        }
        instances.append(seed_data)
        
        # Find similar instances
        for i in range(n_windows):
            if (i == seed_idx or i in used_indices or 
                not self._check_variability_constraints(df, i)):
                continue
                
            # Check distance to seed
            dist = np.linalg.norm(T[:, i:i+self.window_size] - 
                                 T[:, seed_idx:seed_idx+self.window_size])
            
            if dist <= self.radius:
                inst_data = {
                    'start': i,
                    'end': i + self.window_size,
                    'distance': dist,
                    'data': df.iloc[i:i + self.window_size].to_dict('list')
                }
                instances.append(inst_data)
        
        return instances
    
    def _create_motif(self, instances: List[Dict]) -> Motif:
        """Create a Motif object from a list of instances."""
        motif = Motif(motif_id=len(self.motifs) + 1)
        motif.add_metadata('pattern_type', 'density_constraint')
        
        for inst_data in instances:
            instance = MotifInstance(
                start=inst_data['start'],
                end=inst_data['end'],
                distance=inst_data['distance'],
                data=inst_data['data']
            )
            motif.add_instance(instance)
        
        return motif
    
    def _plot_motif(self, motif: Motif, output_path: Path) -> None:
        """Generate a plot for a single motif."""
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        features = ['Ore', 'WaterMill', 'WaterZumpf']
        
        for i, feature in enumerate(features):
            ax = axs[i]
            for inst in motif.instances:
                data = inst.get_feature(feature)
                if data is not None:
                    ax.plot(data, alpha=0.6)
            ax.set_ylabel(feature)
            ax.grid(True)
        
        plt.suptitle(f'Motif {motif.motif_id} - {len(motif.instances)} instances')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
