"""
Pressure Constraint Pattern Discovery.

This module implements pattern discovery for scenarios where PressureHC is stable
while other variables (Ore, WaterMill, WaterZumpf) vary, capturing optimal control regions.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import stumpy
from pathlib import Path
import matplotlib.pyplot as plt

from modeling.motif_discovery import Motif, MotifInstance
from ..core.base_pattern import BasePattern
from ..core.pattern_registry import PatternRegistry

logger = logging.getLogger(__name__)

@PatternRegistry.register('pressure')
class PressurePattern(BasePattern):
    """
    Discovers patterns where PressureHC is stable while other variables vary.
    
    This pattern is useful for identifying optimal control regions where the
    system maintains stable pressure while other variables adjust.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pressure constraint pattern discovery.
        
        Args:
            config: Configuration dictionary with pattern parameters
        """
        super().__init__(config)
        self.window_size = config.get('window_size', 60)
        self.max_motifs = config.get('max_motifs', 10)
        self.radius = config.get('radius', 3.0)
        self.pressure_max_cv = config.get('pressure_max_cv', 0.01)  # 1% CV for stable pressure
        self.min_variation = config.get('min_variation', 0.001)  # Minimum CV for other variables
        self.required_features = ['PressureHC', 'Ore', 'WaterMill', 'WaterZumpf']
    
    def discover(self, data: pd.DataFrame) -> List[Motif]:
        """
        Discover pressure-constrained patterns in the data.
        
        Args:
            data: Input time series data as a DataFrame
            
        Returns:
            List of discovered Motif objects
        """
        logger.info("Discovering pressure-constrained patterns...")
        
        # Check for required columns
        missing = [f for f in self.required_features if f not in data.columns]
        if missing:
            logger.warning(f"Missing required features: {missing}")
            return []
        
        # Prepare time series data (normalized)
        T = self._prepare_time_series(data, self.required_features)
        
        # Calculate matrix profile for pressure-constrained patterns
        logger.info("  Computing pressure-constrained matrix profile...")
        matrix_profile, profile_indices = stumpy.mstump(T, m=self.window_size)
        mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))
        
        # Find pressure-constrained patterns
        motifs = self._find_pressure_patterns(
            data, T, matrix_profile, profile_indices, mp_distances
        )
        
        self.motifs = motifs
        logger.info(f"Discovered {len(motifs)} pressure-constrained motifs")
        return motifs
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the discovered pressure-constrained patterns.
        
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
            'avg_pressure_stability': np.mean([m.metadata.get('pressure_stability', 0) 
                                            for m in self.motifs]),
            'avg_ore_variation': np.mean([m.metadata.get('ore_variation', 0) 
                                        for m in self.motifs]),
            'avg_watermill_variation': np.mean([m.metadata.get('watermill_variation', 0) 
                                              for m in self.motifs]),
            'avg_waterzumpf_variation': np.mean([m.metadata.get('waterzumpf_variation', 0) 
                                               for m in self.motifs])
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
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a figure with subplots for each motif
            for i, motif in enumerate(self.motifs):
                self._plot_motif(motif, output_dir / f'pressure_motif_{i+1}.png')
                
            # Create a summary plot
            self._plot_summary(output_dir / 'pressure_patterns_summary.png')
                
            logger.info(f"Saved visualizations to {output_dir}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _prepare_time_series(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """Prepare normalized time series data for analysis."""
        # Normalize each feature
        normalized = (df[features] - df[features].mean()) / df[features].std()
        return normalized.values.T
    
    def _find_pressure_patterns(
        self,
        df: pd.DataFrame,
        T: np.ndarray,
        matrix_profile: np.ndarray,
        profile_indices: np.ndarray,
        mp_distances: np.ndarray
    ) -> List[Motif]:
        """Find patterns that match the pressure constraints."""
        n_windows = matrix_profile.shape[1]
        used_indices = set()
        motifs = []
        
        for _ in range(self.max_motifs):
            # Find the best seed that meets our constraints
            seed_idx = self._find_pressure_seed(df, mp_distances, used_indices, n_windows)
            if seed_idx is None or mp_distances[seed_idx] > self.radius:
                break
                
            # Find similar instances
            instances = self._find_pressure_instances(
                df, T, seed_idx, n_windows, used_indices, mp_distances
            )
            
            if len(instances) >= 2:  # Require at least 2 instances for a motif
                motif = self._create_motif(instances, df)
                motifs.append(motif)
                
                # Mark these indices as used
                for inst in instances:
                    for offset in range(-self.window_size//2, self.window_size//2):
                        used_indices.add(inst['start'] + offset)
        
        return motifs
    
    def _find_pressure_seed(
        self,
        df: pd.DataFrame,
        mp_distances: np.ndarray,
        used_indices: set,
        n_windows: int
    ) -> Optional[int]:
        """Find the best seed index that meets the pressure constraints."""
        best_idx = None
        best_distance = float('inf')
        
        for i in range(n_windows):
            if i in used_indices:
                continue
                
            # Check pressure stability and other constraints
            if not self._check_pressure_constraints(df, i):
                continue
                
            # Track the best valid seed
            if mp_distances[i] < best_distance:
                best_distance = mp_distances[i]
                best_idx = i
        
        return best_idx
    
    def _check_pressure_constraints(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if the window at idx meets the pressure constraints."""
        window = df.iloc[idx:idx + self.window_size]
        
        # Calculate CV for pressure (should be stable)
        pressure_cv = window['PressureHC'].std() / window['PressureHC'].mean()
        
        # Calculate CV for other variables (should be varying)
        ore_cv = window['Ore'].std() / window['Ore'].mean()
        watermill_cv = window['WaterMill'].std() / window['WaterMill'].mean()
        waterzumpf_cv = window['WaterZumpf'].std() / window['WaterZumpf'].mean()
        
        # Check constraints
        return (pressure_cv <= self.pressure_max_cv and
                (ore_cv >= self.min_variation or 
                 watermill_cv >= self.min_variation or
                 waterzumpf_cv >= self.min_variation))
    
    def _find_pressure_instances(
        self,
        df: pd.DataFrame,
        T: np.ndarray,
        seed_idx: int,
        n_windows: int,
        used_indices: set,
        mp_distances: np.ndarray
    ) -> List[Dict]:
        """Find all instances of the pressure pattern starting at seed_idx."""
        instances = []
        
        # Add the seed instance
        seed_data = {
            'start': seed_idx,
            'end': seed_idx + self.window_size,
            'distance': mp_distances[seed_idx],
            'data': df.iloc[seed_idx:seed_idx + self.window_size].to_dict('list'),
            'pressure_stability': self._get_pressure_stability(df, seed_idx),
            'ore_variation': self._get_variation(df, 'Ore', seed_idx),
            'watermill_variation': self._get_variation(df, 'WaterMill', seed_idx),
            'waterzumpf_variation': self._get_variation(df, 'WaterZumpf', seed_idx)
        }
        instances.append(seed_data)
        
        # Find similar instances
        for i in range(n_windows):
            if (i == seed_idx or i in used_indices or 
                not self._check_pressure_constraints(df, i)):
                continue
                
            # Check distance to seed
            dist = np.linalg.norm(T[:, i:i+self.window_size] - 
                                 T[:, seed_idx:seed_idx+self.window_size])
            
            if dist <= self.radius:
                inst_data = {
                    'start': i,
                    'end': i + self.window_size,
                    'distance': dist,
                    'data': df.iloc[i:i + self.window_size].to_dict('list'),
                    'pressure_stability': self._get_pressure_stability(df, i),
                    'ore_variation': self._get_variation(df, 'Ore', i),
                    'watermill_variation': self._get_variation(df, 'WaterMill', i),
                    'waterzumpf_variation': self._get_variation(df, 'WaterZumpf', i)
                }
                instances.append(inst_data)
        
        return instances
    
    def _get_pressure_stability(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate pressure stability as 1/CV."""
        window = df.iloc[idx:idx + self.window_size]
        pressure_std = window['PressureHC'].std()
        pressure_mean = window['PressureHC'].mean()
        return pressure_mean / pressure_std if pressure_std > 0 else float('inf')
    
    def _get_variation(self, df: pd.DataFrame, col: str, idx: int) -> float:
        """Calculate coefficient of variation for a column in the window."""
        window = df.iloc[idx:idx + self.window_size]
        return window[col].std() / window[col].mean() if window[col].mean() != 0 else 0
    
    def _create_motif(self, instances: List[Dict], df: pd.DataFrame) -> Motif:
        """Create a Motif object from a list of instances with pressure constraints."""
        motif = Motif(motif_id=len(self.motifs) + 1)
        motif.add_metadata('pattern_type', 'pressure_constrained')
        
        # Calculate average metrics across all instances
        metrics = ['pressure_stability', 'ore_variation', 
                  'watermill_variation', 'waterzumpf_variation']
        
        for metric in metrics:
            values = [inst[metric] for inst in instances if metric in inst]
            if values:
                motif.add_metadata(f'avg_{metric}', np.mean(values))
        
        # Add instances to motif
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
        """Generate a plot for a single pressure-constrained motif."""
        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        features = ['PressureHC', 'Ore', 'WaterMill', 'WaterZumpf']
        
        for i, feature in enumerate(features):
            ax = axs[i]
            for inst in motif.instances:
                data = inst.get_feature(feature)
                if data is not None:
                    ax.plot(data, alpha=0.6, label=f'Instance {inst.id}')
            ax.set_ylabel(feature)
            ax.grid(True)
            
            # Add mean and std for pressure
            if feature == 'PressureHC':
                all_data = np.array([inst.get_feature(feature) 
                                   for inst in motif.instances 
                                   if inst.get_feature(feature) is not None])
                if len(all_data) > 0:
                    mean_data = np.mean(all_data, axis=0)
                    std_data = np.std(all_data, axis=0)
                    ax.fill_between(range(len(mean_data)), 
                                  mean_data - std_data, 
                                  mean_data + std_data, 
                                  alpha=0.2, color='blue',
                                  label='±1 std dev')
                    ax.plot(mean_data, 'k-', linewidth=2, label='Mean')
            
            if i == 0:
                ax.legend()
        
        plt.suptitle(f'Pressure-Constrained Motif {motif.motif_id} - {len(motif.instances)} instances')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_summary(self, output_path: Path) -> None:
        """Generate a summary plot of all pressure-constrained motifs."""
        if not self.motifs:
            return
            
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.ravel()
        
        # Plot 1: Number of instances per motif
        instances = [len(motif.instances) for motif in self.motifs]
        axs[0].bar(range(1, len(instances) + 1), instances)
        axs[0].set_title('Instances per Motif')
        axs[0].set_xlabel('Motif ID')
        axs[0].set_ylabel('Number of Instances')
        
        # Plot 2: Pressure stability (1/CV) vs. motif ID
        stability = [m.metadata.get('avg_pressure_stability', 0) for m in self.motifs]
        axs[1].bar(range(1, len(stability) + 1), stability, color='green')
        axs[1].set_title('Average Pressure Stability (1/CV)')
        axs[1].set_xlabel('Motif ID')
        axs[1].set_ylabel('Stability (1/CV)')
        
        # Plot 3: Variation in other variables (boxplot)
        variations = {
            'Ore': [m.metadata.get('avg_ore_variation', 0) for m in self.motifs],
            'WaterMill': [m.metadata.get('avg_watermill_variation', 0) for m in self.motifs],
            'WaterZumpf': [m.metadata.get('avg_waterzumpf_variation', 0) for m in self.motifs]
        }
        axs[2].boxplot(variations.values(), labels=variations.keys())
        axs[2].set_title('Variation in Other Variables')
        axs[2].set_ylabel('Coefficient of Variation (CV)')
        
        # Plot 4: Distance distribution
        distances = []
        for motif in self.motifs:
            distances.extend([inst.distance for inst in motif.instances])
        axs[3].hist(distances, bins=20, alpha=0.7, color='purple')
        axs[3].set_title('Distribution of Instance Distances')
        axs[3].set_xlabel('Distance to Motif Centroid')
        axs[3].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
