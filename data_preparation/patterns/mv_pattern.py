"""
Standard MV (Manipulated Variable) motif discovery pattern.

Discovers repeating patterns in MV features without constraints.
"""

import numpy as np
import pandas as pd
import stumpy
from typing import List, Tuple
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.base_pattern import BasePattern, Motif, MotifInstance
from core.pattern_registry import PatternRegistry

logger = logging.getLogger(__name__)


@PatternRegistry.register('mv')
class MVPattern(BasePattern):
    """
    Standard MV motif discovery without variability constraints.
    
    Discovers repeating patterns in manipulated variables (Ore, WaterMill, WaterZumpf).
    """
    
    def __init__(self, name: str, config: dict):
        """
        Initialize MV pattern.
        
        Args:
            name: Pattern name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        
        self.features = config.get('features', ['Ore', 'WaterMill', 'WaterZumpf'])
        self.apply_correlation_filter = config.get('apply_correlation_filter', False)
        self.correlation_rules = config.get('correlation_rules', {})
        
        logger.info(f"  Features: {self.features}")
        logger.info(f"  Window: {self.window_size}, Max motifs: {self.max_motifs}, Radius: {self.radius}")
    
    def discover(self, df: pd.DataFrame) -> List[Motif]:
        """
        Discover MV motifs.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of discovered motifs
        """
        logger.info(f"\nDiscovering MV motifs: {self.name}")
        
        # Validate data
        if not self.validate_data(df, self.features):
            return []
        
        # Prepare time series
        T = self.prepare_time_series(df, self.features)
        
        # Compute matrix profile
        logger.info("  Computing multivariate matrix profile...")
        matrix_profile, profile_indices = stumpy.mstump(T, m=self.window_size)
        mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))
        
        # Discover motifs
        self.motifs = []
        used_indices = set()
        n_windows = matrix_profile.shape[1]
        
        for motif_idx in range(self.max_motifs):
            # Find best unused seed
            seed_idx, seed_distance = self._find_best_seed(mp_distances, used_indices, n_windows)
            
            if seed_idx is None or seed_distance > self.radius:
                break
            
            # Find similar instances
            instances = self._find_similar_instances(
                df, T, seed_idx, n_windows, used_indices, mp_distances
            )
            
            if len(instances) >= 2:
                motif = Motif(motif_id=len(self.motifs) + 1)
                motif.add_metadata('pattern_type', 'mv')
                
                for inst_data in instances:
                    instance = MotifInstance(
                        start=inst_data['start'],
                        end=inst_data['end'],
                        distance=inst_data['distance'],
                        data=inst_data['data']
                    )
                    instance.add_metadata('pattern_type', 'mv')
                    motif.add_instance(instance)
                
                self.motifs.append(motif)
                
                # Mark as used
                for inst in instances:
                    for offset in range(-self.window_size, self.window_size):
                        neighbor = inst['start'] + offset
                        if 0 <= neighbor < n_windows:
                            used_indices.add(neighbor)
            else:
                # Mark seed as used even if no motif found
                for offset in range(-self.window_size, self.window_size):
                    neighbor = seed_idx + offset
                    if 0 <= neighbor < n_windows:
                        used_indices.add(neighbor)
        
        total_instances = sum(len(m) for m in self.motifs)
        logger.info(f"  ✓ Found {len(self.motifs)} motifs with {total_instances} instances")
        
        return self.motifs
    
    def _find_best_seed(
        self,
        mp_distances: np.ndarray,
        used_indices: set,
        n_windows: int
    ) -> Tuple[int, float]:
        """Find best unused seed point."""
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
        
        return seed_idx, seed_distance
    
    def _find_similar_instances(
        self,
        df: pd.DataFrame,
        T: np.ndarray,
        seed_idx: int,
        n_windows: int,
        used_indices: set,
        mp_distances: np.ndarray
    ) -> List[dict]:
        """Find instances similar to seed."""
        # Compute distance profile for each dimension
        distance_components = []
        for dim in range(T.shape[0]):
            query = T[dim, seed_idx:seed_idx + self.window_size]
            if len(query) < self.window_size:
                continue
            distance_profile = stumpy.mass(query, T[dim])
            distance_components.append(distance_profile[:n_windows])
        
        if not distance_components:
            return []
        
        # Aggregate distances
        distance_components = np.array(distance_components)
        aggregated_profile = np.sqrt(np.mean(distance_components**2, axis=0))
        
        # Find valid instances
        sorted_candidates = np.argsort(aggregated_profile)
        valid_instances = []
        
        for idx in sorted_candidates:
            if len(valid_instances) >= self.max_instances_per_motif:
                break
            
            if idx >= n_windows or idx in used_indices:
                continue
            
            dist = aggregated_profile[idx]
            if np.isnan(dist) or np.isinf(dist) or dist > self.radius:
                continue
            
            # Check for overlap with existing instances
            if any(abs(idx - vi['start']) < self.window_size for vi in valid_instances):
                continue
            
            # Extract data
            data = {}
            for feat in self.features:
                data[feat] = df[feat].iloc[idx:idx + self.window_size].values
            
            if 'TimeStamp' in df.columns:
                data['TimeStamp'] = df['TimeStamp'].iloc[idx:idx + self.window_size].values
            
            valid_instances.append({
                'start': idx,
                'end': idx + self.window_size,
                'distance': dist,
                'data': data
            })
        
        return valid_instances
