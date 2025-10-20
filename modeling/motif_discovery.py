"""
Motif discovery module for time series pattern recognition.

Provides classes and functions for discovering repeating patterns (motifs)
in multivariate time series data using STUMPY.
"""

import numpy as np
import pandas as pd
import stumpy
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MotifInstance:
    """Represents a single instance of a motif pattern."""
    
    def __init__(self, start: int, end: int, distance: float, data: Dict[str, np.ndarray]):
        """
        Initialize motif instance.
        
        Args:
            start: Start index
            end: End index
            distance: Distance from motif seed
            data: Dictionary mapping feature names to data arrays
        """
        self.start = start
        self.end = end
        self.distance = distance
        self.data = data
        self.metadata = {}
    
    def __len__(self):
        """Return length of the instance."""
        return self.end - self.start
    
    def add_metadata(self, key: str, value):
        """Add metadata to instance."""
        self.metadata[key] = value
    
    def get_feature(self, feature_name: str) -> np.ndarray:
        """Get data for a specific feature."""
        return self.data.get(feature_name)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'start': self.start,
            'end': self.end,
            'distance': self.distance,
            'data': self.data,
            'metadata': self.metadata
        }


class Motif:
    """Represents a motif group containing multiple similar instances."""
    
    def __init__(self, motif_id: int):
        """
        Initialize motif.
        
        Args:
            motif_id: Unique identifier for this motif
        """
        self.motif_id = motif_id
        self.instances: List[MotifInstance] = []
        self.avg_distance = 0.0
        self.metadata = {}
    
    def add_instance(self, instance: MotifInstance):
        """Add an instance to this motif."""
        self.instances.append(instance)
        self._update_avg_distance()
    
    def _update_avg_distance(self):
        """Update average distance across instances."""
        if self.instances:
            self.avg_distance = float(np.mean([inst.distance for inst in self.instances]))
    
    def __len__(self):
        """Return number of instances."""
        return len(self.instances)
    
    def add_metadata(self, key: str, value):
        """Add metadata to motif."""
        self.metadata[key] = value
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'motif_id': self.motif_id,
            'instances': [inst.to_dict() for inst in self.instances],
            'distance': self.avg_distance,
            'metadata': self.metadata
        }


class MotifDiscovery:
    """Main class for discovering motifs in multivariate time series."""
    
    def __init__(
        self,
        window_size: int = 60,
        max_motifs: int = 20,
        max_instances_per_motif: int = 1000,
        radius: float = 4.5
    ):
        """
        Initialize motif discovery.
        
        Args:
            window_size: Length of pattern window
            max_motifs: Maximum number of motif groups to find
            max_instances_per_motif: Maximum instances per motif
            radius: Distance threshold for matching
        """
        self.window_size = window_size
        self.max_motifs = max_motifs
        self.max_instances_per_motif = max_instances_per_motif
        self.radius = radius
        
        self.motifs: List[Motif] = []
        self.matrix_profile = None
        self.profile_indices = None
    
    def discover(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[List[Motif], List[Tuple[int, int, int]]]:
        """
        Discover motifs in multivariate time series.
        
        Args:
            df: DataFrame with time series data
            feature_columns: List of column names to use
            
        Returns:
            Tuple of (motifs list, segment tuples)
        """
        logger.info(f"Discovering motifs in {len(feature_columns)} features...")
        logger.info(f"  Window size: {self.window_size}")
        logger.info(f"  Max motifs: {self.max_motifs}")
        logger.info(f"  Radius: {self.radius}")
        
        # Prepare multivariate time series
        T = self._prepare_time_series(df, feature_columns)
        
        # Compute matrix profile
        logger.info("  Computing multivariate matrix profile...")
        self.matrix_profile, self.profile_indices = stumpy.mstump(T, m=self.window_size)
        mp_distances = np.sqrt(np.mean(self.matrix_profile**2, axis=0))
        
        # Discover motifs
        logger.info("  Discovering motif groups...")
        self.motifs = []
        segment_tuples = []
        used_indices = set()
        n_windows = self.matrix_profile.shape[1]
        
        for motif_idx in range(self.max_motifs):
            # Find best seed
            seed_idx, seed_distance = self._find_best_seed(mp_distances, used_indices, n_windows)
            
            if seed_idx is None:
                break
            if self.radius is not None and seed_distance > self.radius:
                break
            
            # Find instances for this motif
            motif_indices, motif_distances = self._find_motif_instances(
                T, seed_idx, n_windows, used_indices
            )
            
            if not motif_indices:
                continue
            
            # Create motif object
            motif = Motif(motif_id=motif_idx + 1)
            
            for idx, dist in zip(motif_indices, motif_distances):
                # Extract data for this instance
                instance_data = {
                    col: df[col].iloc[idx:idx + self.window_size].values
                    for col in feature_columns
                }
                
                instance = MotifInstance(
                    start=idx,
                    end=idx + self.window_size,
                    distance=float(dist),
                    data=instance_data
                )
                
                motif.add_instance(instance)
                segment_tuples.append((idx, idx + self.window_size, motif_idx + 1))
                
                # Mark as used
                used_indices.add(idx)
                for offset in range(-self.window_size, self.window_size):
                    neighbor = idx + offset
                    if 0 <= neighbor < n_windows:
                        used_indices.add(neighbor)
            
            self.motifs.append(motif)
        
        logger.info(f"  ✓ Found {len(self.motifs)} motif groups")
        logger.info(f"  ✓ Total instances: {len(segment_tuples)}")
        if self.motifs:
            avg_dist = np.mean([m.avg_distance for m in self.motifs])
            logger.info(f"  ✓ Average distance: {avg_dist:.3f}")
        
        return self.motifs, segment_tuples
    
    def _prepare_time_series(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Prepare and normalize multivariate time series."""
        ts_list = []
        for col in feature_columns:
            ts = np.array(df[col])
            # Normalize
            ts = (ts - np.mean(ts)) / np.std(ts)
            ts_list.append(ts)
        return np.array(ts_list)
    
    def _find_best_seed(
        self,
        mp_distances: np.ndarray,
        used_indices: set,
        n_windows: int
    ) -> Tuple[Optional[int], float]:
        """Find the best seed for next motif."""
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
    
    def _find_motif_instances(
        self,
        T: np.ndarray,
        seed_idx: int,
        n_windows: int,
        used_indices: set
    ) -> Tuple[List[int], List[float]]:
        """Find all instances for a motif seed."""
        # Compute distance profile for seed
        distance_components = []
        for dim in range(T.shape[0]):
            query = T[dim, seed_idx:seed_idx + self.window_size]
            if len(query) < self.window_size:
                continue
            distance_profile = stumpy.mass(query, T[dim])
            distance_components.append(distance_profile[:n_windows])
        
        if not distance_components:
            return [], []
        
        distance_components = np.array(distance_components)
        aggregated_profile = np.sqrt(np.mean(distance_components**2, axis=0))
        
        # Find candidate instances
        sorted_candidates = np.argsort(aggregated_profile)
        motif_indices = []
        motif_distances = []
        
        for idx in sorted_candidates:
            if len(motif_indices) >= self.max_instances_per_motif:
                break
            if idx >= n_windows:
                continue
            
            dist = aggregated_profile[idx]
            if np.isnan(dist) or np.isinf(dist):
                continue
            if self.radius is not None and dist > self.radius:
                break
            if idx in used_indices:
                continue
            
            # Avoid overlapping instances
            if any(abs(idx - existing_idx) < self.window_size for existing_idx in motif_indices):
                continue
            
            motif_indices.append(idx)
            motif_distances.append(dist)
        
        # Ensure seed is included
        if seed_idx not in motif_indices:
            motif_indices.insert(0, seed_idx)
            motif_distances.insert(0, aggregated_profile[seed_idx])
        
        return motif_indices, motif_distances


class CorrelationFilter:
    """Filter motifs based on cross-correlation constraints."""
    
    def __init__(
        self,
        correlation_rules: Dict[Tuple[str, str], str],
        min_correlation_strength: float = 0.3,
        filter_level: str = 'instance'
    ):
        """
        Initialize correlation filter.
        
        Args:
            correlation_rules: Dict mapping (feat1, feat2) to 'pos' or 'neg'
            min_correlation_strength: Minimum |correlation| to enforce
            filter_level: 'instance' or 'motif'
        """
        self.correlation_rules = correlation_rules
        self.min_correlation_strength = min_correlation_strength
        self.filter_level = filter_level
    
    def filter(self, motifs: List[Motif]) -> Tuple[List[Motif], dict]:
        """
        Filter motifs based on correlation rules.
        
        Args:
            motifs: List of motifs to filter
            
        Returns:
            Tuple of (filtered motifs, correlation stats)
        """
        logger.info("Filtering motifs by correlation constraints...")
        logger.info(f"  Filter level: {self.filter_level}")
        logger.info(f"  Min correlation strength: {self.min_correlation_strength}")
        logger.info(f"  Rules:")
        for (feat1, feat2), sign in self.correlation_rules.items():
            logger.info(f"    {feat1} vs {feat2}: {sign}")
        
        filtered_motifs = []
        correlation_stats = {}
        
        for motif in motifs:
            valid_instances = []
            instance_correlations = []
            
            for instance in motif.instances:
                # Check correlation rules
                instance_corrs = {}
                all_rules_satisfied = True
                
                for (feat1, feat2), expected_sign in self.correlation_rules.items():
                    if feat1 not in instance.data or feat2 not in instance.data:
                        continue
                    
                    data1 = instance.data[feat1]
                    data2 = instance.data[feat2]
                    
                    if len(data1) > 1 and len(data2) > 1:
                        corr, _ = pearsonr(data1, data2)
                        instance_corrs[(feat1, feat2)] = corr
                        
                        # Check constraint
                        if abs(corr) < self.min_correlation_strength:
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
            
            # Store stats
            correlation_stats[motif.motif_id] = {
                'total_instances': len(motif.instances),
                'valid_instances': len(valid_instances),
                'avg_correlations': {}
            }
            
            # Compute average correlations
            if valid_instances:
                for (feat1, feat2) in self.correlation_rules.keys():
                    corrs = [ic.get((feat1, feat2), np.nan) for ic in instance_correlations
                            if (feat1, feat2) in ic]
                    if corrs:
                        correlation_stats[motif.motif_id]['avg_correlations'][(feat1, feat2)] = np.mean(corrs)
            
            # Apply filter level
            if self.filter_level == 'instance':
                if valid_instances:
                    filtered_motif = Motif(motif.motif_id)
                    for inst in valid_instances:
                        filtered_motif.add_instance(inst)
                    filtered_motif.metadata = motif.metadata.copy()
                    filtered_motifs.append(filtered_motif)
            elif self.filter_level == 'motif':
                # Keep motif only if majority of instances are valid
                if len(valid_instances) >= len(motif.instances) * 0.5:
                    filtered_motifs.append(motif)
        
        logger.info(f"  ✓ Filtered: {len(filtered_motifs)} motifs kept")
        
        return filtered_motifs, correlation_stats


def convert_motifs_to_legacy_format(motifs: List[Motif]) -> List[dict]:
    """
    Convert Motif objects to legacy dictionary format for compatibility.
    
    Args:
        motifs: List of Motif objects
        
    Returns:
        List of dictionaries in legacy format
    """
    legacy_motifs = []
    for motif in motifs:
        legacy_motif = {
            'motif_id': motif.motif_id,
            'instances': [],
            'distance': motif.avg_distance
        }
        
        for instance in motif.instances:
            legacy_instance = {
                'start': instance.start,
                'end': instance.end,
                'data': instance.data
            }
            legacy_motif['instances'].append(legacy_instance)
        
        legacy_motifs.append(legacy_motif)
    
    return legacy_motifs
