"""
Universal constraint-based motif discovery pattern.

A single, flexible implementation that handles all constraint types
through configuration rather than separate classes.
"""

import numpy as np
import pandas as pd
import stumpy
from typing import List, Dict, Tuple, Any
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.base_pattern import BasePattern, Motif, MotifInstance
from core.pattern_registry import PatternRegistry

logger = logging.getLogger(__name__)


@PatternRegistry.register('constraint')
class ConstraintPattern(BasePattern):
    """
    Universal constraint-based motif discovery.
    
    Discovers motifs based on configurable variability constraints.
    Features can be marked as 'stable' (low CV) or 'varying' (high CV).
    
    Example constraints:
        - Density pattern: WaterZumpf stable, Ore/WaterMill varying
        - Inverse pattern: Ore/WaterMill stable, WaterZumpf varying
        - Dynamic pattern: All MVs varying
        - Pressure pattern: PressureHC stable, MVs varying
    """
    
    def __init__(self, name: str, config: dict):
        """
        Initialize constraint pattern.
        
        Args:
            name: Pattern name
            config: Configuration dictionary with 'constraints' key
        """
        super().__init__(name, config)
        
        # Parse constraints
        self.constraints = config.get('constraints', {})
        self.features = list(self.constraints.keys())
        self.relative_variability_factor = config.get('relative_variability_factor', 1.2)
        
        # Separate stable and varying features
        self.stable_features = []
        self.varying_features = []
        
        for feature, constraint in self.constraints.items():
            constraint_type = constraint.get('type', 'stable')
            if constraint_type == 'stable':
                self.stable_features.append(feature)
            elif constraint_type == 'varying':
                self.varying_features.append(feature)
        
        logger.info(f"  Stable features: {self.stable_features}")
        logger.info(f"  Varying features: {self.varying_features}")
        logger.info(f"  Window: {self.window_size}, Max motifs: {self.max_motifs}, Radius: {self.radius}")
    
    def discover(self, df: pd.DataFrame) -> List[Motif]:
        """
        Discover constraint-based motifs.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of discovered motifs
        """
        logger.info(f"\nDiscovering constraint motifs: {self.name}")
        
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
            # Find constrained seed
            seed_idx, seed_distance = self._find_constrained_seed(
                df, mp_distances, used_indices, n_windows
            )
            
            if seed_idx is None or seed_distance > self.radius:
                break
            
            # Find constrained instances
            instances = self._find_constrained_instances(
                df, T, seed_idx, n_windows, used_indices, mp_distances
            )
            
            if len(instances) >= 2:
                motif = Motif(motif_id=len(self.motifs) + 1)
                motif.add_metadata('pattern_type', self.name)
                
                for inst_data in instances:
                    instance = MotifInstance(
                        start=inst_data['start'],
                        end=inst_data['end'],
                        distance=inst_data['distance'],
                        data=inst_data['data']
                    )
                    
                    # Add CV metadata for all features
                    for feat in self.features:
                        cv_key = f'{feat.lower()}_cv'
                        if cv_key in inst_data:
                            instance.add_metadata(cv_key, inst_data[cv_key])
                    
                    instance.add_metadata('pattern_type', self.name)
                    motif.add_instance(instance)
                
                self.motifs.append(motif)
                
                # Mark as used
                for inst in instances:
                    for offset in range(-self.window_size, self.window_size):
                        neighbor = inst['start'] + offset
                        if 0 <= neighbor < n_windows:
                            used_indices.add(neighbor)
            else:
                # Mark seed as used
                for offset in range(-self.window_size, self.window_size):
                    neighbor = seed_idx + offset
                    if 0 <= neighbor < n_windows:
                        used_indices.add(neighbor)
        
        total_instances = sum(len(m) for m in self.motifs)
        logger.info(f"  ✓ Found {len(self.motifs)} motifs with {total_instances} instances")
        
        return self.motifs
    
    def _find_constrained_seed(
        self,
        df: pd.DataFrame,
        mp_distances: np.ndarray,
        used_indices: set,
        n_windows: int
    ) -> Tuple[int, float]:
        """Find best seed that satisfies constraints."""
        seed_idx = None
        seed_distance = float('inf')
        
        for i in range(n_windows):
            if i in used_indices:
                continue
            
            dist = mp_distances[i]
            if np.isnan(dist) or np.isinf(dist):
                continue
            
            # Check variability constraints
            if not self._check_constraints(df, i):
                continue
            
            if dist < seed_distance:
                seed_distance = dist
                seed_idx = i
        
        return seed_idx, seed_distance
    
    def _check_constraints(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Check if window at idx satisfies all constraints.
        
        Args:
            df: DataFrame
            idx: Window start index
            
        Returns:
            True if constraints satisfied
        """
        cvs = {}
        
        # Calculate CV for all features
        for feature in self.features:
            data = df[feature].iloc[idx:idx + self.window_size].values
            cvs[feature] = self.calculate_variability(data)
        
        # Check stable features (low CV)
        for feature in self.stable_features:
            constraint = self.constraints[feature]
            max_cv = constraint.get('max_cv', 0.01)
            
            if cvs[feature] > max_cv:
                return False
        
        # Check varying features (high CV)
        for feature in self.varying_features:
            constraint = self.constraints[feature]
            min_cv = constraint.get('min_cv', 0.0008)
            
            if cvs[feature] < min_cv:
                return False
        
        # Check relative variability (varying should be more variable than stable)
        if self.stable_features and self.varying_features:
            max_stable_cv = max(cvs[f] for f in self.stable_features)
            min_varying_cv = min(cvs[f] for f in self.varying_features)
            
            if min_varying_cv < max_stable_cv * self.relative_variability_factor:
                return False
        
        return True
    
    def _find_constrained_instances(
        self,
        df: pd.DataFrame,
        T: np.ndarray,
        seed_idx: int,
        n_windows: int,
        used_indices: set,
        mp_distances: np.ndarray
    ) -> List[dict]:
        """Find instances similar to seed that satisfy constraints."""
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
            
            # Check constraints
            if not self._check_constraints(df, idx):
                continue
            
            # Check for overlap
            if any(abs(idx - vi['start']) < self.window_size for vi in valid_instances):
                continue
            
            # Extract data and calculate CVs
            data = {}
            cvs = {}
            
            for feat in self.features:
                feat_data = df[feat].iloc[idx:idx + self.window_size].values
                data[feat] = feat_data
                cvs[f'{feat.lower()}_cv'] = self.calculate_variability(feat_data)
            
            # Add DensityHC if available (for analysis)
            if 'DensityHC' in df.columns and 'DensityHC' not in data:
                data['DensityHC'] = df['DensityHC'].iloc[idx:idx + self.window_size].values
            
            if 'TimeStamp' in df.columns:
                data['TimeStamp'] = df['TimeStamp'].iloc[idx:idx + self.window_size].values
            
            instance = {
                'start': idx,
                'end': idx + self.window_size,
                'distance': dist,
                'data': data,
                **cvs
            }
            
            valid_instances.append(instance)
        
        return valid_instances
