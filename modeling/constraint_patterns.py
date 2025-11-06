"""
Additional constraint pattern discovery classes.

Provides alternative motif discovery strategies with different variability constraints.
"""

import numpy as np
import pandas as pd
import stumpy
from typing import List, Tuple
import logging

from motif_discovery import Motif, MotifInstance
from density_analysis import calculate_variability

logger = logging.getLogger(__name__)


class InverseConstraintMotifDiscovery:
    """
    Discover motifs with inverse variability constraints.
    
    Finds patterns where Ore and WaterMill are stable but WaterZumpf varies.
    This captures steady-state feed with sump water adjustments.
    """
    
    def __init__(
        self,
        window_size: int = 60,
        max_motifs: int = 10,
        radius: float = 4.5,
        ore_max_cv: float = 0.01,
        watermill_max_cv: float = 0.01,
        waterzumpf_min_cv: float = 0.0008,
        relative_variability_factor: float = 1.2
    ):
        self.window_size = window_size
        self.max_motifs = max_motifs
        self.radius = radius
        self.ore_max_cv = ore_max_cv
        self.watermill_max_cv = watermill_max_cv
        self.waterzumpf_min_cv = waterzumpf_min_cv
        self.relative_variability_factor = relative_variability_factor
        self.motifs: List[Motif] = []
    
    def discover(self, df: pd.DataFrame) -> List[Motif]:
        """Discover inverse constraint motifs."""
        logger.info("Discovering inverse constraint motifs (stable Ore/WaterMill, varying WaterZumpf)...")
        logger.info(f"  Window size: {self.window_size} minutes")
        logger.info(f"  Max motifs: {self.max_motifs}")
        
        features = ['Ore', 'WaterMill', 'WaterZumpf']
        T = self._prepare_time_series(df, features)
        
        logger.info("  Computing multivariate matrix profile...")
        matrix_profile, profile_indices = stumpy.mstump(T, m=self.window_size)
        mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))
        
        self.motifs = []
        used_indices = set()
        n_windows = matrix_profile.shape[1]
        
        for motif_idx in range(self.max_motifs):
            seed_idx, seed_distance = self._find_constrained_seed(
                df, mp_distances, used_indices, n_windows
            )
            
            if seed_idx is None or seed_distance > self.radius:
                break
            
            valid_instances = self._find_constrained_instances(
                df, T, seed_idx, n_windows, used_indices, mp_distances
            )
            
            if len(valid_instances) >= 2:
                motif = Motif(motif_id=len(self.motifs) + 1)
                motif.add_metadata('pattern_type', 'inverse_constraint')
                
                for inst_data in valid_instances:
                    instance = MotifInstance(
                        start=inst_data['start'],
                        end=inst_data['end'],
                        distance=inst_data['distance'],
                        data=inst_data['data']
                    )
                    instance.add_metadata('ore_cv', inst_data['ore_cv'])
                    instance.add_metadata('watermill_cv', inst_data['watermill_cv'])
                    instance.add_metadata('waterzumpf_cv', inst_data['waterzumpf_cv'])
                    instance.add_metadata('pattern_type', 'inverse_constraint')
                    
                    motif.add_instance(instance)
                
                self.motifs.append(motif)
                
                for inst in valid_instances:
                    for offset in range(-self.window_size, self.window_size):
                        neighbor = inst['start'] + offset
                        if 0 <= neighbor < n_windows:
                            used_indices.add(neighbor)
            else:
                for offset in range(-self.window_size, self.window_size):
                    neighbor = seed_idx + offset
                    if 0 <= neighbor < n_windows:
                        used_indices.add(neighbor)
        
        logger.info(f"  ✓ Found {len(self.motifs)} inverse constraint motifs")
        total_instances = sum(len(m.instances) for m in self.motifs)
        logger.info(f"  ✓ Total instances: {total_instances}")
        
        return self.motifs
    
    def _prepare_time_series(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        ts_list = []
        for col in features:
            ts = np.array(df[col])
            ts = (ts - np.mean(ts)) / np.std(ts)
            ts_list.append(ts)
        return np.array(ts_list)
    
    def _find_constrained_seed(self, df: pd.DataFrame, mp_distances: np.ndarray,
                               used_indices: set, n_windows: int) -> Tuple[int, float]:
        seed_idx = None
        seed_distance = float('inf')
        
        for i in range(n_windows):
            if i in used_indices:
                continue
            dist = mp_distances[i]
            if np.isnan(dist) or np.isinf(dist):
                continue
            if not self._check_variability_constraints(df, i):
                continue
            if dist < seed_distance:
                seed_distance = dist
                seed_idx = i
        
        return seed_idx, seed_distance
    
    def _check_variability_constraints(self, df: pd.DataFrame, idx: int) -> bool:
        ore_data = df['Ore'].iloc[idx:idx + self.window_size].values
        watermill_data = df['WaterMill'].iloc[idx:idx + self.window_size].values
        waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + self.window_size].values
        
        ore_cv = calculate_variability(ore_data)
        watermill_cv = calculate_variability(watermill_data)
        waterzumpf_cv = calculate_variability(waterzumpf_data)
        
        return (
            ore_cv <= self.ore_max_cv and
            watermill_cv <= self.watermill_max_cv and
            waterzumpf_cv >= self.waterzumpf_min_cv and
            waterzumpf_cv >= ore_cv * self.relative_variability_factor and
            waterzumpf_cv >= watermill_cv * self.relative_variability_factor
        )
    
    def _find_constrained_instances(self, df: pd.DataFrame, T: np.ndarray, seed_idx: int,
                                    n_windows: int, used_indices: set, mp_distances: np.ndarray) -> List[dict]:
        distance_components = []
        for dim in range(T.shape[0]):
            query = T[dim, seed_idx:seed_idx + self.window_size]
            if len(query) < self.window_size:
                continue
            distance_profile = stumpy.mass(query, T[dim])
            distance_components.append(distance_profile[:n_windows])
        
        if not distance_components:
            return []
        
        distance_components = np.array(distance_components)
        aggregated_profile = np.sqrt(np.mean(distance_components**2, axis=0))
        sorted_candidates = np.argsort(aggregated_profile)
        valid_instances = []
        
        for idx in sorted_candidates:
            if len(valid_instances) >= 20:
                break
            if idx >= n_windows or idx in used_indices:
                continue
            dist = aggregated_profile[idx]
            if np.isnan(dist) or np.isinf(dist) or dist > self.radius:
                continue
            if not self._check_variability_constraints(df, idx):
                continue
            if any(abs(idx - vi['start']) < self.window_size for vi in valid_instances):
                continue
            
            ore_data = df['Ore'].iloc[idx:idx + self.window_size].values
            watermill_data = df['WaterMill'].iloc[idx:idx + self.window_size].values
            waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + self.window_size].values
            density_data = df['DensityHC'].iloc[idx:idx + self.window_size].values
            timestamp_data = df['TimeStamp'].iloc[idx:idx + self.window_size].values
            
            instance = {
                'start': idx,
                'end': idx + self.window_size,
                'distance': dist,
                'ore_cv': calculate_variability(ore_data),
                'watermill_cv': calculate_variability(watermill_data),
                'waterzumpf_cv': calculate_variability(waterzumpf_data),
                'data': {
                    'Ore': ore_data,
                    'WaterMill': watermill_data,
                    'WaterZumpf': waterzumpf_data,
                    'DensityHC': density_data,
                    'TimeStamp': timestamp_data
                }
            }
            valid_instances.append(instance)
        
        return valid_instances


class DynamicMotifDiscovery:
    """
    Discover motifs where all MV features vary together.
    
    Finds patterns where Ore, WaterMill, and WaterZumpf all change simultaneously.
    This captures transient/dynamic operations with coordinated adjustments.
    """
    
    def __init__(
        self,
        window_size: int = 60,
        max_motifs: int = 10,
        radius: float = 4.5,
        ore_min_cv: float = 0.0008,
        watermill_min_cv: float = 0.0015,
        waterzumpf_min_cv: float = 0.0008
    ):
        self.window_size = window_size
        self.max_motifs = max_motifs
        self.radius = radius
        self.ore_min_cv = ore_min_cv
        self.watermill_min_cv = watermill_min_cv
        self.waterzumpf_min_cv = waterzumpf_min_cv
        self.motifs: List[Motif] = []
    
    def discover(self, df: pd.DataFrame) -> List[Motif]:
        """Discover dynamic motifs."""
        logger.info("Discovering dynamic motifs (all MVs varying)...")
        logger.info(f"  Window size: {self.window_size} minutes")
        logger.info(f"  Max motifs: {self.max_motifs}")
        
        features = ['Ore', 'WaterMill', 'WaterZumpf']
        T = self._prepare_time_series(df, features)
        
        logger.info("  Computing multivariate matrix profile...")
        matrix_profile, profile_indices = stumpy.mstump(T, m=self.window_size)
        mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))
        
        self.motifs = []
        used_indices = set()
        n_windows = matrix_profile.shape[1]
        
        for motif_idx in range(self.max_motifs):
            seed_idx, seed_distance = self._find_constrained_seed(
                df, mp_distances, used_indices, n_windows
            )
            
            if seed_idx is None or seed_distance > self.radius:
                break
            
            valid_instances = self._find_constrained_instances(
                df, T, seed_idx, n_windows, used_indices, mp_distances
            )
            
            if len(valid_instances) >= 2:
                motif = Motif(motif_id=len(self.motifs) + 1)
                motif.add_metadata('pattern_type', 'dynamic')
                
                for inst_data in valid_instances:
                    instance = MotifInstance(
                        start=inst_data['start'],
                        end=inst_data['end'],
                        distance=inst_data['distance'],
                        data=inst_data['data']
                    )
                    instance.add_metadata('ore_cv', inst_data['ore_cv'])
                    instance.add_metadata('watermill_cv', inst_data['watermill_cv'])
                    instance.add_metadata('waterzumpf_cv', inst_data['waterzumpf_cv'])
                    instance.add_metadata('pattern_type', 'dynamic')
                    
                    motif.add_instance(instance)
                
                self.motifs.append(motif)
                
                for inst in valid_instances:
                    for offset in range(-self.window_size, self.window_size):
                        neighbor = inst['start'] + offset
                        if 0 <= neighbor < n_windows:
                            used_indices.add(neighbor)
            else:
                for offset in range(-self.window_size, self.window_size):
                    neighbor = seed_idx + offset
                    if 0 <= neighbor < n_windows:
                        used_indices.add(neighbor)
        
        logger.info(f"  ✓ Found {len(self.motifs)} dynamic motifs")
        total_instances = sum(len(m.instances) for m in self.motifs)
        logger.info(f"  ✓ Total instances: {total_instances}")
        
        return self.motifs
    
    def _prepare_time_series(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        ts_list = []
        for col in features:
            ts = np.array(df[col])
            ts = (ts - np.mean(ts)) / np.std(ts)
            ts_list.append(ts)
        return np.array(ts_list)
    
    def _find_constrained_seed(self, df: pd.DataFrame, mp_distances: np.ndarray,
                               used_indices: set, n_windows: int) -> Tuple[int, float]:
        seed_idx = None
        seed_distance = float('inf')
        
        for i in range(n_windows):
            if i in used_indices:
                continue
            dist = mp_distances[i]
            if np.isnan(dist) or np.isinf(dist):
                continue
            if not self._check_variability_constraints(df, i):
                continue
            if dist < seed_distance:
                seed_distance = dist
                seed_idx = i
        
        return seed_idx, seed_distance
    
    def _check_variability_constraints(self, df: pd.DataFrame, idx: int) -> bool:
        ore_data = df['Ore'].iloc[idx:idx + self.window_size].values
        watermill_data = df['WaterMill'].iloc[idx:idx + self.window_size].values
        waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + self.window_size].values
        
        ore_cv = calculate_variability(ore_data)
        watermill_cv = calculate_variability(watermill_data)
        waterzumpf_cv = calculate_variability(waterzumpf_data)
        
        return (
            ore_cv >= self.ore_min_cv and
            watermill_cv >= self.watermill_min_cv and
            waterzumpf_cv >= self.waterzumpf_min_cv
        )
    
    def _find_constrained_instances(self, df: pd.DataFrame, T: np.ndarray, seed_idx: int,
                                    n_windows: int, used_indices: set, mp_distances: np.ndarray) -> List[dict]:
        distance_components = []
        for dim in range(T.shape[0]):
            query = T[dim, seed_idx:seed_idx + self.window_size]
            if len(query) < self.window_size:
                continue
            distance_profile = stumpy.mass(query, T[dim])
            distance_components.append(distance_profile[:n_windows])
        
        if not distance_components:
            return []
        
        distance_components = np.array(distance_components)
        aggregated_profile = np.sqrt(np.mean(distance_components**2, axis=0))
        sorted_candidates = np.argsort(aggregated_profile)
        valid_instances = []
        
        for idx in sorted_candidates:
            if len(valid_instances) >= 20:
                break
            if idx >= n_windows or idx in used_indices:
                continue
            dist = aggregated_profile[idx]
            if np.isnan(dist) or np.isinf(dist) or dist > self.radius:
                continue
            if not self._check_variability_constraints(df, idx):
                continue
            if any(abs(idx - vi['start']) < self.window_size for vi in valid_instances):
                continue
            
            ore_data = df['Ore'].iloc[idx:idx + self.window_size].values
            watermill_data = df['WaterMill'].iloc[idx:idx + self.window_size].values
            waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + self.window_size].values
            density_data = df['DensityHC'].iloc[idx:idx + self.window_size].values
            timestamp_data = df['TimeStamp'].iloc[idx:idx + self.window_size].values
            
            instance = {
                'start': idx,
                'end': idx + self.window_size,
                'distance': dist,
                'ore_cv': calculate_variability(ore_data),
                'watermill_cv': calculate_variability(watermill_data),
                'waterzumpf_cv': calculate_variability(waterzumpf_data),
                'data': {
                    'Ore': ore_data,
                    'WaterMill': watermill_data,
                    'WaterZumpf': waterzumpf_data,
                    'DensityHC': density_data,
                    'TimeStamp': timestamp_data
                }
            }
            valid_instances.append(instance)
        
        return valid_instances


class PressureConstraintMotifDiscovery:
    """
    Discover motifs where PressureHC is stable but MVs vary.
    
    Finds patterns where cyclone pressure remains constant despite MV changes.
    This indicates good process control and potentially optimal operating regions.
    """
    
    def __init__(
        self,
        window_size: int = 60,
        max_motifs: int = 10,
        radius: float = 4.5,
        pressure_max_cv: float = 0.01,
        ore_min_cv: float = 0.0008,
        watermill_min_cv: float = 0.0015,
        waterzumpf_min_cv: float = 0.0008
    ):
        self.window_size = window_size
        self.max_motifs = max_motifs
        self.radius = radius
        self.pressure_max_cv = pressure_max_cv
        self.ore_min_cv = ore_min_cv
        self.watermill_min_cv = watermill_min_cv
        self.waterzumpf_min_cv = waterzumpf_min_cv
        self.motifs: List[Motif] = []
    
    def discover(self, df: pd.DataFrame) -> List[Motif]:
        """Discover pressure-constrained motifs."""
        logger.info("Discovering pressure-constrained motifs (stable PressureHC, varying MVs)...")
        logger.info(f"  Window size: {self.window_size} minutes")
        logger.info(f"  Max motifs: {self.max_motifs}")
        
        features = ['Ore', 'WaterMill', 'WaterZumpf', 'PressureHC']
        T = self._prepare_time_series(df, features)
        
        logger.info("  Computing multivariate matrix profile...")
        matrix_profile, profile_indices = stumpy.mstump(T, m=self.window_size)
        mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))
        
        self.motifs = []
        used_indices = set()
        n_windows = matrix_profile.shape[1]
        
        for motif_idx in range(self.max_motifs):
            seed_idx, seed_distance = self._find_constrained_seed(
                df, mp_distances, used_indices, n_windows
            )
            
            if seed_idx is None or seed_distance > self.radius:
                break
            
            valid_instances = self._find_constrained_instances(
                df, T, seed_idx, n_windows, used_indices, mp_distances
            )
            
            if len(valid_instances) >= 2:
                motif = Motif(motif_id=len(self.motifs) + 1)
                motif.add_metadata('pattern_type', 'pressure_constraint')
                
                for inst_data in valid_instances:
                    instance = MotifInstance(
                        start=inst_data['start'],
                        end=inst_data['end'],
                        distance=inst_data['distance'],
                        data=inst_data['data']
                    )
                    instance.add_metadata('pressure_cv', inst_data['pressure_cv'])
                    instance.add_metadata('ore_cv', inst_data['ore_cv'])
                    instance.add_metadata('watermill_cv', inst_data['watermill_cv'])
                    instance.add_metadata('waterzumpf_cv', inst_data['waterzumpf_cv'])
                    instance.add_metadata('pattern_type', 'pressure_constraint')
                    
                    motif.add_instance(instance)
                
                self.motifs.append(motif)
                
                for inst in valid_instances:
                    for offset in range(-self.window_size, self.window_size):
                        neighbor = inst['start'] + offset
                        if 0 <= neighbor < n_windows:
                            used_indices.add(neighbor)
            else:
                for offset in range(-self.window_size, self.window_size):
                    neighbor = seed_idx + offset
                    if 0 <= neighbor < n_windows:
                        used_indices.add(neighbor)
        
        logger.info(f"  ✓ Found {len(self.motifs)} pressure-constrained motifs")
        total_instances = sum(len(m.instances) for m in self.motifs)
        logger.info(f"  ✓ Total instances: {total_instances}")
        
        return self.motifs
    
    def _prepare_time_series(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        ts_list = []
        for col in features:
            ts = np.array(df[col])
            ts = (ts - np.mean(ts)) / np.std(ts)
            ts_list.append(ts)
        return np.array(ts_list)
    
    def _find_constrained_seed(self, df: pd.DataFrame, mp_distances: np.ndarray,
                               used_indices: set, n_windows: int) -> Tuple[int, float]:
        seed_idx = None
        seed_distance = float('inf')
        
        for i in range(n_windows):
            if i in used_indices:
                continue
            dist = mp_distances[i]
            if np.isnan(dist) or np.isinf(dist):
                continue
            if not self._check_variability_constraints(df, i):
                continue
            if dist < seed_distance:
                seed_distance = dist
                seed_idx = i
        
        return seed_idx, seed_distance
    
    def _check_variability_constraints(self, df: pd.DataFrame, idx: int) -> bool:
        pressure_data = df['PressureHC'].iloc[idx:idx + self.window_size].values
        ore_data = df['Ore'].iloc[idx:idx + self.window_size].values
        watermill_data = df['WaterMill'].iloc[idx:idx + self.window_size].values
        waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + self.window_size].values
        
        pressure_cv = calculate_variability(pressure_data)
        ore_cv = calculate_variability(ore_data)
        watermill_cv = calculate_variability(watermill_data)
        waterzumpf_cv = calculate_variability(waterzumpf_data)
        
        return (
            pressure_cv <= self.pressure_max_cv and
            ore_cv >= self.ore_min_cv and
            watermill_cv >= self.watermill_min_cv and
            waterzumpf_cv >= self.waterzumpf_min_cv
        )
    
    def _find_constrained_instances(self, df: pd.DataFrame, T: np.ndarray, seed_idx: int,
                                    n_windows: int, used_indices: set, mp_distances: np.ndarray) -> List[dict]:
        distance_components = []
        for dim in range(T.shape[0]):
            query = T[dim, seed_idx:seed_idx + self.window_size]
            if len(query) < self.window_size:
                continue
            distance_profile = stumpy.mass(query, T[dim])
            distance_components.append(distance_profile[:n_windows])
        
        if not distance_components:
            return []
        
        distance_components = np.array(distance_components)
        aggregated_profile = np.sqrt(np.mean(distance_components**2, axis=0))
        sorted_candidates = np.argsort(aggregated_profile)
        valid_instances = []
        
        for idx in sorted_candidates:
            if len(valid_instances) >= 20:
                break
            if idx >= n_windows or idx in used_indices:
                continue
            dist = aggregated_profile[idx]
            if np.isnan(dist) or np.isinf(dist) or dist > self.radius:
                continue
            if not self._check_variability_constraints(df, idx):
                continue
            if any(abs(idx - vi['start']) < self.window_size for vi in valid_instances):
                continue
            
            pressure_data = df['PressureHC'].iloc[idx:idx + self.window_size].values
            ore_data = df['Ore'].iloc[idx:idx + self.window_size].values
            watermill_data = df['WaterMill'].iloc[idx:idx + self.window_size].values
            waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + self.window_size].values
            density_data = df['DensityHC'].iloc[idx:idx + self.window_size].values
            timestamp_data = df['TimeStamp'].iloc[idx:idx + self.window_size].values
            
            instance = {
                'start': idx,
                'end': idx + self.window_size,
                'distance': dist,
                'pressure_cv': calculate_variability(pressure_data),
                'ore_cv': calculate_variability(ore_data),
                'watermill_cv': calculate_variability(watermill_data),
                'waterzumpf_cv': calculate_variability(waterzumpf_data),
                'data': {
                    'Ore': ore_data,
                    'WaterMill': watermill_data,
                    'WaterZumpf': waterzumpf_data,
                    'PressureHC': pressure_data,
                    'DensityHC': density_data,
                    'TimeStamp': timestamp_data
                }
            }
            valid_instances.append(instance)
        
        return valid_instances
