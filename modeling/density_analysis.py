"""
Density motif analysis module.

Discovers motifs where WaterZumpf is constant but Ore and WaterMill are changing,
then analyzes DensityHC behavior.
"""

import numpy as np
import pandas as pd
import stumpy
from scipy.stats import pearsonr
from scipy.signal import correlate
from typing import List, Dict, Tuple
import logging

from motif_discovery import Motif, MotifInstance

logger = logging.getLogger(__name__)


def calculate_variability(data: np.ndarray) -> float:
    """
    Calculate coefficient of variation (CV) for a time series.
    
    Args:
        data: Time series data
        
    Returns:
        Coefficient of variation
    """
    std = np.std(data)
    mean = np.mean(data)
    if mean == 0:
        return 0
    return std / abs(mean)


class DensityMotifDiscovery:
    """
    Discover motifs with specific variability constraints.
    
    Finds patterns where WaterZumpf is stable but Ore and WaterMill vary.
    """
    
    def __init__(
        self,
        window_size: int = 60,
        max_motifs: int = 15,
        radius: float = 4.5,
        waterzumpf_max_cv: float = 0.01,
        ore_min_cv: float = 0.0008,
        watermill_min_cv: float = 0.0015,
        relative_variability_factor: float = 1.2
    ):
        """
        Initialize density motif discovery.
        
        Args:
            window_size: Window size in minutes
            max_motifs: Maximum number of motifs
            radius: Distance threshold
            waterzumpf_max_cv: Max CV for WaterZumpf (stable)
            ore_min_cv: Min CV for Ore (variable)
            watermill_min_cv: Min CV for WaterMill (variable)
            relative_variability_factor: Ore/WaterMill should be this much more variable than WaterZumpf
        """
        self.window_size = window_size
        self.max_motifs = max_motifs
        self.radius = radius
        self.waterzumpf_max_cv = waterzumpf_max_cv
        self.ore_min_cv = ore_min_cv
        self.watermill_min_cv = watermill_min_cv
        self.relative_variability_factor = relative_variability_factor
        
        self.motifs: List[Motif] = []
    
    def discover(self, df: pd.DataFrame) -> List[Motif]:
        """
        Discover constrained motifs.
        
        Args:
            df: DataFrame with required columns
            
        Returns:
            List of discovered motifs
        """
        logger.info("Discovering density motifs with variability constraints...")
        logger.info(f"  Constraint: WaterZumpf stable, Ore & WaterMill variable")
        logger.info(f"  Window size: {self.window_size} minutes")
        logger.info(f"  Max motifs: {self.max_motifs}")
        logger.info(f"  Radius: {self.radius}")
        
        # Prepare time series
        features = ['WaterZumpf', 'Ore', 'WaterMill']
        T = self._prepare_time_series(df, features)
        
        # Compute matrix profile
        logger.info("  Computing multivariate matrix profile...")
        matrix_profile, profile_indices = stumpy.mstump(T, m=self.window_size)
        mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))
        
        logger.info("  Discovering motifs with variability filtering...")
        
        self.motifs = []
        used_indices = set()
        n_windows = matrix_profile.shape[1]
        
        for motif_idx in range(self.max_motifs):
            # Find seed that passes variability constraints
            seed_idx, seed_distance = self._find_constrained_seed(
                df, mp_distances, used_indices, n_windows
            )
            
            if seed_idx is None or seed_distance > self.radius:
                break
            
            # Find instances
            valid_instances = self._find_constrained_instances(
                df, T, seed_idx, n_windows, used_indices, mp_distances
            )
            
            if len(valid_instances) >= 2:  # Need at least 2 instances
                motif = Motif(motif_id=len(self.motifs) + 1)
                
                for inst_data in valid_instances:
                    instance = MotifInstance(
                        start=inst_data['start'],
                        end=inst_data['end'],
                        distance=inst_data['distance'],
                        data=inst_data['data']
                    )
                    instance.add_metadata('waterzumpf_cv', inst_data['waterzumpf_cv'])
                    instance.add_metadata('ore_cv', inst_data['ore_cv'])
                    instance.add_metadata('watermill_cv', inst_data['watermill_cv'])
                    
                    motif.add_instance(instance)
                
                self.motifs.append(motif)
                
                # Mark as used
                for inst in valid_instances:
                    for offset in range(-self.window_size, self.window_size):
                        neighbor = inst['start'] + offset
                        if 0 <= neighbor < n_windows:
                            used_indices.add(neighbor)
            else:
                # Mark seed as used to avoid infinite loop
                for offset in range(-self.window_size, self.window_size):
                    neighbor = seed_idx + offset
                    if 0 <= neighbor < n_windows:
                        used_indices.add(neighbor)
        
        logger.info(f"  ✓ Found {len(self.motifs)} motif groups")
        total_instances = sum(len(m.instances) for m in self.motifs)
        logger.info(f"  ✓ Total instances: {total_instances}")
        
        return self.motifs
    
    def _prepare_time_series(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """Prepare and normalize time series."""
        ts_list = []
        for col in features:
            ts = np.array(df[col])
            ts = (ts - np.mean(ts)) / np.std(ts)
            ts_list.append(ts)
        return np.array(ts_list)
    
    def _find_constrained_seed(
        self,
        df: pd.DataFrame,
        mp_distances: np.ndarray,
        used_indices: set,
        n_windows: int
    ) -> Tuple[int, float]:
        """Find seed that passes variability constraints."""
        seed_idx = None
        seed_distance = float('inf')
        
        for i in range(n_windows):
            if i in used_indices:
                continue
            
            dist = mp_distances[i]
            if np.isnan(dist) or np.isinf(dist):
                continue
            
            # Check variability constraints
            if not self._check_variability_constraints(df, i):
                continue
            
            if dist < seed_distance:
                seed_distance = dist
                seed_idx = i
        
        return seed_idx, seed_distance
    
    def _check_variability_constraints(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if window at index passes variability constraints."""
        waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + self.window_size].values
        ore_data = df['Ore'].iloc[idx:idx + self.window_size].values
        watermill_data = df['WaterMill'].iloc[idx:idx + self.window_size].values
        
        waterzumpf_cv = calculate_variability(waterzumpf_data)
        ore_cv = calculate_variability(ore_data)
        watermill_cv = calculate_variability(watermill_data)
        
        # Apply constraints
        return (
            waterzumpf_cv <= self.waterzumpf_max_cv and
            ore_cv >= self.ore_min_cv and
            watermill_cv >= self.watermill_min_cv and
            ore_cv >= waterzumpf_cv * self.relative_variability_factor and
            watermill_cv >= waterzumpf_cv * self.relative_variability_factor
        )
    
    def _find_constrained_instances(
        self,
        df: pd.DataFrame,
        T: np.ndarray,
        seed_idx: int,
        n_windows: int,
        used_indices: set,
        mp_distances: np.ndarray
    ) -> List[dict]:
        """Find instances that pass variability constraints."""
        # Compute distance profile
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
        
        # Find candidates
        sorted_candidates = np.argsort(aggregated_profile)
        valid_instances = []
        
        for idx in sorted_candidates:
            if len(valid_instances) >= 20:  # Max instances per motif
                break
            
            if idx >= n_windows or idx in used_indices:
                continue
            
            dist = aggregated_profile[idx]
            if np.isnan(dist) or np.isinf(dist) or dist > self.radius:
                continue
            
            # Check variability constraints
            if not self._check_variability_constraints(df, idx):
                continue
            
            # Avoid overlapping
            if any(abs(idx - vi['start']) < self.window_size for vi in valid_instances):
                continue
            
            # Extract data
            waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + self.window_size].values
            ore_data = df['Ore'].iloc[idx:idx + self.window_size].values
            watermill_data = df['WaterMill'].iloc[idx:idx + self.window_size].values
            density_data = df['DensityHC'].iloc[idx:idx + self.window_size].values
            timestamp_data = df['TimeStamp'].iloc[idx:idx + self.window_size].values
            
            instance = {
                'start': idx,
                'end': idx + self.window_size,
                'distance': dist,
                'waterzumpf_cv': calculate_variability(waterzumpf_data),
                'ore_cv': calculate_variability(ore_data),
                'watermill_cv': calculate_variability(watermill_data),
                'data': {
                    'WaterZumpf': waterzumpf_data,
                    'Ore': ore_data,
                    'WaterMill': watermill_data,
                    'DensityHC': density_data,
                    'TimeStamp': timestamp_data
                }
            }
            valid_instances.append(instance)
        
        return valid_instances


def find_optimal_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 20) -> int:
    """
    Find the lag that maximizes correlation between x and y.
    
    Args:
        x: First time series
        y: Second time series
        max_lag: Maximum lag to consider
        
    Returns:
        Optimal lag in time steps
    """
    if len(x) != len(y) or len(x) < 2:
        return 0
    
    # Normalize
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)
    
    # Cross-correlation
    correlation = correlate(x_norm, y_norm, mode='same')
    
    # Find peak within max_lag
    center = len(correlation) // 2
    search_range = slice(max(0, center - max_lag), min(len(correlation), center + max_lag + 1))
    correlation_window = correlation[search_range]
    
    if len(correlation_window) == 0:
        return 0
    
    peak_idx = np.argmax(np.abs(correlation_window))
    lag = peak_idx - min(max_lag, center)
    
    return int(lag)


def analyze_density_behavior(motifs: List[Motif]) -> List[dict]:
    """
    Analyze how DensityHC behaves in discovered motifs.
    
    Args:
        motifs: List of motifs to analyze
        
    Returns:
        List of analysis results per motif
    """
    logger.info("Analyzing density behavior in motifs...")
    
    analysis_results = []
    
    for motif in motifs:
        logger.info(f"\nMotif {motif.motif_id} ({len(motif.instances)} instances):")
        
        # Aggregate statistics
        density_changes = []
        ore_density_corrs = []
        watermill_density_corrs = []
        ore_density_lags = []
        watermill_density_lags = []
        
        for instance in motif.instances:
            density = instance.data['DensityHC']
            ore = instance.data['Ore']
            watermill = instance.data['WaterMill']
            
            # Density change
            density_change = density[-1] - density[0]
            density_changes.append(density_change)
            
            # Correlations
            if len(density) > 1:
                ore_corr, _ = pearsonr(ore, density)
                watermill_corr, _ = pearsonr(watermill, density)
                ore_density_corrs.append(ore_corr)
                watermill_density_corrs.append(watermill_corr)
            
            # Lag analysis
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
        
        logger.info(f"  Avg DensityHC change: {avg_density_change:+.2f}")
        logger.info(f"  Ore-Density correlation: {avg_ore_corr:+.3f} (lag: {avg_ore_lag:.0f} min)")
        logger.info(f"  WaterMill-Density correlation: {avg_watermill_corr:+.3f} (lag: {avg_watermill_lag:.0f} min)")
        
        analysis_results.append({
            'motif_id': motif.motif_id,
            'num_instances': len(motif.instances),
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
