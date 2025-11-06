"""
Base classes for motif patterns.

Provides the foundation for all pattern discovery implementations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class MotifInstance:
    """Represents a single instance of a motif pattern."""
    
    start: int
    end: int
    distance: float
    data: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Return length of the instance."""
        return self.end - self.start
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to instance."""
        self.metadata[key] = value
    
    def get_feature(self, feature_name: str) -> Optional[np.ndarray]:
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


@dataclass
class Motif:
    """Represents a motif group containing multiple similar instances."""
    
    motif_id: int
    instances: List[MotifInstance] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    avg_distance: float = 0.0
    
    def add_instance(self, instance: MotifInstance):
        """Add an instance to this motif."""
        self.instances.append(instance)
        self._update_avg_distance()
    
    def _update_avg_distance(self):
        """Update average distance across instances."""
        if self.instances:
            self.avg_distance = float(np.mean([inst.distance for inst in self.instances]))
    
    def __len__(self) -> int:
        """Return number of instances."""
        return len(self.instances)
    
    def add_metadata(self, key: str, value: Any):
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


class BasePattern(ABC):
    """
    Abstract base class for all motif discovery patterns.
    
    Subclasses must implement the discover() method.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize pattern.
        
        Args:
            name: Pattern name/identifier
            config: Pattern configuration dictionary
        """
        self.name = name
        self.config = config
        self.motifs: List[Motif] = []
        self.enabled = config.get('enabled', True)
        
        # Common parameters
        self.window_size = config.get('window_size', 60)
        self.max_motifs = config.get('max_motifs', 15)
        self.radius = config.get('radius', 4.5)
        self.max_instances_per_motif = config.get('max_instances_per_motif', 20)
        
        logger.info(f"Initialized pattern: {name}")
    
    @abstractmethod
    def discover(self, df: pd.DataFrame) -> List[Motif]:
        """
        Discover motifs in the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of discovered motifs
        """
        pass
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid, False otherwise
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.warning(f"Pattern '{self.name}': Missing columns {missing}")
            return False
        return True
    
    def prepare_time_series(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Prepare and normalize time series data.
        
        Args:
            df: Input DataFrame
            features: Feature columns to extract
            
        Returns:
            Normalized time series array (features x time)
        """
        ts_list = []
        for col in features:
            ts = np.array(df[col])
            # Z-score normalization
            ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-8)
            ts_list.append(ts)
        return np.array(ts_list)
    
    def calculate_variability(self, data: np.ndarray) -> float:
        """
        Calculate coefficient of variation (CV).
        
        Args:
            data: Time series data
            
        Returns:
            Coefficient of variation
        """
        std = np.std(data)
        mean = np.mean(data)
        if abs(mean) < 1e-8:
            return 0.0
        return std / abs(mean)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for discovered motifs.
        
        Returns:
            Dictionary with summary information
        """
        if not self.motifs:
            return {
                'pattern_name': self.name,
                'num_motifs': 0,
                'total_instances': 0
            }
        
        return {
            'pattern_name': self.name,
            'num_motifs': len(self.motifs),
            'total_instances': sum(len(m) for m in self.motifs),
            'avg_instances_per_motif': np.mean([len(m) for m in self.motifs]),
            'avg_distance': np.mean([m.avg_distance for m in self.motifs])
        }
