"""
Motif and MotifInstance classes for pattern discovery.

This module provides the core data structures for representing discovered patterns.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import json

@dataclass
class MotifInstance:
    """Represents a single instance of a discovered pattern."""
    start: int
    """Start index of the instance in the original time series."""
    
    end: int
    """End index of the instance in the original time series."""
    
    distance: float
    """Distance of this instance to the motif centroid."""
    
    data: Dict[str, List[float]]
    """The actual time series data for this instance."""
    
    instance_id: int = field(default_factory=lambda: id(object()))
    """Unique identifier for this instance."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about this instance."""
    
    def get_feature(self, feature_name: str) -> Optional[List[float]]:
        """
        Get the time series data for a specific feature.
        
        Args:
            feature_name: Name of the feature to retrieve
            
        Returns:
            List of values for the requested feature, or None if not found
        """
        return self.data.get(feature_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the instance to a dictionary."""
        return {
            'start': self.start,
            'end': self.end,
            'distance': self.distance,
            'data': self.data,
            'instance_id': self.instance_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MotifInstance':
        """Create a MotifInstance from a dictionary."""
        return cls(
            start=data['start'],
            end=data['end'],
            distance=data['distance'],
            data=data['data'],
            instance_id=data.get('instance_id', id(object())),
            metadata=data.get('metadata', {})
        )

@dataclass
class Motif:
    """Represents a discovered pattern (motif) in the time series data."""
    motif_id: int
    """Unique identifier for this motif."""
    
    instances: List[MotifInstance] = field(default_factory=list)
    """List of instances that belong to this motif."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about this motif."""
    
    @property
    def avg_distance(self) -> float:
        """Calculate the average distance of instances to the motif centroid."""
        if not self.instances:
            return 0.0
        return sum(inst.distance for inst in self.instances) / len(self.instances)
    
    def add_instance(self, instance: MotifInstance) -> None:
        """Add an instance to this motif."""
        self.instances.append(instance)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to this motif."""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the motif to a dictionary."""
        return {
            'motif_id': self.motif_id,
            'instances': [inst.to_dict() for inst in self.instances],
            'metadata': self.metadata,
            'avg_distance': self.avg_distance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Motif':
        """Create a Motif from a dictionary."""
        motif = cls(
            motif_id=data['motif_id'],
            metadata=data.get('metadata', {})
        )
        
        for inst_data in data.get('instances', []):
            motif.add_instance(MotifInstance.from_dict(inst_data))
            
        return motif
    
    def save(self, filepath: str) -> None:
        """Save the motif to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Motif':
        """Load a motif from a JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))
