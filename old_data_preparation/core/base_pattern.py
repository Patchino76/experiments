"""
Base classes for pattern discovery in time series data.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, TypeVar, Generic
import pandas as pd
from pathlib import Path
import logging

from data_preparation.motif import Motif

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BasePattern')

class BasePattern(ABC):
    """
    Abstract base class for all pattern discovery implementations.
    
    Subclasses must implement the discover() method and can optionally
    override analyze() and visualize() methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pattern with configuration.
        
        Args:
            config: Configuration dictionary for this pattern
        """
        self.config = config
        self.motifs: List[Motif] = []
        self.analysis_results: Dict[str, Any] = {}
        self._validate_config()
    
    @abstractmethod
    def discover(self, data: pd.DataFrame) -> List[Motif]:
        """
        Discover patterns in the input data.
        
        Args:
            data: Input time series data as a DataFrame
            
        Returns:
            List of discovered Motif objects
        """
        pass
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the discovered patterns.
        
        Args:
            data: Original input data
            
        Returns:
            Dictionary of analysis results
        """
        return {}
    
    def visualize(self, output_dir: Path) -> None:
        """
        Generate visualizations for the discovered patterns.
        
        Args:
            output_dir: Directory to save visualizations
        """
        pass
    
    def _validate_config(self) -> None:
        """Validate the configuration for this pattern."""
        required = ['enabled', 'window_size', 'max_motifs']
        for param in required:
            if param not in self.config:
                raise ValueError(f"Missing required config parameter: {param}")
    
    def is_enabled(self) -> bool:
        """Check if this pattern is enabled in the config."""
        return self.config.get('enabled', False)
    
    def get_name(self) -> str:
        """Get the name of this pattern (derived from class name)."""
        return self.__class__.__name__.replace('Pattern', '').lower()
    
    def save_results(self, output_dir: Path) -> None:
        """
        Save pattern discovery results to disk.
        
        Args:
            output_dir: Base directory to save results to
        """
        if not self.motifs:
            return
            
        # Create pattern-specific output directory
        pattern_dir = output_dir / self.get_name()
        pattern_dir.mkdir(parents=True, exist_ok=True)
        
        # Save motifs
        self._save_motifs(pattern_dir)
        
        # Save analysis
        self._save_analysis(pattern_dir)
        
        # Generate visualizations
        self.visualize(pattern_dir)
    
    def _save_motifs(self, output_dir: Path) -> None:
        """Save discovered motifs to disk."""
        import json
        
        # Convert motifs to serializable format
        motifs_data = [m.to_dict() for m in self.motifs]
        
        # Save as JSON
        output_file = output_dir / 'motifs.json'
        with open(output_file, 'w') as f:
            json.dump(motifs_data, f, indent=2)
    
    def _save_analysis(self, output_dir: Path) -> None:
        """Save analysis results to disk."""
        if not self.analysis_results:
            return
            
        import json
        
        output_file = output_dir / 'analysis.json'
        with open(output_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
