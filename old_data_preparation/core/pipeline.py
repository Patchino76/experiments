"""
Core pipeline for executing pattern discovery and analysis.

This module provides the main Pipeline class that coordinates the execution
of multiple pattern discovery algorithms and handles result aggregation.
"""
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
import pandas as pd

from .base_pattern import BasePattern
from .pattern_registry import PatternRegistry

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Main pipeline for executing pattern discovery and analysis.
    
    This class coordinates the execution of multiple pattern discovery algorithms,
    manages their configurations, and handles result aggregation and persistence.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "output"):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration dictionary
            output_dir: Base directory for output files
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.patterns: Dict[str, BasePattern] = {}
        self.results: Dict[str, Any] = {}
        self._initialize_patterns()
    
    def _initialize_patterns(self) -> None:
        """Initialize pattern instances based on configuration."""
        pattern_configs = self.config.get('patterns', {})
        
        for pattern_name, pattern_config in pattern_configs.items():
            if not pattern_config.get('enabled', False):
                logger.info(f"Skipping disabled pattern: {pattern_name}")
                continue
                
            pattern = PatternRegistry.create_pattern(pattern_name, pattern_config)
            if pattern is not None:
                self.patterns[pattern_name] = pattern
                logger.info(f"Initialized pattern: {pattern_name}")
            else:
                logger.error(f"Failed to initialize pattern: {pattern_name}")
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the pipeline on the input data.
        
        Args:
            data: Input time series data as a DataFrame
            
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting pattern discovery pipeline")
        
        # Create output directory structure
        self._prepare_output_dirs()
        
        # Run each pattern discovery algorithm
        for pattern_name, pattern in self.patterns.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Running pattern: {pattern_name}")
            logger.info(f"{'='*50}")
            
            try:
                # Discover patterns
                motifs = pattern.discover(data)
                
                # Analyze results
                analysis = pattern.analyze(data)
                
                # Save results
                pattern_output_dir = self.output_dir / pattern_name
                pattern.save_results(pattern_output_dir)
                
                # Store results
                self.results[pattern_name] = {
                    'motifs': motifs,
                    'analysis': analysis,
                    'output_dir': str(pattern_output_dir)
                }
                
                logger.info(f"Completed pattern: {pattern_name}")
                
            except Exception as e:
                logger.exception(f"Error running pattern '{pattern_name}': {e}")
                self.results[pattern_name] = {
                    'error': str(e),
                    'success': False
                }
        
        # Combine and save all results
        self._combine_results()
        
        logger.info("\nPipeline execution completed")
        return self.results
    
    def _prepare_output_dirs(self) -> None:
        """Create necessary output directories."""
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create pattern-specific directories
        for pattern_name in self.patterns:
            pattern_dir = self.output_dir / pattern_name
            pattern_dir.mkdir(exist_ok=True)
    
    def _combine_results(self) -> None:
        """Combine results from all patterns."""
        combined_results = {}
        
        for pattern_name, result in self.results.items():
            if 'error' in result:
                continue
                
            pattern_results = {
                'motifs': [m.to_dict() for m in result.get('motifs', [])],
                'analysis': result.get('analysis', {}),
                'output_dir': result.get('output_dir', '')
            }
            combined_results[pattern_name] = pattern_results
        
        # Save combined results
        import json
        output_file = self.output_dir / 'combined_results.json'
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        logger.info(f"Saved combined results to {output_file}")
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the pipeline results.
        
        Returns:
            Dictionary containing all results
        """
        return self.results
