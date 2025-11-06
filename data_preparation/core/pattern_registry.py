"""
Pattern registry for managing and discovering patterns.

Provides a centralized system for registering, configuring, and
executing pattern discovery.
"""

import logging
from typing import List, Dict, Type, Any
from .base_pattern import BasePattern, Motif

logger = logging.getLogger(__name__)


class PatternRegistry:
    """
    Registry for managing pattern discovery implementations.
    
    Patterns can self-register and be dynamically instantiated
    based on configuration.
    """
    
    _patterns: Dict[str, Type[BasePattern]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a pattern class.
        
        Usage:
            @PatternRegistry.register('my_pattern')
            class MyPattern(BasePattern):
                ...
        """
        def decorator(pattern_class: Type[BasePattern]):
            cls._patterns[name] = pattern_class
            logger.debug(f"Registered pattern: {name}")
            return pattern_class
        return decorator
    
    @classmethod
    def create_pattern(cls, name: str, config: Dict[str, Any]) -> BasePattern:
        """
        Create a pattern instance from configuration.
        
        Args:
            name: Pattern type name
            config: Pattern configuration
            
        Returns:
            Instantiated pattern object
            
        Raises:
            ValueError: If pattern type not registered
        """
        pattern_type = config.get('type', name)
        
        if pattern_type not in cls._patterns:
            raise ValueError(
                f"Pattern type '{pattern_type}' not registered. "
                f"Available: {list(cls._patterns.keys())}"
            )
        
        pattern_class = cls._patterns[pattern_type]
        return pattern_class(name=name, config=config)
    
    @classmethod
    def list_patterns(cls) -> List[str]:
        """Get list of registered pattern types."""
        return list(cls._patterns.keys())
    
    @classmethod
    def discover_all(cls, df, pattern_configs: List[Dict[str, Any]]) -> Dict[str, List[Motif]]:
        """
        Discover motifs using multiple patterns.
        
        Args:
            df: Input DataFrame
            pattern_configs: List of pattern configurations
            
        Returns:
            Dictionary mapping pattern names to discovered motifs
        """
        results = {}
        
        for config in pattern_configs:
            name = config.get('name', 'unnamed')
            enabled = config.get('enabled', True)
            
            if not enabled:
                logger.info(f"Pattern '{name}' is disabled, skipping...")
                results[name] = []
                continue
            
            try:
                pattern = cls.create_pattern(name, config)
                motifs = pattern.discover(df)
                results[name] = motifs
                
                summary = pattern.get_summary()
                logger.info(
                    f"Pattern '{name}': {summary['num_motifs']} motifs, "
                    f"{summary['total_instances']} instances"
                )
                
            except Exception as e:
                logger.error(f"Error discovering pattern '{name}': {e}", exc_info=True)
                results[name] = []
        
        return results
