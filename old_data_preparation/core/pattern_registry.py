"""
Pattern registry for managing different types of pattern discovery algorithms.

This module provides a registry system for dynamically loading and instantiating
pattern discovery implementations.
"""
from typing import Dict, Type, Any, Optional, TypeVar, Callable
import logging

from .base_pattern import BasePattern

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BasePattern)

class PatternRegistry:
    """Registry for pattern discovery implementations."""
    
    _instance = None
    _patterns: Dict[str, Type[BasePattern]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, name: str) -> Callable[[Type[BasePattern]], Type[BasePattern]]:
        """
        Decorator to register a pattern class.
        
        Args:
            name: Unique name for this pattern type
            
        Returns:
            Decorator function
        """
        def decorator(pattern_class: Type[BasePattern]) -> Type[BasePattern]:
            if name in cls._patterns:
                logger.warning(f"Pattern '{name}' is already registered. Overwriting.")
            cls._patterns[name] = pattern_class
            logger.debug(f"Registered pattern: {name} -> {pattern_class.__name__}")
            return pattern_class
        return decorator
    
    @classmethod
    def create_pattern(cls, name: str, config: Dict[str, Any]) -> Optional[BasePattern]:
        """
        Create an instance of a registered pattern.
        
        Args:
            name: Name of the pattern to create
            config: Configuration dictionary for the pattern
            
        Returns:
            Instance of the requested pattern, or None if not found
        """
        if name not in cls._patterns:
            logger.error(f"Pattern '{name}' is not registered")
            return None
            
        try:
            return cls._patterns[name](config)
        except Exception as e:
            logger.exception(f"Failed to create pattern '{name}': {e}")
            return None
    
    @classmethod
    def list_patterns(cls) -> Dict[str, Type[BasePattern]]:
        """
        Get all registered patterns.
        
        Returns:
            Dictionary mapping pattern names to their classes
        """
        return dict(cls._patterns)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered patterns."""
        cls._patterns.clear()
        logger.debug("Cleared pattern registry")
