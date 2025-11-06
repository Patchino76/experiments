"""
Quick test script for the new data preparation system.

Tests basic functionality without running the full pipeline.
"""

import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Module Imports")
    logger.info("=" * 80)
    
    try:
        from core.base_pattern import BasePattern, Motif, MotifInstance
        from core.pattern_registry import PatternRegistry
        from core.data_loader import DataLoader
        from core.segmentation import SegmentationEngine
        from patterns.mv_pattern import MVPattern
        from patterns.constraint_pattern import ConstraintPattern
        from config.defaults import PipelineConfig, PatternConfig, DataConfig
        from config.pattern_configs import get_default_patterns
        from analysis.analyzer import PatternAnalyzer
        from analysis.visualizer import PatternVisualizer
        from pipeline import DataPreparationPipeline
        
        logger.info("✓ All modules imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_pattern_registry():
    """Test pattern registry system."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Pattern Registry")
    logger.info("=" * 80)
    
    try:
        from core.pattern_registry import PatternRegistry
        
        # Check registered patterns
        patterns = PatternRegistry.list_patterns()
        logger.info(f"Registered patterns: {patterns}")
        
        expected = ['mv', 'constraint']
        for pattern_type in expected:
            if pattern_type in patterns:
                logger.info(f"  ✓ {pattern_type} registered")
            else:
                logger.error(f"  ✗ {pattern_type} not registered")
                return False
        
        logger.info("✓ Pattern registry working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Pattern registry test failed: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Configuration System")
    logger.info("=" * 80)
    
    try:
        from config.defaults import PipelineConfig
        from config.pattern_configs import get_default_patterns
        
        # Create default config
        config = PipelineConfig.create_default(
            mill_number=8,
            start_date="2025-08-01",
            end_date="2025-11-03"
        )
        
        logger.info(f"  Mill number: {config.data.mill_number}")
        logger.info(f"  Date range: {config.data.start_date} to {config.data.end_date}")
        logger.info(f"  Patterns: {len(config.patterns)}")
        
        # Check patterns
        enabled = config.get_enabled_patterns()
        logger.info(f"  Enabled patterns: {len(enabled)}")
        
        for pattern in enabled:
            logger.info(f"    - {pattern.name} ({pattern.type})")
        
        logger.info("✓ Configuration system working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_pattern_creation():
    """Test pattern creation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Pattern Creation")
    logger.info("=" * 80)
    
    try:
        from core.pattern_registry import PatternRegistry
        from config.pattern_configs import create_mv_pattern, create_density_pattern
        
        # Create MV pattern
        mv_config = create_mv_pattern()
        mv_pattern = PatternRegistry.create_pattern('mv', mv_config.to_dict())
        logger.info(f"  ✓ Created MV pattern: {mv_pattern.name}")
        
        # Create constraint pattern
        density_config = create_density_pattern()
        density_pattern = PatternRegistry.create_pattern('density', density_config.to_dict())
        logger.info(f"  ✓ Created density pattern: {density_pattern.name}")
        
        # Check constraints
        logger.info(f"    Constraints: {list(density_pattern.constraints.keys())}")
        logger.info(f"    Stable features: {density_pattern.stable_features}")
        logger.info(f"    Varying features: {density_pattern.varying_features}")
        
        logger.info("✓ Pattern creation working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Pattern creation test failed: {e}")
        return False


def test_custom_pattern():
    """Test custom pattern creation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Custom Pattern Creation")
    logger.info("=" * 80)
    
    try:
        from config.pattern_configs import create_custom_pattern
        from core.pattern_registry import PatternRegistry
        
        # Create custom pattern
        custom = create_custom_pattern(
            name='test_pattern',
            constraints={
                'Ore': {'type': 'stable', 'max_cv': 0.01},
                'WaterMill': {'type': 'varying', 'min_cv': 0.001}
            },
            window_size=90,
            max_motifs=10
        )
        
        logger.info(f"  Pattern name: {custom.name}")
        logger.info(f"  Pattern type: {custom.type}")
        logger.info(f"  Window size: {custom.window_size}")
        logger.info(f"  Constraints: {list(custom.constraints.keys())}")
        
        # Instantiate pattern
        pattern = PatternRegistry.create_pattern('test_pattern', custom.to_dict())
        logger.info(f"  ✓ Instantiated: {pattern.name}")
        logger.info(f"    Stable: {pattern.stable_features}")
        logger.info(f"    Varying: {pattern.varying_features}")
        
        logger.info("✓ Custom pattern creation working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Custom pattern test failed: {e}")
        return False


def test_analyzer_visualizer():
    """Test analyzer and visualizer initialization."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Analyzer and Visualizer")
    logger.info("=" * 80)
    
    try:
        from analysis.analyzer import PatternAnalyzer
        from analysis.visualizer import PatternVisualizer
        
        analyzer = PatternAnalyzer()
        logger.info("  ✓ PatternAnalyzer initialized")
        
        visualizer = PatternVisualizer()
        logger.info("  ✓ PatternVisualizer initialized")
        
        logger.info("✓ Analyzer and visualizer working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Analyzer/visualizer test failed: {e}")
        return False


def test_pipeline_initialization():
    """Test pipeline initialization."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 7: Pipeline Initialization")
    logger.info("=" * 80)
    
    try:
        from pipeline import DataPreparationPipeline
        from config.defaults import PipelineConfig
        
        config = PipelineConfig.create_default(
            mill_number=8,
            start_date="2025-01-01",
            end_date="2025-11-03"
        )
        
        pipeline = DataPreparationPipeline(config)
        
        logger.info("  ✓ Pipeline initialized")
        logger.info(f"    Data loader: {pipeline.data_loader is not None}")
        logger.info(f"    Segmentation: {pipeline.segmentation is not None}")
        logger.info(f"    Analyzer: {pipeline.analyzer is not None}")
        logger.info(f"    Visualizer: {pipeline.visualizer is not None}")
        
        logger.info("✓ Pipeline initialization working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Pipeline initialization test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("DATA PREPARATION SYSTEM - TEST SUITE")
    logger.info("=" * 80)
    
    tests = [
        ("Module Imports", test_imports),
        ("Pattern Registry", test_pattern_registry),
        ("Configuration System", test_configuration),
        ("Pattern Creation", test_pattern_creation),
        ("Custom Pattern Creation", test_custom_pattern),
        ("Analyzer and Visualizer", test_analyzer_visualizer),
        ("Pipeline Initialization", test_pipeline_initialization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    logger.info("\n" + "-" * 80)
    logger.info(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    logger.info("=" * 80)
    
    if passed == total:
        logger.info("\n🎉 All tests passed! System is ready to use.")
        return True
    else:
        logger.error(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
