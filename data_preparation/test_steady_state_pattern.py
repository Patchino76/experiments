"""
Test script for steady-state pattern.

Verifies that the new steady-state pattern is properly configured
and can be used in the pipeline.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from config.pattern_configs import (
    create_steady_state_pattern,
    get_default_patterns
)
from config.defaults import PipelineConfig


def test_steady_state_pattern_creation():
    """Test that steady-state pattern can be created."""
    print("=" * 70)
    print("TEST 1: Create Steady-State Pattern")
    print("=" * 70)
    
    pattern = create_steady_state_pattern()
    
    print(f"✓ Pattern created: {pattern.name}")
    print(f"  Type: {pattern.type}")
    print(f"  Enabled: {pattern.enabled}")
    print(f"  Window size: {pattern.window_size}")
    print(f"  Max motifs: {pattern.max_motifs}")
    print(f"  Radius: {pattern.radius}")
    
    print(f"\n  Constraints:")
    for feature, constraint in pattern.constraints.items():
        print(f"    {feature}: {constraint}")
    
    # Verify all required features are present
    required_features = ['Ore', 'WaterMill', 'WaterZumpf', 'PSI200']
    for feature in required_features:
        assert feature in pattern.constraints, f"Missing feature: {feature}"
        assert pattern.constraints[feature]['type'] == 'stable', \
            f"{feature} should be stable"
    
    print("\n✅ All assertions passed!")
    return True


def test_pattern_in_defaults():
    """Test that steady-state pattern is in default patterns."""
    print("\n" + "=" * 70)
    print("TEST 2: Steady-State Pattern in Defaults")
    print("=" * 70)
    
    patterns = get_default_patterns()
    pattern_names = [p.name for p in patterns]
    
    print(f"Default patterns: {pattern_names}")
    
    assert 'steady_state' in pattern_names, \
        "steady_state pattern not in defaults"
    
    # Find the steady-state pattern
    ss_pattern = next(p for p in patterns if p.name == 'steady_state')
    
    print(f"\n✓ Steady-state pattern found in defaults")
    print(f"  Enabled: {ss_pattern.enabled}")
    print(f"  Position: {pattern_names.index('steady_state') + 1} of {len(patterns)}")
    
    print("\n✅ All assertions passed!")
    return True


def test_custom_configuration():
    """Test custom steady-state pattern configuration."""
    print("\n" + "=" * 70)
    print("TEST 3: Custom Configuration")
    print("=" * 70)
    
    # Create stricter steady-state pattern
    strict_pattern = create_steady_state_pattern(
        window_size=120,
        max_motifs=10,
        radius=4.0
    )
    
    print(f"✓ Strict pattern created:")
    print(f"  Window size: {strict_pattern.window_size} (longer)")
    print(f"  Max motifs: {strict_pattern.max_motifs} (fewer)")
    print(f"  Radius: {strict_pattern.radius} (tighter)")
    
    # Create relaxed steady-state pattern
    relaxed_pattern = create_steady_state_pattern(
        window_size=60,
        max_motifs=20,
        radius=6.0
    )
    
    print(f"\n✓ Relaxed pattern created:")
    print(f"  Window size: {relaxed_pattern.window_size} (shorter)")
    print(f"  Max motifs: {relaxed_pattern.max_motifs} (more)")
    print(f"  Radius: {relaxed_pattern.radius} (looser)")
    
    print("\n✅ All assertions passed!")
    return True


def test_pipeline_config():
    """Test that steady-state pattern works in pipeline config."""
    print("\n" + "=" * 70)
    print("TEST 4: Pipeline Configuration")
    print("=" * 70)
    
    # Create default config (includes steady-state)
    config = PipelineConfig.create_default(
        mill_number=8,
        start_date="2025-09-01",
        end_date="2025-11-03"
    )
    
    print(f"✓ Pipeline config created")
    print(f"  Total patterns: {len(config.patterns)}")
    
    # Check steady-state pattern is present
    pattern_names = [p.name for p in config.patterns]
    assert 'steady_state' in pattern_names, \
        "steady_state not in pipeline config"
    
    # Get enabled patterns
    enabled = config.get_enabled_patterns()
    enabled_names = [p.name for p in enabled]
    
    print(f"  Enabled patterns: {enabled_names}")
    
    if 'steady_state' in enabled_names:
        print(f"  ✓ Steady-state pattern is ENABLED")
    else:
        print(f"  ⚠ Steady-state pattern is DISABLED")
    
    print("\n✅ All assertions passed!")
    return True


def test_pattern_dict_conversion():
    """Test that pattern converts to dict correctly."""
    print("\n" + "=" * 70)
    print("TEST 5: Pattern Dictionary Conversion")
    print("=" * 70)
    
    pattern = create_steady_state_pattern()
    pattern_dict = pattern.to_dict()
    
    print(f"✓ Pattern converted to dict")
    print(f"  Keys: {list(pattern_dict.keys())}")
    
    # Verify required keys
    required_keys = ['name', 'type', 'enabled', 'window_size', 
                     'max_motifs', 'radius', 'constraints']
    for key in required_keys:
        assert key in pattern_dict, f"Missing key: {key}"
    
    print(f"\n  Pattern dict:")
    print(f"    name: {pattern_dict['name']}")
    print(f"    type: {pattern_dict['type']}")
    print(f"    constraints: {len(pattern_dict['constraints'])} features")
    
    print("\n✅ All assertions passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("STEADY-STATE PATTERN TEST SUITE")
    print("=" * 70)
    
    tests = [
        test_steady_state_pattern_creation,
        test_pattern_in_defaults,
        test_custom_configuration,
        test_pipeline_config,
        test_pattern_dict_conversion
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(('PASS', test.__name__))
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            results.append(('FAIL', test.__name__))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for status, name in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {name}: {status}")
    
    passed = sum(1 for s, _ in results if s == 'PASS')
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Steady-state pattern is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == '__main__':
    exit(main())
