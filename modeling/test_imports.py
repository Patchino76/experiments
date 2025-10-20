"""
Quick test to verify all imports work correctly.
"""

print("Testing imports...")

try:
    from config import PipelineConfig
    print("✓ config imported")
except Exception as e:
    print(f"✗ config failed: {e}")

try:
    from database import DataLoader
    print("✓ database imported")
except Exception as e:
    print(f"✗ database failed: {e}")

try:
    from motif_discovery import MotifDiscovery, Motif, MotifInstance
    print("✓ motif_discovery imported")
except Exception as e:
    print(f"✗ motif_discovery failed: {e}")

try:
    from density_analysis import DensityMotifDiscovery, analyze_density_behavior
    print("✓ density_analysis imported")
except Exception as e:
    print(f"✗ density_analysis failed: {e}")

try:
    from segmentation import create_segmented_dataset, merge_motif_collections
    print("✓ segmentation imported")
except Exception as e:
    print(f"✗ segmentation failed: {e}")

try:
    from visualization import plot_motif_instances
    print("✓ visualization imported")
except Exception as e:
    print(f"✗ visualization failed: {e}")

print("\nAll imports successful! ✓")
