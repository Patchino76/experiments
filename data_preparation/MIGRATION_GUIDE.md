# Migration Guide: Old System → New System

This guide helps you migrate from the old `modeling/prepare_data.py` system to the new `data_preparation/` system.

## 🔄 Key Differences

### Old System
```python
# modeling/prepare_data.py
from config import PipelineConfig
from density_analysis import DensityMotifDiscovery
from constraint_patterns import InverseConstraintMotifDiscovery, DynamicMotifDiscovery

# Hardcoded pattern discovery
density_discovery = DensityMotifDiscovery(window_size=60, max_motifs=15)
inverse_discovery = InverseConstraintMotifDiscovery(window_size=60, max_motifs=10)
# ... separate class for each pattern
```

### New System
```python
# data_preparation/run.py
from data_preparation import DataPreparationPipeline, PipelineConfig
from config.pattern_configs import get_default_patterns

# Unified pattern system
config = PipelineConfig.create_default(mill_number=8, ...)
pipeline = DataPreparationPipeline(config)
pipeline.run()
```

## 📋 Migration Steps

### Step 1: Update Imports

**Old:**
```python
from config import PipelineConfig, DataConfig
from database import DataLoader
from motif_discovery import MotifDiscovery
from density_analysis import DensityMotifDiscovery
from constraint_patterns import InverseConstraintMotifDiscovery
```

**New:**
```python
from data_preparation import DataPreparationPipeline, PipelineConfig
from data_preparation.config.pattern_configs import (
    create_mv_pattern,
    create_density_pattern,
    create_inverse_pattern
)
```

### Step 2: Update Configuration

**Old:**
```python
config = PipelineConfig.create_default(mill_number, start_date, end_date)
config.motif.enable_density_pattern = True
config.motif.enable_inverse_pattern = True
config.motif.density_window_size = 60
config.motif.density_max_motifs = 15
```

**New:**
```python
patterns = [
    create_mv_pattern(window_size=60, max_motifs=20),
    create_density_pattern(window_size=60, max_motifs=15),
    create_inverse_pattern(window_size=60, max_motifs=10)
]

config = PipelineConfig.create_default(
    mill_number=mill_number,
    start_date=start_date,
    end_date=end_date,
    patterns=patterns
)
```

### Step 3: Update Pattern Discovery

**Old:**
```python
# Separate discovery for each pattern
if config.motif.enable_density_pattern:
    density_discovery = DensityMotifDiscovery(
        window_size=config.motif.density_window_size,
        max_motifs=config.motif.density_max_motifs
    )
    density_motifs = density_discovery.discover(df)

if config.motif.enable_inverse_pattern:
    inverse_discovery = InverseConstraintMotifDiscovery(
        window_size=config.motif.inverse_window_size,
        max_motifs=config.motif.inverse_max_motifs
    )
    inverse_motifs = inverse_discovery.discover(df)
```

**New:**
```python
# Unified discovery through pipeline
pipeline = DataPreparationPipeline(config)
pipeline.run()  # Discovers all enabled patterns automatically
```

### Step 4: Update Analysis

**Old:**
```python
from density_analysis import analyze_density_behavior

if density_motifs:
    density_analysis = analyze_density_behavior(density_motifs)
    analysis_df = pd.DataFrame(density_analysis)
    analysis_df.to_csv('density_analysis.csv')
```

**New:**
```python
# Analysis happens automatically in pipeline
# Or manually:
from data_preparation.analysis import PatternAnalyzer

analyzer = PatternAnalyzer()
analyses = analyzer.analyze_density_behavior(motifs)
analyzer.save_analysis(analyses, output_path, 'pattern_name')
```

### Step 5: Update Visualization

**Old:**
```python
from visualization import plot_density_analysis

if density_analysis:
    plot_density_analysis(density_analysis, 'density_analysis.png')
```

**New:**
```python
# Visualization happens automatically in pipeline
# Or manually:
from data_preparation.analysis import PatternVisualizer

visualizer = PatternVisualizer()
visualizer.plot_density_analysis(analyses, output_path, 'pattern_name')
```

## 🎯 Feature Mapping

| Old Feature | New Feature | Notes |
|-------------|-------------|-------|
| `DensityMotifDiscovery` | `ConstraintPattern` with density config | Universal class |
| `InverseConstraintMotifDiscovery` | `ConstraintPattern` with inverse config | Universal class |
| `DynamicMotifDiscovery` | `ConstraintPattern` with dynamic config | Universal class |
| `PressureConstraintMotifDiscovery` | `ConstraintPattern` with pressure config | Universal class |
| `config.motif.enable_*_pattern` | `pattern.enabled` in config | Per-pattern toggle |
| `config.motif.*_window_size` | `pattern.window_size` | Per-pattern parameter |
| `config.motif.*_max_motifs` | `pattern.max_motifs` | Per-pattern parameter |
| `analyze_density_behavior()` | `PatternAnalyzer.analyze_density_behavior()` | Method call |
| `plot_density_analysis()` | `PatternVisualizer.plot_density_analysis()` | Method call |

## 🆕 New Capabilities

### 1. Easy Custom Patterns

**Old:** Create new class, update config, modify pipeline (100+ lines of code)

**New:** Define constraints in config (5 lines)
```python
custom_pattern = create_custom_pattern(
    name='my_pattern',
    constraints={
        'Ore': {'type': 'stable', 'max_cv': 0.01},
        'WaterMill': {'type': 'varying', 'min_cv': 0.001}
    }
)
```

### 2. Runtime Pattern Toggle

**Old:** Modify config file, restart

**New:** Toggle at runtime
```python
for pattern in config.patterns:
    if pattern.name == 'pressure':
        pattern.enabled = False
```

### 3. Pattern-Specific Settings

**Old:** Global settings for all patterns

**New:** Per-pattern settings
```python
for pattern in config.patterns:
    if pattern.name == 'density':
        pattern.save_analysis = True
        pattern.save_plots = True
    elif pattern.name == 'inverse':
        pattern.save_analysis = True
        pattern.save_plots = False
```

## 📊 Output Compatibility

### File Names

| Old | New | Compatible? |
|-----|-----|-------------|
| `segmented_motifsMV_{mill}.csv` | `segmented_motifsMV_{mill}.csv` | ✅ Yes |
| `segmented_motifs_all_{mill}.csv` | `segmented_motifs_all_{mill}.csv` | ✅ Yes |
| `density_analysis.csv` | `density_analysis.csv` | ✅ Yes |
| `inverse_analysis.csv` | `inverse_analysis.csv` | ✅ Yes |
| `dynamic_analysis.csv` | `dynamic_analysis.csv` | ✅ Yes |

### File Structure

The new system produces **identical CSV structures** to the old system, ensuring:
- ✅ Existing model training scripts work without changes
- ✅ Analysis scripts work without changes
- ✅ Database schemas remain compatible

## 🔧 Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'data_preparation'`

**Solution:** Add to Python path
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
```

### Pattern Not Discovered

**Problem:** Pattern enabled but no motifs found

**Solution:** Check constraints and parameters
```python
# Relax constraints
pattern.constraints['Ore']['max_cv'] = 0.02  # Was 0.01
pattern.radius = 5.0  # Was 4.5
```

### Missing Database Module

**Problem:** `ImportError: cannot import name 'load_mill_data'`

**Solution:** Use cached data
```python
config.use_database = False
# Ensure initial_data.csv exists in output directory
```

## 📝 Complete Example

### Old System
```python
# modeling/prepare_data.py
from config import PipelineConfig
from database import DataLoader
from motif_discovery import MotifDiscovery
from density_analysis import DensityMotifDiscovery, analyze_density_behavior
from constraint_patterns import InverseConstraintMotifDiscovery
from segmentation import create_segmented_dataset, merge_motif_collections
from visualization import plot_density_analysis

config = PipelineConfig.create_default(8, "2025-01-01", "2025-11-03")
config.motif.enable_density_pattern = True
config.motif.enable_inverse_pattern = True

# Load data
loader = DataLoader(use_database=True)
df = loader.load_mill_data(...)

# Discover MV motifs
mv_discovery = MotifDiscovery(window_size=60, max_motifs=20)
mv_motifs = mv_discovery.discover(df, ['Ore', 'WaterMill', 'WaterZumpf'])

# Discover density motifs
density_discovery = DensityMotifDiscovery(window_size=60, max_motifs=15)
density_motifs = density_discovery.discover(df)

# Discover inverse motifs
inverse_discovery = InverseConstraintMotifDiscovery(window_size=60, max_motifs=10)
inverse_motifs = inverse_discovery.discover(df)

# Analyze
density_analysis = analyze_density_behavior(density_motifs)
inverse_analysis = analyze_density_behavior(inverse_motifs)

# Merge and segment
all_motifs = merge_motif_collections(mv_motifs, density_motifs + inverse_motifs)
segmented_df = create_segmented_dataset(df, all_motifs, features, additional)

# Visualize
plot_density_analysis(density_analysis, 'density_analysis.png')
plot_density_analysis(inverse_analysis, 'inverse_analysis.png')

# Save
segmented_df.to_csv('segmented_motifs_all_08.csv')
```

### New System
```python
# data_preparation/run.py
from data_preparation import DataPreparationPipeline, PipelineConfig
from data_preparation.config.pattern_configs import (
    create_mv_pattern,
    create_density_pattern,
    create_inverse_pattern
)

# Configure
patterns = [
    create_mv_pattern(window_size=60, max_motifs=20),
    create_density_pattern(window_size=60, max_motifs=15),
    create_inverse_pattern(window_size=60, max_motifs=10)
]

config = PipelineConfig.create_default(
    mill_number=8,
    start_date="2025-01-01",
    end_date="2025-11-03",
    patterns=patterns
)

# Run (does everything above automatically)
pipeline = DataPreparationPipeline(config)
pipeline.run()
```

**Result:** 80% less code, same output! 🎉

## ✅ Verification Checklist

After migration, verify:

- [ ] All expected CSV files are generated
- [ ] CSV files have same structure as old system
- [ ] Analysis files contain expected columns
- [ ] Plots are generated for each pattern
- [ ] Model training scripts work with new data
- [ ] Database save works (if enabled)
- [ ] Log files show no errors

## 🆘 Need Help?

If you encounter issues:

1. Check `data_preparation.log` for errors
2. Compare output files with old system
3. Review `example_usage.py` for patterns
4. Consult `README.md` for detailed documentation

## 🎓 Learning Path

1. Start with `run.py` - understand basic usage
2. Explore `example_usage.py` - see different configurations
3. Read `README.md` - learn all features
4. Modify `pattern_configs.py` - create custom patterns
5. Extend `constraint_pattern.py` - add new pattern types (advanced)
