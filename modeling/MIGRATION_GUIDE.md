# Migration Guide

Guide for transitioning from the original scripts to the new modular solution.

## Overview of Changes

### Old Structure
```
segmentation/
├── motif_mv_search.py           # Monolithic script
├── analyze_density_motifs.py    # Separate script
└── xgboost_cascade.py           # Separate script
```

### New Structure
```
modeling/
├── prepare_data.py              # Consolidated data preparation
├── train_models.py              # Consolidated model training
├── config.py                    # Centralized configuration
├── database.py                  # Database utilities
├── motif_discovery.py           # Core motif logic
├── density_analysis.py          # Density analysis
├── segmentation.py              # Data segmentation
└── visualization.py             # Plotting utilities
```

## Key Improvements

### 1. Configuration Management

**Old Way:**
```python
# Hardcoded in script
WINDOW_SIZE = 60
MAX_MOTIFS = 20
RADIUS = 4.5
```

**New Way:**
```python
from config import PipelineConfig

config = PipelineConfig.create_default(mill_number=6, start_date="2024-01-01", end_date="2024-12-31")
config.motif.mv_window_size = 60
config.motif.mv_max_motifs = 20
config.motif.mv_radius = 4.5
```

### 2. Data Loading

**Old Way:**
```python
# In motif_mv_search.py
def load_data(mill_numbers, start_date, end_date, db_config, ...):
    connector = MillsDataConnector(**db_config)
    # ... lots of code
```

**New Way:**
```python
from database import DataLoader

loader = DataLoader(use_database=True)
df = loader.load_mill_data(mill_number=6, start_date="2024-01-01", end_date="2024-12-31")
```

### 3. Motif Discovery

**Old Way:**
```python
# In motif_mv_search.py
motif_info_list, segment_tuples, mps = discover_multivariate_motifs(
    df, feature_columns, window_size=60, max_motifs=20, radius=4.5
)
```

**New Way:**
```python
from motif_discovery import MotifDiscovery

discovery = MotifDiscovery(window_size=60, max_motifs=20, radius=4.5)
motifs, segment_tuples = discovery.discover(df, feature_columns)
```

### 4. Density Analysis

**Old Way:**
```python
# In analyze_density_motifs.py
motif_info_list = find_constrained_motifs(df, window_size=60, max_motifs=15, radius=3.5)
analysis_results = analyze_density_behavior(motif_info_list)
```

**New Way:**
```python
from density_analysis import DensityMotifDiscovery, analyze_density_behavior

discovery = DensityMotifDiscovery(window_size=60, max_motifs=15, radius=4.5)
density_motifs = discovery.discover(df)
analysis_results = analyze_density_behavior(density_motifs)
```

### 5. Merging Motifs

**Old Way:**
```python
# Manual merging with index conflicts
all_motifs = mv_motifs + density_motifs  # IDs might conflict
```

**New Way:**
```python
from segmentation import merge_motif_collections

all_motifs = merge_motif_collections(
    primary_motifs=mv_motifs,
    secondary_motifs=density_motifs,
    shuffle_indices=True  # Automatically reassigns IDs
)
```

### 6. Model Training

**Old Way:**
```python
# In xgboost_cascade.py
class CascadeModelTrainer:
    def __init__(self, mill_number, data_path, output_dir):
        # Hardcoded paths and config
```

**New Way:**
```python
from config import PipelineConfig
from train_models import CascadeModelTrainer

config = PipelineConfig.create_default(mill_number=6, start_date="2024-01-01", end_date="2024-12-31")
trainer = CascadeModelTrainer(config)
trainer.run()
```

## Feature Mapping

### motif_mv_search.py → New Solution

| Old Function | New Location | Notes |
|-------------|--------------|-------|
| `load_data()` | `database.DataLoader.load_mill_data()` | Cleaner interface |
| `filter_data()` | `database.filter_data()` | Unchanged |
| `discover_multivariate_motifs()` | `motif_discovery.MotifDiscovery.discover()` | OOP approach |
| `filter_motifs_by_correlation()` | `motif_discovery.CorrelationFilter.filter()` | Separate class |
| `plot_motif_instances()` | `visualization.plot_motif_instances()` | Enhanced |
| `create_segmented_dataset()` | `segmentation.create_segmented_dataset()` | Unchanged |

### analyze_density_motifs.py → New Solution

| Old Function | New Location | Notes |
|-------------|--------------|-------|
| `calculate_variability()` | `density_analysis.calculate_variability()` | Unchanged |
| `find_constrained_motifs()` | `density_analysis.DensityMotifDiscovery.discover()` | OOP approach |
| `analyze_density_behavior()` | `density_analysis.analyze_density_behavior()` | Unchanged |
| `find_optimal_lag()` | `density_analysis.find_optimal_lag()` | Unchanged |

### xgboost_cascade.py → New Solution

| Old Class/Method | New Location | Notes |
|-----------------|--------------|-------|
| `CascadeModelTrainer` | `train_models.CascadeModelTrainer` | Enhanced with config |
| `load_data()` | `train_models.CascadeModelTrainer.load_data()` | Simplified |
| `train_process_model()` | `train_models.CascadeModelTrainer._train_single_model()` | Generalized |
| `train_quality_model()` | `train_models.CascadeModelTrainer.train_quality_model()` | Unchanged |
| `validate_cascade()` | `train_models.CascadeModelTrainer.validate_cascade()` | Unchanged |

## Migration Steps

### Step 1: Update Imports

**Old:**
```python
from motif_mv_search import discover_multivariate_motifs, load_data
from analyze_density_motifs import find_constrained_motifs
from xgboost_cascade import CascadeModelTrainer
```

**New:**
```python
from config import PipelineConfig
from prepare_data import DataPreparationPipeline
from train_models import CascadeModelTrainer
```

### Step 2: Replace Configuration

**Old:**
```python
# Scattered throughout scripts
WINDOW_SIZE = 60
MAX_MOTIFS = 20
mill_number = 6
start_date = "2024-01-01"
```

**New:**
```python
config = PipelineConfig.create_default(
    mill_number=6,
    start_date="2024-01-01",
    end_date="2024-12-31"
)
config.motif.mv_window_size = 60
config.motif.mv_max_motifs = 20
```

### Step 3: Replace Execution

**Old:**
```python
# Run motif_mv_search.py
# Then run analyze_density_motifs.py
# Then run xgboost_cascade.py
```

**New:**
```python
# Run prepare_data.py
pipeline = DataPreparationPipeline(config)
pipeline.run()

# Run train_models.py
trainer = CascadeModelTrainer(config)
trainer.run()
```

## Backward Compatibility

### Legacy Format Conversion

If you need the old dictionary format:

```python
from motif_discovery import convert_motifs_to_legacy_format

# New format (Motif objects)
motifs = discovery.discover(df, features)

# Convert to old format
legacy_motifs = convert_motifs_to_legacy_format(motifs)
# Returns: [{'motif_id': 1, 'instances': [...], 'distance': 2.5}, ...]
```

### Using Old Data Files

The new solution can read old CSV files:

```python
config.use_database = False
config.use_cached_data = True
# Will use output/initial_data.csv if it exists
```

## Benefits of Migration

1. **Cleaner Code**: Modular design with clear separation of concerns
2. **Easier Configuration**: Single source of truth for all parameters
3. **Better Organization**: Helper classes for motifs and instances
4. **Enhanced Analysis**: More comprehensive output files
5. **Improved Visualization**: Better plots and reports
6. **Easier Maintenance**: Changes in one place affect all uses
7. **Better Testing**: Modular components are easier to test
8. **Extensibility**: Easy to add new features or analysis types

## Common Issues

### Issue: "Cannot import from old scripts"

**Solution:** Use the new imports from the modeling package.

### Issue: "Old configuration variables not working"

**Solution:** Update to use `PipelineConfig` object.

### Issue: "Different output file names"

**Solution:** Check the new output structure in `README.md`.

### Issue: "Motif IDs are different"

**Solution:** This is expected due to index shuffling. Use `shuffle_indices=False` if you need consistent IDs.

## Getting Help

- Check `README.md` for full documentation
- See `QUICK_START.md` for basic usage
- Run `example_usage.py` for working examples
- Review logs in `prepare_data.log` and `train_models.log`

## Rollback Plan

If you need to revert to old scripts:

1. Old scripts are still in `segmentation/` folder
2. They remain unchanged and functional
3. You can run them independently
4. Database connection still works the same way

The new solution is designed to coexist with the old scripts during the transition period.
