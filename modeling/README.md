# Mill Modeling Pipeline

A comprehensive, modular solution for mill process optimization using motif discovery and cascade XGBoost models.

## Overview

This pipeline consolidates functionality from:
- `motif_mv_search.py` - Multivariate motif discovery
- `analyze_density_motifs.py` - Density-constrained motif analysis
- `xgboost_cascade.py` - Cascade model training
- Database scripts - Data loading utilities

## Architecture

The solution is split into two main scripts with supporting modules:

### Main Scripts

1. **`prepare_data.py`** - Data preparation pipeline
   - Loads data from PostgreSQL database
   - Discovers motifs in MV features
   - Discovers density-constrained motifs
   - Analyzes density behavior
   - Creates segmented datasets
   - Generates comprehensive analysis outputs
   - Creates visualizations

2. **`train_models.py`** - Model training pipeline
   - Loads prepared segmented data
   - Trains process models (MV → CV)
   - Trains quality model (CV → DV)
   - Validates cascade performance
   - Saves models with metadata

### Supporting Modules

- **`config.py`** - Centralized configuration management
- **`database.py`** - Database connection and data loading
- **`motif_discovery.py`** - Core motif discovery with helper classes
- **`density_analysis.py`** - Density motif discovery and analysis
- **`segmentation.py`** - Dataset segmentation and feature engineering
- **`visualization.py`** - Plotting and visualization utilities

## Features

### Configuration
- **Flexible feature selection**: Easily configure MV, CV, DV features
- **Centralized parameters**: All settings in one place
- **Mill-specific outputs**: Separate folders per mill

### Data Processing
- **Database integration**: Direct PostgreSQL connection
- **Data caching**: Optional CSV caching for faster iteration
- **Data validation**: Automatic column and quality checks
- **Filtering**: Configurable threshold-based filtering

### Motif Discovery
- **MV motif discovery**: Find patterns in manipulated variables
- **Density motif discovery**: Find patterns with variability constraints
- **Correlation filtering**: Filter motifs by cross-correlation rules
- **Motif merging**: Combine multiple motif collections with index shuffling

### Analysis Outputs
- Segmented datasets (CSV)
- Motif summaries (CSV)
- Instance catalogs (CSV)
- Segment statistics (CSV)
- Density analysis results (CSV)
- Correlation statistics (JSON)
- Text summary reports

### Visualizations
- Individual motif plots
- Motif overview plots
- Density analysis plots
- Correlation heatmaps
- Feature distribution plots

### Model Training
- **Cascade architecture**: MV → CV → DV
- **Hyperparameter tuning**: GridSearchCV with time-series cross-validation
- **Model metadata**: Compatible with API endpoints
- **Performance tracking**: Comprehensive metrics and validation

## Usage

### 1. Configure Pipeline

Edit the configuration in `prepare_data.py` or `train_models.py`:

```python
# Basic configuration
mill_number = 6
start_date = "2024-01-01"
end_date = "2024-12-31"

# Create configuration
config = PipelineConfig.create_default(mill_number, start_date, end_date)

# Customize if needed
config.motif.mv_max_motifs = 25
config.model.test_size = 0.25
```

### 2. Prepare Data

```bash
python prepare_data.py
```

This will:
- Load data from database
- Discover motifs
- Perform analysis
- Save outputs to `output/`, `output/analysis/`, and `output/plots/`

### 3. Train Models

```bash
python train_models.py
```

This will:
- Load segmented data
- Train cascade models
- Validate performance
- Save models to `models/mill_X/`

## Configuration Options

### Data Configuration

```python
DataConfig(
    mill_number=6,
    start_date="2024-01-01",
    end_date="2024-12-31",
    mv_features=['Ore', 'WaterMill', 'WaterZumpf', 'MotorAmp'],
    cv_features=['DensityHC', 'PulpHC', 'PressureHC'],
    dv_features=[],  # Optional: ['Class_15', 'Shisti', 'Daiki', 'FE']
    target='PSI200',
    filter_thresholds={
        'Ore': (100, 220),
        'PulpHC': (400, 600),
        'DensityHC': (1600, 1800),
    }
)
```

### Motif Configuration

```python
MotifConfig(
    mv_window_size=60,  # minutes
    mv_max_motifs=20,
    mv_radius=4.5,
    apply_correlation_filter=True,
    correlation_rules={
        ('PressureHC', 'PulpHC'): 'pos',
        ('WaterZumpf', 'PressureHC'): 'pos',
    },
    min_correlation_strength=0.1
)
```

### Model Configuration

```python
ModelConfig(
    test_size=0.2,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    cv_splits=5
)
```

## Output Structure

```
modeling/
├── output/
│   ├── initial_data.csv              # Raw data from database
│   ├── segmented_motifsMV.csv        # Segmented data for modeling
│   ├── segmented_motifs_all.csv      # Complete segmented data
│   ├── analysis/
│   │   ├── motif_summary.csv
│   │   ├── instance_catalog.csv
│   │   ├── segment_statistics.csv
│   │   ├── density_analysis.csv
│   │   ├── correlation_stats.json
│   │   └── summary_report.txt
│   └── plots/
│       └── mill_6/
│           ├── motifs/
│           │   ├── motif_01.png
│           │   ├── motif_02.png
│           │   └── ...
│           ├── motif_overview.png
│           ├── density_analysis.png
│           ├── correlation_heatmap.png
│           └── feature_distributions.png
├── models/
│   └── mill_6/
│       ├── process_model_PulpHC.pkl
│       ├── process_model_DensityHC.pkl
│       ├── process_model_PressureHC.pkl
│       ├── quality_model.pkl
│       ├── scaler_*.pkl
│       ├── metadata.json
│       └── training_results.json
└── logs/
    ├── prepare_data.log
    └── train_models.log
```

## Key Improvements

1. **Modular Design**: Clean separation of concerns with reusable components
2. **Configuration Management**: Single source of truth for all parameters
3. **Helper Classes**: `Motif` and `MotifInstance` classes for better organization
4. **Motif Merging**: Easy integration of density motifs with MV motifs via index shuffling
5. **Comprehensive Analysis**: More detailed analysis files and visualizations
6. **Error Handling**: Robust validation and logging
7. **Metadata Preservation**: Model metadata compatible with API endpoints
8. **Extensibility**: Easy to add new features or analysis types

## Advanced Usage

### Using Cached Data

To skip database loading and use cached data:

```python
config = PipelineConfig.create_default(mill_number, start_date, end_date)
config.use_database = False
config.use_cached_data = True
```

### Merging Multiple Motif Collections

```python
from segmentation import merge_motif_collections

# Merge with index shuffling
all_motifs = merge_motif_collections(
    primary_motifs=mv_motifs,
    secondary_motifs=density_motifs,
    shuffle_indices=True  # Reassign all IDs sequentially
)
```

### Custom Correlation Rules

```python
config.motif.correlation_rules = {
    ('PressureHC', 'PulpHC'): 'pos',      # Positive correlation
    ('WaterZumpf', 'DensityHC'): 'neg',   # Negative correlation
    ('Ore', 'DensityHC'): 'pos',
}
config.motif.min_correlation_strength = 0.15
```

## Requirements

- Python 3.8+
- pandas
- numpy
- stumpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- sqlalchemy
- psycopg2

## Notes

- Models are saved with metadata compatible with existing API endpoints
- The pipeline preserves the exact model structure from `xgboost_cascade.py`
- All analysis outputs are saved as CSV for easy inspection
- Visualizations are saved as high-resolution PNG files
- Logging is comprehensive for debugging and monitoring

## License

Internal use only.
