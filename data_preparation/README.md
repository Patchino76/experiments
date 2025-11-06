# Data Preparation Pipeline v2.0

A modular, extensible system for discovering motifs and preparing segmented datasets for ball mill modeling.

## 🎯 Key Features

- **Universal Pattern System**: Single constraint pattern class handles all variability constraints
- **Pattern Registry**: Patterns self-register and can be toggled via simple configuration
- **Easy Extension**: Add new patterns by defining constraints in config (no code changes needed)
- **Generic Analysis**: Analysis and visualization work for any pattern type
- **Clean Architecture**: ~70% reduction in code duplication vs. old system
- **Flexible Configuration**: Multiple configuration levels from simple to fully custom

## 📁 Structure

```
data_preparation/
├── core/                      # Core infrastructure
│   ├── base_pattern.py        # Base classes for patterns
│   ├── pattern_registry.py    # Pattern registration system
│   ├── data_loader.py         # Data loading and preprocessing
│   └── segmentation.py        # Motif to dataset conversion
├── patterns/                  # Pattern implementations
│   ├── mv_pattern.py          # Standard MV motif discovery
│   └── constraint_pattern.py  # Universal constraint pattern
├── config/                    # Configuration system
│   ├── defaults.py            # Default configurations
│   └── pattern_configs.py     # Pattern factory functions
├── analysis/                  # Analysis and visualization
│   ├── analyzer.py            # Generic analysis functions
│   └── visualizer.py          # Generic visualization
├── pipeline.py                # Main pipeline orchestrator
├── run.py                     # Entry point script
├── example_usage.py           # Usage examples
└── README.md                  # This file
```

## 🚀 Quick Start

### Basic Usage

```python
from data_preparation import DataPreparationPipeline, PipelineConfig

# Create configuration
config = PipelineConfig.create_default(
    mill_number=8,
    start_date="2025-01-01",
    end_date="2025-11-03"
)

# Run pipeline
pipeline = DataPreparationPipeline(config)
pipeline.run()
```

### Run from Command Line

```bash
cd data_preparation
python run.py
```

## 📋 Pattern Types

### 1. MV Pattern (Standard)
Discovers repeating patterns in manipulated variables without constraints.

```python
from config.pattern_configs import create_mv_pattern

pattern = create_mv_pattern(
    enabled=True,
    window_size=60,
    max_motifs=20,
    radius=4.5
)
```

### 2. Density Pattern
Stable WaterZumpf, varying Ore/WaterMill - captures steady sump water with varying feed.

```python
from config.pattern_configs import create_density_pattern

pattern = create_density_pattern(
    enabled=True,
    window_size=60,
    max_motifs=15
)
```

### 3. Inverse Pattern
Stable Ore/WaterMill, varying WaterZumpf - captures steady feed with sump water adjustments.

```python
from config.pattern_configs import create_inverse_pattern

pattern = create_inverse_pattern(enabled=True)
```

### 4. Dynamic Pattern
All MVs varying - captures transient/coordinated operations.

```python
from config.pattern_configs import create_dynamic_pattern

pattern = create_dynamic_pattern(enabled=True)
```

### 5. Pressure Pattern (Optional)
Stable PressureHC, varying MVs - captures optimal control regions.

```python
from config.pattern_configs import create_pressure_pattern

pattern = create_pressure_pattern(enabled=False)  # Disabled by default
```

## 🎨 Creating Custom Patterns

### Simple Custom Pattern

```python
from config.pattern_configs import create_custom_pattern

# Define a pattern where Ore is stable but water flows vary
custom_pattern = create_custom_pattern(
    name='stable_ore',
    constraints={
        'Ore': {
            'type': 'stable',
            'max_cv': 0.005  # Maximum coefficient of variation
        },
        'WaterMill': {
            'type': 'varying',
            'min_cv': 0.001  # Minimum coefficient of variation
        },
        'WaterZumpf': {
            'type': 'varying',
            'min_cv': 0.001
        }
    },
    window_size=60,
    max_motifs=10
)
```

### Advanced Custom Pattern

```python
from config.defaults import PatternConfig

pattern = PatternConfig(
    name='my_pattern',
    type='constraint',
    enabled=True,
    window_size=90,
    max_motifs=15,
    radius=5.0,
    max_instances_per_motif=25,
    constraints={
        'Ore': {'type': 'stable', 'max_cv': 0.01},
        'WaterMill': {'type': 'varying', 'min_cv': 0.002},
        'WaterZumpf': {'type': 'varying', 'min_cv': 0.001}
    },
    save_analysis=True,
    save_plots=True
)
```

## ⚙️ Configuration Options

### Data Configuration

```python
from config.defaults import DataConfig

data_config = DataConfig(
    mill_number=8,
    start_date="2025-01-01",
    end_date="2025-11-03",
    resample_freq='1min',
    mv_features=['Ore', 'WaterMill', 'WaterZumpf'],
    cv_features=['DensityHC', 'PulpHC', 'PressureHC', 'CirculativeLoad'],
    dv_features=['Class_15', 'Daiki', 'FE'],
    target='PSI200',
    filter_thresholds={
        'Ore': (100, 220),
        'PulpHC': (400, 600),
        'DensityHC': (1600, 1800),
    }
)
```

### Pipeline Configuration

```python
from config.defaults import PipelineConfig

config = PipelineConfig(
    data=data_config,
    patterns=patterns,
    use_database=True,           # Load from database
    save_mv_only=True,            # Save MV motifs separately
    save_combined=True,           # Save all motifs combined
    save_to_database=False        # Save results to database
)
```

## 📊 Output Files

### Data Files
- `initial_data.csv` - Filtered and preprocessed data with circulative load
- `segmented_motifsMV_{mill}.csv` - MV motifs only (for training)
- `segmented_motifs_all_{mill}.csv` - All motifs combined (for analysis)

### Analysis Files
- `{pattern}_analysis.csv` - Density behavior analysis per pattern
- `motif_summary.csv` - Summary statistics for all motifs
- `instance_catalog.csv` - Catalog of all motif instances
- `segment_statistics.csv` - Statistics per motif segment
- `summary_report.txt` - Text summary report

### Visualization Files
- `motif_overview.png` - Overview of all motifs
- `{pattern}_analysis.png` - Density analysis per pattern
- `motifs/{pattern}/motif_{id}_{pattern}.png` - Individual motif plots
- `correlation_heatmap.png` - Feature correlation heatmap
- `feature_distributions.png` - Feature distribution plots

## 🔧 Advanced Usage

### Toggle Patterns at Runtime

```python
config = PipelineConfig.create_default(mill_number=8, ...)

# Disable specific patterns
for pattern in config.patterns:
    if pattern.name in ['inverse', 'pressure']:
        pattern.enabled = False

# Modify parameters
for pattern in config.patterns:
    if pattern.name == 'mv':
        pattern.max_motifs = 30
```

### Custom Analysis

```python
from analysis.analyzer import PatternAnalyzer
from analysis.visualizer import PatternVisualizer

analyzer = PatternAnalyzer()
visualizer = PatternVisualizer()

# Analyze motifs
analyses = analyzer.analyze_density_behavior(motifs)

# Create visualizations
visualizer.plot_density_analysis(analyses, output_path, 'my_pattern')
```

## 🆚 Comparison with Old System

| Feature | Old System | New System |
|---------|-----------|------------|
| Pattern Classes | 4 separate classes (~800 lines) | 1 universal class (~250 lines) |
| Adding Patterns | Create new class + update config + modify pipeline | Define constraints in config |
| Code Duplication | ~90% duplicated code | Minimal duplication |
| Configuration | Scattered across multiple files | Centralized, hierarchical |
| Extensibility | Low (requires code changes) | High (config-driven) |
| Analysis | Pattern-specific | Generic, works for all patterns |
| Visualization | Pattern-specific | Generic, works for all patterns |

## 📝 Examples

See `example_usage.py` for complete examples:

1. **Default Configuration** - Quick start with defaults
2. **Custom Pattern Selection** - Choose which patterns to run
3. **Custom Constraint Pattern** - Define your own constraints
4. **Fully Manual Configuration** - Complete control
5. **Toggle Patterns at Runtime** - Dynamic configuration
6. **Analysis-Focused** - Focus on specific patterns

## 🔍 How It Works

### Pattern Discovery Process

1. **Data Loading**: Load and filter mill data from database or cache
2. **Feature Engineering**: Calculate circulative load and other derived features
3. **Pattern Discovery**: For each enabled pattern:
   - Compute multivariate matrix profile using STUMPY
   - Find seed points that satisfy constraints
   - Discover similar instances
   - Create motif objects with metadata
4. **Analysis**: Analyze density behavior, correlations, and lags
5. **Segmentation**: Convert motifs to segmented datasets
6. **Visualization**: Generate plots and reports
7. **Output**: Save CSV files, plots, and analysis results

### Constraint System

Constraints are defined per feature with two types:

- **Stable**: Feature should have low variability (CV ≤ max_cv)
- **Varying**: Feature should have high variability (CV ≥ min_cv)

The system automatically:
- Validates constraints for each window
- Ensures relative variability (varying > stable * factor)
- Filters instances that don't meet constraints

## 🐛 Troubleshooting

### No motifs discovered
- Check if data meets filter thresholds
- Adjust radius parameter (try larger values)
- Relax constraint thresholds (max_cv, min_cv)
- Increase window_size if data is too noisy

### Too many/few instances
- Adjust `max_motifs` parameter
- Adjust `max_instances_per_motif` parameter
- Modify `radius` threshold

### Missing columns error
- Ensure all required features exist in data
- Check `filter_thresholds` for excluded columns
- Verify database query returns expected columns

## 📚 Dependencies

- numpy
- pandas
- stumpy (matrix profile)
- scipy (correlation analysis)
- matplotlib (visualization)
- seaborn (visualization)

## 🤝 Contributing

To add a new pattern type:

1. Register pattern class with `@PatternRegistry.register('name')`
2. Inherit from `BasePattern`
3. Implement `discover()` method
4. Add factory function in `pattern_configs.py`

## 📄 License

Internal use only - Ball Mill Modeling Project

## 👥 Authors

Data Science Team - Ball Mill Optimization Project
