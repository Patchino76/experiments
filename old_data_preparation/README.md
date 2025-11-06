# Data Preparation Pipeline

A modular and extensible pipeline for discovering patterns in time series data, specifically designed for ball mill operational data analysis.

## Overview

This pipeline implements several pattern discovery algorithms to identify different operational regimes in ball mill data. The system is designed to be modular, allowing for easy addition of new pattern types.

## Available Patterns

1. **Density Constraint Patterns**
   - Identifies patterns where WaterZumpf is stable while Ore and WaterMill vary
   - Useful for understanding steady-state operations

2. **Inverse Constraint Patterns**
   - Identifies patterns where Ore and WaterMill are stable while WaterZumpf varies
   - Useful for understanding feed control adjustments

3. **Dynamic Patterns**
   - Identifies patterns where all variables (Ore, WaterMill, WaterZumpf) are changing
   - Captures dynamic operational states and transitions

4. **Pressure Constraint Patterns**
   - Identifies patterns where PressureHC is stable while other variables vary
   - Useful for understanding optimal control regions

## Project Structure

```
data_preparation/
├── config/
│   └── defaults.py      # Default configuration parameters
├── core/
│   ├── __init__.py
│   ├── base_pattern.py  # Abstract base class for patterns
│   ├── pattern_registry.py  # Pattern registration system
│   └── pipeline.py      # Main pipeline implementation
├── patterns/
│   ├── __init__.py      # Pattern imports and registration
│   ├── density.py       # Density constraint pattern
│   ├── inverse.py       # Inverse constraint pattern
│   ├── dynamic.py       # Dynamic pattern
│   └── pressure.py      # Pressure constraint pattern
├── example_usage.py     # Example script
└── README.md            # This file
```

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```python
from data_preparation.core.pipeline import Pipeline
from data_preparation.config.defaults import get_default_config
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv', parse_dates=['timestamp'], index_col='timestamp')

# Get default configuration
config = get_default_config()

# Customize configuration if needed
config['patterns']['density']['enabled'] = True
config['patterns']['inverse']['enabled'] = True
config['patterns']['dynamic']['enabled'] = True
config['patterns']['pressure']['enabled'] = True

# Initialize and run the pipeline
pipeline = Pipeline(config, output_dir='output/pattern_discovery')
results = pipeline.run(data)
```

### Example Script

Run the example script to see the pipeline in action:

```bash
python -m data_preparation.example_usage
```

## Configuration

Configuration is handled through a nested dictionary structure. The default configuration can be obtained using `get_default_config()` from `data_preparation.config.defaults`.

### Common Parameters

- `window_size`: Size of the sliding window for pattern discovery
- `max_motifs`: Maximum number of motifs to discover per pattern type
- `radius`: Maximum distance for considering instances as part of the same motif
- `enabled`: Whether the pattern type is enabled

### Pattern-Specific Parameters

Each pattern type can have its own set of parameters. Refer to the individual pattern files for details.

## Output

The pipeline generates the following outputs for each pattern type:

- `motifs_<pattern>.json`: JSON file containing discovered motifs
- `analysis_<pattern>.csv`: Analysis results for the discovered patterns
- `plots/` directory with visualizations of the discovered patterns

## Adding New Patterns

To add a new pattern type:

1. Create a new Python file in the `patterns/` directory
2. Create a class that inherits from `BasePattern`
3. Implement the required methods (`discover`, `analyze`, `visualize`)
4. Decorate the class with `@PatternRegistry.register('pattern_name')`
5. Import the class in `patterns/__init__.py`

## Dependencies

- Python 3.8+
- pandas
- numpy
- stumpy
- matplotlib
- scipy
- scikit-learn

## License

[Your License Here]
