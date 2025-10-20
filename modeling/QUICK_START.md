# Quick Start Guide

## Installation

No additional installation needed if you already have the project dependencies.

## Basic Usage

### Step 1: Prepare Data

```bash
cd c:\Projects\experiments\modeling
python prepare_data.py
```

This will:
- ✓ Load data from PostgreSQL for Mill 6
- ✓ Discover MV motifs
- ✓ Discover density motifs
- ✓ Create segmented datasets
- ✓ Generate analysis files
- ✓ Create visualizations

**Outputs:**
- `output/segmented_motifsMV.csv` - Ready for training
- `output/analysis/*.csv` - Analysis results
- `output/plots/mill_6/*.png` - Visualizations

### Step 2: Train Models

```bash
python train_models.py
```

This will:
- ✓ Load segmented data
- ✓ Train 3 process models (MV → CV)
- ✓ Train 1 quality model (CV → DV)
- ✓ Validate cascade performance
- ✓ Save models with metadata

**Outputs:**
- `models/mill_6/*.pkl` - Trained models
- `models/mill_6/metadata.json` - Model metadata
- `models/mill_6/training_results.json` - Performance metrics

## Configuration

Edit the configuration in `prepare_data.py` or `train_models.py`:

```python
# Change mill number
mill_number = 7

# Change date range
start_date = "2024-06-01"
end_date = "2024-12-31"

# Customize motif discovery
config.motif.mv_max_motifs = 30
config.motif.mv_radius = 4.0

# Customize model training
config.model.test_size = 0.25
config.model.n_estimators = 400
```

## Common Tasks

### Use Cached Data (Skip Database)

```python
config = PipelineConfig.create_default(mill_number, start_date, end_date)
config.use_database = False
config.use_cached_data = True
```

### Change Features

```python
config.data.mv_features = ['Ore', 'WaterMill', 'WaterZumpf', 'MotorAmp']
config.data.cv_features = ['DensityHC', 'PulpHC', 'PressureHC']
config.data.target = 'PSI200'
```

### Adjust Correlation Rules

```python
config.motif.correlation_rules = {
    ('PressureHC', 'PulpHC'): 'pos',
    ('WaterZumpf', 'PressureHC'): 'pos',
    ('Ore', 'DensityHC'): 'pos',
}
config.motif.min_correlation_strength = 0.15
```

### Run Only Data Preparation

```python
from prepare_data import DataPreparationPipeline

pipeline = DataPreparationPipeline(config)
pipeline.run()
```

### Run Only Model Training

```python
from train_models import CascadeModelTrainer

trainer = CascadeModelTrainer(config)
trainer.run()
```

## File Structure

```
modeling/
├── prepare_data.py          # Main data preparation script
├── train_models.py          # Main model training script
├── config.py                # Configuration management
├── database.py              # Database utilities
├── motif_discovery.py       # Motif discovery engine
├── density_analysis.py      # Density motif analysis
├── segmentation.py          # Data segmentation
├── visualization.py         # Plotting utilities
├── example_usage.py         # Usage examples
├── README.md                # Full documentation
└── QUICK_START.md          # This file
```

## Troubleshooting

### "Segmented data not found"
Run `prepare_data.py` first before `train_models.py`.

### "Missing required columns"
Check that your database has all required features:
- MV: Ore, WaterMill, WaterZumpf, MotorAmp
- CV: DensityHC, PulpHC, PressureHC
- Target: PSI200

### "Database connection failed"
Check database settings in `db/settings.py`.

### "No motifs discovered"
Try:
- Increasing `mv_radius` (e.g., 5.0 instead of 4.5)
- Increasing `mv_max_motifs`
- Adjusting filter thresholds in `config.data.filter_thresholds`

## Next Steps

1. Review outputs in `output/analysis/summary_report.txt`
2. Check visualizations in `output/plots/mill_6/`
3. Inspect model performance in `models/mill_6/training_results.json`
4. Use trained models via API endpoints

## Support

For issues or questions, check the full README.md or contact the development team.
