# Data Preparation Pipeline - Quick Reference

## 🚀 Quick Start

```bash
cd data_preparation
python run.py
```

## 📋 Common Tasks

### 1. Run with Default Settings
```python
from data_preparation import DataPreparationPipeline, PipelineConfig

config = PipelineConfig.create_default(8, "2025-01-01", "2025-11-03")
pipeline = DataPreparationPipeline(config)
pipeline.run()
```

### 2. Disable a Pattern
```python
config = PipelineConfig.create_default(8, "2025-01-01", "2025-11-03")

for pattern in config.patterns:
    if pattern.name == 'pressure':
        pattern.enabled = False

pipeline = DataPreparationPipeline(config)
pipeline.run()
```

### 3. Create Custom Pattern
```python
from config.pattern_configs import create_custom_pattern

custom = create_custom_pattern(
    name='my_pattern',
    constraints={
        'Ore': {'type': 'stable', 'max_cv': 0.01},
        'WaterMill': {'type': 'varying', 'min_cv': 0.001}
    },
    window_size=60,
    max_motifs=15
)
```

### 4. Modify Pattern Parameters
```python
from config.pattern_configs import create_density_pattern

pattern = create_density_pattern(
    enabled=True,
    window_size=90,      # Changed from 60
    max_motifs=20,       # Changed from 15
    radius=5.0           # Changed from 4.5
)
```

### 5. Use Only Specific Patterns
```python
from config.pattern_configs import create_mv_pattern, create_density_pattern

patterns = [
    create_mv_pattern(),
    create_density_pattern()
]

config = PipelineConfig.create_default(8, "2025-01-01", "2025-11-03", patterns)
```

## 🎨 Pattern Types

| Pattern | Description | Constraints |
|---------|-------------|-------------|
| **mv** | Standard MV motifs | None |
| **density** | Stable WaterZumpf | WaterZumpf stable, Ore/WaterMill varying |
| **inverse** | Stable Ore/WaterMill | Ore/WaterMill stable, WaterZumpf varying |
| **dynamic** | All varying | All MVs varying |
| **pressure** | Stable PressureHC | PressureHC stable, MVs varying |

## ⚙️ Configuration Quick Reference

### Pattern Config
```python
PatternConfig(
    name='my_pattern',           # Pattern name
    type='constraint',           # 'mv' or 'constraint'
    enabled=True,                # Enable/disable
    window_size=60,              # Window in minutes
    max_motifs=15,               # Max motifs to find
    radius=4.5,                  # Distance threshold
    max_instances_per_motif=20,  # Max instances per motif
    constraints={...},           # Constraint definition
    save_analysis=True,          # Save analysis CSV
    save_plots=True              # Save plots
)
```

### Constraint Definition
```python
constraints={
    'FeatureName': {
        'type': 'stable',        # 'stable' or 'varying'
        'max_cv': 0.01,          # For stable features
        'min_cv': 0.001          # For varying features
    }
}
```

### Pipeline Config
```python
PipelineConfig(
    data=data_config,            # Data configuration
    patterns=patterns,           # List of patterns
    use_database=True,           # Load from database
    save_mv_only=True,           # Save MV motifs separately
    save_combined=True,          # Save all motifs combined
    save_to_database=False       # Save to database
)
```

## 📊 Output Files

### Data Files
- `initial_data.csv` - Preprocessed data
- `segmented_motifsMV_{mill}.csv` - MV motifs only
- `segmented_motifs_all_{mill}.csv` - All motifs

### Analysis Files
- `{pattern}_analysis.csv` - Per-pattern analysis
- `motif_summary.csv` - Motif summaries
- `instance_catalog.csv` - Instance catalog
- `segment_statistics.csv` - Segment stats
- `summary_report.txt` - Text report

### Plots
- `motif_overview.png` - Overview
- `{pattern}_analysis.png` - Per-pattern analysis
- `motifs/{pattern}/motif_{id}.png` - Individual motifs
- `correlation_heatmap.png` - Correlations
- `feature_distributions.png` - Distributions

## 🔧 Troubleshooting

### No motifs found
```python
# Increase radius
pattern.radius = 5.0  # or higher

# Relax constraints
pattern.constraints['Ore']['max_cv'] = 0.02  # was 0.01
```

### Too many/few instances
```python
# Adjust max_motifs
pattern.max_motifs = 25  # increase

# Adjust max_instances_per_motif
pattern.max_instances_per_motif = 30  # increase
```

### Database not available
```python
# Use cached data
config.use_database = False
# Ensure initial_data.csv exists
```

## 📚 Documentation

- **README.md** - Complete user guide
- **MIGRATION_GUIDE.md** - Migrate from old system
- **IMPLEMENTATION_SUMMARY.md** - Technical details
- **example_usage.py** - 6 usage examples

## 🧪 Testing

```bash
python test_system.py
```

Expected: 7/7 tests passing ✅

## 💡 Tips

1. **Start simple** - Use default config first
2. **One pattern at a time** - Test patterns individually
3. **Check logs** - Review `data_preparation.log`
4. **Verify output** - Check CSV files before training
5. **Use examples** - See `example_usage.py`

## 🆘 Help

1. Check `data_preparation.log` for errors
2. Run `test_system.py` to verify setup
3. Review `README.md` for detailed docs
4. Check `example_usage.py` for patterns

## 📞 Common Commands

```bash
# Run pipeline
python run.py

# Run tests
python test_system.py

# Run examples
python example_usage.py

# View logs
cat data_preparation.log
```

## 🎯 Best Practices

1. **Always test first** - Run `test_system.py`
2. **Use version control** - Commit config changes
3. **Document custom patterns** - Add comments
4. **Verify outputs** - Check CSV files
5. **Monitor logs** - Watch for warnings

---

**Quick Links:**
- [Full Documentation](README.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Examples](example_usage.py)
- [Tests](test_system.py)
