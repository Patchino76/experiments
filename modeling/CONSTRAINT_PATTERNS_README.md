# Constraint Pattern Discovery

## Overview

The modeling pipeline now supports **four different constraint-based motif discovery patterns**, each capturing distinct operational regimes in ball mill operations. All patterns can be independently enabled/disabled via configuration.

## Pattern Types

### 1. **Density Constraint Pattern** (Original)
**Constraint:** Stable WaterZumpf, Varying Ore/WaterMill

**Operational Meaning:** Steady sump water supply with varying feed conditions

**Use Case:** Captures how density responds when feed rate and mill water change but sump water remains constant

**Configuration:**
```python
config.motif.enable_density_pattern = True  # Default: True
config.motif.density_max_motifs = 15
config.motif.density_window_size = 60
config.motif.density_radius = 4.5
```

**Constraints:**
- WaterZumpf CV ≤ 1%
- Ore CV ≥ 0.08%
- WaterMill CV ≥ 0.15%

---

### 2. **Inverse Constraint Pattern** (NEW)
**Constraint:** Stable Ore/WaterMill, Varying WaterZumpf

**Operational Meaning:** Steady feed conditions with sump water adjustments

**Use Case:** Captures density control operations where operators adjust sump water while maintaining stable feed

**Configuration:**
```python
config.motif.enable_inverse_pattern = True  # Default: True
config.motif.inverse_max_motifs = 10
config.motif.inverse_window_size = 60
config.motif.inverse_radius = 4.5
```

**Constraints:**
- Ore CV ≤ 1%
- WaterMill CV ≤ 1%
- WaterZumpf CV ≥ 0.08%

---

### 3. **Dynamic Pattern** (NEW)
**Constraint:** All MVs Varying Simultaneously

**Operational Meaning:** Transient/dynamic operations with coordinated adjustments

**Use Case:** Captures periods where operators adjust multiple variables together during process transitions

**Configuration:**
```python
config.motif.enable_dynamic_pattern = True  # Default: True
config.motif.dynamic_max_motifs = 10
config.motif.dynamic_window_size = 60
config.motif.dynamic_radius = 4.5
```

**Constraints:**
- Ore CV ≥ 0.08%
- WaterMill CV ≥ 0.15%
- WaterZumpf CV ≥ 0.08%

---

### 4. **Pressure Constraint Pattern** (NEW)
**Constraint:** Stable PressureHC, Varying MVs

**Operational Meaning:** Good process control with stable cyclone pressure

**Use Case:** Identifies optimal operating regions where pressure remains constant despite MV changes

**Configuration:**
```python
config.motif.enable_pressure_pattern = False  # Default: False (optional)
config.motif.pressure_max_motifs = 10
config.motif.pressure_window_size = 60
config.motif.pressure_radius = 4.5
```

**Constraints:**
- PressureHC CV ≤ 1%
- Ore CV ≥ 0.08%
- WaterMill CV ≥ 0.15%
- WaterZumpf CV ≥ 0.08%

---

## Usage

### Basic Configuration

```python
from config import PipelineConfig

# Create default configuration
config = PipelineConfig.create_default(
    mill_number=6,
    start_date="2025-06-26",
    end_date="2025-10-26"
)

# Enable/disable patterns as needed
config.motif.enable_density_pattern = True    # Original pattern
config.motif.enable_inverse_pattern = True    # Inverse pattern
config.motif.enable_dynamic_pattern = True    # Dynamic pattern
config.motif.enable_pressure_pattern = False  # Optional pattern

# Run pipeline
from prepare_data import DataPreparationPipeline
pipeline = DataPreparationPipeline(config)
pipeline.run()
```

### Selective Pattern Usage

**Example 1: Only MV motifs + Density pattern**
```python
config.motif.enable_density_pattern = True
config.motif.enable_inverse_pattern = False
config.motif.enable_dynamic_pattern = False
config.motif.enable_pressure_pattern = False
```

**Example 2: All patterns except pressure**
```python
config.motif.enable_density_pattern = True
config.motif.enable_inverse_pattern = True
config.motif.enable_dynamic_pattern = True
config.motif.enable_pressure_pattern = False  # Requires PressureHC column
```

**Example 3: Only dynamic operations**
```python
config.motif.enable_density_pattern = False
config.motif.enable_inverse_pattern = False
config.motif.enable_dynamic_pattern = True
config.motif.enable_pressure_pattern = False
```

---

## Output Files

### Analysis Files (in `modeling/output/analysis/`)

Each enabled pattern generates its own analysis file:

- `density_analysis.csv` - Density constraint pattern results
- `inverse_analysis.csv` - Inverse constraint pattern results
- `dynamic_analysis.csv` - Dynamic pattern results
- `pressure_analysis.csv` - Pressure constraint pattern results (if enabled)
- `summary_report.txt` - Combined summary of all patterns

### Visualization Files (in `modeling/output/plots/mill_XX/`)

- `density_analysis.png` - Density pattern visualization
- `inverse_analysis.png` - Inverse pattern visualization
- `dynamic_analysis.png` - Dynamic pattern visualization
- `pressure_analysis.png` - Pressure pattern visualization (if enabled)

### Data Files (in `modeling/output/`)

- `segmented_motifsMV_XX.csv` - MV motifs only (for training)
- `segmented_motifs_all_XX.csv` - All motifs combined (MV + enabled constraint patterns)

---

## Implementation Details

### File Structure

```
modeling/
├── constraint_patterns.py          # NEW: Additional pattern classes
├── density_analysis.py             # Original density pattern + analysis
├── prepare_data.py                 # Updated: Integrates all patterns
├── config.py                       # Updated: Toggle parameters
└── CONSTRAINT_PATTERNS_README.md   # This file
```

### Key Classes

**In `constraint_patterns.py`:**
- `InverseConstraintMotifDiscovery` - Pattern 2
- `DynamicMotifDiscovery` - Pattern 3
- `PressureConstraintMotifDiscovery` - Pattern 4

**In `density_analysis.py`:**
- `DensityMotifDiscovery` - Pattern 1 (original)
- `analyze_density_behavior()` - Analysis function (works for all patterns)

### Pattern Metadata

Each motif instance includes metadata identifying its pattern type:

```python
instance.metadata['pattern_type'] = 'density'           # Pattern 1
instance.metadata['pattern_type'] = 'inverse_constraint' # Pattern 2
instance.metadata['pattern_type'] = 'dynamic'           # Pattern 3
instance.metadata['pattern_type'] = 'pressure_constraint' # Pattern 4
```

---

## Expected Benefits

### Data Quality Improvements

| Metric | Expected Improvement |
|--------|---------------------|
| **Operational Coverage** | +30-50% more regimes captured |
| **Data Diversity** | Better representation of control strategies |
| **Model Robustness** | Improved generalization across conditions |
| **Edge Case Handling** | Captures transient behaviors |

### Model Performance

| Model Type | Expected R² Improvement |
|-----------|------------------------|
| **Process Models (MV→CV)** | +5-10% |
| **Quality Model (CV+DV→PSI200)** | +3-7% |
| **Extrapolation** | Better predictions outside training envelope |

---

## Best Practices

### 1. **Start Conservative**
Begin with density + inverse patterns only:
```python
config.motif.enable_density_pattern = True
config.motif.enable_inverse_pattern = True
config.motif.enable_dynamic_pattern = False
config.motif.enable_pressure_pattern = False
```

### 2. **Evaluate Impact**
After training models, check if R² improved by >3%. If yes, add dynamic pattern.

### 3. **Monitor Data Balance**
Ensure each pattern contributes at least 100 instances. Check analysis files:
```bash
# Check instance counts
grep "num_instances" modeling/output/analysis/*_analysis.csv
```

### 4. **Avoid Over-Fragmentation**
Don't enable all patterns if total instances < 1000. This fragments training data too much.

### 5. **Pressure Pattern is Optional**
Only enable if:
- PressureHC column exists
- You specifically want to study pressure control
- Other patterns are already working well

---

## Troubleshooting

### Issue: No motifs found for a pattern

**Possible Causes:**
- Data doesn't contain that operational regime
- Constraints too strict (CV thresholds)
- Window size too large

**Solution:**
```python
# Relax constraints slightly
config.motif.inverse_max_motifs = 5  # Reduce target
# Or adjust CV thresholds in the discovery class
```

### Issue: Too many overlapping motifs

**Solution:**
```python
# Increase radius to be more selective
config.motif.dynamic_radius = 5.5  # More strict matching
```

### Issue: Computation takes too long

**Solution:**
```python
# Reduce max motifs per pattern
config.motif.density_max_motifs = 10
config.motif.inverse_max_motifs = 8
config.motif.dynamic_max_motifs = 8
```

---

## Technical Notes

### Computational Complexity

Each pattern discovery runs independently:
- **Time:** O(n²) per pattern (STUMPY matrix profile)
- **Memory:** O(n) per pattern
- **Parallelizable:** Patterns can be discovered in parallel (future enhancement)

### Data Requirements

| Pattern | Required Columns |
|---------|-----------------|
| Density | Ore, WaterMill, WaterZumpf, DensityHC |
| Inverse | Ore, WaterMill, WaterZumpf, DensityHC |
| Dynamic | Ore, WaterMill, WaterZumpf, DensityHC |
| Pressure | Ore, WaterMill, WaterZumpf, DensityHC, **PressureHC** |

---

## Future Enhancements

1. **Parallel Discovery:** Run patterns in parallel for faster processing
2. **Auto-Tuning:** Automatically adjust CV thresholds based on data statistics
3. **Pattern Ranking:** Score patterns by operational importance
4. **Overlap Detection:** Identify and handle overlapping instances across patterns
5. **Custom Patterns:** User-defined constraint combinations

---

## References

- Original density analysis: `modeling/docs/README_density_analysis.md`
- STUMPY documentation: https://stumpy.readthedocs.io/
- Matrix profile: https://www.cs.ucr.edu/~eamonn/MatrixProfile.html
