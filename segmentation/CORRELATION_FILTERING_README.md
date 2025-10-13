# Cross-Correlation Filtering for Motif Discovery

## Overview

This document explains the elegant solution for filtering discovered motifs based on cross-correlation constraints between feature pairs.

## The Problem

You want to ensure that discovered motifs exhibit specific correlation patterns between features:
- **WaterZumpf vs DensityHC**: Negative correlation
- **WaterZumpf vs PulpHC**: Positive correlation  
- **WaterZumpf vs PressureHC**: Positive correlation
- **Ore vs DensityHC**: Positive correlation (if Ore is included)

## The Solution

### Function: `filter_motifs_by_correlation()`

Located in `motif_mv_search.py` (lines 149-263), this function:

1. **Computes Pearson correlations** for each feature pair within each motif instance
2. **Validates against rules** - checks if correlations match expected signs (pos/neg)
3. **Filters intelligently** - supports two filtering levels:
   - `'instance'`: Keeps motifs with at least one valid instance (removes invalid instances)
   - `'motif'`: Only keeps motifs where ALL instances are valid
4. **Returns statistics** - provides correlation metrics for analysis

### Key Parameters

```python
CORRELATION_RULES = {
    ('WaterZumpf', 'DensityHC'): 'neg',
    ('WaterZumpf', 'PulpHC'): 'pos',
    ('WaterZumpf', 'PressureHC'): 'pos',
}
MIN_CORRELATION_STRENGTH = 0.3  # Minimum |correlation| to enforce
FILTER_LEVEL = 'instance'  # or 'motif'
```

### How It Works

1. **For each motif instance**:
   - Extract time series data for each feature pair
   - Compute Pearson correlation coefficient
   - Check if |correlation| >= `MIN_CORRELATION_STRENGTH`
   - If strong enough, verify sign matches the rule (pos/neg)

2. **Filtering logic**:
   - If correlation is too weak (< threshold), the rule is **skipped** (not enforced)
   - If correlation is strong but wrong sign, instance is **rejected**
   - Only instances passing all rules are kept

3. **Output**:
   - Filtered motif list with only valid instances
   - Updated segment tuples for downstream processing
   - Correlation statistics showing average correlations per motif

## Usage

### Enable/Disable Filtering

```python
APPLY_CORRELATION_FILTER = True  # Set to False to disable
```

### Configure Rules

```python
CORRELATION_RULES = {
    ('Feature1', 'Feature2'): 'pos',  # Expect positive correlation
    ('Feature3', 'Feature4'): 'neg',  # Expect negative correlation
}
```

### Adjust Sensitivity

```python
MIN_CORRELATION_STRENGTH = 0.3  # Lower = more lenient, Higher = stricter
```

### Choose Filter Level

```python
FILTER_LEVEL = 'instance'  # Remove bad instances, keep motif if any valid
FILTER_LEVEL = 'motif'     # Remove entire motif if any instance is invalid
```

## Example Output

```
============================================================
CORRELATION FILTERING
============================================================

  Filtering motifs by correlation constraints...
  Filter level: instance
  Min correlation strength: 0.3
  Rules:
    WaterZumpf vs DensityHC: neg
    WaterZumpf vs PulpHC: pos
    WaterZumpf vs PressureHC: pos
  Filtered: 20 motifs -> 15 motifs
  Total instances: 450 -> 320

============================================================
CORRELATION STATISTICS
============================================================

Motif 1:
  Valid instances: 25/30
  Average correlations:
    WaterZumpf vs DensityHC: -0.654
    WaterZumpf vs PulpHC: +0.721
    WaterZumpf vs PressureHC: +0.543
```

## Advantages of This Approach

1. **Elegant & Modular**: Single function, easy to enable/disable
2. **Flexible**: Configurable rules, thresholds, and filter levels
3. **Informative**: Provides detailed statistics about correlations
4. **Non-destructive**: Original discovery runs first, filtering is post-processing
5. **Efficient**: Uses scipy's optimized Pearson correlation
6. **Robust**: Handles weak correlations gracefully (skips enforcement)

## Integration

The filtering is seamlessly integrated into the main workflow:

```
Motif Discovery → Correlation Filtering → Plotting → Segmentation
```

All downstream functions (plotting, segmentation) automatically use the filtered results.

## Notes

- **Weak correlations**: If |correlation| < `MIN_CORRELATION_STRENGTH`, the rule is not enforced (too noisy)
- **Missing features**: If a feature pair is not in the data, that rule is skipped
- **Motif IDs preserved**: Original motif IDs are maintained for traceability
- **Statistics tracking**: All correlation values are computed and reported for analysis
