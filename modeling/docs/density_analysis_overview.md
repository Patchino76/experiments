# Density Analysis Module - Overview

## Purpose

The `density_analysis.py` module discovers and analyzes specific operational patterns (motifs) in ball mill grinding circuits where water flow to the sump (`WaterZumpf`) remains stable while ore feed and mill water (`Ore` and `WaterMill`) vary. The goal is to understand how pulp density (`DensityHC`) responds to these controlled variations.

## Why This Matters

In ball mill operations, understanding the relationship between water additions and pulp density is critical for:

1. **Process Control**: Maintaining optimal density for grinding efficiency
2. **Product Quality**: Density affects particle size distribution
3. **Energy Efficiency**: Proper density reduces energy consumption
4. **Operational Stability**: Predictable density response enables better control

## Key Concept: Constrained Motif Discovery

Traditional motif discovery finds any recurring patterns. This module implements **constrained motif discovery** that only finds patterns matching specific operational criteria:

- **WaterZumpf (sump water)**: Must be STABLE (CV ≤ 1%)
- **Ore (feed rate)**: Must be VARYING (CV ≥ 0.08%)
- **WaterMill (mill water)**: Must be VARYING (CV ≥ 0.15%)

This represents controlled experiments where operators hold one variable constant while manipulating others.

## Module Components

### 1. Variability Measurement
- `calculate_variability()`: Computes Coefficient of Variation (CV)
- CV = σ / |μ| (standard deviation / mean)
- Scale-independent measure allowing comparison across variables

### 2. Motif Discovery
- `DensityMotifDiscovery` class: Finds constrained patterns
- Uses multivariate matrix profile (STUMPY library)
- Filters patterns based on variability constraints
- Returns groups of similar operational scenarios

### 3. Lag Analysis
- `find_optimal_lag()`: Finds time delays in process response
- Uses cross-correlation to detect delays
- Critical for understanding process dynamics

### 4. Behavior Analysis
- `analyze_density_behavior()`: Analyzes density response in motifs
- Computes correlations, changes, and lags
- Aggregates statistics across instances

## Workflow

```
Input Data (DataFrame)
    ↓
[DensityMotifDiscovery.discover()]
    ├─ Normalize time series
    ├─ Compute matrix profile
    ├─ Find seeds with constraints
    ├─ Find similar instances
    └─ Return motifs
    ↓
[analyze_density_behavior()]
    ├─ Calculate density changes
    ├─ Compute correlations
    ├─ Find optimal lags
    └─ Return analysis results
    ↓
Output: Insights on density behavior
```

## Expected Outputs

For each discovered motif, you get:

1. **Density Change**: Net change from start to end (g/L or %)
2. **Ore-Density Correlation**: Strength of relationship (-1 to +1)
3. **WaterMill-Density Correlation**: Strength of relationship (-1 to +1)
4. **Ore Lag**: Time delay for ore effect (minutes)
5. **WaterMill Lag**: Time delay for water effect (minutes)
6. **Number of Instances**: How many times this pattern occurred

## Typical Use Cases

1. **Model Validation**: Compare discovered patterns with process models
2. **Control Strategy Design**: Use lag information for feedforward control
3. **Operator Training**: Show typical responses to manipulations
4. **Anomaly Detection**: Identify when density doesn't respond as expected
5. **Optimization**: Find conditions that produce desired density changes

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 60 | Pattern length in minutes |
| `max_motifs` | 15 | Maximum motif groups to find |
| `radius` | 4.5 | Distance threshold for similarity |
| `waterzumpf_max_cv` | 0.01 | Max CV for stable WaterZumpf (1%) |
| `ore_min_cv` | 0.0008 | Min CV for varying Ore (0.08%) |
| `watermill_min_cv` | 0.0015 | Min CV for varying WaterMill (0.15%) |
| `relative_variability_factor` | 1.2 | Ore/WaterMill must be 1.2× more variable than WaterZumpf |

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data handling
- `stumpy`: Matrix profile computation
- `scipy.stats`: Statistical functions (Pearson correlation)
- `scipy.signal`: Signal processing (cross-correlation)
- `motif_discovery`: Base motif classes (Motif, MotifInstance)

## Next Steps

See companion documentation files:
- `density_analysis_code_details.md`: Line-by-line code explanation
- `density_analysis_improvements.md`: Suggested improvements and simplifications
