# Density Analysis Module - Documentation Index

Complete documentation for the `density_analysis.py` module used in ball mill grinding circuit analysis.

## 📚 Documentation Files

### 1. [Overview](./density_analysis_overview.md)
**Start here** - High-level explanation of what the module does and why it matters.

**Contents**:
- Purpose and motivation
- Key concepts (constrained motif discovery)
- Module components overview
- Workflow diagram
- Expected outputs
- Use cases
- Parameters reference
- Dependencies

**Best for**: Understanding the big picture, explaining to stakeholders, getting started.

---

### 2. [Code Details](./density_analysis_code_details.md)
**Deep dive** - Line-by-line explanation of every function and method.

**Contents**:
- `calculate_variability()` - CV calculation
- `DensityMotifDiscovery` class
  - Constructor and parameters
  - `discover()` - main algorithm
  - `_prepare_time_series()` - normalization
  - `_find_constrained_seed()` - seed selection
  - `_check_variability_constraints()` - filtering logic
  - `_find_constrained_instances()` - instance finding
- `find_optimal_lag()` - lag detection
- `analyze_density_behavior()` - response analysis

**Best for**: Understanding implementation details, debugging, modifying code.

---

### 3. [Improvements](./density_analysis_improvements.md)
**Optimization guide** - Suggested improvements and simplifications.

**Contents**:
- Code simplifications (CV caching, data extraction)
- Performance optimizations (overlap checking, array operations)
- Robustness improvements (validation, error handling)
- Code organization (method splitting, separation of concerns)
- Additional features (progress reporting, export)
- Testing recommendations
- Documentation enhancements
- Priority ranking and impact analysis

**Best for**: Refactoring, optimization, extending functionality.

---

## 🚀 Quick Start

### Basic Usage

```python
from density_analysis import DensityMotifDiscovery, analyze_density_behavior
import pandas as pd

# Load your data
df = pd.read_csv('mill_data.csv')
# Required columns: WaterZumpf, Ore, WaterMill, DensityHC, TimeStamp

# Initialize discovery
discovery = DensityMotifDiscovery(
    window_size=60,      # 60-minute windows
    max_motifs=15,       # Find up to 15 motif groups
    radius=4.5,          # Distance threshold
    waterzumpf_max_cv=0.01,    # WaterZumpf stable (≤1% CV)
    ore_min_cv=0.0008,         # Ore varying (≥0.08% CV)
    watermill_min_cv=0.0015    # WaterMill varying (≥0.15% CV)
)

# Discover motifs
motifs = discovery.discover(df)
print(f"Found {len(motifs)} motif groups")

# Analyze density behavior
results = analyze_density_behavior(motifs)

# Examine results
for result in results:
    print(f"\nMotif {result['motif_id']}:")
    print(f"  Instances: {result['num_instances']}")
    print(f"  Avg density change: {result['avg_density_change']:+.2f}")
    print(f"  Ore-Density correlation: {result['avg_ore_density_corr']:+.3f}")
    print(f"  Ore lag: {result['avg_ore_lag']:.0f} minutes")
```

---

## 📊 What This Module Does

### Input
Time series data from ball mill operations with:
- **WaterZumpf**: Water flow to sump (m³/h)
- **Ore**: Ore feed rate (t/h)
- **WaterMill**: Water addition to mill (m³/h)
- **DensityHC**: Pulp density in hydrocyclone (g/L or %)
- **TimeStamp**: Time index

### Process
1. **Finds patterns** where WaterZumpf is stable but Ore and WaterMill vary
2. **Groups similar patterns** into motifs using matrix profile
3. **Analyzes density response** in each motif

### Output
For each motif:
- Number of occurrences
- Average density change
- Correlations with Ore and WaterMill
- Time lags (process dynamics)
- Statistical confidence

---

## 🎯 Key Concepts

### Coefficient of Variation (CV)
- **Formula**: CV = σ / |μ| (standard deviation / mean)
- **Purpose**: Scale-independent variability measure
- **Example**: CV = 0.01 means 1% variability

### Constrained Motif Discovery
Unlike traditional motif discovery that finds any recurring pattern, this module finds patterns matching **specific operational constraints**:

| Variable | Constraint | Meaning |
|----------|-----------|---------|
| WaterZumpf | CV ≤ 1% | Held constant (controlled) |
| Ore | CV ≥ 0.08% | Actively varied (manipulated) |
| WaterMill | CV ≥ 0.15% | Actively varied (manipulated) |

This represents **controlled experiments** embedded in operational data.

### Matrix Profile
- Efficient algorithm for finding similar patterns in time series
- Computes distance to nearest neighbor for each window
- O(n²) complexity (vs. naive O(n³))
- Multivariate version handles multiple variables simultaneously

### Lag Analysis
- Detects time delays in process response
- Uses cross-correlation to find optimal lag
- Critical for understanding process dynamics
- Example: "Density responds 5 minutes after Ore changes"

---

## 🔧 Configuration Guide

### Window Size
- **Default**: 60 minutes
- **Smaller** (30-45): Captures rapid changes, more motifs
- **Larger** (90-120): Captures slow dynamics, fewer motifs
- **Recommendation**: Match to typical operational cycle time

### Radius
- **Default**: 4.5
- **Smaller** (3-4): Stricter matching, fewer instances
- **Larger** (5-6): More lenient, more instances
- **Recommendation**: Tune based on data quality and desired strictness

### Variability Thresholds
- **WaterZumpf max CV** (default 0.01): How stable must sump water be?
- **Ore min CV** (default 0.0008): How much must ore vary?
- **WaterMill min CV** (default 0.0015): How much must mill water vary?
- **Recommendation**: Analyze your data's typical CV values first

---

## 📈 Interpreting Results

### Density Change
- **Positive**: Density increased during the pattern
- **Negative**: Density decreased during the pattern
- **Magnitude**: How much density changed (g/L or %)

### Correlations
- **+1.0**: Perfect positive correlation (both increase together)
- **0.0**: No linear relationship
- **-1.0**: Perfect negative correlation (one increases, other decreases)
- **Typical**: 0.3-0.7 for real process data

### Lags
- **Positive lag**: Response follows input (normal)
- **Zero lag**: Instantaneous response (rare)
- **Negative lag**: Response precedes input (unusual, check data)
- **Typical**: 2-10 minutes for grinding circuits

---

## 🐛 Troubleshooting

### No motifs found
**Possible causes**:
- Constraints too strict
- Window size too large
- Radius too small
- Data doesn't contain desired patterns

**Solutions**:
1. Relax variability thresholds
2. Increase radius to 5-6
3. Reduce window size
4. Check data quality and coverage

### Too many motifs (all similar)
**Possible causes**:
- Constraints too loose
- Radius too large
- Window size too small

**Solutions**:
1. Tighten variability thresholds
2. Reduce radius to 3-4
3. Increase window size
4. Increase exclusion range

### Poor correlations
**Possible causes**:
- Process has high noise
- Non-linear relationships
- Multiple confounding factors
- Incorrect lag

**Solutions**:
1. Check data quality
2. Try different window sizes
3. Examine individual instances
4. Consider non-linear analysis

---

## 📦 Dependencies

```python
numpy>=1.20.0
pandas>=1.3.0
stumpy>=1.11.0  # Matrix profile computation
scipy>=1.7.0    # Statistical functions
```

Install with:
```bash
pip install numpy pandas stumpy scipy
```

---

## 🔗 Related Modules

- **`motif_discovery.py`**: Base classes (Motif, MotifInstance)
- **`config.py`**: Configuration for modeling pipeline
- **`prepare_data.py`**: Data preprocessing
- **`gp_modelling/`**: Gaussian Process models using discovered patterns

---

## 📝 Citation

If you use this module in research or publications, please cite:

```
Density Motif Analysis Module
Ball Mill Grinding Circuit Optimization
[Your Organization], 2024
```

---

## 👥 Contributing

To improve this module:

1. Read the [Improvements](./density_analysis_improvements.md) document
2. Implement Priority 1 changes first
3. Add unit tests for new features
4. Update documentation
5. Submit for review

---

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the [Code Details](./density_analysis_code_details.md) document
3. Examine log output for diagnostic information
4. Contact the development team

---

## 📅 Version History

- **v1.0** (2024): Initial implementation
  - Constrained motif discovery
  - Lag analysis
  - Density behavior analysis

---

## 🎓 Learning Path

**Beginner**: 
1. Read [Overview](./density_analysis_overview.md)
2. Run basic usage example
3. Experiment with parameters

**Intermediate**:
1. Read [Code Details](./density_analysis_code_details.md)
2. Understand matrix profile algorithm
3. Tune parameters for your data

**Advanced**:
1. Read [Improvements](./density_analysis_improvements.md)
2. Implement optimizations
3. Extend functionality
4. Integrate with other modules

---

**Last Updated**: October 24, 2025
