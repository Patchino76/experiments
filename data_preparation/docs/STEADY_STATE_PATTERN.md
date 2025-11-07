# Steady-State Operating Point Pattern

## Overview

The **steady-state pattern** is a specialized constraint pattern that discovers operating points where **both manipulated variables (MVs) and the target variable (PSI200) are stable**. This pattern is crucial for understanding how different MV settings affect product quality at steady state.

## Purpose

### What It Captures

```
┌─────────────────────────────────────────────────────────────┐
│              Steady-State Operating Point                    │
├─────────────────────────────────────────────────────────────┤
│ Constraints:                                                 │
│   Ore:        STABLE  (CV ≤ 0.8%)                           │
│   WaterMill:  STABLE  (CV ≤ 1.0%)                           │
│   WaterZumpf: STABLE  (CV ≤ 0.8%)                           │
│   PSI200:     STABLE  (CV ≤ 1.5%)                           │
├─────────────────────────────────────────────────────────────┤
│ Operational Meaning:                                         │
│   • All MVs at constant levels                              │
│   • Product quality (PSI200) stable                         │
│   • System at equilibrium                                   │
│   • Reveals MV → PSI200 relationship                        │
└─────────────────────────────────────────────────────────────┘
```

### Why It's Important

1. **Operating Point Identification**: Find different stable operating regimes
2. **MV-Target Relationship**: Understand how MV settings affect PSI200
3. **Model Validation**: Validate process models at steady state
4. **Optimal Settings**: Identify best MV combinations for target PSI200
5. **Operating Regimes**: Distinguish between coarse vs fine grinding modes

## Example Scenarios

### Scenario 1: Coarse Grinding Mode
```
Operating Point A:
  Ore:        150 t/h  (stable)
  WaterMill:  12 m³/h  (stable)
  WaterZumpf: 45 m³/h  (stable)
  PSI200:     28%      (stable)
  
→ Lower feed, moderate water → Coarser product
```

### Scenario 2: Fine Grinding Mode
```
Operating Point B:
  Ore:        140 t/h  (stable)
  WaterMill:  14 m³/h  (stable)
  WaterZumpf: 50 m³/h  (stable)
  PSI200:     35%      (stable)
  
→ Lower feed, higher water → Finer product
```

### Scenario 3: High Throughput Mode
```
Operating Point C:
  Ore:        165 t/h  (stable)
  WaterMill:  11 m³/h  (stable)
  WaterZumpf: 42 m³/h  (stable)
  PSI200:     25%      (stable)
  
→ Higher feed, lower water → Coarser but higher throughput
```

## Configuration

### Default Configuration

```python
def create_steady_state_pattern(
    enabled: bool = True,
    window_size: int = 90,      # Longer window for steady state
    max_motifs: int = 15,
    radius: float = 5.0         # Slightly higher radius
) -> PatternConfig:
    return PatternConfig(
        name='steady_state',
        type='constraint',
        enabled=enabled,
        window_size=window_size,
        max_motifs=max_motifs,
        radius=radius,
        constraints={
            'Ore': {
                'type': 'stable',
                'max_cv': 0.008  # 0.8%
            },
            'WaterMill': {
                'type': 'stable',
                'max_cv': 0.01   # 1.0%
            },
            'WaterZumpf': {
                'type': 'stable',
                'max_cv': 0.008  # 0.8%
            },
            'PSI200': {
                'type': 'stable',
                'max_cv': 0.015  # 1.5%
            }
        }
    )
```

### Why These Parameters?

1. **Window Size = 90 minutes**
   - Longer than other patterns (60 min)
   - Ensures true steady state, not just temporary stability
   - Allows transients to settle

2. **Radius = 5.0**
   - Slightly higher than other patterns (4.5)
   - Accounts for natural process variation at steady state
   - Still strict enough to group similar operating points

3. **CV Thresholds**
   - **MVs (0.8-1.0%)**: Tight control, minimal variation
   - **PSI200 (1.5%)**: Allows for measurement noise and natural variation
   - Stricter than "stable" in other patterns to ensure true steady state

## Usage

### Basic Usage

```python
from data_preparation import DataPreparationPipeline, PipelineConfig
from data_preparation.config.pattern_configs import get_default_patterns

# Steady-state pattern is included by default
config = PipelineConfig.create_default(
    mill_number=8,
    start_date="2025-09-01",
    end_date="2025-11-03"
)

pipeline = DataPreparationPipeline(config)
pipeline.run()
```

### Custom Configuration

```python
from data_preparation.config.pattern_configs import create_steady_state_pattern

# Stricter steady state (for very stable periods only)
strict_ss = create_steady_state_pattern(
    window_size=120,  # 2 hours
    max_motifs=10,
    radius=4.0
)

# More relaxed steady state (for noisy data)
relaxed_ss = create_steady_state_pattern(
    window_size=60,
    max_motifs=20,
    radius=6.0
)
```

### Disable If Not Needed

```python
from data_preparation.config.pattern_configs import create_steady_state_pattern

# Disable steady-state pattern
ss_pattern = create_steady_state_pattern(enabled=False)
```

## Output Files

### 1. Analysis CSV: `steady_state_analysis.csv`

Contains detailed analysis of each discovered motif:

| Column | Description |
|--------|-------------|
| `motif_id` | Motif identifier |
| `n_instances` | Number of instances |
| `avg_distance` | Average distance between instances |
| `avg_Ore` | Average Ore feed rate |
| `avg_WaterMill` | Average mill water flow |
| `avg_WaterZumpf` | Average sump water flow |
| `avg_PSI200` | Average PSI200 value |
| `cv_Ore` | CV of Ore (should be low) |
| `cv_WaterMill` | CV of WaterMill (should be low) |
| `cv_WaterZumpf` | CV of WaterZumpf (should be low) |
| `cv_PSI200` | CV of PSI200 (should be low) |

### 2. Visualization: `steady_state_analysis.png`

Shows:
- Scatter plot: MV combinations vs PSI200
- Operating point clusters
- CV distributions
- Time series of discovered instances

### 3. Segmented Data: `segmented_motifs_all_08.csv`

All steady-state instances are included with:
- `pattern_type = 'steady_state'`
- `motif_id` for grouping
- All features (MVs, CVs, DVs, PSI200)

## Analysis Insights

### What You Can Learn

1. **Operating Point Map**
   ```python
   # Load analysis results
   import pandas as pd
   df = pd.read_csv('output/analysis/steady_state_analysis.csv')
   
   # Plot MV vs PSI200
   import matplotlib.pyplot as plt
   
   plt.scatter(df['avg_Ore'], df['avg_PSI200'], 
               s=df['n_instances']*10, alpha=0.6)
   plt.xlabel('Ore Feed Rate (t/h)')
   plt.ylabel('PSI200 (%)')
   plt.title('Steady-State Operating Points')
   plt.show()
   ```

2. **Optimal Operating Regions**
   ```python
   # Find operating points with target PSI200
   target_psi = 32.0
   tolerance = 2.0
   
   optimal = df[
       (df['avg_PSI200'] >= target_psi - tolerance) &
       (df['avg_PSI200'] <= target_psi + tolerance)
   ]
   
   print("Optimal MV settings for PSI200 = 32%:")
   print(optimal[['avg_Ore', 'avg_WaterMill', 'avg_WaterZumpf']])
   ```

3. **Operating Regime Classification**
   ```python
   # Cluster operating points
   from sklearn.cluster import KMeans
   
   X = df[['avg_Ore', 'avg_WaterMill', 'avg_WaterZumpf']].values
   kmeans = KMeans(n_clusters=3)
   df['regime'] = kmeans.fit_predict(X)
   
   print("Operating Regimes:")
   for regime in df['regime'].unique():
       regime_data = df[df['regime'] == regime]
       print(f"\nRegime {regime}:")
       print(f"  Avg PSI200: {regime_data['avg_PSI200'].mean():.1f}%")
       print(f"  Avg Ore: {regime_data['avg_Ore'].mean():.1f} t/h")
   ```

## Comparison with Other Patterns

| Pattern | MVs | PSI200 | Purpose |
|---------|-----|--------|---------|
| **MV** | Varying | Any | General MV patterns |
| **Density** | WaterZumpf stable, others varying | Any | Density control |
| **Inverse** | Ore/WaterMill stable, WaterZumpf varying | Any | Feed-forward control |
| **Dynamic** | All varying | Any | Transient operations |
| **Steady-State** ⭐ | **All stable** | **Stable** | **Operating points** |

## Tips for Best Results

### 1. Data Quality
- Ensure PSI200 measurements are available and reliable
- Remove periods with sensor failures
- Filter out startup/shutdown periods

### 2. Parameter Tuning
- **If too few motifs found**: Relax CV thresholds (increase max_cv)
- **If too many motifs found**: Tighten CV thresholds (decrease max_cv)
- **If motifs too similar**: Decrease radius
- **If motifs too sparse**: Increase radius

### 3. Window Size Selection
```python
# For very stable process
window_size = 120  # 2 hours

# For moderately stable process
window_size = 90   # 1.5 hours (default)

# For less stable process
window_size = 60   # 1 hour
```

### 4. CV Threshold Tuning
```python
# Analyze actual CV distribution first
import pandas as pd
import numpy as np

df = pd.read_csv('output/data/initial_data.csv')

for feature in ['Ore', 'WaterMill', 'WaterZumpf', 'PSI200']:
    # Calculate rolling CV
    rolling_cv = df[feature].rolling(90).std() / df[feature].rolling(90).mean()
    
    print(f"\n{feature} CV Distribution:")
    print(f"  10th percentile: {rolling_cv.quantile(0.1):.4f}")
    print(f"  25th percentile: {rolling_cv.quantile(0.25):.4f}")
    print(f"  50th percentile: {rolling_cv.quantile(0.5):.4f}")

# Use 10th-20th percentile as max_cv for steady state
```

## Expected Results

### Typical Findings

For a 2-month dataset (Mill 8):
- **10-15 steady-state motifs** discovered
- **3-5 distinct operating regimes** identified
- **5-20 instances per motif**
- **Clear MV → PSI200 relationships** revealed

### Example Output

```
Steady-State Pattern Discovery Results:
  Motifs discovered: 12
  Total instances: 156
  
Operating Point Summary:
  Motif 1: Ore=145, WaterMill=13, WaterZumpf=48 → PSI200=32%
  Motif 2: Ore=155, WaterMill=12, WaterZumpf=45 → PSI200=28%
  Motif 3: Ore=140, WaterMill=14, WaterZumpf=50 → PSI200=35%
  ...
```

## Troubleshooting

### Problem: No Motifs Found

**Cause**: Constraints too strict or no true steady-state periods

**Solutions**:
1. Relax CV thresholds:
   ```python
   constraints={
       'Ore': {'type': 'stable', 'max_cv': 0.012},      # Was 0.008
       'WaterMill': {'type': 'stable', 'max_cv': 0.015}, # Was 0.01
       'WaterZumpf': {'type': 'stable', 'max_cv': 0.012}, # Was 0.008
       'PSI200': {'type': 'stable', 'max_cv': 0.020}     # Was 0.015
   }
   ```

2. Reduce window size:
   ```python
   window_size = 60  # Was 90
   ```

3. Check data quality:
   ```python
   # Are there any stable periods?
   df['Ore_cv'] = df['Ore'].rolling(90).std() / df['Ore'].rolling(90).mean()
   print(f"Periods with CV < 0.01: {(df['Ore_cv'] < 0.01).sum()} / {len(df)}")
   ```

### Problem: All Motifs Very Similar

**Cause**: Radius too high or process operates at single setpoint

**Solutions**:
1. Decrease radius:
   ```python
   radius = 4.0  # Was 5.0
   ```

2. This might be expected if process is well-controlled!

### Problem: PSI200 Not Available

**Cause**: PSI200 column missing from data

**Solutions**:
1. Check data loading:
   ```python
   print(df.columns)
   ```

2. Use alternative target:
   ```python
   constraints={
       'Ore': {'type': 'stable', 'max_cv': 0.008},
       'WaterMill': {'type': 'stable', 'max_cv': 0.01},
       'WaterZumpf': {'type': 'stable', 'max_cv': 0.008},
       'PSI80': {'type': 'stable', 'max_cv': 0.015}  # Use PSI80 instead
   }
   ```

## Integration with Modeling

### Use in Model Training

The steady-state data is particularly valuable for:

1. **Model Validation**
   - Test model predictions at known steady-state points
   - Verify model captures MV → PSI200 relationship

2. **Model Training**
   - Include in training set for better steady-state performance
   - Weight steady-state data higher if steady-state accuracy is critical

3. **Operating Point Optimization**
   - Use discovered points as starting points for optimization
   - Validate optimized setpoints against historical steady states

### Example: Model Validation

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load steady-state data
ss_data = pd.read_csv('output/data/segmented_motifs_all_08.csv')
ss_data = ss_data[ss_data['pattern_type'] == 'steady_state']

# Train model
X = ss_data[['Ore', 'WaterMill', 'WaterZumpf']]
y = ss_data['PSI200']

model = RandomForestRegressor()
model.fit(X, y)

# Validate at steady-state points
predictions = model.predict(X)
errors = predictions - y

print(f"Steady-State Prediction Error:")
print(f"  MAE: {abs(errors).mean():.2f}%")
print(f"  RMSE: {(errors**2).mean()**0.5:.2f}%")
```

---

## Summary

The **steady-state pattern** is a powerful tool for:
- ✅ Identifying stable operating points
- ✅ Understanding MV → PSI200 relationships
- ✅ Finding optimal operating conditions
- ✅ Validating process models
- ✅ Classifying operating regimes

**Key Takeaway**: This pattern reveals **how different MV settings affect product quality** when the system is at equilibrium, providing crucial insights for process optimization and control.

---

**Document Version:** 1.0  
**Last Updated:** November 7, 2025  
**Pattern Added:** Steady-State Operating Point Pattern
