# Pattern Summary - All Available Patterns

## Quick Reference

| Pattern | MVs | Target | Purpose | Default |
|---------|-----|--------|---------|---------|
| **MV** | Varying | Any | General MV patterns | ✅ ON |
| **Density** | WaterZumpf stable, others varying | Any | Density control | ✅ ON |
| **Inverse** | Ore/WaterMill stable, WaterZumpf varying | Any | Feed-forward control | ✅ ON |
| **Dynamic** | All varying | Any | Transient operations | ✅ ON |
| **Steady-State** ⭐ | **All stable** | **PSI200 stable** | **Operating points** | ✅ ON |
| **Pressure** | All varying, PressureHC stable | Any | Optimal control | ❌ OFF |

## Pattern Descriptions

### 1. MV Pattern (Standard)
- **Type:** `mv`
- **Features:** Ore, WaterMill, WaterZumpf
- **Constraints:** None
- **Purpose:** Discover general repeating MV patterns
- **Window:** 60 min
- **Max Motifs:** 20

### 2. Density Pattern
- **Type:** `constraint`
- **Stable:** WaterZumpf (CV ≤ 1%)
- **Varying:** Ore (CV ≥ 0.08%), WaterMill (CV ≥ 0.15%)
- **Purpose:** Steady sump water with varying feed
- **Window:** 60 min
- **Max Motifs:** 15

### 3. Inverse Pattern
- **Type:** `constraint`
- **Stable:** Ore (CV ≤ 1%), WaterMill (CV ≤ 1%)
- **Varying:** WaterZumpf (CV ≥ 0.1%)
- **Purpose:** Steady feed with sump water adjustments
- **Window:** 60 min
- **Max Motifs:** 10

### 4. Dynamic Pattern
- **Type:** `constraint`
- **Varying:** Ore (CV ≥ 0.08%), WaterMill (CV ≥ 0.15%), WaterZumpf (CV ≥ 0.1%)
- **Purpose:** Coordinated/transient operations
- **Window:** 60 min
- **Max Motifs:** 10

### 5. Steady-State Pattern ⭐ NEW
- **Type:** `constraint`
- **Stable:** Ore (CV ≤ 0.8%), WaterMill (CV ≤ 1%), WaterZumpf (CV ≤ 0.8%), PSI200 (CV ≤ 1.5%)
- **Purpose:** Identify stable operating points, reveal MV → PSI200 relationship
- **Window:** 90 min (longer for true steady state)
- **Max Motifs:** 15
- **Key Feature:** Includes target variable (PSI200) in constraints!

### 6. Pressure Pattern (Optional)
- **Type:** `constraint`
- **Stable:** PressureHC (CV ≤ 1%)
- **Varying:** Ore (CV ≥ 0.08%), WaterMill (CV ≥ 0.15%), WaterZumpf (CV ≥ 0.1%)
- **Purpose:** Optimal pressure control regions
- **Window:** 60 min
- **Max Motifs:** 10
- **Default:** Disabled

## Output Files

Each pattern generates:
- `{pattern}_analysis.csv` - Detailed analysis
- `{pattern}_analysis.png` - Visualization
- Instances in `segmented_motifs_all_08.csv` with `pattern_type` tag

## Usage Examples

### Use All Default Patterns (Including Steady-State)
```python
from data_preparation import DataPreparationPipeline, PipelineConfig

config = PipelineConfig.create_default(
    mill_number=8,
    start_date="2025-09-01",
    end_date="2025-11-03"
)

pipeline = DataPreparationPipeline(config)
pipeline.run()
```

### Use Only Steady-State Pattern
```python
from data_preparation.config.pattern_configs import create_steady_state_pattern

config = PipelineConfig.create_default(
    mill_number=8,
    start_date="2025-09-01",
    end_date="2025-11-03",
    patterns=[create_steady_state_pattern()]
)

pipeline = DataPreparationPipeline(config)
pipeline.run()
```

### Customize Steady-State Pattern
```python
from data_preparation.config.pattern_configs import create_steady_state_pattern

# Stricter steady state
strict_ss = create_steady_state_pattern(
    window_size=120,  # 2 hours
    max_motifs=10,
    radius=4.0
)

# More relaxed steady state
relaxed_ss = create_steady_state_pattern(
    window_size=60,   # 1 hour
    max_motifs=20,
    radius=6.0
)
```

## When to Use Each Pattern

### Use MV Pattern When:
- You want general MV patterns without constraints
- You're exploring the data
- You need maximum flexibility

### Use Density Pattern When:
- Sump water level is tightly controlled
- You want to understand density control strategies
- Feed and mill water are being adjusted

### Use Inverse Pattern When:
- Feed rate is steady
- Sump water is being adjusted for control
- You want feed-forward control patterns

### Use Dynamic Pattern When:
- All variables are changing together
- You want to capture transitions
- Coordinated adjustments are happening

### Use Steady-State Pattern When: ⭐
- **You want to understand MV → PSI200 relationships**
- **You need to identify optimal operating points**
- **You want to validate models at steady state**
- **You need to find different operating regimes**
- **Product quality (PSI200) is stable**

### Use Pressure Pattern When:
- Pressure control is critical
- You want optimal control regions
- You have good pressure control

## Pattern Comparison Matrix

| Aspect | MV | Density | Inverse | Dynamic | **Steady-State** ⭐ | Pressure |
|--------|----|---------|---------|---------|--------------------|----------|
| Constraints | None | 1 stable, 2 varying | 2 stable, 1 varying | 3 varying | **4 stable** | 1 stable, 3 varying |
| Includes Target | No | No | No | No | **Yes (PSI200)** | No |
| Window Size | 60 | 60 | 60 | 60 | **90** | 60 |
| Purpose | General | Control | Feed-forward | Transient | **Operating Points** | Optimal |
| Default Status | ON | ON | ON | ON | **ON** | OFF |

## Key Insights from Each Pattern

### MV Pattern
- General operational patterns
- Baseline for comparison
- No operational interpretation

### Density Pattern
- How operators maintain density
- Feed-water coordination
- Density control strategies

### Inverse Pattern
- Feed-forward control approach
- Sump water as control variable
- Alternative control strategy

### Dynamic Pattern
- Transient behaviors
- Coordinated changes
- System dynamics

### Steady-State Pattern ⭐
- **MV → PSI200 relationships**
- **Optimal operating points**
- **Operating regime classification**
- **Model validation points**
- **Process optimization targets**

### Pressure Pattern
- Pressure control quality
- Optimal control regions
- Process stability indicators

## Documentation

- **Main Guide:** `CONSTRAINT_PATTERNS_GUIDE.md`
- **Steady-State Details:** `STEADY_STATE_PATTERN.md`
- **Code vs Docs:** `CODE_VS_DOCS_NOTES.md`
- **User Guide:** `../README.md`
- **Migration:** `../MIGRATION_GUIDE.md`

---

**Last Updated:** November 7, 2025  
**Total Patterns:** 6 (5 enabled by default)  
**New Pattern:** Steady-State Operating Point Pattern ⭐
