# Constraint Patterns System - Complete Guide

> **📝 Note:** This documentation reflects the **actual implementation** in `patterns/constraint_pattern.py`.  
> All code snippets are taken directly from or closely match the real codebase.

## 📚 Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [How It Works](#how-it-works)
5. [Pattern Types](#pattern-types)
6. [Code Structure](#code-structure)
7. [Data Flow](#data-flow)
8. [How to Add New Patterns](#how-to-add-new-patterns)
9. [Advanced Topics](#advanced-topics)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### What Are Constraint Patterns?

Constraint patterns are a **universal system for discovering repeating operational patterns** in ball mill data where certain features exhibit specific variability characteristics. Instead of having separate classes for each pattern type, we use **one flexible class** configured through constraints.

### Key Innovation

```
Old System (modeling/):
├── DensityMotifDiscovery        (~200 lines)
├── InverseConstraintMotifDiscovery (~200 lines)
├── DynamicMotifDiscovery        (~200 lines)
└── PressureConstraintMotifDiscovery (~200 lines)
    Total: ~800 lines, 90% duplication

New System (data_preparation/):
└── ConstraintPattern            (~250 lines)
    Total: ~250 lines, <10% duplication
    
Reduction: 70% less code!
```

### Why Constraint Patterns?

1. **Operational Insight**: Different constraint combinations represent different operational regimes
2. **Model Training**: More diverse patterns → better model generalization
3. **Process Understanding**: Stable/varying features reveal control strategies
4. **Flexibility**: Easy to add new patterns without coding

---

## Core Concepts

### 1. Coefficient of Variation (CV)

**Definition:**
```
CV = σ / μ = Standard Deviation / Mean
```

**Purpose:** Measures relative variability independent of scale

**Example:**
```python
# Feature A: mean=100, std=10 → CV = 10/100 = 0.10 (10%)
# Feature B: mean=1000, std=10 → CV = 10/1000 = 0.01 (1%)
# Feature B is more stable despite same absolute variation
```

### 2. Constraint Types

#### **Stable Constraint**
Feature should have **low variability** (tightly controlled)

```python
{
    'type': 'stable',
    'max_cv': 0.01  # CV must be ≤ 1%
}
```

**Operational Meaning:** Variable is being controlled tightly, not changing much

#### **Varying Constraint**
Feature should have **high variability** (actively changing)

```python
{
    'type': 'varying',
    'min_cv': 0.001  # CV must be ≥ 0.1%
}
```

**Operational Meaning:** Variable is being actively adjusted or responding to changes

### 3. Constraint Validation

For a window to be valid, **ALL constraints must be satisfied**:

```python
def _check_constraints(self, df: pd.DataFrame, idx: int) -> bool:
    """
    Check if window at idx satisfies all constraints.
    
    Args:
        df: DataFrame
        idx: Window start index
        
    Returns:
        True if constraints satisfied
    """
    cvs = {}
    
    # Calculate CV for all features
    for feature in self.features:
        data = df[feature].iloc[idx:idx + self.window_size].values
        cvs[feature] = self.calculate_variability(data)
    
    # Check stable features (low CV)
    for feature in self.stable_features:
        constraint = self.constraints[feature]
        max_cv = constraint.get('max_cv', 0.01)
        
        if cvs[feature] > max_cv:
            return False  # ❌ Too variable
    
    # Check varying features (high CV)
    for feature in self.varying_features:
        constraint = self.constraints[feature]
        min_cv = constraint.get('min_cv', 0.0008)
        
        if cvs[feature] < min_cv:
            return False  # ❌ Not varying enough
    
    # Check relative variability (varying should be more variable than stable)
    if self.stable_features and self.varying_features:
        max_stable_cv = max(cvs[f] for f in self.stable_features)
        min_varying_cv = min(cvs[f] for f in self.varying_features)
        
        if min_varying_cv < max_stable_cv * self.relative_variability_factor:
            return False  # ❌ Not relatively variable enough
    
    return True  # ✅ All constraints satisfied
```

**Key Implementation Details:**

1. **Pre-calculate all CVs**: More efficient than calculating on-demand
2. **Use `calculate_variability()` method**: Handles edge cases (division by zero, NaN)
3. **Default values**: Uses `.get()` with defaults for robustness
4. **Relative variability check**: Ensures varying features are significantly more variable than stable features (by `relative_variability_factor`, typically 2.0)

### 4. Relative Variability Check

Ensures varying features are **significantly more variable** than stable features:

```python
# From actual implementation:
if self.stable_features and self.varying_features:
    max_stable_cv = max(cvs[f] for f in self.stable_features)
    min_varying_cv = min(cvs[f] for f in self.varying_features)
    
    # Varying should be at least 2x more variable than stable
    if min_varying_cv < max_stable_cv * self.relative_variability_factor:
        return False  # Not different enough
```

**Why This Matters:**
- Prevents patterns where "stable" and "varying" features have similar variability
- Ensures meaningful operational distinctions
- Default `relative_variability_factor = 2.0` (configurable)

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Preparation Pipeline                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Pattern Registry                         │
│  • Registers pattern classes                                │
│  • Creates pattern instances from config                    │
│  • Discovers all enabled patterns                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   ConstraintPattern Class                    │
│  • Universal pattern discovery                              │
│  • Configurable constraints                                 │
│  • Matrix profile computation                               │
│  • Constraint validation                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Pattern Configurations                    │
│  • Density Pattern                                          │
│  • Inverse Pattern                                          │
│  • Dynamic Pattern                                          │
│  • Pressure Pattern                                         │
│  • Custom Patterns                                          │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
data_preparation/
├── patterns/
│   ├── __init__.py
│   ├── mv_pattern.py              # Standard MV pattern
│   └── constraint_pattern.py      # ⭐ Universal constraint pattern
│
├── config/
│   ├── defaults.py                # PatternConfig dataclass
│   └── pattern_configs.py         # ⭐ Pattern factory functions
│
├── core/
│   ├── base_pattern.py            # BasePattern abstract class
│   ├── pattern_registry.py        # ⭐ Registration system
│   ├── data_loader.py             # Data loading
│   └── segmentation.py            # Motif to dataset conversion
│
├── analysis/
│   ├── analyzer.py                # Generic analysis
│   └── visualizer.py              # Generic visualization
│
└── pipeline.py                    # Main orchestrator
```

---

## How It Works

### Complete Discovery Process

```
Step 1: Load Configuration
┌─────────────────────────────────────────────────────────────┐
│ config = create_density_pattern()                           │
│ constraints = {                                             │
│   'WaterZumpf': {'type': 'stable', 'max_cv': 0.01},       │
│   'Ore': {'type': 'varying', 'min_cv': 0.0008},           │
│   'WaterMill': {'type': 'varying', 'min_cv': 0.0015}      │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
Step 2: Create Pattern Instance
┌─────────────────────────────────────────────────────────────┐
│ pattern = PatternRegistry.create_pattern('density', config) │
│ # Creates ConstraintPattern instance                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
Step 3: Prepare Time Series
┌─────────────────────────────────────────────────────────────┐
│ features = ['Ore', 'WaterMill', 'WaterZumpf']              │
│ time_series = df[features].values                          │
│ # Shape: (n_samples, n_features)                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
Step 4: Compute Matrix Profile (STUMPY)
┌─────────────────────────────────────────────────────────────┐
│ mp = stumpy.mstump(T=time_series, m=window_size)           │
│ # Returns: distance to nearest neighbor for each window    │
└─────────────────────────────────────────────────────────────┘
                              ↓
Step 5: Find Seed Points
┌─────────────────────────────────────────────────────────────┐
│ for idx in sorted_by_distance:                             │
│     if _check_constraints(df, idx):                        │
│         seeds.append(idx)  # ✅ Satisfies constraints      │
└─────────────────────────────────────────────────────────────┘
                              ↓
Step 6: Find Similar Instances
┌─────────────────────────────────────────────────────────────┐
│ for candidate in sorted_by_distance:                       │
│     if distance <= radius AND                              │
│        _check_constraints(df, candidate) AND               │
│        not overlaps:                                       │
│         instances.append(candidate)  # ✅ Valid instance   │
└─────────────────────────────────────────────────────────────┘
                              ↓
Step 7: Create Motif Objects
┌─────────────────────────────────────────────────────────────┐
│ motif = Motif(motif_id, instances, metadata)               │
└─────────────────────────────────────────────────────────────┘
```

## Pattern Types

### 1. Density Pattern

**Operational Scenario:** Steady sump water with varying feed

```
┌─────────────────────────────────────────────────────────────┐
│                     Density Pattern                          │
├─────────────────────────────────────────────────────────────┤
│ Constraints:                                                 │
│   WaterZumpf: STABLE   (CV ≤ 1.0%)                          │
│   Ore:        VARYING  (CV ≥ 0.08%)                         │
│   WaterMill:  VARYING  (CV ≥ 0.15%)                         │
├─────────────────────────────────────────────────────────────┤
│ Operational Meaning:                                         │
│   • Sump water level controlled tightly                     │
│   • Feed rate being adjusted                                │
│   • Mill water being adjusted                               │
│   • Represents density control operations                   │
└─────────────────────────────────────────────────────────────┘
```

**Configuration Code:**
```python
def create_density_pattern(
    enabled: bool = True,
    window_size: int = 60,
    max_motifs: int = 15,
    radius: float = 4.5
) -> PatternConfig:
    return PatternConfig(
        name='density',
        type='constraint',
        enabled=enabled,
        window_size=window_size,
        max_motifs=max_motifs,
        radius=radius,
        constraints={
            'WaterZumpf': {
                'type': 'stable',
                'max_cv': 0.01  # 1%
            },
            'Ore': {
                'type': 'varying',
                'min_cv': 0.0008  # 0.08%
            },
            'WaterMill': {
                'type': 'varying',
                'min_cv': 0.0015  # 0.15%
            }
        }
    )
```

### 2. Inverse Pattern

**Operational Scenario:** Steady feed with sump water adjustments

```
┌─────────────────────────────────────────────────────────────┐
│                     Inverse Pattern                          │
├─────────────────────────────────────────────────────────────┤
│ Constraints:                                                 │
│   Ore:        STABLE   (CV ≤ 1.0%)                          │
│   WaterMill:  STABLE   (CV ≤ 1.0%)                          │
│   WaterZumpf: VARYING  (CV ≥ 0.1%)                          │
├─────────────────────────────────────────────────────────────┤
│ Operational Meaning:                                         │
│   • Feed rate steady                                        │
│   • Mill water steady                                       │
│   • Sump water being adjusted                               │
│   • Represents feed-forward control                         │
└─────────────────────────────────────────────────────────────┘
```

### 3. Dynamic Pattern

**Operational Scenario:** Coordinated/transient operations

```
┌─────────────────────────────────────────────────────────────┐
│                     Dynamic Pattern                          │
├─────────────────────────────────────────────────────────────┤
│ Constraints:                                                 │
│   Ore:        VARYING  (CV ≥ 0.08%)                         │
│   WaterMill:  VARYING  (CV ≥ 0.15%)                         │
│   WaterZumpf: VARYING  (CV ≥ 0.1%)                          │
├─────────────────────────────────────────────────────────────┤
│ Operational Meaning:                                         │
│   • All variables changing together                         │
│   • Coordinated adjustments                                 │
│   • Transition periods                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Code Structure

### Key Files and Their Roles

#### 1. `patterns/constraint_pattern.py` - The Universal Pattern Class

```python
@PatternRegistry.register('constraint')
class ConstraintPattern(BasePattern):
    """
    Universal constraint-based motif discovery.
    
    Discovers motifs based on configurable variability constraints.
    Features can be marked as 'stable' (low CV) or 'varying' (high CV).
    """
    
    def __init__(self, name: str, config: dict):
        """
        Initialize constraint pattern.
        
        Args:
            name: Pattern name
            config: Configuration dictionary with 'constraints' key
        """
        super().__init__(name, config)
        
        # Parse constraints
        self.constraints = config.get('constraints', {})
        self.features = list(self.constraints.keys())
        self.relative_variability_factor = config.get('relative_variability_factor', 1.2)
        
        # Separate stable and varying features
        self.stable_features = []
        self.varying_features = []
        
        for feature, constraint in self.constraints.items():
            constraint_type = constraint.get('type', 'stable')
            if constraint_type == 'stable':
                self.stable_features.append(feature)
            elif constraint_type == 'varying':
                self.varying_features.append(feature)
    
    def discover(self, df: pd.DataFrame) -> List[Motif]:
        """
        Discover constraint-based motifs.
        
        Steps:
        1. Validate data
        2. Prepare time series
        3. Compute matrix profile (STUMPY mstump)
        4. Find constrained seeds
        5. Find similar instances for each seed
        6. Create Motif objects
        """
        # Implementation in actual code
        pass
    
    def _check_constraints(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Check if window at idx satisfies all constraints.
        
        Returns True if:
        - All stable features have CV <= max_cv
        - All varying features have CV >= min_cv
        - Varying features are more variable than stable (by factor)
        """
        # See Core Concepts section for full implementation
        pass
    
    def _find_constrained_seed(self, df, mp_distances, n_windows, used) -> Tuple[int, float]:
        """Find best seed that satisfies constraints."""
        pass
    
    def _find_constrained_instances(self, df, T, seed_idx, n_windows, used, mp) -> List[dict]:
        """Find instances similar to seed that satisfy constraints."""
        pass
```

#### 2. `config/pattern_configs.py` - Pattern Factory Functions

```python
def create_density_pattern(...) -> PatternConfig:
    """Create density pattern configuration."""
    return PatternConfig(name='density', type='constraint', ...)

def create_inverse_pattern(...) -> PatternConfig:
    """Create inverse pattern configuration."""
    return PatternConfig(name='inverse', type='constraint', ...)

def create_dynamic_pattern(...) -> PatternConfig:
    """Create dynamic pattern configuration."""
    return PatternConfig(name='dynamic', type='constraint', ...)

def create_custom_pattern(name, constraints, ...) -> PatternConfig:
    """Create custom pattern configuration."""
    return PatternConfig(name=name, type='constraint', 
                        constraints=constraints, ...)
```

#### 3. `core/pattern_registry.py` - Registration System

```python
class PatternRegistry:
    """Registry for pattern classes."""
    
    _patterns = {}  # Class-level registry
    
    @classmethod
    def register(cls, pattern_type):
        """Decorator to register pattern classes."""
        def decorator(pattern_class):
            cls._patterns[pattern_type] = pattern_class
            logger.info(f"Registered pattern: {pattern_type}")
            return pattern_class
        return decorator
    
    @classmethod
    def create_pattern(cls, name, config):
        """Create pattern instance from config."""
        pattern_type = config.get('type', name)
        
        if pattern_type not in cls._patterns:
            raise ValueError(f"Pattern type '{pattern_type}' not registered")
        
        pattern_class = cls._patterns[pattern_type]
        return pattern_class(name=name, **config)
    
    @classmethod
    def discover_all(cls, df, pattern_configs):
        """Discover all patterns."""
        results = {}
        for config in pattern_configs:
            name = config['name']
            try:
                pattern = cls.create_pattern(name, config)
                motifs = pattern.discover(df)
                results[name] = motifs
            except Exception as e:
                logger.error(f"Error discovering pattern '{name}': {e}")
                results[name] = []
        return results
```

---

## Data Flow

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER CONFIGURATION                                        │
│    • Define patterns in run.py or config                    │
│    • Enable/disable patterns                                │
│    • Set parameters (window_size, max_motifs, etc.)         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. PATTERN REGISTRATION                                      │
│    • Import patterns module                                 │
│    • @register decorator triggers registration              │
│    • Patterns added to registry                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. DATA LOADING                                              │
│    • Load from database or cache                            │
│    • Filter data                                            │
│    • Calculate circulative load                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. PATTERN DISCOVERY (for each enabled pattern)             │
│    ├─ Create pattern instance from config                   │
│    ├─ Prepare time series data                              │
│    ├─ Compute matrix profile (STUMPY)                       │
│    ├─ Find seed points (check constraints)                  │
│    ├─ Find similar instances (check constraints + distance) │
│    └─ Create Motif objects                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ANALYSIS                                                  │
│    • Analyze density behavior                               │
│    • Calculate correlations and lags                        │
│    • Generate summary statistics                            │
│    • Save analysis CSVs                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. SEGMENTATION                                              │
│    • Merge all motif collections                            │
│    • Shuffle instances                                      │
│    • Create segmented datasets                              │
│    • Save CSV files                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. VISUALIZATION                                             │
│    • Create overview plots                                  │
│    • Create pattern-specific plots                          │
│    • Create correlation heatmaps                            │
│    • Save PNG files                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. OUTPUT                                                    │
│    • segmented_motifsMV_{mill}.csv                          │
│    • segmented_motifs_all_{mill}.csv                        │
│    • {pattern}_analysis.csv                                 │
│    • {pattern}_analysis.png                                 │
│    • motif_summary.csv                                      │
│    • instance_catalog.csv                                   │
└─────────────────────────────────────────────────────────────┘
```

### Constraint Checking Flow

```
Window at index i
       ↓
┌─────────────────────────────────────────────────────────────┐
│ Extract window data                                          │
│ window = df.iloc[i:i+window_size]                           │
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│ For each feature in constraints:                            │
│   1. Get feature data                                       │
│   2. Calculate CV = std / mean                              │
│   3. Check constraint type                                  │
└─────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────┬──────────────────────────────────────┐
│ If STABLE:           │ If VARYING:                          │
│   CV <= max_cv?      │   CV >= min_cv?                      │
│   ✅ Yes → Continue  │   ✅ Yes → Continue                  │
│   ❌ No → Reject     │   ❌ No → Reject                     │
└──────────────────────┴──────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│ Check relative variability                                   │
│ varying_cv > stable_cv * 2.0?                               │
│   ✅ Yes → Window is VALID                                  │
│   ❌ No → Window is INVALID                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## How to Add New Patterns

### Method 1: Using Factory Function (Recommended)

**Step 1: Define Your Pattern**

Think about what operational regime you want to capture:
- Which features should be stable?
- Which features should be varying?
- What are appropriate CV thresholds?

**Step 2: Create Pattern Configuration**

```python
# In your script or in config/pattern_configs.py

from data_preparation.config.pattern_configs import create_custom_pattern

# Example: Stable ore feed with varying water flows
stable_ore_pattern = create_custom_pattern(
    name='stable_ore',
    constraints={
        'Ore': {
            'type': 'stable',
            'max_cv': 0.005  # Very stable (0.5%)
        },
        'WaterMill': {
            'type': 'varying',
            'min_cv': 0.002  # Actively varying (0.2%)
        },
        'WaterZumpf': {
            'type': 'varying',
            'min_cv': 0.001  # Actively varying (0.1%)
        }
    },
    window_size=90,  # 90-minute windows
    max_motifs=10,
    radius=5.0
)
```

**Step 3: Use in Pipeline**

```python
from data_preparation import DataPreparationPipeline, PipelineConfig
from data_preparation.config.pattern_configs import create_mv_pattern

# Create pattern list
patterns = [
    create_mv_pattern(),
    stable_ore_pattern  # Your custom pattern
]

# Create config
config = PipelineConfig.create_default(
    mill_number=8,
    start_date="2025-09-01",
    end_date="2025-11-03",
    patterns=patterns
)

# Run pipeline
pipeline = DataPreparationPipeline(config)
pipeline.run()
```

### Method 2: Direct PatternConfig Creation

```python
from data_preparation.config.defaults import PatternConfig

my_pattern = PatternConfig(
    name='my_custom_pattern',
    type='constraint',
    enabled=True,
    window_size=60,
    max_motifs=15,
    radius=4.5,
    max_instances_per_motif=20,
    constraints={
        'Feature1': {'type': 'stable', 'max_cv': 0.01},
        'Feature2': {'type': 'varying', 'min_cv': 0.001},
        'Feature3': {'type': 'varying', 'min_cv': 0.002}
    },
    save_analysis=True,
    save_plots=True
)
```

### Method 3: Add to pattern_configs.py (For Reusable Patterns)

If you want your pattern to be available as a standard option:

```python
# In config/pattern_configs.py

def create_my_pattern(
    enabled: bool = True,
    window_size: int = 60,
    max_motifs: int = 15,
    radius: float = 4.5
) -> PatternConfig:
    """
    Create my custom pattern configuration.
    
    Operational scenario: [Describe what this pattern captures]
    
    Args:
        enabled: Enable/disable pattern
        window_size: Window size in minutes
        max_motifs: Maximum motifs to discover
        radius: Distance threshold
        
    Returns:
        PatternConfig instance
    """
    return PatternConfig(
        name='my_pattern',
        type='constraint',
        enabled=enabled,
        window_size=window_size,
        max_motifs=max_motifs,
        radius=radius,
        constraints={
            'Feature1': {'type': 'stable', 'max_cv': 0.01},
            'Feature2': {'type': 'varying', 'min_cv': 0.001}
        },
        save_analysis=True,
        save_plots=True
    )
```

### Complete Example: Adding a "Stable Pressure" Pattern

```python
# Step 1: Define the pattern
from data_preparation.config.pattern_configs import create_custom_pattern

stable_pressure_pattern = create_custom_pattern(
    name='stable_pressure',
    constraints={
        # Pressure is tightly controlled
        'PressureHC': {
            'type': 'stable',
            'max_cv': 0.003  # 0.3% - very tight control
        },
        # Density is also stable
        'DensityHC': {
            'type': 'stable',
            'max_cv': 0.005  # 0.5%
        },
        # But MVs are varying
        'Ore': {
            'type': 'varying',
            'min_cv': 0.0008
        },
        'WaterMill': {
            'type': 'varying',
            'min_cv': 0.0015
        }
    },
    window_size=60,
    max_motifs=10,
    radius=4.5
)

# Step 2: Use in pipeline
from data_preparation import DataPreparationPipeline, PipelineConfig
from data_preparation.config.pattern_configs import get_default_patterns

# Get default patterns and add yours
patterns = get_default_patterns()
patterns.append(stable_pressure_pattern)

# Create config
config = PipelineConfig.create_default(
    mill_number=8,
    start_date="2025-09-01",
    end_date="2025-11-03",
    patterns=patterns
)

# Run
pipeline = DataPreparationPipeline(config)
pipeline.run()

# Step 3: Check outputs
# - stable_pressure_analysis.csv
# - stable_pressure_analysis.png
# - Motifs included in segmented_motifs_all_08.csv
```

### Pattern Design Guidelines

1. **Choose Meaningful Constraints**
   - Reflect actual operational scenarios
   - Use domain knowledge
   - Consider physical relationships

2. **Set Appropriate Thresholds**
   - Stable: CV typically 0.005 - 0.01 (0.5% - 1%)
   - Varying: CV typically 0.0008 - 0.002 (0.08% - 0.2%)
   - Adjust based on feature characteristics

3. **Balance Strictness**
   - Too strict → No motifs found
   - Too loose → Too many low-quality motifs
   - Start conservative, relax if needed

4. **Test and Iterate**
   - Run with small max_motifs first
   - Check analysis outputs
   - Adjust constraints based on results

---

## Advanced Topics

### 1. Understanding Matrix Profiles

**What is a Matrix Profile?**
- For each subsequence (window), stores distance to its nearest neighbor
- Computed using STUMPY library
- Enables fast motif discovery

**Why Multivariate (mstump)?**
- Considers all features simultaneously
- Captures relationships between features
- More robust than univariate

**Distance Calculation:**
```python
# For two windows A and B with features [f1, f2, f3]
distance = sqrt(mean((A - B)^2))
```

### 2. Constraint Tuning

**Finding Optimal CV Thresholds:**

```python
# Analyze feature variability
import pandas as pd
import numpy as np

def analyze_feature_variability(df, features, window_size=60):
    """Analyze CV distribution for features."""
    results = []
    
    for i in range(0, len(df) - window_size, window_size):
        window = df.iloc[i:i+window_size]
        
        for feature in features:
            data = window[feature].values
            cv = np.std(data) / np.mean(data)
            results.append({
                'feature': feature,
                'cv': cv
            })
    
    results_df = pd.DataFrame(results)
    
    # Print statistics
    for feature in features:
        feature_cvs = results_df[results_df['feature'] == feature]['cv']
        print(f"\n{feature}:")
        print(f"  Mean CV: {feature_cvs.mean():.4f}")
        print(f"  Median CV: {feature_cvs.median():.4f}")
        print(f"  10th percentile: {feature_cvs.quantile(0.1):.4f}")
        print(f"  90th percentile: {feature_cvs.quantile(0.9):.4f}")
    
    return results_df

# Use it
features = ['Ore', 'WaterMill', 'WaterZumpf', 'DensityHC', 'PressureHC']
cv_analysis = analyze_feature_variability(df, features)

# Set thresholds based on percentiles
# Stable: Use 10th-20th percentile as max_cv
# Varying: Use 70th-80th percentile as min_cv
```

### 3. Performance Optimization

**For Large Datasets:**

```python
# 1. Reduce window_size
pattern.window_size = 30  # Instead of 60

# 2. Reduce max_motifs
pattern.max_motifs = 10  # Instead of 20

# 3. Increase radius (less strict)
pattern.radius = 5.0  # Instead of 4.5

# 4. Use data subsampling for testing
df_sample = df.sample(n=10000)  # Test on subset first
```

### 4. Debugging Patterns

**Enable Detailed Logging:**

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

**Check Constraint Satisfaction:**

```python
# Add to your script
def check_constraint_coverage(df, pattern, sample_size=1000):
    """Check how many windows satisfy constraints."""
    
    satisfied = 0
    for i in range(0, min(len(df) - pattern.window_size, sample_size)):
        if pattern._check_constraints(df, i):
            satisfied += 1
    
    coverage = satisfied / sample_size * 100
    print(f"Constraint coverage: {coverage:.1f}%")
    
    if coverage < 1:
        print("⚠️  Very few windows satisfy constraints - consider relaxing")
    elif coverage > 50:
        print("⚠️  Too many windows satisfy constraints - consider tightening")
    else:
        print("✅ Constraint coverage looks good")
    
    return coverage
```

---

## Troubleshooting

### Problem 1: No Motifs Discovered

**Symptoms:**
```
✓ Pattern discovery complete:
  Total motifs: 0
  Total instances: 0
```

**Solutions:**

1. **Check Constraint Coverage**
   ```python
   # Are any windows satisfying constraints?
   check_constraint_coverage(df, pattern)
   ```

2. **Relax Constraints**
   ```python
   # Increase max_cv for stable features
   'WaterZumpf': {'type': 'stable', 'max_cv': 0.02}  # Was 0.01
   
   # Decrease min_cv for varying features
   'Ore': {'type': 'varying', 'min_cv': 0.0005}  # Was 0.0008
   ```

3. **Increase Radius**
   ```python
   pattern.radius = 6.0  # Was 4.5
   ```

4. **Check Data Quality**
   ```python
   # Are there NaN values?
   print(df[features].isna().sum())
   
   # Is data filtered too aggressively?
   print(f"Data rows: {len(df)}")
   ```

### Problem 2: Too Many Motifs

**Symptoms:**
- Motifs have very high instance counts
- Low-quality patterns

**Solutions:**

1. **Tighten Constraints**
   ```python
   # Decrease max_cv for stable
   'WaterZumpf': {'type': 'stable', 'max_cv': 0.005}  # Was 0.01
   
   # Increase min_cv for varying
   'Ore': {'type': 'varying', 'min_cv': 0.001}  # Was 0.0008
   ```

2. **Decrease Radius**
   ```python
   pattern.radius = 3.5  # Was 4.5
   ```

3. **Reduce max_instances_per_motif**
   ```python
   pattern.max_instances_per_motif = 15  # Was 20
   ```

### Problem 3: Pattern Not Registered

**Symptoms:**
```
ValueError: Pattern type 'constraint' not registered. Available: []
```

**Solutions:**

1. **Import patterns module**
   ```python
   # In pipeline.py or run.py
   import patterns  # Triggers registration
   ```

2. **Check decorator**
   ```python
   # In constraint_pattern.py
   @PatternRegistry.register('constraint')  # Must be present
   class ConstraintPattern(BasePattern):
       ...
   ```

### Problem 4: Slow Performance

**Symptoms:**
- Matrix profile computation takes too long
- Pattern discovery hangs

**Solutions:**

1. **Reduce Window Size**
   ```python
   window_size = 30  # Instead of 60
   ```

2. **Reduce Data Size**
   ```python
   # Use shorter date range
   start_date = "2025-10-01"  # Instead of "2025-09-01"
   ```

3. **Use GPU (if available)**
   ```python
   # STUMPY can use GPU
   import stumpy
   stumpy.config.STUMPY_EXCL_ZONE_DENOM = 4  # Optimize
   ```

### Problem 5: Inconsistent Results

**Symptoms:**
- Different motifs each run
- Unstable patterns

**Solutions:**

1. **Set Random Seed**
   ```python
   import numpy as np
   np.random.seed(42)
   ```

2. **Check Data Consistency**
   ```python
   # Ensure same data each time
   df = df.sort_values('TimeStamp').reset_index(drop=True)
   ```

3. **Increase min_instances**
   - Require more instances per motif for stability

---

## Summary

### Key Takeaways

1. **Universal System**: One class handles all constraint patterns
2. **Configuration-Driven**: Add patterns without coding
3. **Flexible**: Easy to create custom patterns
4. **Efficient**: 70% code reduction vs old system
5. **Extensible**: Pattern registry enables dynamic discovery

### Quick Reference

**Create Custom Pattern:**
```python
from data_preparation.config.pattern_configs import create_custom_pattern

pattern = create_custom_pattern(
    name='my_pattern',
    constraints={
        'Feature1': {'type': 'stable', 'max_cv': 0.01},
        'Feature2': {'type': 'varying', 'min_cv': 0.001}
    }
)
```

**Use in Pipeline:**
```python
config = PipelineConfig.create_default(8, "2025-09-01", "2025-11-03", [pattern])
pipeline = DataPreparationPipeline(config)
pipeline.run()
```

**Check Outputs:**
- `{pattern}_analysis.csv` - Analysis results
- `{pattern}_analysis.png` - Visualization
- `segmented_motifs_all_08.csv` - Combined data

### Further Reading

- `README.md` - User guide
- `MIGRATION_GUIDE.md` - Migration from old system
- `QUICK_REFERENCE.md` - Quick reference card
- `example_usage.py` - Usage examples

---

**Document Version:** 1.0  
**Last Updated:** November 6, 2025  
**Author:** Data Science Team - Ball Mill Optimization Project
