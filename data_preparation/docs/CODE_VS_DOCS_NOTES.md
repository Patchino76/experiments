# Documentation vs Implementation Notes

## Overview

This document explains the differences between the documentation and the actual implementation in `patterns/constraint_pattern.py`.

## Why Differences Exist

The documentation was initially written with **simplified examples** for clarity and teaching purposes. However, the **actual implementation** is more sophisticated and handles edge cases, performance optimizations, and additional features.

## Key Differences

### 1. `__init__` Method

**Documentation (Initial - Simplified):**
```python
def __init__(self, name, constraints, window_size, max_motifs, 
             radius, max_instances_per_motif, **kwargs):
    super().__init__(name, window_size, max_motifs, radius)
    self.constraints = constraints
    self.max_instances_per_motif = max_instances_per_motif
```

**Actual Implementation:**
```python
def __init__(self, name: str, config: dict):
    super().__init__(name, config)
    
    # Parse constraints from config
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
```

**Why Different:**
- Real implementation uses a single `config` dict for all parameters
- Extracts features list automatically
- Includes `relative_variability_factor` for constraint checking
- More robust with `.get()` and default values

### 2. `_check_constraints` Method

**Documentation (Initial - Simplified):**
```python
def _check_constraints(self, df, start_idx):
    window = df.iloc[start_idx:start_idx + self.window_size]
    
    for feature, constraint in self.constraints.items():
        data = window[feature].values
        cv = np.std(data) / np.mean(data)  # Simple CV calculation
        
        if constraint['type'] == 'stable':
            if cv > constraint['max_cv']:
                return False
        elif constraint['type'] == 'varying':
            if cv < constraint['min_cv']:
                return False
    
    return True
```

**Actual Implementation:**
```python
def _check_constraints(self, df: pd.DataFrame, idx: int) -> bool:
    cvs = {}
    
    # Calculate CV for all features FIRST
    for feature in self.features:
        data = df[feature].iloc[idx:idx + self.window_size].values
        cvs[feature] = self.calculate_variability(data)  # Uses base class method
    
    # Check stable features
    for feature in self.stable_features:
        constraint = self.constraints[feature]
        max_cv = constraint.get('max_cv', 0.01)  # Default value
        
        if cvs[feature] > max_cv:
            return False
    
    # Check varying features
    for feature in self.varying_features:
        constraint = self.constraints[feature]
        min_cv = constraint.get('min_cv', 0.0008)  # Default value
        
        if cvs[feature] < min_cv:
            return False
    
    # Check RELATIVE variability
    if self.stable_features and self.varying_features:
        max_stable_cv = max(cvs[f] for f in self.stable_features)
        min_varying_cv = min(cvs[f] for f in self.varying_features)
        
        if min_varying_cv < max_stable_cv * self.relative_variability_factor:
            return False  # Not different enough
    
    return True
```

**Why Different:**
- **Pre-calculates all CVs**: More efficient, avoids redundant calculations
- **Uses `calculate_variability()` method**: Handles edge cases (NaN, division by zero)
- **Default values with `.get()`**: More robust, won't crash if config incomplete
- **Relative variability check**: Ensures meaningful distinction between stable/varying
- **Separate loops**: Clearer logic, easier to debug

### 3. Method Names

**Documentation:**
- `_find_seed_points()`
- `_find_similar_instances()`

**Actual Implementation:**
- `_find_constrained_seed()` - Finds ONE best seed
- `_find_constrained_instances()` - Finds instances for that seed

**Why Different:**
- Real implementation finds seeds one at a time in a loop
- More memory efficient
- Allows early stopping when max_motifs reached

### 4. `discover` Method

**Documentation (Simplified):**
```python
def discover(self, df):
    # 1. Prepare time series
    # 2. Compute matrix profile
    # 3. Find seed points
    # 4. Find similar instances
    # 5. Create motifs
```

**Actual Implementation:**
```python
def discover(self, df: pd.DataFrame) -> List[Motif]:
    # 1. Validate data
    if not self.validate_data(df, self.features):
        return []
    
    # 2. Prepare time series
    T = self.prepare_time_series(df, self.features)
    
    # 3. Compute matrix profile
    matrix_profile, profile_indices = stumpy.mstump(T, m=self.window_size)
    mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))
    
    # 4. Loop to find motifs
    self.motifs = []
    used_indices = set()
    
    for motif_idx in range(self.max_motifs):
        # Find ONE constrained seed
        seed_idx, seed_distance = self._find_constrained_seed(...)
        
        if seed_idx is None:
            break  # No more valid seeds
        
        # Find instances for this seed
        instances = self._find_constrained_instances(...)
        
        if len(instances) >= 2:
            motif = self._create_motif(...)
            self.motifs.append(motif)
            used_indices.update(...)
    
    return self.motifs
```

**Why Different:**
- Real implementation has validation step
- Computes aggregated distances from matrix profile
- Uses iterative loop instead of batch processing
- Tracks used indices to avoid overlap
- Has early stopping logic

## Status: UPDATED ✅

The documentation in `CONSTRAINT_PATTERNS_GUIDE.md` has been **updated** to reflect the actual implementation:

- ✅ `_check_constraints` - Now shows full implementation
- ✅ `__init__` - Now shows actual signature and logic
- ✅ Added note at top about reflecting actual code
- ✅ Added "Key Implementation Details" section
- ✅ Explained relative variability factor

## Remaining Simplified Sections

Some sections remain simplified for pedagogical purposes:
- High-level flow diagrams (intentionally simplified)
- Example usage (shows user-facing API, not internals)
- Troubleshooting (focuses on symptoms, not internal details)

These are **intentionally** simplified because they're meant to help users understand concepts, not implementation details.

## Recommendation

When reading the documentation:
1. **Core Concepts section** - Shows actual implementation
2. **Code Structure section** - Shows actual class structure
3. **How It Works section** - Shows simplified flow for understanding
4. **How to Add New Patterns** - Shows user-facing API (correct)

For implementation details, always refer to:
- `patterns/constraint_pattern.py` - Source of truth
- This document - Explains differences

## Version History

- **v1.0** (Initial) - Simplified examples for clarity
- **v1.1** (Current) - Updated to match actual implementation in core sections

---

**Last Updated:** November 6, 2025  
**Reflects Code Version:** patterns/constraint_pattern.py (current)
