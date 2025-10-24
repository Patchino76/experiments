# Density Analysis Module - Improvements & Simplifications

## Code Analysis Summary

After detailed review of `density_analysis.py`, here are identified areas for improvement:

---

## 1. Code Simplifications

### Issue 1.1: Redundant CV Calculations

**Current Code** (Lines 265-275):
```python
# Extracted in _find_constrained_instances
waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + self.window_size].values
ore_data = df['Ore'].iloc[idx:idx + self.window_size].values
watermill_data = df['WaterMill'].iloc[idx:idx + self.window_size].values
density_data = df['DensityHC'].iloc[idx:idx + self.window_size].values

# Then calculated again
instance = {
    'waterzumpf_cv': calculate_variability(waterzumpf_data),
    'ore_cv': calculate_variability(ore_data),
    'watermill_cv': calculate_variability(watermill_data),
}
```

**Problem**: CVs are calculated twice - once in `_check_variability_constraints()` and again when creating instance.

**Solution**: Cache CV values to avoid redundant computation.

**Improved Code**:
```python
def _check_variability_constraints(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Dict[str, float]]:
    """Check constraints and return CVs for reuse."""
    waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + self.window_size].values
    ore_data = df['Ore'].iloc[idx:idx + self.window_size].values
    watermill_data = df['WaterMill'].iloc[idx:idx + self.window_size].values
    
    cvs = {
        'waterzumpf_cv': calculate_variability(waterzumpf_data),
        'ore_cv': calculate_variability(ore_data),
        'watermill_cv': calculate_variability(watermill_data)
    }
    
    passes = (
        cvs['waterzumpf_cv'] <= self.waterzumpf_max_cv and
        cvs['ore_cv'] >= self.ore_min_cv and
        cvs['watermill_cv'] >= self.watermill_min_cv and
        cvs['ore_cv'] >= cvs['waterzumpf_cv'] * self.relative_variability_factor and
        cvs['watermill_cv'] >= cvs['waterzumpf_cv'] * self.relative_variability_factor
    )
    
    return passes, cvs
```

**Impact**: Reduces computation by ~50% for CV calculations.

---

### Issue 1.2: Repeated Data Extraction

**Current Code**: Data is extracted multiple times in different methods.

**Solution**: Create a helper method for data extraction.

**Improved Code**:
```python
def _extract_window_data(self, df: pd.DataFrame, idx: int) -> Dict[str, np.ndarray]:
    """Extract all required data for a window."""
    return {
        'WaterZumpf': df['WaterZumpf'].iloc[idx:idx + self.window_size].values,
        'Ore': df['Ore'].iloc[idx:idx + self.window_size].values,
        'WaterMill': df['WaterMill'].iloc[idx:idx + self.window_size].values,
        'DensityHC': df['DensityHC'].iloc[idx:idx + self.window_size].values,
        'TimeStamp': df['TimeStamp'].iloc[idx:idx + self.window_size].values
    }
```

**Impact**: DRY principle, easier maintenance.

---

### Issue 1.3: Magic Numbers

**Current Code**: Hard-coded values scattered throughout.

**Problem**:
- Line 246: `if len(valid_instances) >= 20:` - Why 20?
- Line 291: `max_lag: int = 20` - Why 20?
- Line 307: `+ 1e-10` - Why this epsilon?

**Solution**: Define as class constants or parameters.

**Improved Code**:
```python
class DensityMotifDiscovery:
    # Class constants
    MAX_INSTANCES_PER_MOTIF = 20
    MIN_INSTANCES_PER_MOTIF = 2
    NORMALIZATION_EPSILON = 1e-10
    DEFAULT_MAX_LAG = 20
    
    def __init__(self, ...):
        # Existing parameters
        self.max_instances_per_motif = max_instances_per_motif or self.MAX_INSTANCES_PER_MOTIF
```

**Impact**: Better readability, easier tuning.

---

## 2. Performance Optimizations

### Issue 2.1: Inefficient Overlap Check

**Current Code** (Line 261):
```python
if any(abs(idx - vi['start']) < self.window_size for vi in valid_instances):
    continue
```

**Problem**: O(n) check for each candidate, becomes O(n²) overall.

**Solution**: Use a set-based approach.

**Improved Code**:
```python
# In _find_constrained_instances
excluded_ranges = set()

for idx in sorted_candidates:
    # Check if in excluded range
    if idx in excluded_ranges:
        continue
    
    # ... other checks ...
    
    valid_instances.append(instance)
    
    # Add exclusion range
    for offset in range(-self.window_size, self.window_size):
        neighbor = idx + offset
        if 0 <= neighbor < n_windows:
            excluded_ranges.add(neighbor)
```

**Impact**: Reduces complexity from O(n²) to O(n).

---

### Issue 2.2: Unnecessary Array Copies

**Current Code**: Multiple array slicing creates copies.

**Solution**: Use views where possible, or extract once and reuse.

**Improved Code**:
```python
# Instead of multiple iloc calls
window_data = self._extract_window_data(df, idx)  # Extract once
waterzumpf_cv = calculate_variability(window_data['WaterZumpf'])  # Reuse
```

---

## 3. Robustness Improvements

### Issue 3.1: Missing Input Validation

**Current Code**: No validation of input DataFrame.

**Problem**: Could fail with cryptic errors if columns missing.

**Solution**: Add validation method.

**Improved Code**:
```python
def _validate_input(self, df: pd.DataFrame) -> None:
    """Validate input DataFrame has required columns."""
    required_cols = ['WaterZumpf', 'Ore', 'WaterMill', 'DensityHC', 'TimeStamp']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if len(df) < self.window_size:
        raise ValueError(f"DataFrame too short: {len(df)} < {self.window_size}")
    
    # Check for NaN values
    for col in required_cols[:-1]:  # Exclude TimeStamp
        if df[col].isna().any():
            logger.warning(f"Column '{col}' contains NaN values")

# Call in discover()
def discover(self, df: pd.DataFrame) -> List[Motif]:
    self._validate_input(df)  # Add this
    # ... rest of method
```

**Impact**: Better error messages, early failure detection.

---

### Issue 3.2: No Handling of Edge Cases

**Current Code**: Assumes data is well-behaved.

**Problems**:
- What if all windows fail constraints?
- What if matrix profile computation fails?
- What if no motifs found?

**Solution**: Add error handling and informative messages.

**Improved Code**:
```python
# In discover()
if len(self.motifs) == 0:
    logger.warning("No motifs found matching constraints. Consider:")
    logger.warning("  - Relaxing variability thresholds")
    logger.warning("  - Increasing radius")
    logger.warning("  - Checking data quality")
    
# In _find_constrained_seed()
if seed_idx is None:
    logger.debug(f"No valid seed found in iteration {motif_idx}")
    logger.debug(f"  Remaining windows: {n_windows - len(used_indices)}")
```

---

### Issue 3.3: Silent Failures in Lag Calculation

**Current Code** (Line 303-304):
```python
if len(x) != len(y) or len(x) < 2:
    return 0
```

**Problem**: Returns 0 for invalid input, which could be confused with "no lag".

**Solution**: Return None or raise exception.

**Improved Code**:
```python
def find_optimal_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 20) -> Optional[int]:
    if len(x) != len(y):
        logger.warning(f"Length mismatch: x={len(x)}, y={len(y)}")
        return None
    
    if len(x) < 2:
        logger.warning("Insufficient data for lag calculation")
        return None
    
    # ... rest of function
```

**Impact**: Clearer semantics, easier debugging.

---

## 4. Code Organization

### Issue 4.1: Large Method

**Current Code**: `_find_constrained_instances()` is 73 lines (Lines 216-288).

**Problem**: Does too many things - computes distance profile, filters candidates, extracts data.

**Solution**: Split into smaller methods.

**Improved Code**:
```python
def _find_constrained_instances(self, df, T, seed_idx, n_windows, used_indices, mp_distances):
    """Find instances similar to seed."""
    # Step 1: Compute distance profile
    aggregated_profile = self._compute_distance_profile(T, seed_idx, n_windows)
    
    # Step 2: Find valid candidates
    valid_instances = self._filter_candidates(
        df, aggregated_profile, n_windows, used_indices
    )
    
    return valid_instances

def _compute_distance_profile(self, T, seed_idx, n_windows):
    """Compute distance from seed to all windows."""
    # Lines 227-239
    ...

def _filter_candidates(self, df, aggregated_profile, n_windows, used_indices):
    """Filter candidates based on constraints."""
    # Lines 242-286
    ...
```

**Impact**: Better readability, easier testing.

---

### Issue 4.2: Mixed Concerns

**Current Code**: `analyze_density_behavior()` computes and logs results.

**Solution**: Separate computation from presentation.

**Improved Code**:
```python
def analyze_density_behavior(motifs: List[Motif], verbose: bool = True) -> List[dict]:
    """Analyze density behavior."""
    analysis_results = []
    
    for motif in motifs:
        result = _analyze_single_motif(motif)
        analysis_results.append(result)
        
        if verbose:
            _log_motif_analysis(result)
    
    return analysis_results

def _analyze_single_motif(motif: Motif) -> dict:
    """Analyze a single motif."""
    # Computation only
    ...

def _log_motif_analysis(result: dict) -> None:
    """Log analysis results."""
    # Logging only
    ...
```

---

## 5. Additional Features

### Feature 5.1: Progress Reporting

**Current Code**: No progress indication for long-running operations.

**Solution**: Add progress logging or tqdm integration.

**Improved Code**:
```python
from tqdm import tqdm

# In discover()
for motif_idx in tqdm(range(self.max_motifs), desc="Discovering motifs"):
    # ... existing code
```

---

### Feature 5.2: Configurable Constraints

**Current Code**: Constraints are fixed at initialization.

**Solution**: Allow dynamic constraint adjustment.

**Improved Code**:
```python
def update_constraints(self, **kwargs):
    """Update variability constraints."""
    for key, value in kwargs.items():
        if hasattr(self, key):
            setattr(self, key, value)
            logger.info(f"Updated {key} = {value}")

# Usage
discovery.update_constraints(waterzumpf_max_cv=0.02, ore_min_cv=0.001)
```

---

### Feature 5.3: Export Functionality

**Current Code**: No built-in export of results.

**Solution**: Add export methods.

**Improved Code**:
```python
def export_motifs_to_csv(self, motifs: List[Motif], filepath: str):
    """Export motifs to CSV."""
    rows = []
    for motif in motifs:
        for instance in motif.instances:
            rows.append({
                'motif_id': motif.motif_id,
                'start': instance.start,
                'end': instance.end,
                'distance': instance.distance,
                'waterzumpf_cv': instance.metadata['waterzumpf_cv'],
                'ore_cv': instance.metadata['ore_cv'],
                'watermill_cv': instance.metadata['watermill_cv']
            })
    
    pd.DataFrame(rows).to_csv(filepath, index=False)
```

---

## 6. Testing Improvements

### Issue 6.1: No Unit Tests

**Solution**: Add comprehensive tests.

**Test Structure**:
```python
# test_density_analysis.py

def test_calculate_variability():
    # Test normal case
    data = np.array([1, 2, 3, 4, 5])
    cv = calculate_variability(data)
    assert cv > 0
    
    # Test zero mean
    data = np.array([0, 0, 0])
    cv = calculate_variability(data)
    assert cv == 0
    
    # Test constant data
    data = np.array([5, 5, 5, 5])
    cv = calculate_variability(data)
    assert cv == 0

def test_check_variability_constraints():
    # Create test data
    df = create_test_dataframe()
    discovery = DensityMotifDiscovery()
    
    # Test passing case
    passes, cvs = discovery._check_variability_constraints(df, 0)
    assert passes == True
    
    # Test failing case
    # ...
```

---

## 7. Documentation Improvements

### Issue 7.1: Missing Docstring Examples

**Solution**: Add usage examples to docstrings.

**Improved Code**:
```python
def discover(self, df: pd.DataFrame) -> List[Motif]:
    """
    Discover constrained motifs.
    
    Args:
        df: DataFrame with required columns
        
    Returns:
        List of discovered motifs
        
    Example:
        >>> discovery = DensityMotifDiscovery(window_size=60)
        >>> motifs = discovery.discover(df)
        >>> print(f"Found {len(motifs)} motifs")
        >>> for motif in motifs:
        ...     print(f"Motif {motif.motif_id}: {len(motif.instances)} instances")
    """
```

---

## 8. Summary of Recommended Changes

### Priority 1 (High Impact, Low Effort)
1. ✅ Add input validation
2. ✅ Cache CV calculations
3. ✅ Define magic numbers as constants
4. ✅ Add progress reporting

### Priority 2 (Medium Impact, Medium Effort)
1. ✅ Optimize overlap checking
2. ✅ Split large methods
3. ✅ Add error handling
4. ✅ Improve logging

### Priority 3 (Nice to Have)
1. ✅ Add export functionality
2. ✅ Add unit tests
3. ✅ Improve docstrings
4. ✅ Add configuration flexibility

---

## 9. Estimated Impact

| Improvement | Code Reduction | Performance Gain | Maintainability |
|-------------|----------------|------------------|-----------------|
| Cache CVs | -20 lines | +50% (CV calc) | ⭐⭐⭐ |
| Optimize overlap | -5 lines | +30% (large datasets) | ⭐⭐⭐ |
| Split methods | +30 lines | 0% | ⭐⭐⭐⭐⭐ |
| Add validation | +20 lines | 0% | ⭐⭐⭐⭐ |
| Constants | +10 lines | 0% | ⭐⭐⭐⭐ |
| **Total** | **+35 lines** | **+80% overall** | **Much better** |

---

## 10. Refactored Code Structure

**Proposed new structure**:

```
density_analysis.py
├── calculate_variability()
├── find_optimal_lag()
├── analyze_density_behavior()
│   ├── _analyze_single_motif()
│   └── _log_motif_analysis()
└── DensityMotifDiscovery
    ├── Constants (class-level)
    ├── __init__()
    ├── discover()
    │   ├── _validate_input()
    │   ├── _prepare_time_series()
    │   ├── _compute_matrix_profile()
    │   └── _discover_motifs_iteratively()
    ├── Constraint checking
    │   ├── _check_variability_constraints()
    │   └── _extract_window_data()
    ├── Seed finding
    │   └── _find_constrained_seed()
    ├── Instance finding
    │   ├── _find_constrained_instances()
    │   ├── _compute_distance_profile()
    │   └── _filter_candidates()
    └── Utilities
        ├── update_constraints()
        └── export_motifs_to_csv()
```

**Benefits**:
- Clear separation of concerns
- Easier to test individual components
- Better code reuse
- Improved readability

---

## Conclusion

The current code is **functional and well-structured**, but has room for improvement in:
1. **Performance**: CV caching, overlap optimization
2. **Robustness**: Input validation, error handling
3. **Maintainability**: Split large methods, remove magic numbers
4. **Usability**: Progress reporting, export functionality

**Recommended approach**: Implement Priority 1 changes first (high impact, low effort), then gradually add Priority 2 and 3 improvements as needed.
