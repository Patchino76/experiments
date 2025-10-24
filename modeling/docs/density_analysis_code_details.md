# Density Analysis Module - Code Details

## Function 1: `calculate_variability()`

**Location**: Lines 21-35

**Purpose**: Calculate Coefficient of Variation (CV) for a time series

**Code**:
```python
def calculate_variability(data: np.ndarray) -> float:
    std = np.std(data)
    mean = np.mean(data)
    if mean == 0:
        return 0
    return std / abs(mean)
```

**Explanation**:
- **CV Formula**: CV = σ / |μ|
- **Why CV?**: Unlike standard deviation, CV is dimensionless and scale-independent
  - Allows comparison: "Is Ore more variable than WaterZumpf?"
  - A CV of 0.01 = 1% variability regardless of units
- **Zero handling**: Returns 0 if mean is zero to avoid division by zero
- **Absolute value**: Uses |mean| to handle negative values correctly

**Example**:
```python
# WaterZumpf: mean=100, std=0.5 → CV = 0.005 (0.5% - very stable)
# Ore: mean=200, std=2.0 → CV = 0.01 (1% - more variable)
```

---

## Class: `DensityMotifDiscovery`

### Constructor: `__init__()`

**Location**: Lines 45-73

**Parameters Explained**:

1. **`window_size=60`**: 
   - Pattern length in minutes
   - 60 minutes = 1 hour of operation
   - Longer windows capture slower dynamics
   - Shorter windows find rapid changes

2. **`max_motifs=15`**:
   - Maximum number of distinct motif groups
   - Prevents finding too many similar patterns
   - Balance: enough diversity, not overwhelming

3. **`radius=4.5`**:
   - Distance threshold for similarity
   - Calculated on normalized data (z-scores)
   - Smaller = stricter matching
   - Larger = more lenient matching
   - 4.5 is empirically tuned for this application

4. **`waterzumpf_max_cv=0.01`**:
   - WaterZumpf must have CV ≤ 1%
   - Ensures sump water is held constant
   - Represents controlled variable

5. **`ore_min_cv=0.0008`**:
   - Ore must have CV ≥ 0.08%
   - Ensures ore feed is being manipulated
   - Represents manipulated variable

6. **`watermill_min_cv=0.0015`**:
   - WaterMill must have CV ≥ 0.15%
   - Ensures mill water is being manipulated
   - Higher threshold than Ore (water changes faster)

7. **`relative_variability_factor=1.2`**:
   - Ore/WaterMill must be 1.2× more variable than WaterZumpf
   - Ensures clear distinction between controlled and manipulated
   - Prevents false positives from noise

**Storage**:
```python
self.motifs: List[Motif] = []  # Discovered motifs stored here
```

---

### Method: `discover()`

**Location**: Lines 77-157

**Purpose**: Main discovery algorithm - finds all constrained motifs

**Step-by-Step Breakdown**:

#### Step 1: Prepare Time Series (Line 95)
```python
T = self._prepare_time_series(df, ['WaterZumpf', 'Ore', 'WaterMill'])
```
- Extracts 3 features from DataFrame
- Z-score normalizes each: `(x - μ) / σ`
- Returns shape: `(3, n_samples)` - 3 dimensions × time points
- **Why normalize?** So all features contribute equally to distance

#### Step 2: Compute Matrix Profile (Lines 98-100)
```python
matrix_profile, profile_indices = stumpy.mstump(T, m=self.window_size)
mp_distances = np.sqrt(np.mean(matrix_profile**2, axis=0))
```

**What is Matrix Profile?**
- For each window, finds distance to its nearest neighbor
- `mstump` = Multivariate STUMP (Scalable Time series Anytime Matrix Profile)
- Efficient: O(n²) instead of naive O(n³)

**Distance Aggregation**:
- `matrix_profile` shape: `(3, n_windows)` - distance per dimension
- Aggregates using RMS: `√(mean(d₁² + d₂² + d₃²))`
- Single distance value per window

#### Step 3: Iterative Motif Discovery (Lines 108-151)

**Loop Structure**:
```python
for motif_idx in range(self.max_motifs):
    # Find seed
    # Find instances
    # Create motif if valid
    # Mark as used
```

**3a. Find Constrained Seed** (Lines 110-115)
```python
seed_idx, seed_distance = self._find_constrained_seed(
    df, mp_distances, used_indices, n_windows
)
```
- Searches for window with smallest distance
- **Must pass variability constraints**
- Skips already-used indices
- Breaks if no valid seed or distance > radius

**3b. Find Constrained Instances** (Lines 118-120)
```python
valid_instances = self._find_constrained_instances(
    df, T, seed_idx, n_windows, used_indices, mp_distances
)
```
- Finds all windows similar to seed
- Each must also pass variability constraints
- Returns list of valid instances with metadata

**3c. Create Motif** (Lines 122-138)
```python
if len(valid_instances) >= 2:  # Need at least 2 instances
    motif = Motif(motif_id=len(self.motifs) + 1)
    
    for inst_data in valid_instances:
        instance = MotifInstance(
            start=inst_data['start'],
            end=inst_data['end'],
            distance=inst_data['distance'],
            data=inst_data['data']
        )
        instance.add_metadata('waterzumpf_cv', inst_data['waterzumpf_cv'])
        instance.add_metadata('ore_cv', inst_data['ore_cv'])
        instance.add_metadata('watermill_cv', inst_data['watermill_cv'])
        
        motif.add_instance(instance)
    
    self.motifs.append(motif)
```
- Requires minimum 2 instances (a pattern must repeat)
- Creates `Motif` object (container for related instances)
- Creates `MotifInstance` for each occurrence
- Stores CV values as metadata for later analysis

**3d. Mark as Used** (Lines 141-151)
```python
for inst in valid_instances:
    for offset in range(-self.window_size, self.window_size):
        neighbor = inst['start'] + offset
        if 0 <= neighbor < n_windows:
            used_indices.add(neighbor)
```
- Prevents overlapping motifs
- Marks ±window_size range around each instance
- Example: If instance at index 100, marks 40-160 as used (for window_size=60)
- Ensures spatial diversity in discovered motifs

---

### Method: `_prepare_time_series()`

**Location**: Lines 159-166

**Code**:
```python
def _prepare_time_series(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    ts_list = []
    for col in features:
        ts = np.array(df[col])
        ts = (ts - np.mean(ts)) / np.std(ts)  # Z-score normalization
        ts_list.append(ts)
    return np.array(ts_list)
```

**Normalization Formula**: z = (x - μ) / σ

**Why Z-score?**
- Centers data at 0 (mean = 0)
- Scales to unit variance (std = 1)
- Makes features comparable
- Example:
  - Ore: 150-250 t/h → normalized to -2 to +2
  - WaterMill: 50-100 m³/h → normalized to -2 to +2
  - Now both contribute equally to distance

---

### Method: `_find_constrained_seed()`

**Location**: Lines 168-195

**Purpose**: Find the best seed window that passes all constraints

**Algorithm**:
```python
seed_idx = None
seed_distance = float('inf')

for i in range(n_windows):
    if i in used_indices:
        continue  # Skip used
    
    dist = mp_distances[i]
    if np.isnan(dist) or np.isinf(dist):
        continue  # Skip invalid
    
    if not self._check_variability_constraints(df, i):
        continue  # Skip if doesn't pass constraints
    
    if dist < seed_distance:
        seed_distance = dist
        seed_idx = i  # Update best seed

return seed_idx, seed_distance
```

**Key Points**:
1. Iterates through all windows
2. Filters out: used, invalid, non-conforming
3. Selects window with **smallest distance** among valid candidates
4. Returns None if no valid seed found

---

### Method: `_check_variability_constraints()`

**Location**: Lines 197-214

**Purpose**: Core filtering logic - enforces operational constraints

**Code**:
```python
def _check_variability_constraints(self, df: pd.DataFrame, idx: int) -> bool:
    # Extract window data
    waterzumpf_data = df['WaterZumpf'].iloc[idx:idx + self.window_size].values
    ore_data = df['Ore'].iloc[idx:idx + self.window_size].values
    watermill_data = df['WaterMill'].iloc[idx:idx + self.window_size].values
    
    # Calculate CVs
    waterzumpf_cv = calculate_variability(waterzumpf_data)
    ore_cv = calculate_variability(ore_data)
    watermill_cv = calculate_variability(watermill_data)
    
    # Apply ALL constraints (must pass all)
    return (
        waterzumpf_cv <= self.waterzumpf_max_cv and  # Constraint 1: WaterZumpf stable
        ore_cv >= self.ore_min_cv and                # Constraint 2: Ore varying
        watermill_cv >= self.watermill_min_cv and    # Constraint 3: WaterMill varying
        ore_cv >= waterzumpf_cv * self.relative_variability_factor and      # Constraint 4: Relative
        watermill_cv >= waterzumpf_cv * self.relative_variability_factor    # Constraint 5: Relative
    )
```

**Constraint Logic**:

| Constraint | Condition | Meaning |
|------------|-----------|---------|
| 1 | `waterzumpf_cv ≤ 0.01` | Sump water held constant (≤1% variation) |
| 2 | `ore_cv ≥ 0.0008` | Ore feed is changing (≥0.08% variation) |
| 3 | `watermill_cv ≥ 0.0015` | Mill water is changing (≥0.15% variation) |
| 4 | `ore_cv ≥ 1.2 × waterzumpf_cv` | Ore variation is significant vs. noise |
| 5 | `watermill_cv ≥ 1.2 × waterzumpf_cv` | WaterMill variation is significant vs. noise |

**All must be True** - this is an AND operation.

---

### Method: `_find_constrained_instances()`

**Location**: Lines 216-288

**Purpose**: Find all instances similar to seed that also pass constraints

**Step 1: Compute Distance Profile** (Lines 227-239)
```python
distance_components = []
for dim in range(T.shape[0]):  # For each dimension (WaterZumpf, Ore, WaterMill)
    query = T[dim, seed_idx:seed_idx + self.window_size]  # Extract seed window
    distance_profile = stumpy.mass(query, T[dim])  # Compare to all windows
    distance_components.append(distance_profile[:n_windows])

distance_components = np.array(distance_components)
aggregated_profile = np.sqrt(np.mean(distance_components**2, axis=0))  # RMS
```

**MASS Algorithm**:
- Mueen's Algorithm for Similarity Search
- Efficiently computes distance from query to all windows
- O(n log n) using FFT
- Returns distance for each possible window position

**Aggregation**:
- Computes RMS across dimensions
- Single distance value per window
- Lower distance = more similar to seed

**Step 2: Find Valid Candidates** (Lines 242-286)
```python
sorted_candidates = np.argsort(aggregated_profile)  # Sort by distance (closest first)
valid_instances = []

for idx in sorted_candidates:
    if len(valid_instances) >= 20:  # Limit instances per motif
        break
    
    # Filter 1: Check bounds and usage
    if idx >= n_windows or idx in used_indices:
        continue
    
    # Filter 2: Check distance
    dist = aggregated_profile[idx]
    if np.isnan(dist) or np.isinf(dist) or dist > self.radius:
        continue
    
    # Filter 3: Check variability constraints
    if not self._check_variability_constraints(df, idx):
        continue
    
    # Filter 4: Avoid overlapping
    if any(abs(idx - vi['start']) < self.window_size for vi in valid_instances):
        continue
    
    # Extract all data for this instance
    instance = {
        'start': idx,
        'end': idx + self.window_size,
        'distance': dist,
        'waterzumpf_cv': calculate_variability(waterzumpf_data),
        'ore_cv': calculate_variability(ore_data),
        'watermill_cv': calculate_variability(watermill_data),
        'data': {
            'WaterZumpf': waterzumpf_data,
            'Ore': ore_data,
            'WaterMill': watermill_data,
            'DensityHC': density_data,  # Response variable
            'TimeStamp': timestamp_data
        }
    }
    valid_instances.append(instance)
```

**Four-Stage Filtering**:
1. **Bounds check**: Valid index, not used
2. **Distance check**: Within radius threshold
3. **Variability check**: Passes operational constraints
4. **Overlap check**: Not too close to existing instances

**Data Extraction**:
- Stores all 5 variables for the window
- Includes CV metadata
- Includes distance for quality assessment

---

## Function 2: `find_optimal_lag()`

**Location**: Lines 291-324

**Purpose**: Find time delay that maximizes correlation between two signals

**Code Breakdown**:

```python
def find_optimal_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 20) -> int:
    # Step 1: Validation
    if len(x) != len(y) or len(x) < 2:
        return 0
    
    # Step 2: Normalize
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)
    
    # Step 3: Cross-correlation
    correlation = correlate(x_norm, y_norm, mode='same')
    
    # Step 4: Find peak within max_lag
    center = len(correlation) // 2
    search_range = slice(max(0, center - max_lag), min(len(correlation), center + max_lag + 1))
    correlation_window = correlation[search_range]
    
    peak_idx = np.argmax(np.abs(correlation_window))
    lag = peak_idx - min(max_lag, center)
    
    return int(lag)
```

**Cross-Correlation Explained**:
- Slides one signal past the other
- Computes correlation at each offset
- Peak correlation indicates optimal lag

**Example**:
```
Ore:     [100, 110, 120, 130, 140]
Density: [1.5, 1.5, 1.6, 1.7, 1.8]  (delayed response)

Cross-correlation finds: lag = 1
Meaning: Density responds 1 minute after Ore changes
```

**Why `mode='same'`?**
- Returns correlation array same length as input
- Centers the zero-lag at middle index
- Simplifies lag calculation

**Why `+1e-10` in normalization?**
- Prevents division by zero if signal is constant
- Tiny epsilon doesn't affect results

---

## Function 3: `analyze_density_behavior()`

**Location**: Lines 327-397

**Purpose**: Analyze how DensityHC responds in discovered motifs

**Code Structure**:

```python
def analyze_density_behavior(motifs: List[Motif]) -> List[dict]:
    analysis_results = []
    
    for motif in motifs:
        # Initialize collectors
        density_changes = []
        ore_density_corrs = []
        watermill_density_corrs = []
        ore_density_lags = []
        watermill_density_lags = []
        
        # Analyze each instance
        for instance in motif.instances:
            # ... compute metrics ...
        
        # Aggregate statistics
        # ... compute averages ...
        
        # Store results
        analysis_results.append({...})
    
    return analysis_results
```

**Metrics Computed Per Instance**:

1. **Density Change** (Lines 356-358):
```python
density_change = density[-1] - density[0]
```
- Net change from start to end
- Positive = density increased
- Negative = density decreased
- Units: g/L or % depending on data

2. **Correlations** (Lines 361-365):
```python
ore_corr, _ = pearsonr(ore, density)
watermill_corr, _ = pearsonr(watermill, density)
```
- Pearson correlation coefficient
- Range: -1 (perfect negative) to +1 (perfect positive)
- 0 = no linear relationship
- Measures strength and direction of relationship

3. **Lags** (Lines 368-371):
```python
ore_lag = find_optimal_lag(ore, density)
watermill_lag = find_optimal_lag(watermill, density)
```
- Time delay in minutes
- Positive lag: density follows input
- Negative lag: density leads input (unusual)

**Aggregation** (Lines 374-378):
```python
avg_density_change = np.mean(density_changes)
avg_ore_corr = np.mean(ore_density_corrs)
avg_watermill_corr = np.mean(watermill_density_corrs)
avg_ore_lag = np.median(ore_density_lags)  # Median for lags (robust)
avg_watermill_lag = np.median(watermill_density_lags)
```

**Why median for lags?**
- Lags can have outliers
- Median is robust to extreme values
- Mean would be skewed by anomalies

**Output Dictionary** (Lines 384-395):
```python
{
    'motif_id': motif.motif_id,
    'num_instances': len(motif.instances),
    'avg_density_change': avg_density_change,
    'avg_ore_density_corr': avg_ore_corr,
    'avg_watermill_density_corr': avg_watermill_corr,
    'avg_ore_lag': avg_ore_lag,
    'avg_watermill_lag': avg_watermill_lag,
    'density_changes': density_changes,  # Raw data for further analysis
    'ore_density_corrs': ore_density_corrs,
    'watermill_density_corrs': watermill_density_corrs
}
```

Stores both aggregated statistics and raw values for detailed analysis.
