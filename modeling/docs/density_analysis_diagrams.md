# Density Analysis Module - Visual Diagrams

## 1. Overall Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                               │
│  DataFrame with: WaterZumpf, Ore, WaterMill, DensityHC, Time   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: PREPARE TIME SERIES                         │
│  • Extract 3 features: WaterZumpf, Ore, WaterMill              │
│  • Z-score normalize each: (x - μ) / σ                         │
│  • Shape: (3, n_samples)                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 2: COMPUTE MATRIX PROFILE                         │
│  • Use STUMPY's mstump (multivariate)                           │
│  • Find nearest neighbor distance for each window               │
│  • Aggregate across dimensions using RMS                        │
│  • Output: distance array (n_windows,)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 3: ITERATIVE MOTIF DISCOVERY                      │
│  Loop for each motif (up to max_motifs):                        │
│    ┌──────────────────────────────────────────────────┐        │
│    │ 3a. Find Constrained Seed                        │        │
│    │  • Search for window with smallest distance      │        │
│    │  • Must pass variability constraints             │        │
│    │  • Skip already-used indices                     │        │
│    └──────────────┬───────────────────────────────────┘        │
│                   ▼                                              │
│    ┌──────────────────────────────────────────────────┐        │
│    │ 3b. Find Constrained Instances                   │        │
│    │  • Compute distance profile from seed            │        │
│    │  • Filter by distance, constraints, overlap      │        │
│    │  • Extract data for valid instances              │        │
│    └──────────────┬───────────────────────────────────┘        │
│                   ▼                                              │
│    ┌──────────────────────────────────────────────────┐        │
│    │ 3c. Create Motif (if ≥2 instances)               │        │
│    │  • Create Motif object                           │        │
│    │  • Add MotifInstance for each occurrence         │        │
│    │  • Store CV metadata                             │        │
│    └──────────────┬───────────────────────────────────┘        │
│                   ▼                                              │
│    ┌──────────────────────────────────────────────────┐        │
│    │ 3d. Mark as Used                                 │        │
│    │  • Exclude ±window_size around each instance     │        │
│    │  • Prevents overlapping motifs                   │        │
│    └──────────────────────────────────────────────────┘        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OUTPUT: LIST OF MOTIFS                          │
│  Each motif contains multiple instances of similar patterns     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 4: ANALYZE DENSITY BEHAVIOR                       │
│  For each motif:                                                 │
│    • Calculate density changes (end - start)                    │
│    • Compute correlations (Ore-Density, WaterMill-Density)     │
│    • Find optimal lags (cross-correlation)                      │
│    • Aggregate statistics across instances                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL RESULTS                                 │
│  • Density change per motif                                     │
│  • Correlation strengths                                        │
│  • Time lags (process dynamics)                                 │
│  • Statistical summaries                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Variability Constraint Logic

```
                    ┌─────────────────────┐
                    │  Extract Window     │
                    │  (60 minutes)       │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
         ┌──────────┐   ┌──────────┐   ┌──────────┐
         │WaterZumpf│   │   Ore    │   │WaterMill │
         └─────┬────┘   └─────┬────┘   └─────┬────┘
               │              │              │
               ▼              ▼              ▼
         ┌──────────┐   ┌──────────┐   ┌──────────┐
         │ CV = σ/μ │   │ CV = σ/μ │   │ CV = σ/μ │
         └─────┬────┘   └─────┬────┘   └─────┬────┘
               │              │              │
               ▼              ▼              ▼
         ┌──────────┐   ┌──────────┐   ┌──────────┐
         │CV ≤ 0.01?│   │CV ≥ 0.0008?│ │CV ≥ 0.0015?│
         │  (≤1%)   │   │  (≥0.08%) │   │  (≥0.15%) │
         └─────┬────┘   └─────┬────┘   └─────┬────┘
               │              │              │
               └──────────────┼──────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  AND Operation      │
                    │  All must be TRUE   │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
         ┌─────────────┐              ┌─────────────┐
         │   PASS      │              │   FAIL      │
         │ Use window  │              │ Skip window │
         └─────────────┘              └─────────────┘
```

**Additional Relative Checks**:
```
Ore CV ≥ 1.2 × WaterZumpf CV
WaterMill CV ≥ 1.2 × WaterZumpf CV
```

---

## 3. Matrix Profile Concept

```
Time Series (normalized):
WaterZumpf: [─────────────────────────────────────]
Ore:        [─────────────────────────────────────]
WaterMill:  [─────────────────────────────────────]

Window Size = 60 minutes

                Window 1        Window 2        Window 3
                ┌──────┐        ┌──────┐        ┌──────┐
WaterZumpf:     │██████│        │      │        │      │
                └──────┘        └──────┘        └──────┘
                    │               │               │
                    └───────┬───────┴───────┬───────┘
                            ▼               ▼
                    Compare all windows to find
                    nearest neighbor distance

Matrix Profile:
Index:    0    1    2    3    4    5    6  ...
Distance: 2.1  3.4  1.8  5.2  2.9  1.5  4.1 ...
          ↑                        ↑
          Low distance =           Low distance =
          Has similar pattern      Has similar pattern
```

**Distance Calculation** (Multivariate):
```
For each dimension (WaterZumpf, Ore, WaterMill):
  d₁ = distance in dimension 1
  d₂ = distance in dimension 2
  d₃ = distance in dimension 3

Aggregated distance = √((d₁² + d₂² + d₃²) / 3)
```

---

## 4. Instance Finding Process

```
Given Seed Window at index 100:

Step 1: Compute Distance Profile
┌─────────────────────────────────────────────────────┐
│ Distance from seed to all other windows             │
│                                                      │
│  Index:  0   10   20   30  ...  90  100  110  120  │
│  Dist:  4.2  3.1  2.8  5.1 ... 1.9  0.0  2.1  4.5  │
│          ↑    ↑    ↑              ↑    ↑    ↑       │
│          │    │    │              │    │    │       │
│       Similar Similar Similar  Similar Seed Similar │
└─────────────────────────────────────────────────────┘

Step 2: Sort by Distance (closest first)
Sorted: [100, 90, 110, 20, 10, 30, 0, 120, ...]
         ↑    ↑    ↑    ↑
         Seed Close Close Close

Step 3: Filter Each Candidate
For each candidate index:
  ┌─────────────────────────────────┐
  │ 1. Check if within radius       │ → Pass/Fail
  ├─────────────────────────────────┤
  │ 2. Check variability constraints│ → Pass/Fail
  ├─────────────────────────────────┤
  │ 3. Check not already used       │ → Pass/Fail
  ├─────────────────────────────────┤
  │ 4. Check not overlapping        │ → Pass/Fail
  └─────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────┐
  │ All pass? → Add to instances    │
  │ Any fail? → Skip                │
  └─────────────────────────────────┘

Step 4: Extract Data for Valid Instances
For each valid instance:
  Extract: WaterZumpf, Ore, WaterMill, DensityHC, TimeStamp
  Store: start, end, distance, CVs, data
```

---

## 5. Lag Analysis Visualization

```
Cross-Correlation Example:

Ore:     ████░░░░░░░░░░░░░░░░░░░░░░░░
Density: ░░░░████░░░░░░░░░░░░░░░░░░░░
         ↑   ↑
         │   └─ Density peak
         └───── Ore peak
         
Lag = 4 minutes (Density follows Ore by 4 minutes)

Cross-Correlation Function:
     Correlation
        ↑
    1.0 │           ╱╲
        │          ╱  ╲
    0.5 │         ╱    ╲
        │        ╱      ╲
    0.0 ├───────┼────────┼───────→ Lag
        │      ╱│        │╲
   -0.5 │     ╱ │        │ ╲
        │    ╱  │        │  ╲
   -1.0 │   ╱   │        │   ╲
        └────────┴────────┴────────
           -10   0   +4   +10
                     ↑
                  Peak at lag=4
                  
Interpretation:
• Peak at positive lag → Response follows input
• Peak at zero lag → Instantaneous response
• Peak at negative lag → Response precedes input (unusual)
```

---

## 6. Data Flow Through Functions

```
┌──────────────────────────────────────────────────────────┐
│                    discover()                             │
│  ┌────────────────────────────────────────────────────┐  │
│  │ _prepare_time_series()                             │  │
│  │   Input:  DataFrame                                │  │
│  │   Output: Normalized array (3, n_samples)         │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                 │
│  ┌────────────────────────────────────────────────────┐  │
│  │ stumpy.mstump()                                    │  │
│  │   Input:  Normalized array                         │  │
│  │   Output: Matrix profile, indices                  │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                 │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Loop: for each motif                               │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │ _find_constrained_seed()                     │  │  │
│  │  │   Input:  mp_distances, used_indices         │  │  │
│  │  │   Calls:  _check_variability_constraints()   │  │  │
│  │  │   Output: seed_idx, seed_distance            │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                     │                               │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │ _find_constrained_instances()                │  │  │
│  │  │   Input:  seed_idx, T, df                    │  │  │
│  │  │   Calls:  stumpy.mass() for distance profile │  │  │
│  │  │   Calls:  _check_variability_constraints()   │  │  │
│  │  │   Output: list of valid instances            │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                     │                               │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │ Create Motif and MotifInstances              │  │  │
│  │  │   Store: data, metadata, CVs                 │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                 │
│  Output: List[Motif]                                     │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              analyze_density_behavior()                   │
│  ┌────────────────────────────────────────────────────┐  │
│  │ For each motif:                                    │  │
│  │   For each instance:                               │  │
│  │     • density_change = density[-1] - density[0]   │  │
│  │     • ore_corr = pearsonr(ore, density)           │  │
│  │     • watermill_corr = pearsonr(watermill, density)│ │
│  │     • ore_lag = find_optimal_lag(ore, density)    │  │
│  │     • watermill_lag = find_optimal_lag(watermill, │  │
│  │                                        density)    │  │
│  │   Aggregate:                                       │  │
│  │     • mean(density_changes)                        │  │
│  │     • mean(correlations)                           │  │
│  │     • median(lags)                                 │  │
│  └────────────────────────────────────────────────────┘  │
│  Output: List[dict] with analysis results                │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Motif Structure

```
Motif (ID=1)
├── Instance 1 (start=100, end=160, distance=0.0)
│   ├── data
│   │   ├── WaterZumpf: [50.1, 50.2, 50.1, ...]  (60 values)
│   │   ├── Ore:        [200, 210, 220, ...]     (60 values)
│   │   ├── WaterMill:  [80, 85, 90, ...]        (60 values)
│   │   ├── DensityHC:  [1.5, 1.6, 1.7, ...]     (60 values)
│   │   └── TimeStamp:  [t0, t1, t2, ...]        (60 values)
│   └── metadata
│       ├── waterzumpf_cv: 0.008
│       ├── ore_cv: 0.012
│       └── watermill_cv: 0.018
│
├── Instance 2 (start=500, end=560, distance=1.8)
│   ├── data: {...}
│   └── metadata: {...}
│
├── Instance 3 (start=890, end=950, distance=2.1)
│   ├── data: {...}
│   └── metadata: {...}
│
└── ... (more instances)
```

---

## 8. Constraint Satisfaction Example

```
Example Window (60 minutes):

WaterZumpf: [50.0, 50.1, 50.0, 50.2, 50.1, ...]
  Mean: 50.08
  Std:  0.08
  CV:   0.08/50.08 = 0.0016 (0.16%)
  ✓ PASS: 0.0016 ≤ 0.01 (stable)

Ore: [200, 205, 210, 215, 220, ...]
  Mean: 210
  Std:  7.07
  CV:   7.07/210 = 0.0337 (3.37%)
  ✓ PASS: 0.0337 ≥ 0.0008 (varying)
  ✓ PASS: 0.0337 ≥ 1.2 × 0.0016 (relative)

WaterMill: [80, 82, 85, 88, 90, ...]
  Mean: 85
  Std:  3.54
  CV:   3.54/85 = 0.0416 (4.16%)
  ✓ PASS: 0.0416 ≥ 0.0015 (varying)
  ✓ PASS: 0.0416 ≥ 1.2 × 0.0016 (relative)

RESULT: All constraints satisfied → Window is valid
```

---

## 9. Performance Characteristics

```
Computational Complexity:

┌─────────────────────────┬──────────────┬─────────────┐
│ Operation               │ Complexity   │ Bottleneck? │
├─────────────────────────┼──────────────┼─────────────┤
│ Matrix Profile (mstump) │ O(n²)        │ ✓ Yes       │
│ Distance Profile (mass) │ O(n log n)   │             │
│ Variability Check       │ O(w)         │             │
│ Instance Finding        │ O(n × m)     │             │
│ Lag Calculation         │ O(w log w)   │             │
└─────────────────────────┴──────────────┴─────────────┘

Where:
  n = number of time points
  w = window size
  m = max_motifs

Memory Usage:
  Matrix Profile: O(n)
  Time Series:    O(3n)
  Motif Storage:  O(m × instances × w × 5)
  
Typical Performance (n=10,000, w=60, m=15):
  Matrix Profile: ~30 seconds
  Motif Discovery: ~10 seconds
  Analysis:        ~2 seconds
  Total:           ~42 seconds
```

---

## 10. Decision Tree for Parameter Tuning

```
                    Start
                      │
                      ▼
            ┌─────────────────────┐
            │ Found motifs?       │
            └─────────┬───────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
        Yes                        No
         │                         │
         ▼                         ▼
┌─────────────────┐    ┌─────────────────────┐
│ Too many motifs?│    │ Relax constraints:  │
└────────┬────────┘    │ • Increase radius   │
         │             │ • Reduce CV thresh  │
    ┌────┴────┐        │ • Smaller window    │
   Yes       No        └─────────────────────┘
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│ Tighten │ │ Good! Now    │
│ • Reduce│ │ analyze:     │
│   radius│ │ • Correlations│
│ • Raise │ │ • Lags       │
│   CV    │ │ • Changes    │
│ • Larger│ └──────────────┘
│   window│
└─────────┘
```

---

These diagrams provide visual understanding of the density analysis module's operation, data flow, and decision-making processes.
