# Simplified Discontinuity Markers for Motif Segmentation

## Problem Solved

Chronologically stacked motif segments create temporal gaps and spikes that degrade ML model performance. Instead of adding 42 temporal features, we use **3 simple discontinuity markers** that keep the dataset clean.

## Solution: Minimal Feature Approach

### Output Dataset Structure

**Clean and simple:**
```
TimeStamp, segment_id, motif_id, segment_start, segment_end,
Ore, WaterMill, WaterZumpf, MotorAmp, PulpHC, DensityHC, PressureHC, mill_id,
discontinuity_score, is_segment_start, is_segment_end
```

**Total: 16 columns** (13 original + 3 markers)

### The 3 Discontinuity Markers

#### 1. `discontinuity_score` (float)
- **What it measures**: Normalized magnitude of jumps across all features
- **How it works**: 
  - Computes z-score of the change for each feature
  - Averages across all features
  - Higher values = larger discontinuity
- **Typical values**:
  - 0-1: Smooth transition
  - 1-2: Moderate change
  - >2: Significant discontinuity (likely segment boundary)

#### 2. `is_segment_start` (0 or 1)
- **What it marks**: First row of each segment
- **Purpose**: Explicitly tells the model "this is a boundary"
- **At inference**: Can be inferred from `discontinuity_score > 2.0`

#### 3. `is_segment_end` (0 or 1)
- **What it marks**: Last row of each segment
- **Purpose**: Helps model recognize upcoming transition
- **At inference**: Unknown (set to 0)

## How It Works

### Training Time (motif_mv_search.py)

```python
# For each row, compute discontinuity score
for each feature (Ore, WaterMill, etc.):
    1. Calculate: jump = current_value - previous_value
    2. Get recent history (last 10 values)
    3. Compute: z_score = jump / std(recent_history)

discontinuity_score = average(all z_scores)
```

### Example at Segment Boundary

```
Row 99 (last of segment 1):
  Ore: 161.08
  discontinuity_score: 0.82  ← smooth
  is_segment_end: 1

Row 100 (first of segment 2):
  Ore: 171.69  ← jumped by 10.6!
  discontinuity_score: 5.43  ← HIGH! Boundary detected
  is_segment_start: 1
```

### Inference Time (streaming data)

```python
# Same calculation on new data
new_data_with_markers = add_discontinuity_markers_inference(new_data, base_features)

# Predict
predictions = model.predict(new_data_with_markers[feature_columns])
```

## Advantages Over Full Temporal Features

| Aspect | Full Temporal (47 features) | Discontinuity Markers (3 features) |
|--------|----------------------------|-----------------------------------|
| **Dataset size** | 55 columns | 16 columns |
| **Complexity** | High | Low |
| **Interpretability** | Difficult | Easy |
| **Computation** | Slow | Fast |
| **Effectiveness** | Good | Good (simpler is better) |

## Model Training

The XGBoost model learns:
- **Base patterns** from the 5 raw features (Ore, WaterMill, etc.)
- **Boundary handling** from the 3 discontinuity markers
- **When to be cautious** (high discontinuity_score)
- **When to trust predictions** (low discontinuity_score)

## Feature Importance

After training, you'll likely see:
1. **High importance**: Base features (Ore, WaterMill, etc.)
2. **Medium importance**: `discontinuity_score` (helps at boundaries)
3. **Low importance**: `is_segment_start`, `is_segment_end` (binary flags)

## Implementation Files

1. **motif_mv_search.py**
   - Function: `add_discontinuity_markers()`
   - Computes the 3 markers after segment extraction

2. **xgboost.py**
   - Uses 8 features total: 5 base + 3 markers
   - Automatically detects if markers exist

3. **inference_example.py**
   - Shows how to compute markers on streaming data
   - No segment metadata needed

## Usage

### Step 1: Generate Segmented Data
```bash
python motif_mv_search.py
```
Output: `output/segmented_motifs.csv` with 16 columns

### Step 2: Train Model
```bash
python xgboost.py
```
Uses 8 features (5 base + 3 markers)

### Step 3: Inference
```python
from inference_example import add_discontinuity_markers_inference, predict_on_new_data

# New sensor readings (just 5 base features)
new_data = pd.DataFrame({...})

# Add markers
new_data_with_markers = add_discontinuity_markers_inference(new_data, base_features)

# Predict
predictions = model.predict(new_data_with_markers[feature_columns])
```

## Expected Results

- **Cleaner dataset**: 16 columns vs 55 columns
- **Better interpretability**: Easy to understand what the model is using
- **Good performance**: The 3 markers encode the key information
- **Faster training**: Fewer features = faster computation
- **Easier debugging**: Can inspect discontinuity_score to find issues

## Key Insight

**You don't need 42 features to handle discontinuities. Just 3 well-designed markers that capture:**
1. How big is the jump? (`discontinuity_score`)
2. Is this a segment start? (`is_segment_start`)
3. Is this a segment end? (`is_segment_end`)

The model learns the rest from the base features.
