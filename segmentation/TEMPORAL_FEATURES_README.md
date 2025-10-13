# Temporal Features Implementation for Motif Segmentation

## Problem Statement

When stacking motif segments chronologically, we create **temporal discontinuities** (gaps and spikes) that degrade ML model performance. The test set shows poor predictions because the model struggles with these artificial boundaries.

## Solution: Temporal Feature Engineering

We encode temporal patterns and discontinuities directly into features that can be computed from raw data at both training and inference time.

## Implementation Overview

### 1. **motif_mv_search.py** - Feature Generation

The `add_temporal_features()` function creates 5 types of features:

#### A. Lag Features (Previous Values)
```python
Ore_lag1, Ore_lag2, Ore_lag3
WaterMill_lag1, WaterMill_lag2, WaterMill_lag3
...
```
- Captures recent history
- Provides temporal context

#### B. Rate of Change (First Derivative)
```python
Ore_diff1 = Ore - Ore_lag1
Ore_diff2 = Ore_lag1 - Ore_lag2
...
```
- **Detects sudden jumps** at segment boundaries
- Large values indicate discontinuities

#### C. Rolling Statistics (Smoothing)
```python
Ore_rolling_mean_5 = rolling_mean(Ore, window=5)
Ore_rolling_std_5 = rolling_std(Ore, window=5)
...
```
- Smooths noise
- Provides trend information

#### D. Acceleration (Second Derivative)
```python
Ore_accel = Ore_diff1 - Ore_diff2
...
```
- Detects when rate of change is changing
- Identifies pattern shifts

#### E. Segment Boundary Indicators
```python
is_segment_start = 1 if new segment, else 0
is_segment_end = 1 if segment ending, else 0
```
- Explicitly marks discontinuities
- Helps model recognize boundaries

### 2. **xgboost.py** - Model Training

The script now:
1. Loads pre-computed temporal features from `segmented_motifs.csv`
2. Uses all features (base + temporal) for training
3. Learns to handle discontinuities through the temporal features

### 3. **inference_example.py** - Production Use

Shows how to apply the same feature engineering to new streaming data:
```python
# At inference time:
new_data = get_sensor_readings()  # Just raw features
new_data_with_features = add_temporal_features(new_data, base_features)
predictions = model.predict(new_data_with_features)
```

## Key Benefits

### ✅ Feature Parity
- **Training features = Inference features**
- No segment metadata needed at inference time
- All features computable from raw sensor data

### ✅ Discontinuity Handling
- Model learns that large `diff1` values indicate boundaries
- Boundary indicators explicitly mark transitions
- Rolling statistics smooth out noise

### ✅ Improved Performance
- Temporal context helps predictions
- Model understands when to be cautious (at boundaries)
- Better generalization to unseen patterns

## Feature Count

For 5 base features (Ore, WaterMill, WaterZumpf, PulpHC, PressureHC):
- Base features: 5
- Lag features: 5 × 3 = 15
- Diff features: 5 × 2 = 10
- Rolling features: 5 × 2 = 10
- Acceleration: 5 × 1 = 5
- Boundary indicators: 2
- **Total: 47 features**

## Usage

### Step 1: Generate Segmented Data with Temporal Features
```bash
python motif_mv_search.py
```
This creates `output/segmented_motifs.csv` with all temporal features.

### Step 2: Train XGBoost Model
```bash
python xgboost.py
```
This trains on the enriched feature set.

### Step 3: Inference on New Data
```python
from inference_example import add_temporal_features_inference, predict_on_new_data

# Get new sensor readings
new_data = pd.DataFrame({...})  # Just raw features

# Add temporal features
new_data_enriched = add_temporal_features_inference(new_data, base_features)

# Predict
predictions = model.predict(new_data_enriched[feature_columns])
```

## How It Solves the Problem

### Before (Without Temporal Features)
```
Segment 1 ends: DensityHC = 1650
Segment 2 starts: DensityHC = 1720  ← 70-point jump!
Model sees: [Ore=170, Water=240, ...] → predicts poorly
```

### After (With Temporal Features)
```
Segment 1 ends: DensityHC = 1650
Segment 2 starts: DensityHC = 1720
Model sees: 
  - Ore = 175
  - DensityHC_diff1 = 70  ← LARGE! Boundary detected
  - is_segment_start = 1   ← Explicit boundary marker
  - Ore_rolling_mean_5 = 172  ← Smoothed context
  
Model learns: "Large diff + boundary flag → be cautious, adjust prediction"
```

## Expected Results

- **Better test metrics**: R² should improve significantly
- **Smoother predictions**: Fewer spikes at segment boundaries
- **Better generalization**: Model handles new motif patterns better

## Notes

- Temporal features are computed **after** stacking segments
- Features use pandas `.shift()` which respects row order
- NaN values from shifts are filled with backward/forward fill
- All features are inference-ready (no training-only metadata)
