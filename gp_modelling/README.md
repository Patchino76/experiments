# Gaussian Process Regression Modeling

This folder contains Gaussian Process Regression (GPR) models for ball mill optimization with **uncertainty quantification**.

## Overview

Gaussian Process models provide:
- **Uncertainty estimates** - Critical for optimization decisions
- **Better extrapolation** than XGBoost - More reliable predictions outside training range
- **Works well with sparse data** - Effective even with limited samples
- **Uncertainty propagation** - Track confidence through the cascade

## Model Architecture

### Cascade Structure

```
MV (Manipulated Variables)
  ↓
Process Models (GP)
  ↓
CV (Controlled Variables)
  ↓
Quality Model (GP)
  ↓
PSI200 (Target)
```

### Process Models (MV → CV)
- **Input**: `Ore`, `WaterMill`, `WaterZumpf`
- **Output**: `DensityHC`, `PulpHC`, `PressureHC`, `CirculativeLoad`
- **Kernel**: Matern(nu=2.5) + WhiteKernel
- **Purpose**: Predict controlled variables from manipulated variables with uncertainty

### Quality Model (CV + DV → PSI200)
- **Input**: `DensityHC`, `PulpHC`, `PressureHC`, `CirculativeLoad`, `Class_15`, `Daiki`, `FE`
- **Output**: `PSI200` (particle size)
- **Kernel**: Matern(nu=2.5) + WhiteKernel
- **Purpose**: Predict product quality with uncertainty estimates

## Kernel Choice

**Matern(nu=2.5) + WhiteKernel**
- **Matern(nu=2.5)**: Good balance between smoothness and flexibility
- **WhiteKernel**: Captures measurement noise
- **ConstantKernel**: Scales the overall variance

## Files

- `train_gp_models.py` - Main training script
- `gp_visualization.py` - Visualization functions with uncertainty bands
- `models/` - Saved GP models and scalers
- `output/` - Training results and analysis
  - `mill_gp_XX/` - Model artifacts for mill XX
  - `analysis/` - Performance metrics
  - `plots/` - Visualizations with uncertainty bands

## Usage

### Training Models

```bash
python train_gp_models.py
```

The script will:
1. Load segmented motif data from `modeling/output/`
2. Train process models (MV → CV) with uncertainty
3. Train quality model (CV + DV → PSI200) with uncertainty
4. Validate cascade with uncertainty propagation
5. Save models, scalers, and visualizations

### Configuration

Edit `train_gp_models.py` to change:
- `mill_number` - Which mill to train (6 or 8)
- `start_date` / `end_date` - Date range for data
- Kernel parameters in `_train_single_gp_model()`

## Model Outputs

### Saved Artifacts

For each model:
- `{model_name}.pkl` - Trained GP model
- `{model_name}_scaler.pkl` - Feature scaler
- `metadata.json` - Training configuration and metrics

### Visualizations

For each model:
- `{model_name}_train_predictions.png` - Training predictions with 95% confidence bands
- `{model_name}_test_predictions.png` - Test predictions with 95% confidence bands
- `{model_name}_uncertainty.png` - Uncertainty distribution analysis
- `cascade_validation.png` - Full cascade validation with uncertainty

### Metrics Tracked

- **R²** - Coefficient of determination
- **RMSE** - Root mean squared error
- **MAE** - Mean absolute error
- **Mean σ** - Average uncertainty (standard deviation)
- **Std σ** - Uncertainty variability

## Uncertainty Quantification

### What is Uncertainty?

The GP model provides a **standard deviation (σ)** for each prediction:
- **Low σ**: High confidence (close to training data)
- **High σ**: Low confidence (extrapolating, sparse data)

### 95% Confidence Interval

Predictions are shown with **±2σ bands**:
- 95% of true values should fall within this range
- Wider bands indicate more uncertainty

### Uncertainty Propagation

In cascade validation:
1. MV → CV predictions have uncertainty σ₁
2. CV → PSI200 predictions have uncertainty σ₂
3. Total uncertainty combines both sources

## Comparison with Other Models

| Model Type | Interpolation | Extrapolation | Uncertainty | Training Speed |
|------------|---------------|---------------|-------------|----------------|
| **XGBoost** | Excellent | Poor | No | Fast |
| **Polynomial** | Good | Moderate | No | Very Fast |
| **Gaussian Process** | Excellent | Good | **Yes** | Slow |

## Best Practices

### When to Use GP Models

✅ **Use GP when:**
- You need uncertainty estimates for optimization
- Operating near boundaries of training data
- Data is sparse or limited
- Extrapolation is required

❌ **Avoid GP when:**
- Dataset is very large (>10,000 samples)
- Only interpolation is needed
- Speed is critical
- Uncertainty is not important

### Hyperparameter Tuning

The GP automatically optimizes kernel hyperparameters via:
- **n_restarts_optimizer=10** - Multiple optimization attempts
- **Log-marginal likelihood** - Objective function
- **Bounded parameters** - Prevents unrealistic values

### Computational Complexity

- **Training**: O(n³) where n = number of samples
- **Prediction**: O(n) per sample
- **Memory**: O(n²)

For large datasets, consider:
- Using a subset for training
- Sparse GP approximations
- Switching to polynomial models

## Data Requirements

### Input Data

The script expects segmented motif data from `modeling/output/`:
- `segmented_motifs_all_{mill_number:02d}.csv`

### Required Columns

- **MV**: Ore, WaterMill, WaterZumpf
- **CV**: DensityHC, PulpHC, PressureHC, CirculativeLoad
- **DV**: Class_15, Daiki, FE
- **Target**: PSI200

### Data Preprocessing

- NaN values are removed
- Features are standardized (zero mean, unit variance)
- Target values are normalized internally by GP
- Quality model filters PSI200 ∈ (10, 35)

## Troubleshooting

### Model Training Fails

**Issue**: Kernel optimization fails
- **Solution**: Adjust kernel bounds in `_train_single_gp_model()`
- Try different initial values

**Issue**: Memory error
- **Solution**: Reduce training data size
- Use a subset of samples

### Poor Predictions

**Issue**: High uncertainty everywhere
- **Solution**: Check data quality and coverage
- May need more training data

**Issue**: Low R² scores
- **Solution**: Try different kernel (RBF, Matern with different nu)
- Check feature scaling

### Slow Training

**Issue**: Training takes too long
- **Solution**: Reduce `n_restarts_optimizer`
- Use fewer training samples
- Consider polynomial models instead

## References

- Rasmussen & Williams (2006). "Gaussian Processes for Machine Learning"
- scikit-learn GP documentation: https://scikit-learn.org/stable/modules/gaussian_process.html
- Matern kernel: Balances smoothness and flexibility for process data
