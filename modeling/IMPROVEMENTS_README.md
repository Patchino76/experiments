# Polynomial Model Training Improvements

## Overview

The upgraded training pipeline (`train_poly_upgraded_models.py`) includes significant enhancements over the original version to improve model performance, robustness, and interpretability.

## File Structure

```
modeling/
├── train_poly_upgraded_models.py    # Main improved training script
├── model_improvements.py            # Helper utilities module
├── train_poly_models.py             # Original training script (for comparison)
└── IMPROVEMENTS_README.md           # This file
```

## Key Improvements

### 1. Multiple Regularization Methods

**Original:** Only Ridge regression
**Improved:** Tests Ridge, ElasticNet, and Lasso

- **Ridge (L2):** Good for correlated features, keeps all features
- **ElasticNet (L1+L2):** Balances feature selection and regularization
- **Lasso (L1):** Aggressive feature selection, creates sparse models

The best method is automatically selected based on cross-validation performance.

### 2. Enhanced Hyperparameter Search

**Original:**
```python
param_grid = {
    "poly__degree": [2, 3],
    "ridge__alpha": [0.1, 1.0, 10.0],
}
```

**Improved:**
```python
# Ridge example
param_grid = {
    "poly__degree": [1, 2, 3],                    # Include linear baseline
    "poly__interaction_only": [False, True],      # Test interaction-only
    "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],  # Wider range
}
```

**Benefits:**
- Wider alpha range finds better bias-variance tradeoff
- Interaction-only option reduces feature explosion
- Linear baseline provides comparison point

### 3. Better Scoring Metric

**Original:** `scoring="neg_mean_absolute_error"`
**Improved:** `scoring="r2"`

R² is more appropriate for:
- Measuring explained variance
- Comparing models across different scales
- Detecting overfitting (can be negative for poor models)

### 4. Domain-Specific Feature Engineering

New engineered features for mill operations:

```python
# Ratios (more stable than raw values)
- ore_water_ratio = Ore / WaterMill
- water_mill_zumpf_ratio = WaterMill / WaterZumpf

# Totals
- total_water = WaterMill + WaterZumpf

# Energy indicators
- specific_energy = MotorAmp / Ore

# Dilution
- dilution_factor = (WaterMill + WaterZumpf) / Ore
```

**Benefits:**
- Captures physical relationships
- More interpretable coefficients
- Often more stable than raw measurements

### 5. Sample Weighting for Quality Model

**Original:** Hard filtering (16 < PSI200 < 30)
```python
train_mask = (train_df[self.dv_target] > 16) & (train_df[self.dv_target] < 30)
train_df_filtered = train_df[train_mask]
```

**Improved:** Wider range (10 < PSI200 < 35) with sample weighting
```python
train_mask = (train_df[self.dv_target] > 10) & (train_df[self.dv_target] < 35)
weights = compute_target_weights(y_train, target_min=16, target_max=30, 
                                 in_range_weight=2.0, out_range_weight=1.0)
```

**Benefits:**
- More training data (better generalization)
- Model can predict outside target range
- Still emphasizes important range

### 6. Data Quality Diagnostics

Automatic analysis of:
- **Outliers:** Using IQR method (>3 IQR from quartiles)
- **Variance:** Coefficient of variation (CV)
- **Missing values:** Count and percentage
- **Distribution:** Mean, std, min, max

**Warnings for:**
- CV < 1% (very low variance)
- Outliers > 5% of data
- Missing values > 10%

### 7. Comprehensive Visualization

**New plots created:**

1. **Residual Analysis** (4-panel):
   - Residuals vs Predicted
   - Residual histogram
   - Q-Q plot (normality check)
   - Predicted vs Actual

2. **CV Results**:
   - Top 10 parameter combinations
   - Score distribution across all combinations

3. **Cascade Validation**:
   - Full pipeline performance visualization

### 8. Enhanced Logging and Metrics

**Additional metrics tracked:**
- Train and test R², RMSE, MAE
- Overfitting gap (train R² - test R²)
- Feature sparsity (% of zero coefficients)
- Polynomial complexity breakdown
- CV score statistics (mean ± std)

**Overfitting detection:**
```
R² gap > 0.15  → Warning: possible overfitting
R² gap < 0     → Good: test better than train
0 < gap < 0.15 → Healthy bias-variance tradeoff
```

### 9. Cross-Validation Diagnostics

Logs top 5 parameter combinations with:
- Mean CV score ± standard deviation
- Full parameter configuration
- Rank

Helps understand:
- Parameter sensitivity
- Model stability across folds
- Alternative good configurations

### 10. Polynomial Complexity Analysis

Breaks down polynomial features into:
- **Linear features:** Original variables
- **Interaction features:** Products of 2+ variables
- **Power features:** Squared, cubed terms

Example output:
```
Total features: 120
Linear: 8
Interactions: 84
Powers: 28
```

## Expected Performance Improvements

Based on the enhancements:

1. **R² improvement:** +0.05 to +0.15 (5-15% better explained variance)
2. **Better extrapolation:** ElasticNet/Lasso remove irrelevant terms
3. **More stable:** Wider alpha range finds optimal regularization
4. **Less overfitting:** Interaction-only reduces feature explosion
5. **Better interpretability:** Feature engineering creates meaningful terms

## Usage

### Basic Usage

```bash
python train_poly_upgraded_models.py
```

### Customization

Edit the `main()` function in `train_poly_upgraded_models.py`:

```python
def main():
    mill_number = 8
    start_date = "2025-01-01"
    end_date = "2025-10-19"
    
    config = PipelineConfig.create_default(mill_number, start_date, end_date)
    
    # Optional: Customize configuration
    # config.model.test_size = 0.25
    # config.model.cv_splits = 3
    
    trainer = ImprovedPolynomialCascadeTrainer(config)
    trainer.run()
```

## Output Files

### Models Directory: `output/mill_poly_upgraded_XX/`

```
mill_poly_upgraded_08/
├── process_model_DensityHC.pkl
├── process_model_PulpHC.pkl
├── process_model_PressureHC.pkl
├── process_model_CirculativeLoad.pkl
├── quality_model.pkl
├── metadata_poly_upgraded.json
├── training_results_poly_upgraded.json
└── plots/
    ├── process_model_DensityHC_residuals.png
    ├── process_model_DensityHC_cv_results.png
    ├── process_model_PulpHC_residuals.png
    ├── process_model_PulpHC_cv_results.png
    ├── process_model_PressureHC_residuals.png
    ├── process_model_PressureHC_cv_results.png
    ├── process_model_CirculativeLoad_residuals.png
    ├── process_model_CirculativeLoad_cv_results.png
    ├── quality_model_residuals.png
    ├── quality_model_cv_results.png
    └── cascade_validation.png
```

## Comparison with Original

| Aspect | Original | Improved | Benefit |
|--------|----------|----------|---------|
| Regularization | Ridge only | Ridge + ElasticNet + Lasso | Better feature selection |
| Hyperparameters | 6 combinations | 21-126 combinations | Better optimization |
| Scoring | MAE | R² | Better for regression |
| Feature Engineering | None | 5+ engineered features | Physical meaning |
| Quality Model Filter | Hard (16-30) | Soft (10-35 + weights) | More data, better generalization |
| Diagnostics | Basic | Comprehensive | Better insights |
| Visualization | None | 11+ plots | Better understanding |
| Logging | Minimal | Detailed | Better debugging |

## Performance Monitoring

### Key Metrics to Watch

1. **R² Score:**
   - Train R² > 0.7: Good fit
   - Test R² > 0.6: Good generalization
   - Gap < 0.15: Not overfitting

2. **RMSE:**
   - Should be < 10% of target variable range
   - Lower is better

3. **Feature Sparsity:**
   - ElasticNet/Lasso: 30-70% sparsity is typical
   - Ridge: 0% sparsity (keeps all features)

4. **CV Score Stability:**
   - Std < 0.1: Stable across folds
   - Std > 0.2: Unstable, consider more data

## Troubleshooting

### Issue: All models fail to train

**Solution:** Check data quality diagnostics for:
- Missing values
- Low variance features
- Outliers

### Issue: High overfitting (R² gap > 0.15)

**Solutions:**
1. Increase regularization (higher alpha)
2. Use interaction_only=True
3. Try ElasticNet or Lasso
4. Reduce polynomial degree

### Issue: Poor test performance (R² < 0.5)

**Solutions:**
1. Check feature engineering relevance
2. Increase polynomial degree
3. Add more training data
4. Check for data quality issues

### Issue: Training too slow

**Solutions:**
1. Reduce hyperparameter grid size
2. Reduce cv_splits (default: 5)
3. Use fewer polynomial degrees
4. Set n_jobs=-1 (already default)

## Next Steps

1. **Run the improved pipeline:**
   ```bash
   python train_poly_upgraded_models.py
   ```

2. **Compare results:**
   - Check R² improvements
   - Review residual plots
   - Examine feature importance

3. **Fine-tune if needed:**
   - Adjust sample weights
   - Modify feature engineering
   - Customize hyperparameter grids

4. **Deploy best models:**
   - Use the .pkl files for predictions
   - Integrate into optimization workflow

## References

- **Regularization:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- **Feature Engineering:** Kuhn, M., & Johnson, K. (2013). Applied Predictive Modeling
- **Cross-Validation:** Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation
