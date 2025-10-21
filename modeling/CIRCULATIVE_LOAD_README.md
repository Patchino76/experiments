# Circulative Load Calculation for Ball Mills

## Overview

The circulative load calculation has been integrated into the modeling pipeline to provide insights into the recirculation of material in closed-circuit grinding operations with hydrocyclones.

## What is Circulative Load?

**Circulative Load (CL)** represents the ratio of material recirculated back to the mill from the cyclone compared to the fresh feed entering the system.

**Formula:** `CL = (M_solid_to_cyclone - Fresh_Feed) / Fresh_Feed`

**Typical Range:** 1.5 to 3.0 (150-300%) for closed-circuit grinding with hydrocyclone

## Implementation

### Files Created/Modified

1. **`circulative_load.py`** (NEW)
   - Main calculation module
   - Contains `calculate_circulative_load()` function
   - Contains `validate_circulative_load()` function

2. **`prepare_data.py`** (MODIFIED)
   - Integrated circulative load calculation in `load_data()` method
   - Calculation happens after data filtering
   - Results saved to `initial_data.csv`

3. **`test_circulative_load.py`** (NEW)
   - Test script to verify the calculation
   - Demonstrates usage with sample data

### Calculation Steps

The calculation follows these steps:

1. **Calculate Volumetric Concentration (C_v)**
   ```
   C_v = (ρ_pulp - ρ_water) / (ρ_solid - ρ_water)
   ```
   - ρ_pulp = DensityHC (from data)
   - ρ_water = 1000 kg/m³
   - ρ_solid = 2900 kg/m³ (default for copper ore)

2. **Calculate Mass Concentration (C_m)**
   ```
   C_m = (C_v × ρ_solid) / (C_v × ρ_solid + (1 - C_v) × ρ_water)
   ```

3. **Calculate Mass Flow of Solids to Cyclone (t/h)**
   ```
   M_solid_to_cyclone = (PulpHC × DensityHC × C_m) / 1000
   ```

4. **Calculate Circulative Load Ratio**
   ```
   CirculativeLoad = (M_solid_to_cyclone - Ore) / Ore
   ```

### Required Columns

The calculation requires these columns in your data:
- **Ore**: Fresh feed ore flow rate (t/h)
- **PulpHC**: Pulp flow to hydrocyclone (m³/h)
- **DensityHC**: Pulp density at hydrocyclone (kg/m³)

### Added Columns

The calculation adds these columns to your DataFrame:
- **C_v**: Volumetric concentration (fraction)
- **C_m**: Mass concentration (fraction)
- **M_solid_to_cyclone**: Mass flow of solids to cyclone (t/h)
- **CirculativeLoad**: Circulative load ratio (dimensionless)

## Usage

### Automatic Integration

When you run `prepare_data.py`, the circulative load is automatically calculated:

```python
python prepare_data.py
```

The calculation happens in the data loading step and the results are saved to:
- `modeling/output/initial_data.csv` (includes CirculativeLoad column)

### Manual Usage

You can also use the function directly:

```python
from circulative_load import calculate_circulative_load, validate_circulative_load
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Calculate circulative load
df = calculate_circulative_load(df, rho_solid=2900)

# Validate results
validate_circulative_load(df)
```

### Testing

Run the test script to verify the calculation:

```python
python test_circulative_load.py
```

## Configuration

### Solid Density (rho_solid)

The default solid density is **2900 kg/m³** (typical for copper ore). You can adjust this if needed:

```python
df = calculate_circulative_load(df, rho_solid=2800)  # Custom density
```

### Validation Ranges

The validation function checks if values are within reasonable ranges:
- Default valid range: 0.5 to 5.0
- Typical operational range: 1.5 to 3.0

## Interpretation

### Typical Values
- **CL < 1.5**: Low recirculation, may indicate inefficient classification
- **CL = 1.5 - 3.0**: Normal operating range for most grinding circuits
- **CL > 3.0**: High recirculation, may indicate overloading or classification issues

### What Affects Circulative Load?
- Cyclone operating pressure
- Feed density to cyclone
- Particle size distribution
- Cyclone geometry and wear
- Fresh feed rate

## Error Handling

The integration includes robust error handling:
- Missing columns: Warning logged, continues without calculation
- Invalid values: NaN values handled gracefully
- Out-of-range values: Logged for review

## Logging

The calculation provides detailed logging:
- Number of rows processed
- Statistical summary (mean, median, std, min, max)
- Percentage of values in typical range
- Warnings for unusual values

## Example Output

```
Calculating circulative load...
  Using rho_solid = 2900 kg/m³
  ✓ Circulative load calculated for 8640 rows
    Mean: 2.234
    Median: 2.198
    Std: 0.456
    Min: 1.123
    Max: 3.987
    Values in typical range [1.5, 3.0]: 7234/8640 (83.7%)
  ✓ Circulative load calculation complete
```

## Model Integration

### Feature Classification

**CirculativeLoad is classified as a CV (Controlled Variable)** in the cascade model structure:

```
MV → CV → Quality
     ↓
CirculativeLoad is here (CV feature)
```

### Usage in Cascade Models

**✅ Quality Model (CV + DV → PSI200):**
- CirculativeLoad is used as an INPUT feature
- Provides grinding circuit efficiency information
- Helps predict particle size distribution (PSI200)

**❌ Process Models (MV → CV):**
- NOT used in process models to avoid circular dependencies
- CirculativeLoad is calculated FROM CVs (PulpHC, DensityHC)
- Process models predict individual CVs from MVs

### Why This Approach?

1. **Physical Relationship:** CirculativeLoad directly affects grinding efficiency and particle size
2. **Process State Indicator:** Like density and pressure, it describes current mill operation
3. **Avoids Circularity:** It's derived from CVs, so shouldn't be predicted by them
4. **Improves Quality Predictions:** Provides additional context for PSI200 prediction

### Configuration

CirculativeLoad is automatically added to `cv_features` in `config.py`:

```python
cv_features = ['DensityHC', 'PulpHC', 'PressureHC', 'CirculativeLoad']
```

This means:
- Process models: MV → [DensityHC, PulpHC, PressureHC] (CirculativeLoad not predicted)
- Quality model: [DensityHC, PulpHC, PressureHC, CirculativeLoad] + DV → PSI200

## Notes

- The calculation assumes steady-state operation
- Extreme values may indicate measurement errors or transient conditions
- The circulative load column is automatically included in all downstream analyses
- The column is available for use in model training and predictions
- CirculativeLoad is used ONLY in the quality model, not in process models

## Support

For questions or issues with the circulative load calculation, check:
1. Required columns are present in your data
2. Data values are within reasonable physical ranges
3. Log files for detailed error messages
