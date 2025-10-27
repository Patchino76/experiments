# GPU-Accelerated Gaussian Process Training

This is a GPU-accelerated version of the Gaussian Process Regression training pipeline using **GPyTorch** and **PyTorch**.

## Features

✅ **GPU Acceleration**: Leverages CUDA GPUs for faster training  
✅ **Memory-Efficient Chunking**: Automatically chunks data to fit within 16GB VRAM  
✅ **Identical Architecture**: Same cascade structure as CPU version (Process Models → Quality Model)  
✅ **Uncertainty Quantification**: Full uncertainty estimates with confidence intervals  
✅ **Automatic Fallback**: Uses CPU if GPU is not available  

## Requirements

### Hardware
- **Recommended**: NVIDIA GPU with 16GB VRAM (e.g., RTX 4080, RTX 4090, A4000)
- **Minimum**: NVIDIA GPU with 8GB VRAM (will use smaller chunks)
- **Fallback**: Works on CPU if no GPU available

### Software
```bash
pip install -r requirements_gpu.txt
```

For GPU support, install PyTorch with CUDA:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check your CUDA version:
```bash
nvidia-smi
```

## Usage

### Basic Usage
```bash
python train_gp_models_gpu.py
```

### Configuration
Edit the `main()` function in `train_gp_models_gpu.py`:

```python
def main():
    mill_number = 8
    start_date = "2025-08-01"
    end_date = "2025-10-19"
    
    config = PipelineConfig.create_default(mill_number, start_date, end_date)
    
    # Adjust max VRAM usage (default: 14GB, leaving 2GB buffer)
    trainer = GaussianProcessCascadeTrainerGPU(config, max_vram_gb=14.0)
    trainer.run()
```

### Memory Management

The script automatically calculates optimal chunk sizes based on:
- Available VRAM (default: 14GB usable from 16GB total)
- Number of training samples
- Number of features

**Memory estimation formula:**
```
VRAM ≈ 3 × (n² × 4 bytes + n × d × 4 bytes)
```
Where:
- `n` = number of samples in chunk
- `d` = number of features
- Factor of 3 accounts for gradients and intermediate computations

**Example chunk sizes for 16GB VRAM:**
- 3 features: ~6,000 samples per chunk
- 7 features: ~5,500 samples per chunk
- 10 features: ~5,000 samples per chunk

### Adjusting VRAM Usage

If you encounter out-of-memory errors:

1. **Reduce max_vram_gb:**
   ```python
   trainer = GaussianProcessCascadeTrainerGPU(config, max_vram_gb=12.0)
   ```

2. **Reduce prediction chunk size** (in `_predict_in_chunks`):
   ```python
   pred, sigma = self._predict_in_chunks(model, likelihood, X_scaled, chunk_size=1000)
   ```

3. **Close other GPU applications** to free VRAM

## Differences from CPU Version

| Feature | CPU Version | GPU Version |
|---------|-------------|-------------|
| **Library** | scikit-learn | GPyTorch + PyTorch |
| **Device** | CPU only | GPU (with CPU fallback) |
| **Training Speed** | Baseline | 5-20x faster (depending on data size) |
| **Memory** | RAM | VRAM (chunked) |
| **Model Files** | `.pkl` | `.pth` (PyTorch state dicts) |
| **Output Dir** | `models/` | `models_gpu/` |
| **Kernel** | Matérn + White | Matérn + Gaussian Noise |

## Model Architecture

Same cascade structure as CPU version:

```
Process Models (MV → CV):
  - MV (Ore, WaterMill, WaterZumpf) → DensityHC
  - MV (Ore, WaterMill, WaterZumpf) → PulpHC
  - MV (Ore, WaterMill, WaterZumpf) → PressureHC
  - MV (Ore, WaterMill, WaterZumpf) → CirculativeLoad

Quality Model (CV + DV → Target):
  - CV (DensityHC, PulpHC, PressureHC, CirculativeLoad) + 
    DV (Class_15, Daiki, FE) → PSI200
```

## Output Files

All outputs are saved to `gp_modelling/output/mill_gp_gpu_XX/`:

### Models
- `process_model_*.pth` - Process model state dicts
- `quality_model.pth` - Quality model state dict
- `*_scaler.pkl` - Feature scalers (StandardScaler)

### Metadata
- `metadata.json` - Training configuration and metrics
- `gpu_info.json` - GPU device info and peak VRAM usage

### Plots
- `*_train_predictions.png` - Training predictions with uncertainty bands
- `*_test_predictions.png` - Test predictions with uncertainty bands
- `*_uncertainty.png` - Uncertainty distribution analysis
- `cascade_validation.png` - Full cascade validation

## Performance Tips

### For Large Datasets (>10,000 samples)
- GPU provides 5-20x speedup over CPU
- Training time: ~2-5 minutes per model (vs 30-60 minutes on CPU)

### For Small Datasets (<5,000 samples)
- GPU speedup is less significant (~2-3x)
- CPU version may be sufficient

### Monitoring GPU Usage
```bash
# Watch GPU usage in real-time
nvidia-smi -l 1

# Or use
watch -n 1 nvidia-smi
```

### Troubleshooting

**Error: "CUDA out of memory"**
- Reduce `max_vram_gb` parameter
- Reduce `chunk_size` in predictions
- Close other GPU applications

**Error: "No GPU detected"**
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Script will automatically fall back to CPU

**Slow training despite GPU**
- Check if model is actually on GPU (look for "GPU detected" in logs)
- Verify CUDA version matches PyTorch installation
- Monitor GPU utilization with `nvidia-smi`

## Loading Trained Models

```python
import torch
import pickle
from pathlib import Path

# Load model
model_path = Path("output/mill_gp_gpu_08/process_model_DensityHC.pth")
checkpoint = torch.load(model_path)

# Load scaler
scaler_path = Path("output/mill_gp_gpu_08/process_model_DensityHC_scaler.pkl")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Reconstruct model (you'll need to recreate the ExactGPModel class)
# See train_gp_models_gpu.py for model definition
```

## Comparison with CPU Version

Run both versions and compare:

```bash
# CPU version
python train_gp_models.py

# GPU version
python train_gp_models_gpu.py
```

Results should be nearly identical (small differences due to numerical precision and optimization randomness).

## References

- **GPyTorch**: https://gpytorch.ai/
- **PyTorch**: https://pytorch.org/
- **Gaussian Processes**: Rasmussen & Williams (2006)

## Support

For issues or questions:
1. Check GPU is detected in logs
2. Verify CUDA installation with `nvidia-smi`
3. Check PyTorch CUDA support: `torch.cuda.is_available()`
4. Review memory usage in logs
