"""
GPU Setup Checker for Gaussian Process Training

Checks GPU availability, VRAM, and estimates optimal chunk sizes.

Usage:
    python check_gpu_setup.py
"""

import sys
import numpy as np

def check_pytorch():
    """Check if PyTorch is installed."""
    try:
        import torch
        print("✓ PyTorch installed")
        print(f"  Version: {torch.__version__}")
        return True, torch
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch")
        return False, None


def check_gpytorch():
    """Check if GPyTorch is installed."""
    try:
        import gpytorch
        print("✓ GPyTorch installed")
        print(f"  Version: {gpytorch.__version__}")
        return True
    except ImportError:
        print("✗ GPyTorch not installed")
        print("  Install with: pip install gpytorch")
        return False


def check_cuda(torch):
    """Check CUDA availability and GPU info."""
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        print("  PyTorch will use CPU")
        print("  To enable GPU:")
        print("    1. Install NVIDIA GPU drivers")
        print("    2. Install CUDA toolkit")
        print("    3. Reinstall PyTorch with CUDA support")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return False, None
    
    print("✓ CUDA available")
    print(f"  CUDA Version: {torch.version.cuda}")
    
    # Get GPU info
    n_gpus = torch.cuda.device_count()
    print(f"  Number of GPUs: {n_gpus}")
    
    gpu_info = []
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        name = torch.cuda.get_device_name(i)
        total_memory = props.total_memory / 1e9
        
        print(f"\n  GPU {i}: {name}")
        print(f"    Total VRAM: {total_memory:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
        
        gpu_info.append({
            'index': i,
            'name': name,
            'total_memory_gb': total_memory,
            'compute_capability': (props.major, props.minor)
        })
    
    return True, gpu_info


def estimate_memory_usage(n_samples, n_features):
    """Estimate GPU memory usage for GP training."""
    # Covariance matrix: n x n (float32)
    covar_memory = n_samples * n_samples * 4 / 1e9  # GB
    # Feature matrix: n x d (float32)
    feature_memory = n_samples * n_features * 4 / 1e9  # GB
    # Overhead (gradients, intermediate computations): ~3x
    total_memory = (covar_memory + feature_memory) * 3
    return total_memory


def calculate_optimal_chunk_size(total_vram_gb, n_features, buffer_gb=2.0):
    """Calculate optimal chunk size for given VRAM."""
    max_vram_gb = total_vram_gb - buffer_gb
    
    # Binary search for optimal chunk size
    low, high = 100, 50000
    optimal_chunk = 100
    
    while low <= high:
        mid = (low + high) // 2
        mem_usage = estimate_memory_usage(mid, n_features)
        
        if mem_usage <= max_vram_gb:
            optimal_chunk = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return optimal_chunk, max_vram_gb


def print_recommendations(gpu_info):
    """Print recommendations based on GPU info."""
    if not gpu_info:
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        print("No GPU detected. The script will run on CPU.")
        print("For GPU acceleration, install CUDA-enabled PyTorch.")
        return
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    # Use first GPU
    gpu = gpu_info[0]
    total_vram = gpu['total_memory_gb']
    
    print(f"\nUsing GPU 0: {gpu['name']}")
    print(f"Total VRAM: {total_vram:.2f} GB")
    
    # Test different feature counts
    feature_counts = [3, 7, 10]
    
    print("\nOptimal chunk sizes for different feature counts:")
    print("-" * 70)
    print(f"{'Features':<12} {'Chunk Size':<15} {'Usable VRAM':<15} {'Est. Memory':<15}")
    print("-" * 70)
    
    for n_features in feature_counts:
        chunk_size, max_vram = calculate_optimal_chunk_size(total_vram, n_features)
        est_mem = estimate_memory_usage(chunk_size, n_features)
        print(f"{n_features:<12} {chunk_size:<15} {max_vram:.2f} GB{'':<7} {est_mem:.2f} GB")
    
    print("-" * 70)
    
    # Recommendations
    print("\nConfiguration recommendations:")
    print(f"  max_vram_gb = {total_vram - 2.0:.1f}  # Leaving 2GB buffer")
    
    if total_vram >= 16:
        print("\n✓ Your GPU has sufficient VRAM for large datasets")
        print("  Expected performance: 5-20x faster than CPU")
    elif total_vram >= 8:
        print("\n⚠ Your GPU has moderate VRAM")
        print("  Will use smaller chunks, but still faster than CPU")
        print("  Expected performance: 3-10x faster than CPU")
    else:
        print("\n⚠ Your GPU has limited VRAM")
        print("  May need to use very small chunks")
        print("  Consider using CPU version for large datasets")
    
    print("\nTo adjust VRAM usage in train_gp_models_gpu.py:")
    print("  trainer = GaussianProcessCascadeTrainerGPU(config, max_vram_gb=XX.X)")


def test_simple_gp(torch):
    """Test a simple GP training on GPU."""
    print("\n" + "=" * 70)
    print("TESTING SIMPLE GP TRAINING")
    print("=" * 70)
    
    try:
        import gpytorch
        from gpytorch.models import ExactGP
        from gpytorch.means import ConstantMean
        from gpytorch.kernels import ScaleKernel, MaternKernel
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.mlls import ExactMarginalLogLikelihood
        
        # Simple test data
        n_train = 100
        n_features = 3
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Generate synthetic data
        X_train = torch.randn(n_train, n_features).to(device)
        y_train = torch.sin(X_train[:, 0]) + 0.1 * torch.randn(n_train).to(device)
        
        # Define simple GP model
        class SimpleGP(ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = ConstantMean()
                self.covar_module = ScaleKernel(MaternKernel(nu=2.5))
            
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        # Initialize
        likelihood = GaussianLikelihood().to(device)
        model = SimpleGP(X_train, y_train, likelihood).to(device)
        
        # Training mode
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(likelihood, model)
        
        # Train for a few iterations
        print("Training for 10 iterations...")
        for i in range(10):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 5 == 0:
                print(f"  Iteration {i+1}/10 - Loss: {loss.item():.4f}")
        
        # Test prediction
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            X_test = torch.randn(10, n_features).to(device)
            pred = likelihood(model(X_test))
            print(f"\n✓ Test prediction successful")
            print(f"  Mean shape: {pred.mean.shape}")
            print(f"  Variance shape: {pred.variance.shape}")
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.max_memory_allocated(0) / 1e9
            print(f"\n✓ Peak VRAM usage: {mem_allocated:.4f} GB")
        
        print("\n✓ GPU setup is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during GP training test:")
        print(f"  {str(e)}")
        return False


def main():
    """Main entry point."""
    print("=" * 70)
    print("GPU SETUP CHECKER FOR GAUSSIAN PROCESS TRAINING")
    print("=" * 70)
    print()
    
    # Check installations
    pytorch_ok, torch = check_pytorch()
    gpytorch_ok = check_gpytorch()
    
    if not pytorch_ok or not gpytorch_ok:
        print("\n✗ Missing required packages")
        print("  Install with: pip install -r requirements_gpu.txt")
        return
    
    print()
    
    # Check CUDA
    cuda_ok, gpu_info = check_cuda(torch)
    
    # Print recommendations
    print_recommendations(gpu_info)
    
    # Test simple GP if GPU available
    if cuda_ok:
        test_simple_gp(torch)
    
    print("\n" + "=" * 70)
    print("SETUP CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
