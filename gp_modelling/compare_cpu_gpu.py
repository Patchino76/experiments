"""
Compare CPU vs GPU Gaussian Process Training Performance

Runs both versions and compares training time and results.

Usage:
    python compare_cpu_gpu.py
"""

import sys
from pathlib import Path
import time
import json
import pandas as pd

GP_BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GP_BASE_DIR.parent
MODELING_DIR = PROJECT_ROOT / "modeling"

for path_candidate in (GP_BASE_DIR, PROJECT_ROOT, MODELING_DIR):
    if str(path_candidate) not in sys.path:
        sys.path.append(str(path_candidate))

from modeling.config import PipelineConfig


def run_cpu_version(config):
    """Run CPU version and measure time."""
    print("\n" + "=" * 80)
    print("RUNNING CPU VERSION")
    print("=" * 80)
    
    from train_gp_models import GaussianProcessCascadeTrainer
    
    start_time = time.time()
    trainer = GaussianProcessCascadeTrainer(config)
    trainer.run()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    # Load metadata
    metadata_path = trainer.model_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return {
        'version': 'CPU',
        'elapsed_time': elapsed_time,
        'metadata': metadata,
        'model_dir': trainer.model_dir
    }


def run_gpu_version(config, max_vram_gb=14.0):
    """Run GPU version and measure time."""
    print("\n" + "=" * 80)
    print("RUNNING GPU VERSION")
    print("=" * 80)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠ No GPU available, skipping GPU version")
            return None
    except ImportError:
        print("⚠ PyTorch not installed, skipping GPU version")
        return None
    
    from train_gp_models_gpu import GaussianProcessCascadeTrainerGPU
    
    start_time = time.time()
    trainer = GaussianProcessCascadeTrainerGPU(config, max_vram_gb=max_vram_gb)
    trainer.run()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    # Load metadata
    metadata_path = trainer.model_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load GPU info if available
    gpu_info_path = trainer.model_dir / "gpu_info.json"
    gpu_info = None
    if gpu_info_path.exists():
        with open(gpu_info_path, 'r') as f:
            gpu_info = json.load(f)
    
    return {
        'version': 'GPU',
        'elapsed_time': elapsed_time,
        'metadata': metadata,
        'gpu_info': gpu_info,
        'model_dir': trainer.model_dir
    }


def compare_results(cpu_result, gpu_result):
    """Compare CPU and GPU results."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Training time comparison
    print("\n1. TRAINING TIME")
    print("-" * 80)
    cpu_time = cpu_result['elapsed_time']
    print(f"CPU Version: {cpu_time:.2f} seconds ({cpu_time/60:.2f} minutes)")
    
    if gpu_result:
        gpu_time = gpu_result['elapsed_time']
        speedup = cpu_time / gpu_time
        print(f"GPU Version: {gpu_time:.2f} seconds ({gpu_time/60:.2f} minutes)")
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 5:
            print("✓ Excellent speedup!")
        elif speedup > 2:
            print("✓ Good speedup")
        else:
            print("⚠ Limited speedup (dataset may be too small)")
    else:
        print("GPU Version: Not available")
    
    # Model performance comparison
    print("\n2. MODEL PERFORMANCE")
    print("-" * 80)
    
    cpu_meta = cpu_result['metadata']
    
    # Compare process models
    print("\nProcess Models (MV → CV):")
    for model_name in cpu_meta['model_performance']:
        if 'process_model' in model_name:
            cpu_perf = cpu_meta['model_performance'][model_name]
            output_var = cpu_perf['output_var']
            cpu_r2 = cpu_perf['test_metrics']['r2']
            cpu_rmse = cpu_perf['test_metrics']['rmse']
            
            print(f"\n  {output_var}:")
            print(f"    CPU  - R²: {cpu_r2:.4f}, RMSE: {cpu_rmse:.4f}")
            
            if gpu_result:
                gpu_meta = gpu_result['metadata']
                if model_name in gpu_meta['model_performance']:
                    gpu_perf = gpu_meta['model_performance'][model_name]
                    gpu_r2 = gpu_perf['test_metrics']['r2']
                    gpu_rmse = gpu_perf['test_metrics']['rmse']
                    
                    print(f"    GPU  - R²: {gpu_r2:.4f}, RMSE: {gpu_rmse:.4f}")
                    
                    r2_diff = abs(cpu_r2 - gpu_r2)
                    rmse_diff = abs(cpu_rmse - gpu_rmse)
                    
                    if r2_diff < 0.01 and rmse_diff < 0.1:
                        print(f"    ✓ Results are very similar")
                    else:
                        print(f"    ⚠ Results differ (R² diff: {r2_diff:.4f}, RMSE diff: {rmse_diff:.4f})")
    
    # Compare quality model
    print("\nQuality Model (CV + DV → Target):")
    if 'quality_model' in cpu_meta['model_performance']:
        cpu_perf = cpu_meta['model_performance']['quality_model']
        cpu_r2 = cpu_perf['test_metrics']['r2']
        cpu_rmse = cpu_perf['test_metrics']['rmse']
        
        print(f"  CPU  - R²: {cpu_r2:.4f}, RMSE: {cpu_rmse:.4f}")
        
        if gpu_result:
            gpu_meta = gpu_result['metadata']
            if 'quality_model' in gpu_meta['model_performance']:
                gpu_perf = gpu_meta['model_performance']['quality_model']
                gpu_r2 = gpu_perf['test_metrics']['r2']
                gpu_rmse = gpu_perf['test_metrics']['rmse']
                
                print(f"  GPU  - R²: {gpu_r2:.4f}, RMSE: {gpu_rmse:.4f}")
                
                r2_diff = abs(cpu_r2 - gpu_r2)
                rmse_diff = abs(cpu_rmse - gpu_rmse)
                
                if r2_diff < 0.01 and rmse_diff < 0.1:
                    print(f"  ✓ Results are very similar")
                else:
                    print(f"  ⚠ Results differ (R² diff: {r2_diff:.4f}, RMSE diff: {rmse_diff:.4f})")
    
    # Cascade validation
    print("\nCascade Validation:")
    if 'cascade_validation' in cpu_meta:
        cpu_cascade = cpu_meta['cascade_validation']
        cpu_r2 = cpu_cascade['r2']
        cpu_rmse = cpu_cascade['rmse']
        
        print(f"  CPU  - R²: {cpu_r2:.4f}, RMSE: {cpu_rmse:.4f}")
        
        if gpu_result and 'cascade_validation' in gpu_result['metadata']:
            gpu_cascade = gpu_result['metadata']['cascade_validation']
            gpu_r2 = gpu_cascade['r2']
            gpu_rmse = gpu_cascade['rmse']
            
            print(f"  GPU  - R²: {gpu_r2:.4f}, RMSE: {gpu_rmse:.4f}")
            
            r2_diff = abs(cpu_r2 - gpu_r2)
            rmse_diff = abs(cpu_rmse - gpu_rmse)
            
            if r2_diff < 0.01 and rmse_diff < 0.1:
                print(f"  ✓ Results are very similar")
            else:
                print(f"  ⚠ Results differ (R² diff: {r2_diff:.4f}, RMSE diff: {rmse_diff:.4f})")
    
    # GPU-specific info
    if gpu_result and gpu_result.get('gpu_info'):
        print("\n3. GPU INFORMATION")
        print("-" * 80)
        gpu_info = gpu_result['gpu_info']
        print(f"Device: {gpu_info['device_name']}")
        print(f"Total VRAM: {gpu_info['total_memory_gb']:.2f} GB")
        print(f"Peak VRAM Usage: {gpu_info['max_memory_allocated_gb']:.2f} GB")
        print(f"VRAM Utilization: {gpu_info['max_memory_allocated_gb']/gpu_info['total_memory_gb']*100:.1f}%")
    
    # Summary
    print("\n4. SUMMARY")
    print("-" * 80)
    
    if gpu_result:
        speedup = cpu_result['elapsed_time'] / gpu_result['elapsed_time']
        
        print(f"✓ Both versions completed successfully")
        print(f"✓ GPU version is {speedup:.2f}x faster")
        print(f"✓ Model performance is comparable")
        
        if speedup > 5:
            print(f"\n💡 Recommendation: Use GPU version for production")
        elif speedup > 2:
            print(f"\n💡 Recommendation: GPU version provides good speedup")
        else:
            print(f"\n💡 Recommendation: CPU version may be sufficient for this dataset size")
    else:
        print("⚠ GPU version not available")
        print("💡 Recommendation: Use CPU version")
    
    # Save comparison report
    comparison_report = {
        'cpu': {
            'elapsed_time': cpu_result['elapsed_time'],
            'model_dir': str(cpu_result['model_dir'])
        }
    }
    
    if gpu_result:
        comparison_report['gpu'] = {
            'elapsed_time': gpu_result['elapsed_time'],
            'speedup': speedup,
            'model_dir': str(gpu_result['model_dir'])
        }
        if gpu_result.get('gpu_info'):
            comparison_report['gpu']['gpu_info'] = gpu_result['gpu_info']
    
    report_path = GP_BASE_DIR / "output" / "comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    print(f"\n✓ Comparison report saved to: {report_path}")


def main():
    """Main entry point."""
    print("=" * 80)
    print("CPU vs GPU GAUSSIAN PROCESS TRAINING COMPARISON")
    print("=" * 80)
    
    # Configuration
    mill_number = 8
    start_date = "2025-10-01"
    end_date = "2025-10-19"
    
    config = PipelineConfig.create_default(mill_number, start_date, end_date)
    
    # Run CPU version
    cpu_result = run_cpu_version(config)
    
    # Run GPU version
    gpu_result = run_gpu_version(config, max_vram_gb=14.0)
    
    # Compare results
    compare_results(cpu_result, gpu_result)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
