"""
Gaussian Process Regression Model Training Pipeline - GPU Accelerated Version

Implements GPR models using GPyTorch with GPU support and memory-efficient chunking
to fit within 16GB VRAM constraints.

Requirements:
    pip install gpytorch torch

Usage:
    python train_gp_models_gpu.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

GP_BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GP_BASE_DIR.parent
MODELING_DIR = PROJECT_ROOT / "modeling"

for path_candidate in (GP_BASE_DIR, PROJECT_ROOT, MODELING_DIR):
    if str(path_candidate) not in sys.path:
        sys.path.append(str(path_candidate))

from modeling.config import PipelineConfig
from gp_visualization import plot_gp_predictions_with_uncertainty, plot_uncertainty_analysis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(GP_BASE_DIR / 'train_gp_models_gpu.log', encoding='utf-8')
    ]
)

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
logger = logging.getLogger(__name__)


def get_device():
    """Get the best available device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.info("⚠ No GPU detected, using CPU")
    return device


def estimate_memory_usage(n_samples, n_features):
    """
    Estimate GPU memory usage for GP training.
    
    GP requires O(n²) memory for covariance matrix.
    Each float32 element = 4 bytes.
    """
    # Covariance matrix: n x n
    covar_memory = n_samples * n_samples * 4 / 1e9  # GB
    # Feature matrix: n x d
    feature_memory = n_samples * n_features * 4 / 1e9  # GB
    # Overhead (gradients, intermediate computations): ~3x
    total_memory = (covar_memory + feature_memory) * 3
    return total_memory


def calculate_optimal_chunk_size(n_samples, n_features, max_vram_gb=14.0):
    """
    Calculate optimal chunk size to fit within VRAM budget.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features
        max_vram_gb: Maximum VRAM to use (leave 2GB buffer from 16GB)
    
    Returns:
        chunk_size: Optimal chunk size
    """
    # Binary search for optimal chunk size
    low, high = 100, n_samples
    optimal_chunk = 100
    
    while low <= high:
        mid = (low + high) // 2
        mem_usage = estimate_memory_usage(mid, n_features)
        
        if mem_usage <= max_vram_gb:
            optimal_chunk = mid
            low = mid + 1
        else:
            high = mid - 1
    
    logger.info(f"  Optimal chunk size: {optimal_chunk} samples")
    logger.info(f"  Estimated VRAM per chunk: {estimate_memory_usage(optimal_chunk, n_features):.2f} GB")
    
    return optimal_chunk


class ExactGPModel(ExactGP):
    """Exact Gaussian Process model with Matérn kernel."""
    
    def __init__(self, train_x, train_y, likelihood, nu=2.5):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        
        # Matérn kernel with automatic relevance determination (ARD)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=nu, ard_num_dims=train_x.shape[-1])
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def rebase_paths_to_gp(config: PipelineConfig) -> PipelineConfig:
    """Ensure config paths point to the gp_modelling directory."""
    output_dir = GP_BASE_DIR / "output"
    models_dir = GP_BASE_DIR / "models_gpu"
    analysis_dir = output_dir / "analysis_gpu"
    plots_dir = output_dir / "plots_gpu"

    for path in [output_dir, models_dir, analysis_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    config.paths.output_dir = output_dir
    config.paths.models_dir = models_dir
    config.paths.analysis_dir = analysis_dir
    config.paths.plots_dir = plots_dir
    return config


class GaussianProcessCascadeTrainerGPU:
    """Trains Gaussian Process Regression cascade models using GPyTorch with GPU support."""

    def __init__(self, config: PipelineConfig, max_vram_gb=14.0):
        self.config = rebase_paths_to_gp(config)
        self.mill_number = config.data.mill_number
        self.model_dir = config.paths.output_dir / f"mill_gp_gpu_{self.mill_number:02d}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.model_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.mv_features = config.data.mv_features
        self.cv_features = config.data.cv_features
        self.dv_features = config.data.dv_features
        self.dv_target = config.data.target

        self.models = {}
        self.likelihoods = {}
        self.scalers = {}
        self.training_results = {}
        self.metadata = {}
        self.df = None
        self.random_state = config.model.random_state
        
        self.device = get_device()
        self.max_vram_gb = max_vram_gb
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.backends.cudnn.deterministic = True

    def run(self):
        """Execute the complete training pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info(f"STARTING GPU-ACCELERATED GAUSSIAN PROCESS CASCADE TRAINING - MILL {self.mill_number}")
        logger.info("=" * 80)

        self.load_data()
        train_df, test_df = self.split_data()
        self.initialize_metadata()
        self.train_process_models(train_df, test_df)
        self.train_quality_model(train_df, test_df)
        self.validate_cascade(test_df)
        self.save_results()

        logger.info("\n" + "=" * 80)
        logger.info("GPU-ACCELERATED GAUSSIAN PROCESS MODEL TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Models saved to: {self.model_dir}")

    def load_data(self):
        """Load segmented data."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("-" * 80)

        modeling_output_dir = PROJECT_ROOT / "modeling" / "output"
        possible_files = [
            self.config.paths.output_dir / f'segmented_motifs_all_{self.mill_number:02d}.csv',
            modeling_output_dir / f'segmented_motifs_all_{self.mill_number:02d}.csv',
            modeling_output_dir / "segmented_motifs_all.csv",
        ]
        
        data_path = None
        for path in possible_files:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(f"Segmented data not found. Please run prepare_data.py first.")

        logger.info(f"Loading data from {data_path}...")
        preview = pd.read_csv(data_path, nrows=0)
        parse_dates = ['TimeStamp'] if 'TimeStamp' in preview.columns else None
        self.df = pd.read_csv(data_path, parse_dates=parse_dates)

        # Filter by date range if TimeStamp column exists
        if 'TimeStamp' in self.df.columns:
            original_rows = len(self.df)
            start_date = pd.to_datetime(self.config.data.start_date)
            end_date = pd.to_datetime(self.config.data.end_date)
            self.df = self.df[(self.df['TimeStamp'] >= start_date) & (self.df['TimeStamp'] <= end_date)]
            logger.info(f"  Date filtering: {original_rows} → {len(self.df)} rows")
            logger.info(f"  Date range: {self.config.data.start_date} to {self.config.data.end_date}")

        required_cols = self.mv_features + self.cv_features + self.dv_features + [self.dv_target]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        original_shape = self.df.shape
        self.df = self.df.dropna(subset=required_cols)
        cleaned_shape = self.df.shape

        rows_removed = original_shape[0] - cleaned_shape[0]
        logger.info(f"  Data shape: {cleaned_shape}, Rows removed (NaN): {rows_removed}")

    def split_data(self):
        """Split data into train and test sets."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 2: SPLITTING DATA")
        logger.info("-" * 80)

        train_size = int(len(self.df) * (1 - self.config.model.test_size))
        train_df = self.df.iloc[:train_size]
        test_df = self.df.iloc[train_size:]

        logger.info(f"  Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df

    def initialize_metadata(self):
        """Initialize model metadata."""
        self.metadata = {
            "mill_number": self.mill_number,
            "created_at": datetime.now().isoformat(),
            "model_version": "1.0.0-GPU",
            "model_type": "Gaussian Process Regression (GPyTorch)",
            "device": str(self.device),
            "kernel": "Matern(nu=2.5) + Gaussian Noise",
            "features": {
                "mv_features": self.mv_features,
                "cv_features": self.cv_features,
                "dv_features": self.dv_features,
                "target_variable": self.dv_target,
            },
            "training_config": {
                "test_size": self.config.model.test_size,
                "data_shape": list(self.df.shape),
                "max_vram_gb": self.max_vram_gb,
                "training_iterations": 100
            },
            "model_performance": {}
        }

    def train_process_models(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train process models (MV → CV)."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 3: TRAINING PROCESS MODELS (MV → CV) ON GPU")
        logger.info("-" * 80)

        for cv_var in self.cv_features:
            logger.info(f"\nTraining GP model: MV → {cv_var}")

            X_train = train_df[self.mv_features].values
            y_train = train_df[cv_var].values
            X_test = test_df[self.mv_features].values
            y_test = test_df[cv_var].values

            result = self._train_single_gp_model(
                X_train, y_train, X_test, y_test,
                model_name=f"process_model_{cv_var}",
                output_var=cv_var,
                input_features=self.mv_features
            )

            self.metadata["model_performance"][f"process_model_{cv_var}"] = result

    def train_quality_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train quality model (CV + DV → Target)."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 4: TRAINING QUALITY MODEL (CV + DV → Target) ON GPU")
        logger.info("-" * 80)

        train_mask = (train_df[self.dv_target] > 10) & (train_df[self.dv_target] < 35)
        test_mask = (test_df[self.dv_target] > 10) & (test_df[self.dv_target] < 35)

        train_df_filtered = train_df[train_mask]
        test_df_filtered = test_df[test_mask]

        logger.info(f"  Filtered: Train {len(train_df_filtered)}, Test {len(test_df_filtered)}")

        input_features = self.cv_features + self.dv_features

        X_train = train_df_filtered[input_features].values
        y_train = train_df_filtered[self.dv_target].values
        X_test = test_df_filtered[input_features].values
        y_test = test_df_filtered[self.dv_target].values

        result = self._train_single_gp_model(
            X_train, y_train, X_test, y_test,
            model_name="quality_model",
            output_var=self.dv_target,
            input_features=input_features
        )

        result["model_type"] = "quality_model"
        self.metadata["model_performance"]["quality_model"] = result

    def _train_single_gp_model(
        self, X_train, y_train, X_test, y_test,
        model_name, output_var, input_features,
        training_iter=100
    ):
        """Train a single Gaussian Process model with GPU acceleration and chunking."""
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        n_train = len(X_train_scaled)
        n_features = X_train_scaled.shape[1]
        
        # Calculate optimal chunk size for memory efficiency
        chunk_size = calculate_optimal_chunk_size(n_train, n_features, self.max_vram_gb)
        
        # If data fits in memory, train on full dataset
        if n_train <= chunk_size:
            logger.info(f"  Training on full dataset ({n_train} samples)...")
            model, likelihood = self._train_gp_chunk(
                X_train_scaled, y_train, training_iter
            )
        else:
            # Use subset of data for training (most recent samples)
            logger.info(f"  Dataset too large ({n_train} samples), using last {chunk_size} samples...")
            X_train_chunk = X_train_scaled[-chunk_size:]
            y_train_chunk = y_train[-chunk_size:]
            model, likelihood = self._train_gp_chunk(
                X_train_chunk, y_train_chunk, training_iter
            )
        
        # Predictions with uncertainty (in chunks to avoid OOM)
        logger.info(f"  Making predictions...")
        train_pred, train_sigma = self._predict_in_chunks(
            model, likelihood, X_train_scaled, chunk_size=2000
        )
        test_pred, test_sigma = self._predict_in_chunks(
            model, likelihood, X_test_scaled, chunk_size=2000
        )
        
        # Metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        logger.info(f"  Train: R²={train_r2:.4f}, RMSE={train_rmse:.4f}, σ={np.mean(train_sigma):.4f}")
        logger.info(f"  Test:  R²={test_r2:.4f}, RMSE={test_rmse:.4f}, σ={np.mean(test_sigma):.4f}")
        
        # Store model, likelihood, and scaler
        self.models[model_name] = model
        self.likelihoods[model_name] = likelihood
        self.scalers[model_name] = scaler
        
        # Visualizations
        train_plot = self.plots_dir / f"{model_name}_train_predictions.png"
        plot_gp_predictions_with_uncertainty(
            y_train, train_pred, train_sigma, train_r2,
            f"{model_name} - {output_var} (Train)", str(train_plot), "Train"
        )
        
        test_plot = self.plots_dir / f"{model_name}_test_predictions.png"
        plot_gp_predictions_with_uncertainty(
            y_test, test_pred, test_sigma, test_r2,
            f"{model_name} - {output_var} (Test)", str(test_plot), "Test"
        )
        
        uncertainty_plot = self.plots_dir / f"{model_name}_uncertainty.png"
        plot_uncertainty_analysis(
            train_sigma, test_sigma,
            f"{model_name} - Uncertainty Analysis", str(uncertainty_plot)
        )
        
        return {
            "model_name": model_name,
            "output_var": output_var,
            "input_features": input_features,
            "kernel": str(model.covar_module),
            "train_samples": int(n_train),
            "chunk_size": int(chunk_size),
            "train_metrics": {
                "r2": float(train_r2),
                "rmse": float(train_rmse),
                "mae": float(train_mae),
                "mean_uncertainty": float(np.mean(train_sigma))
            },
            "test_metrics": {
                "r2": float(test_r2),
                "rmse": float(test_rmse),
                "mae": float(test_mae),
                "mean_uncertainty": float(np.mean(test_sigma))
            }
        }

    def _train_gp_chunk(self, X_train, y_train, training_iter):
        """Train GP model on a single chunk of data."""
        # Convert to tensors
        train_x = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        # Initialize likelihood and model
        likelihood = GaussianLikelihood().to(self.device)
        model = ExactGPModel(train_x, train_y, likelihood, nu=2.5).to(self.device)
        
        # Training mode
        model.train()
        likelihood.train()
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        # Loss function
        mll = ExactMarginalLogLikelihood(likelihood, model)
        
        # Training loop
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 20 == 0:
                logger.info(f"    Iteration {i+1}/{training_iter} - Loss: {loss.item():.4f}")
        
        # Set to eval mode
        model.eval()
        likelihood.eval()
        
        return model, likelihood

    def _predict_in_chunks(self, model, likelihood, X, chunk_size=2000):
        """Make predictions in chunks to avoid GPU memory overflow."""
        n_samples = len(X)
        predictions = []
        uncertainties = []
        
        model.eval()
        likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for start_idx in range(0, n_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, n_samples)
                X_chunk = X[start_idx:end_idx]
                
                # Convert to tensor
                test_x = torch.tensor(X_chunk, dtype=torch.float32).to(self.device)
                
                # Make predictions
                observed_pred = likelihood(model(test_x))
                
                # Get mean and variance
                pred_mean = observed_pred.mean.cpu().numpy()
                pred_var = observed_pred.variance.cpu().numpy()
                pred_std = np.sqrt(pred_var)
                
                predictions.append(pred_mean)
                uncertainties.append(pred_std)
                
                # Clear GPU cache
                del test_x, observed_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        predictions = np.concatenate(predictions)
        uncertainties = np.concatenate(uncertainties)
        
        return predictions, uncertainties

    def validate_cascade(self, test_df: pd.DataFrame):
        """Validate cascade with uncertainty propagation."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 5: CASCADE VALIDATION WITH UNCERTAINTY PROPAGATION")
        logger.info("-" * 80)

        test_mask = (test_df[self.dv_target] > 10) & (test_df[self.dv_target] < 35)
        test_df_filtered = test_df[test_mask].copy()
        
        if len(test_df_filtered) == 0:
            logger.warning("  No test samples after filtering")
            return

        logger.info(f"  Using {len(test_df_filtered)} test samples")

        # Predict CV from MV
        X_mv = test_df_filtered[self.mv_features].values
        cv_predictions = {}
        cv_uncertainties = {}
        
        for cv_var in self.cv_features:
            model_name = f"process_model_{cv_var}"
            if model_name not in self.models:
                cv_predictions[cv_var] = test_df_filtered[cv_var].values
                cv_uncertainties[cv_var] = np.zeros(len(test_df_filtered))
                continue
            
            scaler = self.scalers[model_name]
            model = self.models[model_name]
            likelihood = self.likelihoods[model_name]
            
            X_scaled = scaler.transform(X_mv)
            pred, sigma = self._predict_in_chunks(model, likelihood, X_scaled, chunk_size=2000)
            
            cv_predictions[cv_var] = pred
            cv_uncertainties[cv_var] = sigma
            logger.info(f"  {cv_var}: σ={np.mean(sigma):.4f}")

        # Predict target from predicted CV + actual DV
        quality_model_name = "quality_model"
        if quality_model_name not in self.models:
            logger.warning("  Quality model not found")
            return

        X_quality = np.column_stack([
            cv_predictions[cv_var] for cv_var in self.cv_features
        ] + [
            test_df_filtered[dv_var].values for dv_var in self.dv_features
        ])
        
        scaler = self.scalers[quality_model_name]
        model = self.models[quality_model_name]
        likelihood = self.likelihoods[quality_model_name]
        
        X_quality_scaled = scaler.transform(X_quality)
        y_cascade_pred, y_cascade_sigma = self._predict_in_chunks(
            model, likelihood, X_quality_scaled, chunk_size=2000
        )
        
        y_true = test_df_filtered[self.dv_target].values
        
        # Metrics
        cascade_r2 = r2_score(y_true, y_cascade_pred)
        cascade_rmse = np.sqrt(mean_squared_error(y_true, y_cascade_pred))
        cascade_mae = mean_absolute_error(y_true, y_cascade_pred)
        
        logger.info(f"\n  Cascade Performance:")
        logger.info(f"    R²={cascade_r2:.4f}, RMSE={cascade_rmse:.4f}, MAE={cascade_mae:.4f}")
        logger.info(f"    Mean σ={np.mean(y_cascade_sigma):.4f}")
        
        self.metadata["cascade_validation"] = {
            "n_samples": int(len(test_df_filtered)),
            "r2": float(cascade_r2),
            "rmse": float(cascade_rmse),
            "mae": float(cascade_mae),
            "mean_uncertainty": float(np.mean(y_cascade_sigma))
        }
        
        # Visualization
        cascade_plot = self.plots_dir / "cascade_validation.png"
        plot_gp_predictions_with_uncertainty(
            y_true, y_cascade_pred, y_cascade_sigma, cascade_r2,
            f"Cascade Validation - {self.dv_target}", str(cascade_plot), "Cascade"
        )

    def save_results(self):
        """Save models, likelihoods, scalers, and metadata."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 6: SAVING RESULTS")
        logger.info("-" * 80)

        # Save models and likelihoods (move to CPU first)
        for model_name, model in self.models.items():
            model_path = self.model_dir / f"{model_name}.pth"
            likelihood = self.likelihoods[model_name]
            
            # Move to CPU for saving
            model_cpu = model.cpu()
            likelihood_cpu = likelihood.cpu()
            
            # Save state dicts
            torch.save({
                'model_state_dict': model_cpu.state_dict(),
                'likelihood_state_dict': likelihood_cpu.state_dict(),
            }, model_path)
            
            logger.info(f"  ✓ Saved {model_name}")
            
            # Move back to device if needed
            if self.device.type == 'cuda':
                model.to(self.device)
                likelihood.to(self.device)

        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = self.model_dir / f"{scaler_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

        # Save metadata
        metadata_path = self.model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"  ✓ Saved metadata")
        
        # Save GPU info
        if torch.cuda.is_available():
            gpu_info = {
                "device_name": torch.cuda.get_device_name(0),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated(0) / 1e9,
            }
            gpu_info_path = self.model_dir / "gpu_info.json"
            with open(gpu_info_path, 'w') as f:
                json.dump(gpu_info, f, indent=2)
            logger.info(f"  ✓ Saved GPU info")
            logger.info(f"    Peak VRAM usage: {gpu_info['max_memory_allocated_gb']:.2f} GB")


def main():
    """Main entry point."""
    mill_number = 6
    start_date = "2025-08-26"
    end_date = "2025-10-26"
    
    config = PipelineConfig.create_default(mill_number, start_date, end_date)
    
    # Set max VRAM to 14GB (leaving 2GB buffer from 16GB)
    trainer = GaussianProcessCascadeTrainerGPU(config, max_vram_gb=14.0)
    trainer.run()
    
    logger.info("\n✓ GPU-accelerated Gaussian Process training completed successfully!")


if __name__ == "__main__":
    main()
