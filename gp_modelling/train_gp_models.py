"""
Gaussian Process Regression Model Training Pipeline

Implements GPR models for ball mill optimization with uncertainty quantification.

Usage:
    python train_gp_models.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

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
        logging.FileHandler(GP_BASE_DIR / 'train_gp_models.log', encoding='utf-8')
    ]
)

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
logger = logging.getLogger(__name__)


def rebase_paths_to_gp(config: PipelineConfig) -> PipelineConfig:
    """Ensure config paths point to the gp_modelling directory."""
    output_dir = GP_BASE_DIR / "output"
    models_dir = GP_BASE_DIR / "models"
    analysis_dir = output_dir / "analysis"
    plots_dir = output_dir / "plots"

    for path in [output_dir, models_dir, analysis_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    config.paths.output_dir = output_dir
    config.paths.models_dir = models_dir
    config.paths.analysis_dir = analysis_dir
    config.paths.plots_dir = plots_dir
    return config


class GaussianProcessCascadeTrainer:
    """Trains Gaussian Process Regression cascade models for mill optimization."""

    def __init__(self, config: PipelineConfig):
        self.config = rebase_paths_to_gp(config)
        self.mill_number = config.data.mill_number
        self.model_dir = config.paths.output_dir / f"mill_gp_{self.mill_number:02d}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.model_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.mv_features = config.data.mv_features
        self.cv_features = config.data.cv_features
        self.dv_features = config.data.dv_features
        self.dv_target = config.data.target

        self.models = {}
        self.scalers = {}
        self.training_results = {}
        self.metadata = {}
        self.df = None
        self.random_state = config.model.random_state

    def run(self):
        """Execute the complete training pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info(f"STARTING GAUSSIAN PROCESS CASCADE TRAINING - MILL {self.mill_number}")
        logger.info("=" * 80)

        self.load_data()
        train_df, test_df = self.split_data()
        self.initialize_metadata()
        self.train_process_models(train_df, test_df)
        self.train_quality_model(train_df, test_df)
        self.validate_cascade(test_df)
        self.save_results()

        logger.info("\n" + "=" * 80)
        logger.info("GAUSSIAN PROCESS MODEL TRAINING COMPLETE")
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
            "model_version": "1.0.0",
            "model_type": "Gaussian Process Regression",
            "kernel": "Matern(nu=2.5) + WhiteKernel",
            "features": {
                "mv_features": self.mv_features,
                "cv_features": self.cv_features,
                "dv_features": self.dv_features,
                "target_variable": self.dv_target,
            },
            "training_config": {
                "test_size": self.config.model.test_size,
                "data_shape": list(self.df.shape),
                "n_restarts_optimizer": 10
            },
            "model_performance": {}
        }

    def train_process_models(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train process models (MV → CV)."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 3: TRAINING PROCESS MODELS (MV → CV)")
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
        logger.info("STEP 4: TRAINING QUALITY MODEL (CV + DV → Target)")
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
        model_name, output_var, input_features
    ):
        """Train a single Gaussian Process model."""
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define kernel
        kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                 WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        
        logger.info(f"  Training with {len(X_train)} samples...")
        
        # Create and train GP model
        gp_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=self.random_state,
            normalize_y=True,
            alpha=1e-10
        )
        
        gp_model.fit(X_train_scaled, y_train)
        
        logger.info(f"  ✓ Training complete")
        logger.info(f"  Optimized kernel: {gp_model.kernel_}")
        
        # Predictions with uncertainty
        train_pred, train_sigma = gp_model.predict(X_train_scaled, return_std=True)
        test_pred, test_sigma = gp_model.predict(X_test_scaled, return_std=True)
        
        # Metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        logger.info(f"  Train: R²={train_r2:.4f}, RMSE={train_rmse:.4f}, σ={np.mean(train_sigma):.4f}")
        logger.info(f"  Test:  R²={test_r2:.4f}, RMSE={test_rmse:.4f}, σ={np.mean(test_sigma):.4f}")
        
        # Store model and scaler
        self.models[model_name] = gp_model
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
            "kernel": str(gp_model.kernel_),
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
            X_scaled = scaler.transform(X_mv)
            pred, sigma = model.predict(X_scaled, return_std=True)
            
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
        X_quality_scaled = scaler.transform(X_quality)
        y_cascade_pred, y_cascade_sigma = model.predict(X_quality_scaled, return_std=True)
        
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
        """Save models, scalers, and metadata."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 6: SAVING RESULTS")
        logger.info("-" * 80)

        # Save models
        for model_name, model in self.models.items():
            model_path = self.model_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"  ✓ Saved {model_name}")

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


def main():
    """Main entry point."""
    mill_number = 8
    start_date = "2025-08-01"
    end_date = "2025-10-19"
    
    config = PipelineConfig.create_default(mill_number, start_date, end_date)
    
    trainer = GaussianProcessCascadeTrainer(config)
    trainer.run()
    
    logger.info("\n✓ Gaussian Process training completed successfully!")


if __name__ == "__main__":
    main()
