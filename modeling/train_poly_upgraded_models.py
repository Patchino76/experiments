"""
Improved Polynomial Model Training Pipeline

Enhanced version with:
- Multiple regularization methods (Ridge, ElasticNet, Lasso)
- Wider hyperparameter search
- Better scoring metrics (R² instead of MAE)
- Data quality diagnostics
- Sample weighting for quality model (instead of hard filtering)
- Residual analysis and visualization
- Comprehensive logging and metrics

Usage:
    python train_poly_upgraded_models.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import PipelineConfig
from model_improvements import (
    diagnose_data_quality,
    compute_target_weights,
    plot_residuals,
    plot_cv_results,
    log_cv_diagnostics,
    plot_train_test_predictions
)

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train_models.log', encoding='utf-8')
    ]
)

# Set UTF-8 encoding for console output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
logger = logging.getLogger(__name__)


class ImprovedPolynomialCascadeTrainer:
    """Trains improved polynomial regression cascade models for mill optimization."""

    def __init__(self, config: PipelineConfig):
        """Initialize trainer."""
        self.config = config
        self.mill_number = config.data.mill_number
        self.model_dir = config.paths.output_dir / f"mill_poly_upgraded_{self.mill_number:02d}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots subdirectory
        self.plots_dir = self.model_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Feature configuration
        self.mv_features = config.data.mv_features
        self.cv_features = config.data.cv_features
        self.dv_features = config.data.dv_features
        self.dv_target = config.data.target

        # Storage
        self.models = {}
        self.training_results = {}
        self.metadata = {}
        self.data_diagnostics = {}

        self.df = None
        self.random_state = config.model.random_state

    def run(self):
        """Execute the complete training pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info(f"STARTING IMPROVED POLYNOMIAL CASCADE TRAINING - MILL {self.mill_number}")
        logger.info("=" * 80)

        self.load_data()
        self.diagnose_data()
        train_df, test_df = self.split_data()
        self.initialize_metadata()
        self.train_process_models(train_df, test_df)
        self.train_quality_model(train_df, test_df)
        self.validate_cascade()
        self.save_results()

        logger.info("\n" + "=" * 80)
        logger.info("IMPROVED POLYNOMIAL MODEL TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Models saved to: {self.model_dir}")
        logger.info(f"Plots saved to: {self.plots_dir}")

    def load_data(self):
        """Load segmented data."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("-" * 80)

        # Try multiple possible filenames
        possible_files = [
            self.config.paths.output_dir / f'segmented_motifs_all_{self.mill_number:02d}.csv',
        ]
        
        data_path = None
        for path in possible_files:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                f"Segmented data not found. Tried: {[str(p) for p in possible_files]}. "
                "Please run prepare_data.py first."
            )

        logger.info(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path, parse_dates=['TimeStamp'] if 'TimeStamp' in pd.read_csv(data_path, nrows=0).columns else None)

        required_cols = self.mv_features + self.cv_features + self.dv_features + [self.dv_target]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        original_shape = self.df.shape
        self.df = self.df.dropna(subset=required_cols)
        cleaned_shape = self.df.shape

        rows_removed = original_shape[0] - cleaned_shape[0]
        reduction_pct = (rows_removed / original_shape[0] * 100) if original_shape[0] > 0 else 0

        logger.info(f"  Data shape: {cleaned_shape}")
        logger.info(f"  Rows removed (NaN): {rows_removed} ({reduction_pct:.2f}%)")

    def diagnose_data(self):
        """Run data quality diagnostics."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 2: DATA QUALITY DIAGNOSTICS")
        logger.info("-" * 80)
        
        all_features = self.mv_features + self.cv_features + self.dv_features + [self.dv_target]
        self.data_diagnostics = diagnose_data_quality(self.df, all_features)

    def split_data(self):
        """Split data into train and test sets."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 3: SPLITTING DATA")
        logger.info("-" * 80)

        train_size = int(len(self.df) * (1 - self.config.model.test_size))
        train_df = self.df.iloc[:train_size]
        test_df = self.df.iloc[train_size:]

        logger.info(f"  Train size: {len(train_df)}")
        logger.info(f"  Test size: {len(test_df)}")
        logger.info(f"  Test ratio: {self.config.model.test_size:.1%}")

        return train_df, test_df

    def initialize_metadata(self):
        """Initialize model metadata."""
        self.metadata = {
            "mill_number": self.mill_number,
            "created_at": datetime.now().isoformat(),
            "model_version": "2.0.0",
            "improvements": [
                "Multiple regularization methods (Ridge, ElasticNet, Lasso)",
                "Wider hyperparameter search",
                "Sample weighting instead of hard filtering",
                "R² scoring instead of MAE",
                "Comprehensive diagnostics and visualization"
            ],
            "training_config": {
                "test_size": self.config.model.test_size,
                "data_shape": list(self.df.shape),
                "training_timestamp": datetime.now().isoformat(),
                "configured_features": {
                    "mv_features": self.mv_features,
                    "cv_features": self.cv_features,
                    "dv_features": self.dv_features,
                    "target_variable": self.dv_target,
                    "using_custom_features": False
                }
            },
            "model_performance": {},
            "data_diagnostics": self.data_diagnostics
        }

    def train_process_models(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train process models (MV → CV)."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 4: TRAINING PROCESS MODELS (MV → CV)")
        logger.info("-" * 80)

        for cv_var in self.cv_features:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model: MV → {cv_var}")
            logger.info(f"{'='*60}")

            X_train = train_df[self.mv_features]
            y_train = train_df[cv_var]
            X_test = test_df[self.mv_features]
            y_test = test_df[cv_var]

            result = self._train_single_model_improved(
                X_train, y_train, X_test, y_test,
                model_name=f"process_model_{cv_var}",
                output_var=cv_var
            )

            self.metadata["model_performance"][f"process_model_{cv_var}"] = result

    def train_quality_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train quality model (CV + DV → Target) with sample weighting."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 5: TRAINING QUALITY MODEL (CV + DV → Target)")
        logger.info("-" * 80)

        # Use wider range instead of hard filtering
        train_mask = (train_df[self.dv_target] > 10) & (train_df[self.dv_target] < 35)
        test_mask = (test_df[self.dv_target] > 10) & (test_df[self.dv_target] < 35)

        train_df_filtered = train_df[train_mask]
        test_df_filtered = test_df[test_mask]

        logger.info(f"  Filtered training data: {len(train_df)} → {len(train_df_filtered)} samples")
        logger.info(f"  Filtered test data: {len(test_df)} → {len(test_df_filtered)} samples")

        input_features = self.cv_features + self.dv_features
        logger.info(f"\nTraining model: {input_features} → {self.dv_target}")

        X_train = train_df_filtered[input_features]
        y_train = train_df_filtered[self.dv_target]
        X_test = test_df_filtered[input_features]
        y_test = test_df_filtered[self.dv_target]

        # Compute sample weights (emphasize target range 16-30)
        sample_weights = compute_target_weights(y_train, target_min=16, target_max=30, 
                                               in_range_weight=2.0, out_range_weight=1.0)

        result = self._train_single_model_improved(
            X_train, y_train, X_test, y_test,
            model_name="quality_model",
            output_var=self.dv_target,
            sample_weights=sample_weights
        )

        result["model_type"] = "quality_model"
        result["cv_vars"] = self.cv_features
        result["dv_vars"] = self.dv_features

        self.metadata["model_performance"]["quality_model"] = result

    def _train_single_model_improved(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        output_var: str,
        sample_weights: np.ndarray = None
    ) -> dict:
        """
        Improved training with multiple regularization methods and better hyperparameters.
        """
        
        pipelines = {
            'ridge': Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(max_iter=10000))
            ]),
            'elasticnet': Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNet(max_iter=10000, tol=1e-4))
            ]),
            'lasso': Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(max_iter=10000, tol=1e-4))
            ])
        }

        param_grids = {
            'ridge': {
                "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            },
            'elasticnet': {
                "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            },
            'lasso': {
                "model__alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            }
        }

        # Try each model type and keep the best
        best_score = -np.inf
        best_pipeline = None
        best_model_type = None
        best_params = None
        best_search = None
        
        inner_cv = TimeSeriesSplit(n_splits=self.config.model.cv_splits)
        
        for model_type, pipeline in pipelines.items():
            logger.info(f"\n    Testing {model_type.upper()}...")
            
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[model_type],
                scoring="r2",
                cv=inner_cv,
                n_jobs=-1,
                verbose=0,
                refit=True
            )
            
            try:
                # Fit with sample weights if provided
                if sample_weights is not None:
                    search.fit(X_train, y_train, model__sample_weight=sample_weights)
                else:
                    search.fit(X_train, y_train)
                
                # Check test performance
                test_pred = search.best_estimator_.predict(X_test)
                test_r2 = r2_score(y_test, test_pred)
                
                logger.info(f"      Best CV score: {search.best_score_:.4f}")
                logger.info(f"      Test R²: {test_r2:.4f}")
                logger.info(f"      Best params: {search.best_params_}")
                
                # Keep track of best model based on CV score
                if search.best_score_ > best_score:
                    best_score = search.best_score_
                    best_pipeline = search.best_estimator_
                    best_model_type = model_type
                    best_params = search.best_params_
                    best_search = search
                    
            except Exception as e:
                logger.warning(f"      Failed to train {model_type}: {e}")
                continue
        
        if best_pipeline is None:
            raise ValueError("All model types failed to train")
        
        logger.info(f"\n  ✓ Best model type: {best_model_type.upper()}")
        logger.info("  Best parameters:")
        for key, value in best_params.items():
            logger.info(f"    {key}: {value}")
        
        # Log CV diagnostics
        log_cv_diagnostics(best_search, top_n=5)
        
        # Calculate metrics
        train_pred = best_pipeline.predict(X_train)
        test_pred = best_pipeline.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        logger.info(f"\n  Training Performance:")
        logger.info(f"    R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        logger.info(f"  Test Performance:")
        logger.info(f"    R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

        # Create train/test prediction plot
        predictions_plot_path = self.plots_dir / f"{model_name}_predictions.png"
        plot_train_test_predictions(
            y_train.values if isinstance(y_train, pd.Series) else y_train,
            train_pred,
            train_r2,
            y_test.values if isinstance(y_test, pd.Series) else y_test,
            test_pred,
            test_r2,
            title=f"{model_name} - {output_var}",
            save_path=str(predictions_plot_path)
        )

        # Calculate overfitting indicator
        r2_gap = train_r2 - test_r2
        if r2_gap > 0.15:
            logger.warning(f"    ⚠ Possible overfitting detected (R² gap: {r2_gap:.4f})")
        elif r2_gap < 0:
            logger.info(f"    ✓ Good generalization (Test R² > Train R²)")
        else:
            logger.info(f"    ✓ Healthy bias-variance tradeoff (R² gap: {r2_gap:.4f})")
        
        # Extract feature importance
        model_step = best_pipeline.named_steps["model"]
        feature_names = X_train.columns
        coefficients = model_step.coef_
        
        # Sort by absolute coefficient value
        feature_importance = {
            feat: float(coeff)
            for feat, coeff in zip(feature_names, coefficients)
        }
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: abs(x[1]), 
                               reverse=True)
        
        # Log top 15 most important features
        logger.info("\n  Top 15 Feature Coefficients:")
        for feat, coeff in sorted_features[:15]:
            logger.info(f"    {feat}: {coeff:.6f}")
        
        # Count non-zero coefficients (for Lasso/ElasticNet)
        non_zero_coeffs = np.sum(np.abs(coefficients) > 1e-10)
        logger.info(f"\n  Feature Selection:")
        logger.info(f"    Total features: {len(coefficients)}")
        logger.info(f"    Non-zero coefficients: {non_zero_coeffs}")
        logger.info(f"    Sparsity: {(1 - non_zero_coeffs/len(coefficients))*100:.1f}%")
        
        # Create visualizations
        logger.info("\n  Creating visualizations...")
        
        # Residual plots
        residual_path = self.plots_dir / f"{model_name}_residuals.png"
        plot_residuals(y_test.values, test_pred, 
                      title=f"{model_name} - {output_var}", 
                      save_path=str(residual_path))
        
        # CV results plot
        cv_results_df = pd.DataFrame(best_search.cv_results_).sort_values('rank_test_score')
        cv_plot_path = self.plots_dir / f"{model_name}_cv_results.png"
        plot_cv_results(cv_results_df, save_path=str(cv_plot_path))
        
        plt.close('all')  # Clean up
        
        # Save model
        model_path = self.model_dir / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_pipeline, f)
        logger.info(f"  ✓ Model saved to {model_path.name}")
        
        self.models[model_name] = best_pipeline
        
        return {
            "r2_score": float(test_r2),
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "train_r2": float(train_r2),
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "overfitting_gap": float(r2_gap),
            "feature_importance": feature_importance,
            "top_features": dict(sorted_features[:20]),
            "model_type": "process_model",
            "regularization_method": best_model_type,
            "input_vars": list(X_train.columns),
            "output_var": output_var,
            "best_params": best_params,
            "n_features_total": len(coefficients),
            "n_features_nonzero": int(non_zero_coeffs),
            "cv_score": float(best_score)
        }

    def validate_cascade(self):
        """Validate the full cascade: MV → CV, then CV + DV → Target."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 6: VALIDATING CASCADE")
        logger.info("-" * 80)

        test_size = int(len(self.df) * self.config.model.test_size)
        test_df = self.df.iloc[-test_size:].copy()

        n_samples = min(self.config.model.cascade_validation_samples, len(test_df))
        if n_samples == 0:
            raise ValueError("Not enough samples for cascade validation.")

        rng = np.random.default_rng(self.random_state)
        sample_indices = rng.choice(test_df.index, size=n_samples, replace=False)
        sample_df = test_df.loc[sample_indices].copy()

        logger.info(f"  Validating with {n_samples} samples...")

        X_mv = sample_df[self.mv_features]

        cv_predictions = {}
        for cv_var in self.cv_features:
            model = self.models[f"process_model_{cv_var}"]
            cv_predictions[cv_var] = model.predict(X_mv)

        X_cv = pd.DataFrame(cv_predictions, index=sample_df.index)

        X_dv = sample_df[self.dv_features]
        X_quality = pd.concat([X_cv, X_dv], axis=1)

        quality_model = self.models["quality_model"]
        dv_predictions = quality_model.predict(X_quality)

        y_actual = sample_df[self.dv_target].values

        r2 = r2_score(y_actual, dv_predictions)
        rmse = np.sqrt(mean_squared_error(y_actual, dv_predictions))
        mae = mean_absolute_error(y_actual, dv_predictions)

        logger.info("  Cascade Validation Results:")
        logger.info(f"    R²: {r2:.4f}")
        logger.info(f"    RMSE: {rmse:.4f}")
        logger.info(f"    MAE: {mae:.4f}")
        
        # Create cascade validation plot
        cascade_plot_path = self.plots_dir / "cascade_validation.png"
        plot_residuals(y_actual, dv_predictions, 
                      title="Cascade Validation", 
                      save_path=str(cascade_plot_path))
        plt.close('all')

        chain_validation = {
            "r2_score": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "n_samples": int(n_samples),
            "predictions": [float(x) for x in dv_predictions],
            "actuals": [float(x) for x in y_actual],
            "validation_error": None
        }

        self.metadata["model_performance"]["chain_validation"] = chain_validation

    def save_results(self):
        """Save training results and metadata."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 7: SAVING RESULTS")
        logger.info("-" * 80)

        training_results = {
            "process_models": {
                cv_var: self.metadata["model_performance"][f"process_model_{cv_var}"]
                for cv_var in self.cv_features
            },
            "quality_model": self.metadata["model_performance"]["quality_model"],
            "chain_validation": self.metadata["model_performance"]["chain_validation"],
            "training_timestamp": datetime.now().isoformat(),
            "data_shape": list(self.df.shape),
            "model_save_path": str(self.model_dir.absolute()),
            "mill_number": self.mill_number,
            "feature_configuration": {
                "mv_features": self.mv_features,
                "cv_features": self.cv_features,
                "dv_features": self.dv_features,
                "target_variable": self.dv_target,
                "using_custom_features": False
            },
            "improvements_applied": self.metadata["improvements"]
        }

        metadata_path = self.model_dir / "metadata_poly_upgraded.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"  ✓ Metadata saved to {metadata_path.name}")

        results_path = self.model_dir / "training_results_poly_upgraded.json"
        with open(results_path, "w") as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"  ✓ Training results saved to {results_path.name}")

        logger.info(f"\n  Files created in {self.model_dir}:")
        for file in sorted(self.model_dir.iterdir()):
            if file.is_file():
                logger.info(f"    - {file.name}")
        
        logger.info(f"\n  Plots created in {self.plots_dir}:")
        for file in sorted(self.plots_dir.iterdir()):
            logger.info(f"    - {file.name}")


def main():
    """Main entry point."""
    mill_number = 6
    start_date = "2025-01-01"
    end_date = "2025-10-19"

    config = PipelineConfig.create_default(mill_number, start_date, end_date)

    trainer = ImprovedPolynomialCascadeTrainer(config)
    trainer.run()

    logger.info("\n✓ Improved polynomial model training pipeline completed successfully!")


if __name__ == "__main__":
    main()
