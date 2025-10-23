"""
Polynomial Model Training Pipeline

Trains polynomial regression cascade models for mill process optimization.

Usage:
    python train_poly_models.py
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


# Add parent to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import PipelineConfig

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


class PolynomialCascadeTrainer:
    """Trains polynomial regression cascade models for mill optimization."""

    def __init__(self, config: PipelineConfig):
        """Initialize trainer."""
        self.config = config
        self.mill_number = config.data.mill_number
        self.model_dir = config.paths.output_dir / f"mill_polly_{self.mill_number:02d}"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Feature configuration
        self.mv_features = config.data.mv_features
        self.cv_features = config.data.cv_features
        self.dv_features = config.data.dv_features
        self.dv_target = config.data.target

        # Storage
        self.models = {}
        self.training_results = {}
        self.metadata = {}

        self.df = None
        self.random_state = config.model.random_state

    def run(self):
        """Execute the complete training pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info(f"STARTING POLYNOMIAL CASCADE TRAINING - MILL {self.mill_number}")
        logger.info("=" * 80)

        self.load_data()
        train_df, test_df = self.split_data()
        self.initialize_metadata()
        self.train_process_models(train_df, test_df)
        self.train_quality_model(train_df, test_df)
        self.validate_cascade()
        self.save_results()

        logger.info("\n" + "=" * 80)
        logger.info("POLYNOMIAL MODEL TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Models saved to: {self.model_dir}")

    def load_data(self):
        """Load segmented data."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("-" * 80)

        data_path = self.config.paths.output_dir / 'segmented_motifs_all.csv'

        if not data_path.exists():
            raise FileNotFoundError(
                f"Segmented data not found at {data_path}. "
                "Please run prepare_data.py first."
            )

        logger.info(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path, parse_dates=['TimeStamp'])

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

    def split_data(self):
        """Split data into train and test sets."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 2: SPLITTING DATA")
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
            "model_version": "1.0.0",
            "training_config": {
                "test_size": self.config.model.test_size,
                "data_shape": list(self.df.shape),
                "training_timestamp": datetime.now().isoformat(),
                "configured_features": {
                    "mv_features": self.mv_features,
                    "cv_features": self.cv_features,
                    "dv_features": self.dv_features,
                    "target_variable": self.dv_target,
                    "using_custom_features": True
                }
            },
            "model_performance": {},
            "steady_state_processing": {
                "enabled": False
            }
        }

    def train_process_models(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train process models (MV → CV)."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 3: TRAINING PROCESS MODELS (MV → CV)")
        logger.info("-" * 80)

        for cv_var in self.cv_features:
            logger.info(f"\nTraining model: MV → {cv_var}")

            X_train = train_df[self.mv_features]
            y_train = train_df[cv_var]
            X_test = test_df[self.mv_features]
            y_test = test_df[cv_var]

            result = self._train_single_model(
                X_train, y_train, X_test, y_test,
                model_name=f"process_model_{cv_var}",
                output_var=cv_var
            )

            self.metadata["model_performance"][f"process_model_{cv_var}"] = result

    def train_quality_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train quality model (CV + DV → Target)."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 4: TRAINING QUALITY MODEL (CV + DV → Target)")
        logger.info("-" * 80)

        train_mask = (train_df[self.dv_target] > 16) & (train_df[self.dv_target] < 30)
        test_mask = (test_df[self.dv_target] > 16) & (test_df[self.dv_target] < 30)

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

        result = self._train_single_model(
            X_train, y_train, X_test, y_test,
            model_name="quality_model",
            output_var=self.dv_target
        )

        result["model_type"] = "quality_model"
        result["cv_vars"] = self.cv_features
        result["dv_vars"] = self.dv_features

        self.metadata["model_performance"]["quality_model"] = result

    def _train_single_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        output_var: str
    ) -> dict:
        """Train a single polynomial regression model with hyperparameter tuning."""

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(include_bias=False)),
            ("ridge", Ridge())
        ])

        param_grid = {
            "poly__degree": [2, 3],
            "ridge__alpha": [0.1, 1.0, 10.0],
        }

        logger.info("  Running GridSearchCV...")
        inner_cv = TimeSeriesSplit(n_splits=self.config.model.cv_splits)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_

        logger.info("  Best parameters:")
        for key, value in search.best_params_.items():
            logger.info(f"    {key}: {value}")

        train_pred = best_pipeline.predict(X_train)
        test_pred = best_pipeline.predict(X_test)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        logger.info(f"  Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
        logger.info(f"  Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

        poly_step = best_pipeline.named_steps["poly"]
        ridge_step = best_pipeline.named_steps["ridge"]
        feature_names = poly_step.get_feature_names_out(X_train.columns)
        coefficients = ridge_step.coef_
        feature_importance = {
            feat: float(coeff)
            for feat, coeff in zip(feature_names, coefficients)
        }

        logger.info("  Coefficients (sorted by absolute value):")
        for feat, coeff in sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True):
            logger.info(f"    {feat}: {coeff:.4f}")

        with open(self.model_dir / f"{model_name}.pkl", "wb") as f:
            pickle.dump(best_pipeline, f)

        self.models[model_name] = best_pipeline

        return {
            "r2_score": float(test_r2),
            "rmse": float(test_rmse),
            "feature_importance": feature_importance,
            "model_type": "process_model",
            "input_vars": list(X_train.columns),
            "output_var": output_var,
            "best_params": search.best_params_
        }

    def validate_cascade(self):
        """Validate the full cascade: MV → CV, then CV + DV → Target."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 5: VALIDATING CASCADE")
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

        chain_validation = {
            "r2_score": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "n_samples": int(n_samples),
            "n_requested": int(n_samples),
            "predictions": [float(x) for x in dv_predictions],
            "actuals": [float(x) for x in y_actual],
            "validation_error": None
        }

        self.metadata["model_performance"]["chain_validation"] = chain_validation

    def save_results(self):
        """Save training results and metadata."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 6: SAVING RESULTS")
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
                "using_custom_features": 1
            }
        }

        metadata_path = self.model_dir / "metadata_poly.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"  ✓ Metadata saved to {metadata_path.name}")

        results_path = self.model_dir / "training_results_poly.json"
        with open(results_path, "w") as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"  ✓ Training results saved to {results_path.name}")

        logger.info(f"\n  Files created in {self.model_dir}:")
        for file in sorted(self.model_dir.iterdir()):
            logger.info(f"    - {file.name}")


def main():
    """Main entry point."""
    mill_number = 8
    start_date = "2025-01-01"
    end_date = "2025-10-19"

    config = PipelineConfig.create_default(mill_number, start_date, end_date)

    trainer = PolynomialCascadeTrainer(config)
    trainer.run()

    logger.info("\n✓ Polynomial model training pipeline completed successfully!")


if __name__ == "__main__":
    main()
