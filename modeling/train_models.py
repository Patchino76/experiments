"""
Model Training Pipeline

Trains XGBoost cascade models for mill process optimization.

Usage:
    python train_models.py
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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

# Add parent to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import PipelineConfig, ModelConfig

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


class CascadeModelTrainer:
    """Trains cascade models for mill optimization."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize trainer.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.mill_number = config.data.mill_number
        self.model_dir = config.paths.get_mill_model_dir(self.mill_number)
        
        # Feature configuration
        self.mv_features = config.data.mv_features
        self.cv_features = config.data.cv_features
        self.dv_features = config.data.dv_features
        self.dv_target = config.data.target
        
        # Storage
        self.models = {}
        self.scalers = {}
        self.training_results = {}
        self.metadata = {}
        
        self.df = None
    
    def run(self):
        """Execute the complete training pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info(f"STARTING CASCADE MODEL TRAINING - MILL {self.mill_number}")
        logger.info("=" * 80)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Split data
        train_df, test_df = self.split_data()
        
        # Step 3: Initialize metadata
        self.initialize_metadata()
        
        # Step 4: Train process models (MV → CV)
        self.train_process_models(train_df, test_df)
        
        # Step 5: Train quality model (CV → DV)
        self.train_quality_model(train_df, test_df)
        
        # Step 6: Validate cascade
        self.validate_cascade()
        
        # Step 7: Save results
        self.save_results()
        
        logger.info("\n" + "=" * 80)
        logger.info("MODEL TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Models saved to: {self.model_dir}")
    
    def load_data(self):
        """Load segmented data."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("-" * 80)
        
        # data_path = self.config.paths.output_dir / 'segmented_motifsMV.csv'
        data_path = self.config.paths.output_dir / 'segmented_motifs_all.csv'
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Segmented data not found at {data_path}. "
                "Please run prepare_data.py first."
            )
        
        logger.info(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path, parse_dates=['TimeStamp'])
        
        # Validate required columns
        required_cols = self.mv_features + self.cv_features + self.dv_features + [self.dv_target]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove NaN values
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
                scaler_name=f"scaler_mv_to_{cv_var}",
                output_var=cv_var
            )
            
            self.metadata["model_performance"][f"process_model_{cv_var}"] = result
    
    def train_quality_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train quality model (CV + DV → Target)."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 4: TRAINING QUALITY MODEL (CV + DV → Target)")
        logger.info("-" * 80)
        
        # Combine CV and DV features as inputs
        input_features = self.cv_features + self.dv_features
        logger.info(f"\nTraining model: {input_features} → {self.dv_target}")
        
        X_train = train_df[input_features]
        y_train = train_df[self.dv_target]
        X_test = test_df[input_features]
        y_test = test_df[self.dv_target]
        
        result = self._train_single_model(
            X_train, y_train, X_test, y_test,
            model_name="quality_model",
            scaler_name="scaler_quality_model",
            output_var=self.dv_target
        )
        
        # Add additional metadata for quality model
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
        scaler_name: str,
        output_var: str
    ) -> dict:
        """Train a single XGBoost model with hyperparameter tuning."""
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define base model
        base_model = XGBRegressor(
            objective=self.config.model.objective,
            n_estimators=self.config.model.n_estimators,
            learning_rate=self.config.model.learning_rate,
            max_depth=self.config.model.max_depth,
            subsample=self.config.model.subsample,
            colsample_bytree=self.config.model.colsample_bytree,
            random_state=self.config.model.random_state,
        )
        
        # Hyperparameter tuning
        logger.info("  Running GridSearchCV...")
        inner_cv = TimeSeriesSplit(n_splits=self.config.model.cv_splits)
        search = GridSearchCV(
            estimator=base_model,
            param_grid=self.config.model.param_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train_scaled, y_train)
        model = search.best_estimator_
        
        logger.info("  Best parameters:")
        for key, value in search.best_params_.items():
            logger.info(f"    {key}: {value}")
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        logger.info(f"  Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
        logger.info(f"  Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
        
        # Feature importance
        feature_importance = {
            k: float(v) for k, v in zip(X_train.columns, model.feature_importances_)
        }
        logger.info("  Feature Importance:")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {feat}: {imp:.4f}")
        
        # Save model and scaler
        with open(self.model_dir / f"{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(self.model_dir / f"{scaler_name}.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        self.models[model_name] = model
        self.scalers[scaler_name] = scaler
        
        return {
            "r2_score": float(test_r2),
            "rmse": float(test_rmse),
            "feature_importance": feature_importance,
            "model_type": "process_model",
            "input_vars": list(X_train.columns),
            "output_var": output_var
        }
    
    def validate_cascade(self):
        """Validate the full cascade: MV → CV, then CV + DV → Target."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 5: VALIDATING CASCADE")
        logger.info("-" * 80)
        
        # Use test set
        test_size = int(len(self.df) * self.config.model.test_size)
        test_df = self.df.iloc[-test_size:].copy()
        
        # Sample for validation
        n_samples = min(self.config.model.cascade_validation_samples, len(test_df))
        sample_indices = np.random.choice(test_df.index, size=n_samples, replace=False)
        sample_df = test_df.loc[sample_indices].copy()
        
        logger.info(f"  Validating with {n_samples} samples...")
        
        # Get MV inputs
        X_mv = sample_df[self.mv_features].values
        
        # Predict CV variables
        cv_predictions = {}
        for cv_var in self.cv_features:
            model = self.models[f"process_model_{cv_var}"]
            scaler = self.scalers[f"scaler_mv_to_{cv_var}"]
            
            X_mv_df = pd.DataFrame(X_mv, columns=self.mv_features)
            X_scaled = scaler.transform(X_mv_df)
            cv_predictions[cv_var] = model.predict(X_scaled)
        
        # Create CV dataframe
        X_cv = pd.DataFrame(cv_predictions)
        
        # Get actual DV features from sample
        X_dv = sample_df[self.dv_features]
        
        # Combine CV predictions with actual DV features
        X_quality = pd.concat([X_cv, X_dv.reset_index(drop=True)], axis=1)
        
        # Predict target using quality model
        quality_model = self.models["quality_model"]
        quality_scaler = self.scalers["scaler_quality_model"]
        X_quality_scaled = quality_scaler.transform(X_quality)
        dv_predictions = quality_model.predict(X_quality_scaled)
        
        # Get actual values
        y_actual = sample_df[self.dv_target].values
        
        # Calculate metrics
        r2 = r2_score(y_actual, dv_predictions)
        rmse = np.sqrt(mean_squared_error(y_actual, dv_predictions))
        mae = mean_absolute_error(y_actual, dv_predictions)
        
        logger.info(f"  Cascade Validation Results:")
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
        
        # Prepare training results
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
        
        # Save metadata
        metadata_path = self.model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"  ✓ Metadata saved to {metadata_path.name}")
        
        # Save training results
        results_path = self.model_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"  ✓ Training results saved to {results_path.name}")
        
        # List all saved files
        logger.info(f"\n  Files created in {self.model_dir}:")
        for file in sorted(self.model_dir.iterdir()):
            logger.info(f"    - {file.name}")


def main():
    """Main entry point."""
    # Configuration - should match prepare_data.py
    mill_number = 6
    start_date = "2025-08-20"
    end_date = "2025-10-19"
    
    # Create configuration
    config = PipelineConfig.create_default(mill_number, start_date, end_date)
    
    # Optionally customize model configuration
    # config.model.test_size = 0.25
    # config.model.cv_splits = 3
    
    # Run training
    trainer = CascadeModelTrainer(config)
    trainer.run()
    
    logger.info("\n✓ Model training pipeline completed successfully!")


if __name__ == "__main__":
    main()
