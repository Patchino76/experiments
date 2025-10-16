"""XGBoost Cascade Model Training Script

This script trains cascade models for mill process optimization:
1. Process Models (MV → CV): Predict controlled variables from manipulated variables
   - PulpHC from [Ore, WaterMill, WaterZumpf, MotorAmp]
   - DensityHC from [Ore, WaterMill, WaterZumpf, MotorAmp]
   - PressureHC from [Ore, WaterMill, WaterZumpf, MotorAmp]

2. Quality Model (CV → DV): Predict dependent variable from controlled variables
   - PSI200 from [PulpHC, DensityHC, PressureHC]

Models are saved with metadata compatible with API usage.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

# Setup paths
script_dir = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == script_dir:
    sys.path.append(sys.path.pop(0))


class CascadeModelTrainer:
    """Trains and saves cascade models for mill optimization."""
    
    def __init__(self, mill_number: int, data_path: Path, output_dir: Path):
        self.mill_number = mill_number
        self.data_path = data_path
        self.output_dir = output_dir / f"mill_{mill_number}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature configuration
        self.mv_features = ["Ore", "WaterMill", "WaterZumpf", "MotorAmp"]
        self.cv_features = ["PulpHC", "DensityHC", "PressureHC"]
        self.dv_target = "PSI200"
        
        # Storage for results
        self.models = {}
        self.scalers = {}
        self.training_results = {}
        self.metadata = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the segmented motifs data."""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path, parse_dates=["TimeStamp"])
        
        # Check for required columns
        required_cols = self.mv_features + self.cv_features + [self.dv_target]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with NaN in critical columns
        original_shape = df.shape
        df = df.dropna(subset=required_cols)
        cleaned_shape = df.shape
        
        rows_removed = original_shape[0] - cleaned_shape[0]
        reduction_pct = (rows_removed / original_shape[0] * 100) if original_shape[0] > 0 else 0
        
        print(f"Data shape: {cleaned_shape}")
        print(f"Rows removed (NaN): {rows_removed} ({reduction_pct:.2f}%)")
        
        self.metadata["data_info"] = {
            "original_shape": list(original_shape),
            "cleaned_shape": list(cleaned_shape),
            "rows_removed": rows_removed,
            "data_reduction": f"{reduction_pct:.1f}%"
        }
        
        return df
    
    def train_process_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           output_var: str) -> dict:
        """Train a single process model (MV → CV)."""
        print(f"\n{'='*60}")
        print(f"Training Process Model: MV → {output_var}")
        print(f"{'='*60}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define model
        base_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        
        # Hyperparameter tuning
        param_grid = {
            "n_estimators": [150, 300, 400],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 8],
            "subsample": [0.6, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        
        inner_cv = TimeSeriesSplit(n_splits=5)
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1,
            verbose=1
        )
        
        print("Running GridSearchCV...")
        search.fit(X_train_scaled, y_train)
        model = search.best_estimator_
        
        print(f"\nBest parameters:")
        for key, value in search.best_params_.items():
            print(f"  {key}: {value}")
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        print(f"\nTrain R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
        print(f"Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
        
        # Feature importance
        feature_importance = {k: float(v) for k, v in zip(X_train.columns, model.feature_importances_)}
        print(f"\nFeature Importance:")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")
        
        # Save model and scaler
        model_name = f"process_model_{output_var}"
        scaler_name = f"scaler_mv_to_{output_var}"
        
        with open(self.output_dir / f"{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(self.output_dir / f"{scaler_name}.pkl", "wb") as f:
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
    
    def train_quality_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Train quality model (CV → DV)."""
        print(f"\n{'='*60}")
        print(f"Training Quality Model: CV → {self.dv_target}")
        print(f"{'='*60}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define model
        base_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        
        # Hyperparameter tuning
        param_grid = {
            "n_estimators": [150, 300, 400],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 8],
            "subsample": [0.6, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        
        inner_cv = TimeSeriesSplit(n_splits=5)
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1,
            verbose=1
        )
        
        print("Running GridSearchCV...")
        search.fit(X_train_scaled, y_train)
        model = search.best_estimator_
        
        print(f"\nBest parameters:")
        for key, value in search.best_params_.items():
            print(f"  {key}: {value}")
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        print(f"\nTrain R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
        print(f"Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
        
        # Feature importance
        feature_importance = {k: float(v) for k, v in zip(X_train.columns, model.feature_importances_)}
        print(f"\nFeature Importance:")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")
        
        # Save model and scaler
        model_name = "quality_model"
        scaler_name = "scaler_quality_model"
        
        with open(self.output_dir / f"{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(self.output_dir / f"{scaler_name}.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        self.models[model_name] = model
        self.scalers[scaler_name] = scaler
        
        return {
            "r2_score": float(test_r2),
            "rmse": float(test_rmse),
            "feature_importance": feature_importance,
            "model_type": "quality_model",
            "input_vars": list(X_train.columns),
            "output_var": self.dv_target,
            "cv_vars": list(X_train.columns),
            "dv_vars": []
        }
    
    def validate_cascade(self, df: pd.DataFrame, n_samples: int = 200) -> dict:
        """Validate the full cascade: MV → CV → DV."""
        print(f"\n{'='*60}")
        print(f"Validating Full Cascade Chain")
        print(f"{'='*60}")
        
        # Use test set samples
        test_size = int(len(df) * 0.2)
        test_df = df.iloc[-test_size:].copy()
        
        # Sample random points
        n_samples = min(n_samples, len(test_df))
        sample_indices = np.random.choice(test_df.index, size=n_samples, replace=False)
        sample_df = test_df.loc[sample_indices].copy()
        
        # Get MV inputs
        X_mv = sample_df[self.mv_features].values
        
        # Predict CV variables using process models
        cv_predictions = {}
        for cv_var in self.cv_features:
            model = self.models[f"process_model_{cv_var}"]
            scaler = self.scalers[f"scaler_mv_to_{cv_var}"]
            # Create DataFrame with proper feature names to avoid warnings
            X_mv_df = pd.DataFrame(X_mv, columns=self.mv_features)
            X_scaled = scaler.transform(X_mv_df)
            cv_predictions[cv_var] = model.predict(X_scaled)
        
        # Create CV dataframe
        X_cv = pd.DataFrame(cv_predictions)
        
        # Predict DV using quality model
        quality_model = self.models["quality_model"]
        quality_scaler = self.scalers["scaler_quality_model"]
        X_cv_scaled = quality_scaler.transform(X_cv)
        dv_predictions = quality_model.predict(X_cv_scaled)
        
        # Get actual values
        y_actual = sample_df[self.dv_target].values
        
        # Calculate metrics
        r2 = r2_score(y_actual, dv_predictions)
        rmse = np.sqrt(mean_squared_error(y_actual, dv_predictions))
        mae = mean_absolute_error(y_actual, dv_predictions)
        
        print(f"\nCascade Validation Results (n={n_samples}):")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        return {
            "r2_score": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "n_samples": int(n_samples),
            "n_requested": int(n_samples),
            "predictions": [float(x) for x in dv_predictions],
            "actuals": [float(x) for x in y_actual],
            "validation_error": None
        }
    
    def train_all_models(self):
        """Train all cascade models."""
        print(f"\n{'#'*60}")
        print(f"# CASCADE MODEL TRAINING - MILL {self.mill_number}")
        print(f"{'#'*60}\n")
        
        # Load data
        df = self.load_data()
        
        # Split data (80/20 train/test)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")
        
        # Initialize metadata
        self.metadata = {
            "mill_number": self.mill_number,
            "created_at": datetime.now().isoformat(),
            "model_version": "1.0.0",
            "training_config": {
                "test_size": 0.2,
                "data_shape": list(df.shape),
                "training_timestamp": datetime.now().isoformat(),
                "configured_features": {
                    "mv_features": self.mv_features,
                    "cv_features": self.cv_features,
                    "dv_features": [],
                    "target_variable": self.dv_target,
                    "using_custom_features": True
                }
            },
            "model_performance": {},
            "steady_state_processing": {
                "enabled": False
            }
        }
        
        # Train process models (MV → CV)
        process_results = {}
        for cv_var in self.cv_features:
            X_train = train_df[self.mv_features]
            y_train = train_df[cv_var]
            X_test = test_df[self.mv_features]
            y_test = test_df[cv_var]
            
            result = self.train_process_model(X_train, y_train, X_test, y_test, cv_var)
            process_results[cv_var] = result
            self.metadata["model_performance"][f"process_model_{cv_var}"] = result
        
        # Train quality model (CV → DV)
        X_train = train_df[self.cv_features]
        y_train = train_df[self.dv_target]
        X_test = test_df[self.cv_features]
        y_test = test_df[self.dv_target]
        
        quality_result = self.train_quality_model(X_train, y_train, X_test, y_test)
        self.metadata["model_performance"]["quality_model"] = quality_result
        
        # Validate cascade
        chain_validation = self.validate_cascade(df, n_samples=200)
        self.metadata["model_performance"]["chain_validation"] = chain_validation
        
        # Save training results
        training_results = {
            "process_models": process_results,
            "quality_model": quality_result,
            "chain_validation": chain_validation,
            "training_timestamp": datetime.now().isoformat(),
            "data_shape": list(df.shape),
            "model_save_path": str(self.output_dir.absolute()),
            "mill_number": self.mill_number,
            "feature_configuration": {
                "mv_features": self.mv_features,
                "cv_features": self.cv_features,
                "dv_features": [],
                "target_variable": self.dv_target,
                "using_custom_features": 1
            }
        }
        
        # Save metadata and results
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        with open(self.output_dir / "training_results.json", "w") as f:
            json.dump(training_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Models saved to: {self.output_dir}")
        print(f"\nFiles created:")
        for file in sorted(self.output_dir.iterdir()):
            print(f"  - {file.name}")


def main():
    """Main training pipeline."""
    # Configuration
    mill_number = 6
    data_path = script_dir / "output" / "segmented_motifsMV.csv"
    output_dir = script_dir / "cascade_models"
    
    # Check if data file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Train models
    trainer = CascadeModelTrainer(mill_number, data_path, output_dir)
    trainer.train_all_models()
    
    print("\n✓ Cascade model training completed successfully!")


if __name__ == "__main__":
    main()
