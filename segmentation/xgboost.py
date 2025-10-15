import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == script_dir:
    sys.path.append(sys.path.pop(0))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor


def main() -> None:
    data_path = script_dir / "output" / "segmented_motifs.csv"
    df = pd.read_csv(data_path, parse_dates=["TimeStamp"])
    target_column = "DensityHC"
    
    # Base features (available at inference time)
    base_feature_columns = [
        "Ore",
        "WaterMill",
        "WaterZumpf",
        "MotorAmp",
        "Daiki",
        "Shisti",
        "FE",
        "Class_15",
        # "PSI200",
    ]
    # base_feature_columns = [
    #     "PulpHC",
    #     "DensityHC",
    #     "PressureHC",
    #     # "Daiki",
    #     # "Shisti",
    #     # "FE",
    #     # "Class_15",
    #     # "PSI200",
    # ]
    
    # Check if discontinuity markers exist (pre-computed by motif_mv_search.py)
    discontinuity_markers_exist = 'discontinuity_score' in df.columns
    
    if discontinuity_markers_exist:
        print("Using pre-computed discontinuity markers from motif_mv_search.py...")
        # Build feature list: base features + discontinuity markers
        feature_columns = base_feature_columns.copy()
        feature_columns.extend(['discontinuity_score'])
    else:
        print("WARNING: Discontinuity markers not found. Using base features only.")
        feature_columns = base_feature_columns.copy()
    
    # Filter to only existing columns
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    print(f"Total features: {len(feature_columns)}")
    print(f"  Base features: {len(base_feature_columns)}")
    if discontinuity_markers_exist:
        print(f"  Discontinuity markers: 3")
    required_columns = feature_columns + [target_column]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")
    X = df[feature_columns]
    y = df[target_column]
    total_samples = len(X)
    if total_samples < 5:
        raise ValueError("Not enough samples for train/test split.")
    train_cutoff = max(int(total_samples * 0.8), 1)
    X_train, X_test = X.iloc[:train_cutoff], X.iloc[train_cutoff:]
    y_train, y_test = y.iloc[:train_cutoff], y.iloc[train_cutoff:]
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    # model = LinearRegression()
    search = None
    if isinstance(model, XGBRegressor):
        param_grid = {
            "n_estimators": [150, 400],
            "learning_rate": [0.01, 0.09, 0.1],
            "max_depth": [1, 8],
            "subsample": [0.1, 0.9],
            "colsample_bytree": [0.2, 1],
        }
        inner_cv = TimeSeriesSplit(n_splits=6, max_train_size=train_cutoff)
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_rmse = mean_squared_error(y_train, train_predictions) ** 0.5
    test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
    train_metrics = {
        "MAE": mean_absolute_error(y_train, train_predictions),
        "RMSE": train_rmse,
        "R2": r2_score(y_train, train_predictions),
    }
    test_metrics = {
        "MAE": mean_absolute_error(y_test, test_predictions),
        "RMSE": test_rmse,
        "R2": r2_score(y_test, test_predictions),
    }
    if search is not None:
        print("Best XGBoost Params:")
        for key, value in search.best_params_.items():
            print(f"  {key}: {value}")
    print("Train Metrics:")
    for name, value in train_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("Test Metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    if hasattr(model, "feature_importances_"):
        feature_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        feature_values = np.abs(np.asarray(model.coef_).reshape(-1))
    else:
        feature_values = np.zeros(len(X.columns))
    feature_importances = pd.Series(feature_values, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    feature_importances.plot(kind="bar")
    plt.title("Feature Importances")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharey=True)
    axes[0].plot(y_train.index, y_train, label="Actual")
    axes[0].plot(y_train.index, train_predictions, label="Predicted")
    axes[0].set_title("Train Set Predictions")
    axes[0].set_ylabel(target_column)
    axes[0].legend()
    axes[1].plot(y_test.index, y_test, label="Actual")
    axes[1].plot(y_test.index, test_predictions, label="Predicted")
    axes[1].set_title("Test Set Predictions")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(target_column)
    axes[1].legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
