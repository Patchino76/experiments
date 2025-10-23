"""
Model Improvement Utilities

Helper functions for enhanced polynomial model training including:
- Feature engineering for mill operations
- Data quality diagnostics
- Residual analysis and visualization
- Polynomial complexity analysis
- Sample weight computation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def engineer_mill_features(df: pd.DataFrame, mv_features: list) -> pd.DataFrame:
    """
    Add domain-specific engineered features for mill operations.
    
    Creates physically meaningful ratios and interactions that often
    improve model performance and interpretability.
    
    Args:
        df: DataFrame with mill process variables
        mv_features: List of manipulated variable feature names
        
    Returns:
        DataFrame with additional engineered features
    """
    df_eng = df.copy()
    
    # Ratios (often more stable than raw values)
    if 'Ore' in mv_features and 'WaterMill' in mv_features:
        df_eng['ore_water_ratio'] = df['Ore'] / (df['WaterMill'] + 1e-6)
        logger.info("  ✓ Added feature: ore_water_ratio")
    
    if 'WaterMill' in mv_features and 'WaterZumpf' in mv_features:
        df_eng['water_mill_zumpf_ratio'] = df['WaterMill'] / (df['WaterZumpf'] + 1e-6)
        logger.info("  ✓ Added feature: water_mill_zumpf_ratio")
    
    # Total water input
    if 'WaterMill' in mv_features and 'WaterZumpf' in mv_features:
        df_eng['total_water'] = df['WaterMill'] + df['WaterZumpf']
        logger.info("  ✓ Added feature: total_water")
    
    # Energy-related features (if motor current is available)
    if 'MotorAmp' in mv_features and 'Ore' in mv_features:
        df_eng['specific_energy'] = df['MotorAmp'] / (df['Ore'] + 1e-6)
        logger.info("  ✓ Added feature: specific_energy")
    
    # Dilution indicator
    if 'Ore' in mv_features and 'WaterMill' in mv_features and 'WaterZumpf' in mv_features:
        total_water = df['WaterMill'] + df['WaterZumpf']
        df_eng['dilution_factor'] = total_water / (df['Ore'] + 1e-6)
        logger.info("  ✓ Added feature: dilution_factor")
    
    return df_eng


def diagnose_data_quality(df: pd.DataFrame, features: list) -> Dict:
    """
    Diagnose potential data quality issues.
    
    Checks for:
    - Outliers using IQR method
    - Low variance features
    - Missing values
    - Distribution statistics
    
    Args:
        df: DataFrame to diagnose
        features: List of feature names to check
        
    Returns:
        Dictionary with diagnostic results
    """
    logger.info("\n" + "-" * 80)
    logger.info("DATA QUALITY DIAGNOSTICS")
    logger.info("-" * 80)
    
    diagnostics = {}
    
    for feature in features:
        if feature not in df.columns:
            logger.warning(f"  ⚠ Feature '{feature}' not found in dataframe")
            continue
            
        data = df[feature].dropna()
        
        if len(data) == 0:
            logger.warning(f"  ⚠ Feature '{feature}' has no valid data")
            continue
        
        # Check for outliers using IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data < Q1 - 3*IQR) | (data > Q3 + 3*IQR)).sum()
        outlier_pct = outliers / len(data) * 100
        
        # Check variance
        std = data.std()
        mean = data.mean()
        cv = (std / mean * 100) if mean != 0 else 0
        
        # Missing values
        missing = df[feature].isna().sum()
        missing_pct = missing / len(df) * 100
        
        diagnostics[feature] = {
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(mean),
            'std': float(std),
            'cv': float(cv),
            'outliers': int(outliers),
            'outlier_pct': float(outlier_pct),
            'missing': int(missing),
            'missing_pct': float(missing_pct)
        }
        
        logger.info(f"\n{feature}:")
        logger.info(f"  Range: [{data.min():.3f}, {data.max():.3f}]")
        logger.info(f"  Mean±Std: {mean:.3f} ± {std:.3f}")
        logger.info(f"  Coefficient of Variation: {cv:.1f}%")
        logger.info(f"  Outliers (>3 IQR): {outliers} ({outlier_pct:.2f}%)")
        logger.info(f"  Missing values: {missing} ({missing_pct:.2f}%)")
        
        if cv < 1:
            logger.warning(f"  ⚠ Very low variance - may not be informative")
        if outlier_pct > 5:
            logger.warning(f"  ⚠ High outlier percentage")
        if missing_pct > 10:
            logger.warning(f"  ⚠ High missing value percentage")
    
    return diagnostics


def compute_target_weights(
    y: pd.Series,
    target_min: float = 16,
    target_max: float = 30,
    in_range_weight: float = 2.0,
    out_range_weight: float = 1.0
) -> np.ndarray:
    """
    Compute sample weights for quality model training.
    
    Gives higher weight to samples in the target range without
    completely excluding out-of-range samples.
    
    Args:
        y: Target variable values
        target_min: Minimum value of target range
        target_max: Maximum value of target range
        in_range_weight: Weight for samples in target range
        out_range_weight: Weight for samples outside target range
        
    Returns:
        Array of sample weights
    """
    weights = np.ones(len(y)) * out_range_weight
    in_range = (y >= target_min) & (y <= target_max)
    weights[in_range] = in_range_weight
    
    in_range_count = in_range.sum()
    out_range_count = len(y) - in_range_count
    
    logger.info(f"  Sample weighting:")
    logger.info(f"    In-range [{target_min}, {target_max}]: {in_range_count} samples (weight={in_range_weight})")
    logger.info(f"    Out-of-range: {out_range_count} samples (weight={out_range_weight})")
    
    return weights


def analyze_polynomial_complexity(pipeline, X_train: pd.DataFrame) -> Dict:
    """
    Analyze the complexity of polynomial features created.
    
    Args:
        pipeline: Fitted sklearn pipeline with PolynomialFeatures
        X_train: Training features
        
    Returns:
        Dictionary with feature complexity statistics
    """
    poly_step = pipeline.named_steps["poly"]
    feature_names = poly_step.get_feature_names_out(X_train.columns)
    
    # Count feature types
    linear_features = sum(1 for f in feature_names if '^' not in f and ' ' not in f)
    interaction_features = sum(1 for f in feature_names if ' ' in f and '^' not in f)
    power_features = sum(1 for f in feature_names if '^' in f)
    
    complexity = {
        'total': len(feature_names),
        'linear': linear_features,
        'interactions': interaction_features,
        'powers': power_features
    }
    
    logger.info(f"\n  Polynomial Feature Complexity:")
    logger.info(f"    Total features: {complexity['total']}")
    logger.info(f"    Linear: {complexity['linear']}")
    logger.info(f"    Interactions: {complexity['interactions']}")
    logger.info(f"    Powers: {complexity['powers']}")
    
    return complexity


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create residual analysis plot.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals vs predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values', fontsize=11)
    axes[0, 0].set_ylabel('Residuals', fontsize=11)
    axes[0, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Predicted vs Actual
    axes[1, 1].scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('Actual Values', fontsize=11)
    axes[1, 1].set_ylabel('Predicted Values', fontsize=11)
    axes[1, 1].set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Residual plot saved to {save_path}")
    
    return fig


def plot_train_test_predictions(
    y_train: np.ndarray,
    train_pred: np.ndarray,
    train_r2: float,
    y_test: np.ndarray,
    test_pred: np.ndarray,
    test_r2: float,
    title: str = "Prediction Performance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot actual vs predicted values for train and test sets."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Training set plot
    axes[0].scatter(y_train, train_pred, alpha=0.6, edgecolor='k', linewidth=0.5)
    min_train = min(np.min(y_train), np.min(train_pred))
    max_train = max(np.max(y_train), np.max(train_pred))
    axes[0].plot([min_train, max_train], [min_train, max_train], 'r--', linewidth=2)
    axes[0].set_title(f"Train Predictions (R²={train_r2:.3f})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Actual Values', fontsize=11)
    axes[0].set_ylabel('Predicted Values', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Test set plot
    axes[1].scatter(y_test, test_pred, alpha=0.6, edgecolor='k', linewidth=0.5)
    min_test = min(np.min(y_test), np.min(test_pred))
    max_test = max(np.max(y_test), np.max(test_pred))
    axes[1].plot([min_test, max_test], [min_test, max_test], 'r--', linewidth=2)
    axes[1].set_title(f"Test Predictions (R²={test_r2:.3f})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Actual Values', fontsize=11)
    axes[1].set_ylabel('Predicted Values', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Train/Test prediction plot saved to {save_path}")

    return fig


def plot_cv_results(cv_results_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize cross-validation results.
    
    Args:
        cv_results_df: DataFrame with CV results from GridSearchCV
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top parameter combinations
    top_n = min(10, len(cv_results_df))
    top_results = cv_results_df.head(top_n)
    
    axes[0].barh(range(top_n), top_results['mean_test_score'])
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels([f"Rank {i+1}" for i in range(top_n)])
    axes[0].set_xlabel('Mean CV Score (R²)', fontsize=11)
    axes[0].set_title('Top 10 Parameter Combinations', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    # Score distribution
    axes[1].hist(cv_results_df['mean_test_score'], bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=cv_results_df['mean_test_score'].iloc[0], color='r', 
                    linestyle='--', linewidth=2, label='Best Score')
    axes[1].set_xlabel('Mean CV Score (R²)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('CV Score Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ CV results plot saved to {save_path}")
    
    return fig


def log_cv_diagnostics(search: "GridSearchCV", top_n: int = 5):
    """
    Log detailed cross-validation diagnostics.
    
    Args:
        search: Fitted GridSearchCV object
        top_n: Number of top parameter combinations to display
    """
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results = cv_results.sort_values('rank_test_score')
    
    logger.info(f"\n  Cross-Validation Diagnostics:")
    logger.info(f"  Total parameter combinations tested: {len(cv_results)}")
    logger.info(f"\n  Top {top_n} parameter combinations:")
    
    for idx, row in cv_results.head(top_n).iterrows():
        logger.info(f"\n    Rank {int(row['rank_test_score'])}:")
        logger.info(f"      Mean CV Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        logger.info(f"      Parameters: {row['params']}")
