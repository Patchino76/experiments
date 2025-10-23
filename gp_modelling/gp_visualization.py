"""
Visualization functions for Gaussian Process models with uncertainty quantification.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def plot_gp_predictions_with_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    r2: float,
    title: str,
    save_path: str,
    dataset_label: str = "Test"
):
    """
    Plot predictions with uncertainty bands for Gaussian Process models.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        sigma: Standard deviation (uncertainty)
        r2: R² score
        title: Plot title
        save_path: Path to save plot
        dataset_label: Label for dataset (Train/Test)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sort by true values for better visualization
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    sigma_sorted = sigma[sort_idx]
    
    # Plot 1: Predictions with uncertainty bands
    ax1 = axes[0]
    x_range = np.arange(len(y_true_sorted))
    ax1.plot(x_range, y_true_sorted, 'b-', label='True', alpha=0.6, linewidth=1.5)
    ax1.plot(x_range, y_pred_sorted, 'r-', label='Predicted', alpha=0.6, linewidth=1.5)
    ax1.fill_between(
        x_range,
        y_pred_sorted - 2 * sigma_sorted,
        y_pred_sorted + 2 * sigma_sorted,
        alpha=0.2,
        color='red',
        label='95% Confidence'
    )
    ax1.set_xlabel('Sample (sorted by true value)')
    ax1.set_ylabel('Value')
    ax1.set_title(f'{dataset_label} Set - Predictions with Uncertainty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction', linewidth=2)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title(f'{dataset_label} Set - R²={r2:.4f}, RMSE={rmse:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_uncertainty_analysis(
    sigma_train: np.ndarray,
    sigma_test: np.ndarray,
    title: str,
    save_path: str
):
    """
    Plot uncertainty distribution analysis.
    
    Args:
        sigma_train: Training set uncertainties
        sigma_test: Test set uncertainties
        title: Plot title
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Histogram
    ax1 = axes[0]
    ax1.hist(sigma_train, bins=30, alpha=0.6, label='Train', color='blue', density=True)
    ax1.hist(sigma_test, bins=30, alpha=0.6, label='Test', color='red', density=True)
    ax1.set_xlabel('Uncertainty (σ)')
    ax1.set_ylabel('Density')
    ax1.set_title('Uncertainty Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    ax2 = axes[1]
    ax2.boxplot([sigma_train, sigma_test], labels=['Train', 'Test'])
    ax2.set_ylabel('Uncertainty (σ)')
    ax2.set_title('Uncertainty Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
