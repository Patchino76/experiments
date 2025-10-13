"""
Example of how to use the trained model at inference time.

This demonstrates that discontinuity markers can be computed from raw data alone,
without needing segment metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def add_discontinuity_markers_inference(df, base_feature_columns):
    """
    Compute discontinuity markers from raw data stream.
    This can be applied to ANY new data stream.
    
    Note: For real-time streaming, you'd maintain a rolling buffer of recent values.
    """
    df = df.copy()
    
    # Calculate lag differences for each feature
    lag_diffs = {}
    for col in base_feature_columns:
        lag1 = df[col].shift(1)
        diff = df[col] - lag1
        lag_diffs[col] = diff
    
    # Compute normalized discontinuity score
    discontinuity_scores = []
    for idx in range(len(df)):
        if idx == 0:
            discontinuity_scores.append(0.0)
            continue
        
        # Calculate z-score of the jump for each feature
        z_scores = []
        for col in base_feature_columns:
            # Get recent history (last 10 points before current)
            start_idx = max(0, idx - 10)
            recent_values = df[col].iloc[start_idx:idx]
            
            if len(recent_values) > 1:
                std_val = recent_values.std()
                
                if std_val > 0:
                    # Z-score of the jump
                    jump = abs(lag_diffs[col].iloc[idx])
                    z_score = jump / std_val
                    z_scores.append(z_score)
        
        # Average z-score across all features
        if z_scores:
            avg_z_score = np.mean(z_scores)
            discontinuity_scores.append(avg_z_score)
        else:
            discontinuity_scores.append(0.0)
    
    df['discontinuity_score'] = discontinuity_scores
    
    # For streaming data, we don't know segment boundaries in advance
    # So we use the discontinuity score itself as a proxy
    df['is_segment_start'] = (df['discontinuity_score'] > 2.0).astype(int)
    df['is_segment_end'] = 0  # Unknown at inference time
    
    return df


def predict_on_new_data(model, new_data_df, base_feature_columns):
    """
    Make predictions on new incoming data.
    
    Parameters:
    - model: trained XGBoost model
    - new_data_df: DataFrame with columns [TimeStamp, Ore, WaterMill, WaterZumpf, PulpHC, PressureHC]
    - base_feature_columns: list of base feature names
    
    Returns:
    - predictions: array of predicted DensityHC values
    """
    # Add discontinuity markers (same as training)
    df_with_features = add_discontinuity_markers_inference(new_data_df, base_feature_columns)
    
    # Build feature list (same order as training)
    feature_columns = base_feature_columns.copy()
    feature_columns.extend(['discontinuity_score', 'is_segment_start', 'is_segment_end'])
    
    # Extract features
    X = df_with_features[feature_columns]
    
    # Predict
    predictions = model.predict(X)
    
    return predictions


# Example usage
if __name__ == "__main__":
    # Simulate new incoming data (just raw features)
    new_data = pd.DataFrame({
        'TimeStamp': pd.date_range('2025-10-13 14:00', periods=100, freq='1min'),
        'Ore': np.random.uniform(160, 180, 100),
        'WaterMill': np.random.uniform(10, 15, 100),
        'WaterZumpf': np.random.uniform(230, 250, 100),
        'PulpHC': np.random.uniform(500, 550, 100),
        'PressureHC': np.random.uniform(0.4, 0.5, 100),
    })
    
    base_features = ["Ore", "WaterMill", "WaterZumpf", "PulpHC", "PressureHC"]
    
    print("New data shape:", new_data.shape)
    print("\nColumns in new data:", new_data.columns.tolist())
    
    # Add discontinuity markers
    data_with_features = add_discontinuity_markers_inference(new_data, base_features)
    
    print("\nAfter adding discontinuity markers:", data_with_features.shape)
    print("\nSample of markers:")
    print(data_with_features[['Ore', 'discontinuity_score', 'is_segment_start']].head(15))
    
    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("="*60)
    print("✓ At training: We compute discontinuity markers from segmented motifs")
    print("✓ At inference: We compute THE SAME markers from raw stream")
    print("✓ No segment metadata needed - only 3 simple markers!")
    print("✓ Clean dataset: original columns + 3 markers")
    print("="*60)
    
    # If you have a saved model, you can use it like this:
    # model = pickle.load(open('model.pkl', 'rb'))
    # predictions = predict_on_new_data(model, new_data, base_features)
    # print("\nPredictions:", predictions[:10])
