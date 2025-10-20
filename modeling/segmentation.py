"""
Segmentation and feature engineering module.

Converts motif instances into segmented datasets for model training.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

from motif_discovery import Motif

logger = logging.getLogger(__name__)


def create_segmented_dataset(
    df: pd.DataFrame,
    motifs: List[Motif],
    feature_columns: List[str],
    additional_columns: List[str] = None
) -> pd.DataFrame:
    """
    Create segmented dataset from motif instances.
    
    Args:
        df: Original DataFrame
        motifs: List of discovered motifs
        feature_columns: Feature columns to include
        additional_columns: Additional columns to include (e.g., target)
        
    Returns:
        Segmented DataFrame with all motif instances
    """
    logger.info("Creating segmented dataset from motifs...")
    
    if additional_columns is None:
        additional_columns = []
    
    all_columns = feature_columns + additional_columns
    
    # Collect all segments
    segments = []
    segment_info = []
    
    for motif in motifs:
        for instance in motif.instances:
            # Extract segment data
            segment_df = df.iloc[instance.start:instance.end].copy()
            
            # Add motif metadata
            segment_df['motif_id'] = motif.motif_id
            segment_df['segment_start'] = instance.start
            segment_df['segment_end'] = instance.end
            segment_df['motif_distance'] = instance.distance
            
            segments.append(segment_df)
            
            segment_info.append({
                'motif_id': motif.motif_id,
                'start': instance.start,
                'end': instance.end,
                'length': len(segment_df)
            })
    
    if not segments:
        logger.warning("  ⚠ No segments found!")
        return pd.DataFrame()
    
    # Combine all segments
    segmented_df = pd.concat(segments, ignore_index=True)
    
    logger.info(f"  ✓ Created segmented dataset:")
    logger.info(f"    - Total segments: {len(segments)}")
    logger.info(f"    - Total rows: {len(segmented_df)}")
    logger.info(f"    - Unique motifs: {segmented_df['motif_id'].nunique()}")
    
    return segmented_df


def merge_motif_collections(
    primary_motifs: List[Motif],
    secondary_motifs: List[Motif],
    shuffle_indices: bool = True
) -> List[Motif]:
    """
    Merge two collections of motifs, optionally shuffling to avoid ID conflicts.
    
    This allows integrating density motifs with MV motifs by reassigning IDs.
    
    Args:
        primary_motifs: Primary motif collection (keeps original IDs)
        secondary_motifs: Secondary motif collection (IDs will be shifted)
        shuffle_indices: If True, reassign all motif IDs sequentially
        
    Returns:
        Merged list of motifs
    """
    logger.info("Merging motif collections...")
    logger.info(f"  Primary motifs: {len(primary_motifs)}")
    logger.info(f"  Secondary motifs: {len(secondary_motifs)}")
    
    merged = []
    
    if shuffle_indices:
        # Reassign all IDs sequentially
        current_id = 1
        
        for motif in primary_motifs:
            motif.motif_id = current_id
            merged.append(motif)
            current_id += 1
        
        for motif in secondary_motifs:
            motif.motif_id = current_id
            merged.append(motif)
            current_id += 1
    else:
        # Keep primary IDs, shift secondary IDs
        merged.extend(primary_motifs)
        
        if primary_motifs:
            max_id = max(m.motif_id for m in primary_motifs)
        else:
            max_id = 0
        
        for motif in secondary_motifs:
            motif.motif_id = max_id + motif.motif_id
            merged.append(motif)
    
    logger.info(f"  ✓ Merged: {len(merged)} total motifs")
    
    return merged


def extract_motif_summary(motifs: List[Motif]) -> pd.DataFrame:
    """
    Extract summary statistics for each motif.
    
    Args:
        motifs: List of motifs
        
    Returns:
        DataFrame with motif summaries
    """
    logger.info("Extracting motif summaries...")
    
    summary_data = []
    
    for motif in motifs:
        # Get all feature names from first instance
        if not motif.instances:
            continue
        
        feature_names = list(motif.instances[0].data.keys())
        
        # Calculate statistics for each feature
        feature_stats = {}
        for feat in feature_names:
            if feat == 'TimeStamp':
                continue
            
            # Collect all values across instances
            all_values = []
            for instance in motif.instances:
                if feat in instance.data:
                    all_values.extend(instance.data[feat])
            
            if all_values:
                feature_stats[f'{feat}_mean'] = np.mean(all_values)
                feature_stats[f'{feat}_std'] = np.std(all_values)
                feature_stats[f'{feat}_min'] = np.min(all_values)
                feature_stats[f'{feat}_max'] = np.max(all_values)
        
        summary = {
            'motif_id': motif.motif_id,
            'num_instances': len(motif.instances),
            'avg_distance': motif.avg_distance,
            'total_points': sum(len(inst) for inst in motif.instances),
            **feature_stats
        }
        
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    
    logger.info(f"  ✓ Created summary for {len(summary_df)} motifs")
    
    return summary_df


def create_instance_catalog(motifs: List[Motif]) -> pd.DataFrame:
    """
    Create a catalog of all motif instances with their metadata.
    
    Args:
        motifs: List of motifs
        
    Returns:
        DataFrame with instance catalog
    """
    logger.info("Creating instance catalog...")
    
    catalog_data = []
    
    for motif in motifs:
        for idx, instance in enumerate(motif.instances):
            record = {
                'motif_id': motif.motif_id,
                'instance_idx': idx,
                'start': instance.start,
                'end': instance.end,
                'length': len(instance),
                'distance': instance.distance,
            }
            
            # Add metadata
            for key, value in instance.metadata.items():
                record[f'meta_{key}'] = value
            
            catalog_data.append(record)
    
    catalog_df = pd.DataFrame(catalog_data)
    
    logger.info(f"  ✓ Cataloged {len(catalog_df)} instances")
    
    return catalog_df


def calculate_segment_statistics(
    segmented_df: pd.DataFrame,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Calculate statistics for each segment/motif.
    
    Args:
        segmented_df: Segmented DataFrame
        feature_columns: Features to calculate statistics for
        
    Returns:
        DataFrame with segment statistics
    """
    logger.info("Calculating segment statistics...")
    
    stats_list = []
    
    for motif_id in segmented_df['motif_id'].unique():
        motif_data = segmented_df[segmented_df['motif_id'] == motif_id]
        
        stats = {'motif_id': motif_id, 'num_points': len(motif_data)}
        
        for col in feature_columns:
            if col in motif_data.columns:
                stats[f'{col}_mean'] = motif_data[col].mean()
                stats[f'{col}_std'] = motif_data[col].std()
                stats[f'{col}_min'] = motif_data[col].min()
                stats[f'{col}_max'] = motif_data[col].max()
                stats[f'{col}_range'] = motif_data[col].max() - motif_data[col].min()
        
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    
    logger.info(f"  ✓ Calculated statistics for {len(stats_df)} motifs")
    
    return stats_df


def shuffle_motif_indices(motifs: List[Motif], start_id: int = 1) -> List[Motif]:
    """
    Reassign motif IDs sequentially starting from start_id.
    
    Args:
        motifs: List of motifs
        start_id: Starting ID number
        
    Returns:
        List of motifs with reassigned IDs
    """
    logger.info(f"Shuffling motif indices starting from {start_id}...")
    
    for idx, motif in enumerate(motifs):
        motif.motif_id = start_id + idx
    
    logger.info(f"  ✓ Reassigned {len(motifs)} motif IDs")
    
    return motifs
