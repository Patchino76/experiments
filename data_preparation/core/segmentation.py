"""
Segmentation engine for converting motifs to datasets.

Handles merging, shuffling, and creating segmented datasets from motifs.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging
from .base_pattern import Motif

logger = logging.getLogger(__name__)


class SegmentationEngine:
    """
    Converts motif instances into segmented datasets for model training.
    """
    
    def __init__(self):
        """Initialize segmentation engine."""
        pass
    
    def create_segmented_dataset(
        self,
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
            additional_columns: Additional columns (e.g., target, timestamp)
            
        Returns:
            Segmented DataFrame with all motif instances
        """
        logger.info("Creating segmented dataset from motifs...")
        
        if additional_columns is None:
            additional_columns = []
        
        segments = []
        
        for motif in motifs:
            for instance in motif.instances:
                # Extract segment data
                segment_df = df.iloc[instance.start:instance.end].copy()
                
                # Add motif metadata
                segment_df['motif_id'] = motif.motif_id
                segment_df['segment_start'] = instance.start
                segment_df['segment_end'] = instance.end
                segment_df['motif_distance'] = instance.distance
                
                # Add pattern type if available
                if 'pattern_type' in motif.metadata:
                    segment_df['pattern_type'] = motif.metadata['pattern_type']
                
                segments.append(segment_df)
        
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
        self,
        motif_collections: Dict[str, List[Motif]],
        shuffle: bool = True
    ) -> List[Motif]:
        """
        Merge multiple motif collections into one.
        
        Args:
            motif_collections: Dictionary mapping pattern names to motif lists
            shuffle: If True, reassign motif IDs sequentially
            
        Returns:
            Merged list of motifs
        """
        logger.info("Merging motif collections...")
        
        all_motifs = []
        for pattern_name, motifs in motif_collections.items():
            logger.info(f"  {pattern_name}: {len(motifs)} motifs")
            all_motifs.extend(motifs)
        
        if shuffle:
            # Reassign IDs sequentially
            for idx, motif in enumerate(all_motifs, start=1):
                motif.motif_id = idx
            logger.info("  ✓ Reassigned motif IDs sequentially")
        
        logger.info(f"  ✓ Total merged motifs: {len(all_motifs)}")
        
        return all_motifs
    
    def extract_motif_summary(self, motifs: List[Motif]) -> pd.DataFrame:
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
            if not motif.instances:
                continue
            
            # Get pattern type
            pattern_type = motif.metadata.get('pattern_type', 'unknown')
            
            # Basic statistics
            summary = {
                'motif_id': motif.motif_id,
                'pattern_type': pattern_type,
                'num_instances': len(motif.instances),
                'avg_distance': motif.avg_distance,
                'total_points': sum(len(inst) for inst in motif.instances),
            }
            
            # Feature statistics
            feature_names = list(motif.instances[0].data.keys())
            for feat in feature_names:
                if feat == 'TimeStamp':
                    continue
                
                all_values = []
                for instance in motif.instances:
                    if feat in instance.data:
                        all_values.extend(instance.data[feat])
                
                if all_values:
                    summary[f'{feat}_mean'] = np.mean(all_values)
                    summary[f'{feat}_std'] = np.std(all_values)
                    summary[f'{feat}_min'] = np.min(all_values)
                    summary[f'{feat}_max'] = np.max(all_values)
            
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        logger.info(f"  ✓ Created summary for {len(summary_df)} motifs")
        
        return summary_df
    
    def create_instance_catalog(self, motifs: List[Motif]) -> pd.DataFrame:
        """
        Create a catalog of all motif instances with metadata.
        
        Args:
            motifs: List of motifs
            
        Returns:
            DataFrame with instance catalog
        """
        logger.info("Creating instance catalog...")
        
        catalog_data = []
        
        for motif in motifs:
            pattern_type = motif.metadata.get('pattern_type', 'unknown')
            
            for idx, instance in enumerate(motif.instances):
                record = {
                    'motif_id': motif.motif_id,
                    'pattern_type': pattern_type,
                    'instance_idx': idx,
                    'start': instance.start,
                    'end': instance.end,
                    'length': len(instance),
                    'distance': instance.distance,
                }
                
                # Add instance metadata
                for key, value in instance.metadata.items():
                    record[f'meta_{key}'] = value
                
                catalog_data.append(record)
        
        catalog_df = pd.DataFrame(catalog_data)
        logger.info(f"  ✓ Cataloged {len(catalog_df)} instances")
        
        return catalog_df
    
    def calculate_segment_statistics(
        self,
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
            
            stats = {
                'motif_id': motif_id,
                'num_points': len(motif_data)
            }
            
            # Add pattern type if available
            if 'pattern_type' in motif_data.columns:
                stats['pattern_type'] = motif_data['pattern_type'].iloc[0]
            
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
