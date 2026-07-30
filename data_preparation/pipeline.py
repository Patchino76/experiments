"""
Main data preparation pipeline orchestrator.

Coordinates all steps of the data preparation process.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import List, Dict

# Setup path
sys.path.append(str(Path(__file__).parent))

from core.data_loader import DataLoader
from core.pattern_registry import PatternRegistry
from core.segmentation import SegmentationEngine
from core.base_pattern import Motif
from analysis.analyzer import PatternAnalyzer
from analysis.visualizer import PatternVisualizer
from config.defaults import PipelineConfig

# Import patterns to trigger registration
import patterns

logger = logging.getLogger(__name__)


class DataPreparationPipeline:
    """
    Main pipeline for data preparation.
    
    Orchestrates data loading, pattern discovery, analysis, and output generation.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.df = None
        self.pattern_results: Dict[str, List[Motif]] = {}
        self.all_motifs: List[Motif] = []
        self.mv_motifs: List[Motif] = []
        self.segmented_df = None
        
        # Initialize components
        self.data_loader = DataLoader(use_database=config.use_database)
        self.segmentation = SegmentationEngine()
        self.analyzer = PatternAnalyzer()
        self.visualizer = PatternVisualizer()
    
    def run(self):
        """Execute the complete data preparation pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING DATA PREPARATION PIPELINE v2.0")
        logger.info("=" * 80)
        logger.info(self.config.summary())
        
        try:
            # Step 1: Load and prepare data
            self.load_data()
            
            # Step 2: Discover patterns
            self.discover_patterns()
            
            # Step 3: Analyze patterns
            self.analyze_patterns()
            
            # Step 4: Merge and segment
            self.create_segmented_datasets()
            
            # Step 5: Generate visualizations
            self.create_visualizations()
            
            # Step 6: Save to database (optional)
            if self.config.save_to_database:
                self.save_to_database()
            
            logger.info("\n" + "=" * 80)
            logger.info("DATA PREPARATION COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Output directory: {self.config.paths.output_dir}")
            logger.info(f"Analysis directory: {self.config.paths.analysis_dir}")
            logger.info(f"Plots directory: {self.config.paths.plots_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def load_data(self):
        """Load and prepare data."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 1: LOADING AND PREPARING DATA")
        logger.info("-" * 80)
        
        # Load data
        cache_path = self.config.paths.output_dir / 'initial_data.csv'
        
        self.df = self.data_loader.load_mill_data(
            mill_number=self.config.data.mill_number,
            start_date=self.config.data.start_date,
            end_date=self.config.data.end_date,
            resample_freq=self.config.data.resample_freq,
            cache_path=cache_path
        )
        
        # Validate columns (excluding CirculativeLoad which is calculated)
        required_cols = [col for col in self.config.data.get_all_columns() 
                        if col != 'CirculativeLoad']
        self.data_loader.validate_columns(self.df, required_cols)
        
        # Filter data (fixed global bounds - cheap first pass)
        self.df = self.data_loader.filter_data(self.df, self.config.data.filter_thresholds)
        
        # Adaptive filter (rolling median + MAD - adapts to drift, second pass)
        if self.config.data.use_adaptive_filter:
            self.df = self.data_loader.filter_data_adaptive(
                self.df,
                columns=self.config.data.adaptive_filter_columns,
                window=self.config.data.adaptive_filter_window,
                k=self.config.data.adaptive_filter_k
            )
        
        # Calculate circulative load
        self.df = self.data_loader.calculate_circulative_load(self.df, rho_solid=2900)
        
        # Save updated data
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(cache_path, index=False)
        
        logger.info(f"✓ Data ready: {len(self.df)} rows, {len(self.df.columns)} columns")
    
    def discover_patterns(self):
        """Discover all enabled patterns."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 2: DISCOVERING PATTERNS")
        logger.info("-" * 80)
        
        enabled_patterns = self.config.get_enabled_patterns()
        logger.info(f"Enabled patterns: {len(enabled_patterns)}")
        
        # Convert pattern configs to dictionaries
        pattern_configs = [p.to_dict() for p in enabled_patterns]
        
        # Discover all patterns
        self.pattern_results = PatternRegistry.discover_all(self.df, pattern_configs)
        
        # Separate MV motifs for training
        self.mv_motifs = self.pattern_results.get('mv', [])
        
        # Summary
        total_motifs = sum(len(motifs) for motifs in self.pattern_results.values())
        total_instances = sum(
            sum(len(m) for m in motifs) 
            for motifs in self.pattern_results.values()
        )
        
        logger.info(f"\n✓ Pattern discovery complete:")
        logger.info(f"  Total motifs: {total_motifs}")
        logger.info(f"  Total instances: {total_instances}")
        
        for pattern_name, motifs in self.pattern_results.items():
            if motifs:
                instances = sum(len(m) for m in motifs)
                logger.info(f"  {pattern_name}: {len(motifs)} motifs, {instances} instances")
    
    def analyze_patterns(self):
        """Analyze discovered patterns."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 3: ANALYZING PATTERNS")
        logger.info("-" * 80)
        
        for pattern_name, motifs in self.pattern_results.items():
            if not motifs:
                continue
            
            # Get pattern config
            pattern_config = next(
                (p for p in self.config.patterns if p.name == pattern_name),
                None
            )
            
            if pattern_config and pattern_config.save_analysis:
                logger.info(f"\nAnalyzing pattern: {pattern_name}")
                
                # Analyze density behavior
                analyses = self.analyzer.analyze_density_behavior(motifs)
                
                # Save analysis
                self.analyzer.save_analysis(
                    analyses,
                    self.config.paths.analysis_dir,
                    pattern_name
                )
        
        logger.info("\n✓ Pattern analysis complete")
    
    def create_segmented_datasets(self):
        """Create segmented datasets from motifs."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 4: CREATING SEGMENTED DATASETS")
        logger.info("-" * 80)
        
        # Determine feature columns
        feature_columns = self.config.data.mv_features.copy()
        
        # Add CV features if they exist
        for cv_feat in self.config.data.cv_features:
            if cv_feat in self.df.columns and cv_feat not in feature_columns:
                feature_columns.append(cv_feat)
        
        # Additional columns
        additional_columns = [self.config.data.target]
        for dv_feat in self.config.data.dv_features:
            if dv_feat in self.df.columns:
                additional_columns.append(dv_feat)
        
        if 'TimeStamp' in self.df.columns:
            additional_columns.append('TimeStamp')
        
        # Columns to exclude from final output
        columns_to_exclude = [
            'id', 'Date', 'Shift', 'Original_Sheet', 'mill_id',
            'C_v', 'C_m', 'segment_start', 'segment_end', 'motif_distance'
        ]
        
        # Save MV motifs only (for training)
        if self.config.save_mv_only and self.mv_motifs:
            logger.info(f"\nCreating MV-only dataset ({len(self.mv_motifs)} motifs)...")
            
            segmented_mv_df = self.segmentation.create_segmented_dataset(
                self.df,
                self.mv_motifs,
                feature_columns,
                additional_columns
            )
            
            if not segmented_mv_df.empty:
                # Filter columns
                segmented_mv_filtered = segmented_mv_df.drop(
                    columns=[col for col in columns_to_exclude if col in segmented_mv_df.columns],
                    errors='ignore'
                )
                
                # Save
                mv_path = self.config.paths.output_dir / f'segmented_motifsMV_{self.config.data.mill_number:02d}.csv'
                segmented_mv_filtered.to_csv(mv_path, index=False)
                logger.info(f"  ✓ Saved: {mv_path.name} ({len(segmented_mv_filtered)} rows)")
        
        # Save combined dataset (all patterns)
        if self.config.save_combined:
            logger.info(f"\nCreating combined dataset (all patterns)...")
            
            # Merge all motifs
            self.all_motifs = self.segmentation.merge_motif_collections(
                self.pattern_results,
                shuffle=True
            )
            
            # Create segmented dataset
            self.segmented_df = self.segmentation.create_segmented_dataset(
                self.df,
                self.all_motifs,
                feature_columns,
                additional_columns
            )
            
            if not self.segmented_df.empty:
                # Filter columns
                segmented_all_filtered = self.segmented_df.drop(
                    columns=[col for col in columns_to_exclude if col in self.segmented_df.columns],
                    errors='ignore'
                )
                
                # Save
                all_path = self.config.paths.output_dir / f'segmented_motifs_all_{self.config.data.mill_number:02d}.csv'
                segmented_all_filtered.to_csv(all_path, index=False)
                logger.info(f"  ✓ Saved: {all_path.name} ({len(segmented_all_filtered)} rows)")
                
                # Update for database save
                self.segmented_df = segmented_all_filtered
        
        # Generate summary files
        if self.all_motifs:
            # Motif summary
            summary_df = self.segmentation.extract_motif_summary(self.all_motifs)
            summary_path = self.config.paths.analysis_dir / 'motif_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"  ✓ Saved: {summary_path.name}")
            
            # Instance catalog
            catalog_df = self.segmentation.create_instance_catalog(self.all_motifs)
            catalog_path = self.config.paths.analysis_dir / 'instance_catalog.csv'
            catalog_df.to_csv(catalog_path, index=False)
            logger.info(f"  ✓ Saved: {catalog_path.name}")
            
            # Segment statistics
            if self.segmented_df is not None and not self.segmented_df.empty:
                stats_df = self.segmentation.calculate_segment_statistics(
                    self.segmented_df,
                    feature_columns
                )
                stats_path = self.config.paths.analysis_dir / 'segment_statistics.csv'
                stats_df.to_csv(stats_path, index=False)
                logger.info(f"  ✓ Saved: {stats_path.name}")
        
        logger.info("\n✓ Segmented datasets created")
    
    def create_visualizations(self):
        """Create all visualizations."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 5: CREATING VISUALIZATIONS")
        logger.info("-" * 80)
        
        mill_plots_dir = self.config.paths.get_mill_plots_dir(self.config.data.mill_number)
        
        # Overall motif overview
        if self.all_motifs:
            overview_path = mill_plots_dir / 'motif_overview.png'
            self.visualizer.plot_motif_overview(self.all_motifs, overview_path)
        
        # Pattern-specific visualizations
        for pattern_name, motifs in self.pattern_results.items():
            if not motifs:
                continue
            
            # Get pattern config
            pattern_config = next(
                (p for p in self.config.patterns if p.name == pattern_name),
                None
            )
            
            if pattern_config and pattern_config.save_plots:
                logger.info(f"\nCreating plots for pattern: {pattern_name}")
                
                # Density analysis plot
                analyses = self.analyzer.analyze_density_behavior(motifs)
                if analyses:
                    plot_path = mill_plots_dir / f'{pattern_name}_analysis.png'
                    self.visualizer.plot_density_analysis(analyses, plot_path, pattern_name)
                
                # Individual motif plots
                motif_dir = mill_plots_dir / 'motifs' / pattern_name
                self.visualizer.plot_individual_motifs(
                    motifs,
                    self.config.data.mv_features,
                    motif_dir,
                    top_n=5
                )
        
        # Correlation and distribution plots
        if self.segmented_df is not None and not self.segmented_df.empty:
            corr_features = self.config.data.mv_features + self.config.data.cv_features
            corr_features = [f for f in corr_features if f in self.segmented_df.columns]
            
            if corr_features:
                # Correlation heatmap
                corr_path = mill_plots_dir / 'correlation_heatmap.png'
                self.visualizer.plot_correlation_heatmap(
                    self.segmented_df,
                    corr_features,
                    corr_path
                )
                
                # Feature distributions
                dist_path = mill_plots_dir / 'feature_distributions.png'
                self.visualizer.plot_feature_distributions(
                    self.segmented_df,
                    corr_features,
                    dist_path
                )
        
        # Generate summary report
        if self.all_motifs:
            all_analyses = []
            for motifs in self.pattern_results.values():
                if motifs:
                    all_analyses.extend(self.analyzer.analyze_density_behavior(motifs))
            
            report_path = self.config.paths.analysis_dir / 'summary_report.txt'
            self.analyzer.generate_summary_report(
                self.all_motifs,
                all_analyses,
                report_path
            )
        
        logger.info(f"\n✓ Visualizations saved to {mill_plots_dir}")
    
    def save_to_database(self):
        """Save segmented data to database."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 6: SAVING TO DATABASE")
        logger.info("-" * 80)
        
        if self.segmented_df is None or self.segmented_df.empty:
            logger.warning("  ⚠ No segmented data to save")
            return
        
        if not self.config.use_database:
            logger.info("  Database save disabled in configuration")
            return
        
        if self.data_loader is None:
            logger.warning("  ⚠ Data loader not initialized")
            return
        
        try:
            logger.info(f"  Saving segmented motifs for Mill {self.config.data.mill_number}...")
            logger.info(f"  Table will be recreated (if_exists='replace')")
            
            success = self.data_loader.save_motifs_to_database(
                df=self.segmented_df,
                mill_number=self.config.data.mill_number,
                table_suffix='MOTIFS',
                if_exists='replace'  # Always recreate the table
            )
            
            if success:
                logger.info(f"✓ Segmented data saved to database table: MOTIFS_{self.config.data.mill_number:02d}")
            else:
                logger.warning("  ⚠ Failed to save segmented data to database")
                
        except Exception as e:
            logger.error(f"  ⚠ Database save failed: {e}")
