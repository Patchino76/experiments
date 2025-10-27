"""
Data Preparation Pipeline

Loads data from database, discovers motifs, performs analysis,
and prepares segmented datasets for model training.

Usage:
    python prepare_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

# Add parent to path for db imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import PipelineConfig, DataConfig
from database import DataLoader, filter_data, validate_required_columns
from motif_discovery import MotifDiscovery, CorrelationFilter, convert_motifs_to_legacy_format
from density_analysis import DensityMotifDiscovery, analyze_density_behavior
from segmentation import (
    create_segmented_dataset,
    merge_motif_collections,
    extract_motif_summary,
    create_instance_catalog,
    calculate_segment_statistics
)
from visualization import (
    plot_all_motifs,
    plot_motif_overview,
    plot_density_analysis,
    plot_correlation_heatmap,
    plot_feature_distributions,
    save_summary_report
)
from circulative_load import calculate_circulative_load, validate_circulative_load

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('prepare_data.log', encoding='utf-8')
    ]
)

# Set UTF-8 encoding for console output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
logger = logging.getLogger(__name__)


class DataPreparationPipeline:
    """Main pipeline for data preparation."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.df = None
        self.mv_motifs = []
        self.density_motifs = []
        self.all_motifs = []
        self.density_analysis = None
        self.segmented_df = None
        self.data_loader = None
    
    def run(self):
        """Execute the complete data preparation pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING DATA PREPARATION PIPELINE")
        logger.info("=" * 80)
        logger.info(self.config.summary())
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Discover MV motifs
        self.discover_mv_motifs()
        
        # Step 3: Discover density motifs (optional)
        if 'DensityHC' in self.df.columns:
            self.discover_density_motifs()
            self.analyze_density()
        
        # Step 4: Merge motifs
        self.merge_motifs()
        
        # Step 5: Create segmented dataset
        self.create_segments()
        
        # Step 6: Save segmented data to database
        if self.config.use_database:
            self.save_to_database()
        
        # Step 7: Generate analysis outputs
        self.generate_outputs()
        
        # Step 8: Create visualizations
        self.create_visualizations()
        
        logger.info("\n" + "=" * 80)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Segmented data saved to: {self.config.paths.output_dir}")
        logger.info(f"Analysis files saved to: {self.config.paths.analysis_dir}")
        logger.info(f"Plots saved to: {self.config.paths.plots_dir}")
    
    def load_data(self):
        """Load and filter data from database or cache."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("-" * 80)
        
        # Initialize data loader
        self.data_loader = DataLoader(use_database=self.config.use_database)
        loader = self.data_loader
        
        # Load data
        cache_path = self.config.paths.output_dir / 'initial_data.csv'
        
        self.df = loader.load_mill_data(
            mill_number=self.config.data.mill_number,
            start_date=self.config.data.start_date,
            end_date=self.config.data.end_date,
            resample_freq=self.config.data.resample_freq,
            cache_path=cache_path if self.config.use_database else cache_path
        )
        
        # Validate columns (excluding CirculativeLoad which is calculated)
        required_cols = [col for col in self.config.data.get_all_columns() if col != 'CirculativeLoad']
        validate_required_columns(self.df, required_cols)
        
        # Filter data
        self.df = filter_data(self.df, self.config.data.filter_thresholds)
        
        # Calculate circulative load
        try:
            self.df = calculate_circulative_load(self.df, rho_solid=2900)
            validate_circulative_load(self.df)
            logger.info("  ✓ Circulative load calculation complete")
            
            # Validate that CirculativeLoad was added
            if 'CirculativeLoad' not in self.df.columns:
                logger.warning("  ⚠ CirculativeLoad column not found after calculation")
        except Exception as e:
            logger.warning(f"  ⚠ Circulative load calculation failed: {e}")
            logger.warning("  Continuing without circulative load column")
        
        # Save updated data with circulative load to cache
        if self.config.use_database:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(cache_path, index=False)
            logger.info(f"  ✓ Updated data with circulative load saved to {cache_path.name}")
        
        logger.info(f"✓ Data loaded and filtered: {len(self.df)} rows")
    
    def discover_mv_motifs(self):
        """Discover motifs in MV features."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 2: DISCOVERING MV MOTIFS")
        logger.info("-" * 80)
        
        # Initialize motif discovery
        discovery = MotifDiscovery(
            window_size=self.config.motif.mv_window_size,
            max_motifs=self.config.motif.mv_max_motifs,
            max_instances_per_motif=self.config.motif.mv_max_instances_per_motif,
            radius=self.config.motif.mv_radius
        )
        
        # Discover motifs
        motifs, segment_tuples = discovery.discover(
            self.df,
            self.config.data.mv_features
        )
        
        # Apply correlation filter if enabled
        if self.config.motif.apply_correlation_filter and self.config.motif.correlation_rules:
            logger.info("\nApplying correlation filter...")
            
            corr_filter = CorrelationFilter(
                correlation_rules=self.config.motif.correlation_rules,
                min_correlation_strength=self.config.motif.min_correlation_strength,
                filter_level=self.config.motif.filter_level
            )
            
            motifs, corr_stats = corr_filter.filter(motifs)
            
            # Save correlation stats
            corr_stats_path = self.config.paths.analysis_dir / 'correlation_stats.json'
            import json
            with open(corr_stats_path, 'w') as f:
                # Convert tuple keys to strings for JSON
                json_stats = {}
                for motif_id, stats in corr_stats.items():
                    json_stats[str(motif_id)] = {
                        'total_instances': stats['total_instances'],
                        'valid_instances': stats['valid_instances'],
                        'avg_correlations': {
                            f"{k[0]}_vs_{k[1]}": v
                            for k, v in stats['avg_correlations'].items()
                        }
                    }
                json.dump(json_stats, f, indent=2)
            logger.info(f"  ✓ Correlation stats saved to {corr_stats_path.name}")
        
        self.mv_motifs = motifs
        logger.info(f"✓ MV motif discovery complete: {len(self.mv_motifs)} motifs")
    
    def discover_density_motifs(self):
        """Discover density-constrained motifs."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 3: DISCOVERING DENSITY MOTIFS")
        logger.info("-" * 80)
        
        # Initialize density motif discovery
        density_discovery = DensityMotifDiscovery(
            window_size=self.config.motif.density_window_size,
            max_motifs=self.config.motif.density_max_motifs,
            radius=self.config.motif.density_radius
        )
        
        # Discover motifs
        self.density_motifs = density_discovery.discover(self.df)
        
        logger.info(f"✓ Density motif discovery complete: {len(self.density_motifs)} motifs")
    
    def analyze_density(self):
        """Analyze density behavior in motifs."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 4: ANALYZING DENSITY BEHAVIOR")
        logger.info("-" * 80)
        
        if not self.density_motifs:
            logger.info("  No density motifs to analyze")
            return
        
        self.density_analysis = analyze_density_behavior(self.density_motifs)
        
        # Save analysis results
        analysis_df = pd.DataFrame(self.density_analysis)
        analysis_path = self.config.paths.analysis_dir / 'density_analysis.csv'
        analysis_df.to_csv(analysis_path, index=False)
        logger.info(f"  ✓ Density analysis saved to {analysis_path.name}")
    
    def merge_motifs(self):
        """Merge MV and density motifs."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 5: MERGING MOTIF COLLECTIONS")
        logger.info("-" * 80)
        
        if self.density_motifs:
            self.all_motifs = merge_motif_collections(
                self.mv_motifs,
                self.density_motifs,
                shuffle_indices=True
            )
        else:
            self.all_motifs = self.mv_motifs
        
        logger.info(f"✓ Total motifs: {len(self.all_motifs)}")
    
    def create_segments(self):
        """Create segmented dataset from motifs."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 6: CREATING SEGMENTED DATASET")
        logger.info("-" * 80)
        
        # Determine which features to include
        feature_columns = self.config.data.mv_features.copy()
        
        # Add CV features if they exist
        for cv_feat in self.config.data.cv_features:
            if cv_feat in self.df.columns and cv_feat not in feature_columns:
                feature_columns.append(cv_feat)
        
        # Additional columns (target + DV features)
        additional_columns = [self.config.data.target]
        for dv_feat in self.config.data.dv_features:
            if dv_feat in self.df.columns:
                additional_columns.append(dv_feat)
        
        # Add TimeStamp if present
        if 'TimeStamp' in self.df.columns:
            additional_columns.append('TimeStamp')
        
        # Create segmented dataset from MV motifs only (for training)
        logger.info(f"  Creating MV motifs dataset ({len(self.mv_motifs)} motifs)...")
        segmented_mv_df = create_segmented_dataset(
            self.df,
            self.mv_motifs,
            feature_columns,
            additional_columns
        )
        
        # Create segmented dataset from all motifs (MV + density)
        logger.info(f"  Creating complete dataset ({len(self.all_motifs)} motifs)...")
        self.segmented_df = create_segmented_dataset(
            self.df,
            self.all_motifs,
            feature_columns,
            additional_columns
        )
        
        # Define columns to exclude from final output
        columns_to_exclude = [
            'id', 'Date', 'Shift', 'Original_Sheet', 'mill_id',
            'C_v', 'C_m', 'segment_start', 'segment_end', 'motif_distance'
        ]
        
        # Save segmented data
        if not segmented_mv_df.empty:
            # Filter out unwanted columns
            segmented_mv_filtered = segmented_mv_df.drop(
                columns=[col for col in columns_to_exclude if col in segmented_mv_df.columns],
                errors='ignore'
            )
            # Save MV motifs only (for model training)
            mv_path = self.config.paths.output_dir / f'segmented_motifsMV_{self.config.data.mill_number:02d}.csv'
            segmented_mv_filtered.to_csv(mv_path, index=False)
            logger.info(f"  ✓ MV motifs data saved to {mv_path.name} ({len(segmented_mv_filtered)} rows, {len(segmented_mv_filtered.columns)} columns)")
        
        if not self.segmented_df.empty:
            # Filter out unwanted columns
            segmented_all_filtered = self.segmented_df.drop(
                columns=[col for col in columns_to_exclude if col in self.segmented_df.columns],
                errors='ignore'
            )
            # Save all motifs (MV + density for comprehensive analysis)
            all_path = self.config.paths.output_dir / f'segmented_motifs_all_{self.config.data.mill_number:02d}.csv'
            segmented_all_filtered.to_csv(all_path, index=False)
            logger.info(f"  ✓ Complete data saved to {all_path.name} ({len(segmented_all_filtered)} rows, {len(segmented_all_filtered.columns)} columns)")
            
            # Update self.segmented_df to the filtered version for database save
            self.segmented_df = segmented_all_filtered
    
    def save_to_database(self):
        """Save segmented motifs data to database."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 6: SAVING SEGMENTED DATA TO DATABASE")
        logger.info("-" * 80)
        
        if self.data_loader is None:
            logger.warning("  ⚠ Data loader not initialized, skipping database save")
            return
        
        if self.segmented_df is None or self.segmented_df.empty:
            logger.warning("  ⚠ No segmented data to save")
            return
        
        # Save segmented_motifs_all to database (filtered version without excluded columns)
        # Note: self.segmented_df is already filtered in create_segments()
        logger.info(f"  Saving segmented motifs data for Mill {self.config.data.mill_number}...")
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
            logger.warning(f"  ⚠ Failed to save segmented data to database")
    
    def generate_outputs(self):
        """Generate analysis output files."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 7: GENERATING ANALYSIS OUTPUTS")
        logger.info("-" * 80)
        
        # Motif summary
        summary_df = extract_motif_summary(self.all_motifs)
        summary_path = self.config.paths.analysis_dir / 'motif_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"  ✓ Motif summary saved to {summary_path.name}")
        
        # Instance catalog
        catalog_df = create_instance_catalog(self.all_motifs)
        catalog_path = self.config.paths.analysis_dir / 'instance_catalog.csv'
        catalog_df.to_csv(catalog_path, index=False)
        logger.info(f"  ✓ Instance catalog saved to {catalog_path.name}")
        
        # Segment statistics
        if not self.segmented_df.empty:
            stats_df = calculate_segment_statistics(
                self.segmented_df,
                self.config.data.mv_features + self.config.data.cv_features
            )
            stats_path = self.config.paths.analysis_dir / 'segment_statistics.csv'
            stats_df.to_csv(stats_path, index=False)
            logger.info(f"  ✓ Segment statistics saved to {stats_path.name}")
        
        # Text summary report
        report_path = self.config.paths.analysis_dir / 'summary_report.txt'
        save_summary_report(self.all_motifs, self.density_analysis, report_path)
    
    def create_visualizations(self):
        """Create visualization plots."""
        logger.info("\n" + "-" * 80)
        logger.info("STEP 8: CREATING VISUALIZATIONS")
        logger.info("-" * 80)
        
        mill_plots_dir = self.config.paths.get_mill_plots_dir(self.config.data.mill_number)
        
        # Plot individual motifs
        plot_all_motifs(
            self.all_motifs,
            self.config.data.mv_features,
            mill_plots_dir / 'motifs',
            top_n=self.config.motif.top_motifs_to_plot
        )
        
        # Plot motif overview
        overview_path = mill_plots_dir / 'motif_overview.png'
        plot_motif_overview(self.all_motifs, overview_path)
        
        # Plot density analysis if available
        if self.density_analysis:
            density_plot_path = mill_plots_dir / 'density_analysis.png'
            plot_density_analysis(self.density_analysis, density_plot_path)
        
        # Plot correlation heatmap
        if not self.segmented_df.empty:
            corr_features = self.config.data.mv_features + self.config.data.cv_features
            corr_features = [f for f in corr_features if f in self.segmented_df.columns]
            
            if corr_features:
                corr_path = mill_plots_dir / 'correlation_heatmap.png'
                plot_correlation_heatmap(self.segmented_df, corr_features, corr_path)
                
                # Plot feature distributions
                dist_path = mill_plots_dir / 'feature_distributions.png'
                plot_feature_distributions(self.segmented_df, corr_features, dist_path)
        
        logger.info(f"✓ Visualizations saved to {mill_plots_dir}")


def main():
    """Main entry point."""
    # Configuration
    mill_number = 6
    start_date = "2025-06-26"
    end_date = "2025-10-26"
    
    # Create configuration
    config = PipelineConfig.create_default(mill_number, start_date, end_date)
    
    # Optionally customize configuration
    # config.motif.mv_max_motifs = 25
    # config.use_cached_data = True  # Use cached data instead of database
    
    # Run pipeline
    pipeline = DataPreparationPipeline(config)
    pipeline.run()
    
    logger.info("\n✓ Data preparation pipeline completed successfully!")


if __name__ == "__main__":
    main()
