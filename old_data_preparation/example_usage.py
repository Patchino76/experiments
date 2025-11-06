"""
Example usage of the data preparation pipeline with all pattern types.

This script demonstrates how to use the pattern discovery pipeline with all
available pattern types (density, inverse, dynamic, and pressure constraints).
"""
import os
import logging
import pandas as pd
from pathlib import Path

import sys
import os

# Add the parent directory to Python path so we can import data_preparation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation.motif import Motif, MotifInstance
from data_preparation.core.pipeline import Pipeline
from data_preparation.config.defaults import get_default_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_data(filepath: str) -> pd.DataFrame:
    """Load sample data from a CSV file with error handling."""
    try:
        logger.info(f"Loading data from {filepath}")
        data = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
        
        # Ensure required columns are present
        required_columns = ['Ore', 'WaterMill', 'WaterZumpf', 'PressureHC']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            logger.warning(f"Missing columns in data: {missing}")
            
            # Create dummy data for demonstration if running the example
            if all(col not in data.columns for col in required_columns):
                logger.info("Generating sample data for demonstration...")
                np.random.seed(42)
                n_samples = 1000
                time = pd.date_range(start='2023-01-01', periods=n_samples, freq='min')
                data = pd.DataFrame({
                    'timestamp': time,
                    'Ore': np.sin(np.linspace(0, 10, n_samples)) * 10 + 100 + np.random.normal(0, 2, n_samples),
                    'WaterMill': np.cos(np.linspace(0, 8, n_samples)) * 5 + 50 + np.random.normal(0, 1, n_samples),
                    'WaterZumpf': np.sin(np.linspace(0, 6, n_samples)) * 3 + 20 + np.random.normal(0, 0.5, n_samples),
                    'PressureHC': np.random.normal(3.5, 0.2, n_samples)  # Stable pressure
                }).set_index('timestamp')
                logger.info("Generated sample data with columns: %s", list(data.columns))
        
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def run_example():
    """Run the example pipeline with all pattern types."""
    # Create output directory
    output_dir = Path("output/pattern_discovery")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get default configuration
    config = get_default_config()
    
    # Enable all pattern types
    config['patterns']['density']['enabled'] = True
    config['patterns']['inverse']['enabled'] = True
    config['patterns']['dynamic']['enabled'] = True
    config['patterns']['pressure']['enabled'] = True
    
    # Update any specific parameters if needed
    config['patterns']['density']['window_size'] = 60
    config['patterns']['inverse']['window_size'] = 60
    config['patterns']['dynamic']['window_size'] = 60
    config['patterns']['pressure']['window_size'] = 60
    
    # Initialize the pipeline
    pipeline = Pipeline(config, output_dir=str(output_dir))
    
    # Load sample data (replace with your actual data loading logic)
    try:
        # Try to load from the modeling/output directory first
        data_file = Path("modeling/output/initial_data.csv")
        if not data_file.exists():
            # Fall back to the segmentation/output directory
            data_file = Path("segmentation/output/initial_data.csv")
            if not data_file.exists():
                raise FileNotFoundError(
                    "Could not find input data file. Please ensure the data file exists in either "
                    "modeling/output/ or segmentation/output/"
                )
        
        data = load_sample_data(str(data_file))
        
        # Run the pipeline
        logger.info("Starting pattern discovery pipeline...")
        results = pipeline.run(data)
        
        # Print summary of results
        logger.info("\n=== Pattern Discovery Results ===")
        for pattern_name, pattern_results in results.items():
            if pattern_results['motifs']:
                logger.info(f"\n{pattern_name.upper()} PATTERNS:")
                logger.info(f"  - Found {len(pattern_results['motifs'])} motifs")
                logger.info(f"  - Total instances: {sum(len(m.instances) for m in pattern_results['motifs'])}")
                if 'analysis' in pattern_results:
                    for k, v in pattern_results['analysis'].items():
                        logger.info(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")
        
        logger.info(f"\nResults saved to: {output_dir.absolute()}")
        
    except Exception as e:
        logger.error(f"Error running pattern discovery: {e}", exc_info=True)

if __name__ == "__main__":
    run_example()
