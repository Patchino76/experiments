"""
Entry point for data preparation pipeline.

Usage:
    python run.py
"""

import sys
import logging
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from pipeline import DataPreparationPipeline
from config.defaults import PipelineConfig
from config.pattern_configs import get_default_patterns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_preparation.log', encoding='utf-8')
    ]
)

# Set UTF-8 encoding for console output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    
    # Configuration
    mill_number = 6
    start_date = "2025-06-20"
    end_date = "2026-07-16"
    
    # Create configuration with default patterns
    config = PipelineConfig.create_default(
        mill_number=mill_number,
        start_date=start_date,
        end_date=end_date,
        patterns=get_default_patterns()
    )
    
    # Optional: Customize configuration
    # config.use_database = False  # Use cached data only
    config.save_to_database = True  # Save results to database
    
    # Optional: Disable specific patterns
    # for pattern in config.patterns:
    #     if pattern.name == 'pressure':
    #         pattern.enabled = False
    
    # Run pipeline
    pipeline = DataPreparationPipeline(config)
    
    try:
        pipeline.run()
        logger.info("\n✓ Data preparation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
