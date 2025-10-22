"""
Test script for saving segmented motifs to database.

This script demonstrates how to save segmented motifs data to the database
using the new functionality added to the data preparation pipeline.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add parent to path for db imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from database import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_save_motifs():
    """Test saving motifs data to database."""
    
    # Test parameters
    mill_number = 6
    output_dir = Path(__file__).resolve().parent / 'output'
    
    # Path to segmented motifs CSV file (with mill suffix)
    segmented_file = output_dir / f'segmented_motifs_all_{mill_number:02d}.csv'
    
    if not segmented_file.exists():
        logger.error(f"Segmented motifs file not found: {segmented_file}")
        logger.info("Please run prepare_data.py first to generate the segmented data")
        return False
    
    # Load the segmented data
    logger.info(f"Loading segmented motifs from {segmented_file}")
    df = pd.read_csv(segmented_file, parse_dates=['TimeStamp'])
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Initialize data loader with database enabled
    logger.info("\nInitializing database connection...")
    try:
        loader = DataLoader(use_database=True)
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {e}")
        return False
    
    # Save to database
    logger.info(f"\nSaving motifs data to database for Mill {mill_number}...")
    success = loader.save_motifs_to_database(
        df=df,
        mill_number=mill_number,
        table_suffix='MOTIF',
        if_exists='replace'
    )
    
    if success:
        logger.info("\n✅ SUCCESS: Motifs data saved to database!")
        logger.info(f"   Table name: MOTIFS_{mill_number:02d}")
        logger.info(f"   Schema: mills")
        logger.info(f"   Rows saved: {len(df)}")
        return True
    else:
        logger.error("\n❌ FAILED: Could not save motifs data to database")
        return False


def test_with_different_mills():
    """Test saving motifs for different mills."""
    
    output_dir = Path(__file__).resolve().parent / 'output'
    
    # Test with multiple mills if data exists
    for mill_num in [6, 8]:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing Mill {mill_num}")
        logger.info('=' * 80)
        
        # Path to segmented motifs CSV file (with mill suffix)
        segmented_file = output_dir / f'segmented_motifs_all_{mill_num:02d}.csv'
        
        if not segmented_file.exists():
            logger.warning(f"No data file found for Mill {mill_num}, skipping...")
            continue
        
        df = pd.read_csv(segmented_file, parse_dates=['TimeStamp'])
        
        try:
            loader = DataLoader(use_database=True)
            success = loader.save_motifs_to_database(
                df=df,
                mill_number=mill_num,
                table_suffix='MOTIFS',
                if_exists='replace'
            )
            
            if success:
                logger.info(f"✅ Mill {mill_num} data saved successfully")
            else:
                logger.warning(f"⚠ Mill {mill_num} save failed")
                
        except Exception as e:
            logger.error(f"Error processing Mill {mill_num}: {e}")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("TESTING MOTIFS DATABASE SAVE FUNCTIONALITY")
    logger.info("=" * 80)
    
    # Run basic test
    success = test_save_motifs()
    
    if success:
        logger.info("\n" + "=" * 80)
        logger.info("All tests completed successfully!")
        logger.info("=" * 80)
    else:
        logger.error("\n" + "=" * 80)
        logger.error("Tests failed!")
        logger.error("=" * 80)
