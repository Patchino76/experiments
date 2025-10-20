"""
Database utilities for loading mill data.

Provides clean interface for data loading from PostgreSQL or cached files.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from db.db_connector import MillsDataConnector
from db.settings import settings

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading from database or cache."""
    
    def __init__(self, use_database: bool = True):
        """
        Initialize data loader.
        
        Args:
            use_database: If True, load from database. If False, use cached CSV.
        """
        self.use_database = use_database
        self.connector = None
        
        if use_database:
            try:
                db_config = {
                    'host': settings.DB_HOST,
                    'port': settings.DB_PORT,
                    'dbname': settings.DB_NAME,
                    'user': settings.DB_USER,
                    'password': settings.DB_PASSWORD
                }
                self.connector = MillsDataConnector(**db_config)
                logger.info("✅ Database connector initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize database connector: {e}")
                raise
    
    def load_mill_data(
        self,
        mill_number: int,
        start_date: str,
        end_date: str,
        resample_freq: str = '1min',
        cache_path: Path = None
    ) -> pd.DataFrame:
        """
        Load data for a single mill.
        
        Args:
            mill_number: Mill number (6, 7, or 8)
            start_date: Start date string
            end_date: End date string
            resample_freq: Resampling frequency
            cache_path: Path to save/load cached data
            
        Returns:
            DataFrame with mill data
        """
        if self.use_database:
            logger.info(f"Loading data from database for Mill {mill_number}...")
            logger.info(f"  Date range: {start_date} to {end_date}")
            logger.info(f"  Resample frequency: {resample_freq}")
            
            df = self.connector.get_combined_data(
                mill_number=mill_number,
                start_date=start_date,
                end_date=end_date,
                resample_freq=resample_freq,
                save_to_logs=False,
                no_interpolation=True,
            )
            
            if df is None or df.empty:
                raise ValueError(f"No data retrieved for Mill {mill_number}")
            
            # Ensure proper index
            df = df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'TimeStamp' in df.columns:
                    df.set_index('TimeStamp', inplace=True)
                else:
                    raise ValueError("Data must include 'TimeStamp' column")
            
            # Add mill_id and reset index
            df['mill_id'] = mill_number
            df = df.reset_index().rename(columns={'index': 'TimeStamp'})
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
            df.sort_values('TimeStamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            logger.info(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Cache if path provided
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False)
                logger.info(f"  ✓ Cached to {cache_path}")
            
            return df
        
        else:
            # Load from cache
            if not cache_path or not cache_path.exists():
                raise FileNotFoundError(f"Cached data not found at {cache_path}")
            
            logger.info(f"Loading cached data from {cache_path}...")
            df = pd.read_csv(cache_path, parse_dates=['TimeStamp'])
            df.sort_values('TimeStamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.info(f"  ✓ Loaded {len(df)} rows from cache")
            
            return df
    
    def load_multiple_mills(
        self,
        mill_numbers: list,
        start_date: str,
        end_date: str,
        resample_freq: str = '1min',
        cache_path: Path = None
    ) -> pd.DataFrame:
        """
        Load and combine data from multiple mills.
        
        Args:
            mill_numbers: List of mill numbers
            start_date: Start date string
            end_date: End date string
            resample_freq: Resampling frequency
            cache_path: Path to save/load cached combined data
            
        Returns:
            Combined DataFrame
        """
        if not self.use_database and cache_path and cache_path.exists():
            # Load combined cache
            logger.info(f"Loading combined cached data from {cache_path}...")
            df = pd.read_csv(cache_path, parse_dates=['TimeStamp'])
            df.sort_values('TimeStamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.info(f"  ✓ Loaded {len(df)} rows from cache")
            return df
        
        # Load from database
        frames = []
        for mill in mill_numbers:
            try:
                df = self.load_mill_data(
                    mill_number=mill,
                    start_date=start_date,
                    end_date=end_date,
                    resample_freq=resample_freq,
                    cache_path=None  # Don't cache individual mills
                )
                frames.append(df)
            except Exception as e:
                logger.warning(f"  ⚠ Failed to load Mill {mill}: {e}")
                continue
        
        if not frames:
            raise ValueError("No data retrieved for any of the specified mills")
        
        # Combine
        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values('TimeStamp', inplace=True)
        combined.reset_index(drop=True, inplace=True)
        
        logger.info(f"✓ Combined data from {len(frames)} mills: {len(combined)} total rows")
        
        # Cache combined data
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(cache_path, index=False)
            logger.info(f"  ✓ Cached combined data to {cache_path}")
        
        return combined


def filter_data(df: pd.DataFrame, filter_thresholds: dict) -> pd.DataFrame:
    """
    Filter data based on threshold constraints.
    
    Args:
        df: Input DataFrame
        filter_thresholds: Dict mapping column names to (min, max) tuples
        
    Returns:
        Filtered DataFrame
    """
    logger.info("Applying data filters...")
    original_len = len(df)
    
    # Build filter conditions
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col, (min_val, max_val) in filter_thresholds.items():
        if col in df.columns:
            col_mask = (df[col] > min_val) & (df[col] < max_val)
            mask = mask & col_mask
            logger.info(f"  {col}: {min_val} < x < {max_val}")
    
    filtered_df = df[mask].copy()
    filtered_len = len(filtered_df)
    removed = original_len - filtered_len
    pct_removed = (removed / original_len * 100) if original_len > 0 else 0
    
    logger.info(f"  ✓ Filtered: {filtered_len} rows kept, {removed} removed ({pct_removed:.1f}%)")
    
    return filtered_df


def validate_required_columns(df: pd.DataFrame, required_columns: list) -> None:
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info(f"✓ All {len(required_columns)} required columns present")
