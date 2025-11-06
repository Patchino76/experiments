"""
Data loading and preprocessing module.

Handles database connections, data filtering, and feature engineering.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple, List

# Add parent to path for db imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles data loading from database or cache files.
    """
    
    def __init__(self, use_database: bool = True):
        """
        Initialize data loader.
        
        Args:
            use_database: If True, load from database; otherwise use cache
        """
        self.use_database = use_database
        self.connector = None
        
        if use_database:
            try:
                from db.db_connector import MillsDataConnector
                from db.settings import settings
                
                db_config = {
                    'host': settings.DB_HOST,
                    'port': settings.DB_PORT,
                    'dbname': settings.DB_NAME,
                    'user': settings.DB_USER,
                    'password': settings.DB_PASSWORD
                }
                self.connector = MillsDataConnector(**db_config)
                logger.info("✅ Database connector initialized")
            except ImportError as e:
                logger.warning(f"Database module not available: {e}")
                logger.warning("Will use cache only")
                self.use_database = False
            except Exception as e:
                logger.error(f"❌ Failed to initialize database connector: {e}")
                logger.warning("Will use cache only")
                self.use_database = False
    
    def load_mill_data(
        self,
        mill_number: int,
        start_date: str,
        end_date: str,
        resample_freq: str = '1min',
        cache_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Load mill data from database or cache.
        
        Args:
            mill_number: Mill identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            resample_freq: Resampling frequency
            cache_path: Path to cache file
            
        Returns:
            DataFrame with mill data
        """
        logger.info(f"Loading data for Mill {mill_number}")
        logger.info(f"  Date range: {start_date} to {end_date}")
        logger.info(f"  Resample: {resample_freq}")
        
        # Try cache first if it exists
        if cache_path and cache_path.exists():
            logger.info(f"  Loading from cache: {cache_path}")
            df = pd.read_csv(cache_path)
            
            # Convert TimeStamp if present
            if 'TimeStamp' in df.columns:
                df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
            
            logger.info(f"  ✓ Loaded {len(df)} rows from cache")
            return df
        
        # Load from database
        if self.use_database and self.connector:
            logger.info("  Loading from database...")
            df = self.connector.get_combined_data(
                mill_number=mill_number,
                start_date=start_date,
                end_date=end_date,
                resample_freq=resample_freq,
                save_to_logs=False,
                no_interpolation=True
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
            
            logger.info(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns from database")
            
            # Save to cache
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False)
                logger.info(f"  ✓ Cached data to {cache_path}")
            
            return df
        
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            "Database is not available. Please provide cached data."
        )
    
    def filter_data(
        self,
        df: pd.DataFrame,
        thresholds: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Filter data based on thresholds.
        
        Args:
            df: Input DataFrame
            thresholds: Dictionary mapping column names to (min, max) tuples
            
        Returns:
            Filtered DataFrame
        """
        logger.info("Filtering data...")
        initial_rows = len(df)
        
        for col, (min_val, max_val) in thresholds.items():
            if col not in df.columns:
                logger.warning(f"  Column '{col}' not found, skipping filter")
                continue
            
            before = len(df)
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]
            removed = before - len(df)
            
            if removed > 0:
                logger.info(f"  {col}: [{min_val}, {max_val}] - removed {removed} rows")
        
        final_rows = len(df)
        logger.info(f"  ✓ Filtered: {initial_rows} → {final_rows} rows ({final_rows/initial_rows*100:.1f}% retained)")
        
        return df
    
    def calculate_circulative_load(
        self,
        df: pd.DataFrame,
        rho_solid: float = 2900
    ) -> pd.DataFrame:
        """
        Calculate circulative load for grinding circuit.
        
        Args:
            df: DataFrame with Ore, PulpHC, DensityHC columns
            rho_solid: Solid density (kg/m³)
            
        Returns:
            DataFrame with added CirculativeLoad column
        """
        logger.info("Calculating circulative load...")
        
        required_cols = ['Ore', 'PulpHC', 'DensityHC']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"  Missing required columns: {required_cols}")
            return df
        
        try:
            # Volumetric concentration
            df['C_v'] = (df['DensityHC'] - 1000) / (rho_solid - 1000)
            
            # Mass concentration
            df['C_m'] = (df['C_v'] * rho_solid) / (df['C_v'] * rho_solid + (1 - df['C_v']) * 1000)
            
            # Mass flow of solids to cyclone
            df['M_solid_to_cyclone'] = df['PulpHC'] * df['C_m']
            
            # Circulative load
            df['CirculativeLoad'] = (df['M_solid_to_cyclone'] - df['Ore']) / df['Ore']
            
            # Validate
            valid_range = (df['CirculativeLoad'] >= 0.5) & (df['CirculativeLoad'] <= 5.0)
            valid_pct = valid_range.sum() / len(df) * 100
            
            logger.info(f"  ✓ Circulative load calculated")
            logger.info(f"  Valid range [0.5, 5.0]: {valid_pct:.1f}% of data")
            logger.info(f"  Mean: {df['CirculativeLoad'].mean():.2f}, Std: {df['CirculativeLoad'].std():.2f}")
            
        except Exception as e:
            logger.error(f"  Error calculating circulative load: {e}")
        
        return df
    
    def validate_columns(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if all columns present
            
        Raises:
            ValueError: If columns are missing
        """
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"  ✓ All required columns present: {len(required_columns)} columns")
        return True
