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
        Load mill data from database.
        
        Args:
            mill_number: Mill identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            resample_freq: Resampling frequency
            cache_path: Ignored - always loads from database
            
        Returns:
            DataFrame with mill data
        """
        logger.info(f"Loading data for Mill {mill_number}")
        logger.info(f"  Date range: {start_date} to {end_date}")
        logger.info(f"  Resample: {resample_freq}")
        
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
            
            logger.info(f"  ✓ Loaded {len(df)} rows from database")
            return df
        else:
            raise RuntimeError("Database not available. Please ensure use_database=True and database connection is working.")
    
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
    
    def filter_data_adaptive(
        self,
        df: pd.DataFrame,
        columns: List[str],
        window: int = 1440,
        k: float = 5.0
    ) -> pd.DataFrame:
        """
        Remove statistical outliers using rolling median + MAD (median absolute deviation).
        
        Complements filter_data() (fixed global bounds) with bounds that adapt to
        slow drift (liner wear, seasonal ore hardness, gradual sensor calibration
        shift) instead of a single hand-tuned static range.
        
        A point is removed if it deviates from its local rolling median by more
        than k * 1.4826 * MAD (1.4826 scales MAD to be comparable to a standard
        deviation under normality).
        
        Args:
            df: Input DataFrame (must be time-ordered)
            columns: Columns to apply adaptive filtering to
            window: Rolling window size in rows (e.g. 1440 = 24h at 1-min resample)
            k: Number of scaled-MAD deviations allowed before a point is flagged
            
        Returns:
            Filtered DataFrame
        """
        logger.info("Applying adaptive (rolling MAD) filtering...")
        initial_rows = len(df)
        min_periods = max(10, window // 10)
        mask = pd.Series(True, index=df.index)
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"  Column '{col}' not found, skipping adaptive filter")
                continue
            
            rolling_median = df[col].rolling(window=window, center=True, min_periods=min_periods).median()
            abs_dev = (df[col] - rolling_median).abs()
            rolling_mad = abs_dev.rolling(window=window, center=True, min_periods=min_periods).median()
            
            threshold = k * 1.4826 * rolling_mad
            
            # Keep the row if within threshold, OR if we don't have enough
            # neighboring data to compute a reliable bound (avoids over-filtering
            # at the very start/end of the series).
            col_mask = (abs_dev <= threshold) | rolling_median.isna() | rolling_mad.isna()
            
            removed = int((~col_mask).sum())
            if removed > 0:
                logger.info(f"  {col}: adaptive filter removed {removed} rows (k={k}, window={window})")
            
            mask &= col_mask
        
        filtered_df = df[mask]
        final_rows = len(filtered_df)
        retained_pct = (final_rows / initial_rows * 100) if initial_rows > 0 else 0
        logger.info(f"  ✓ Adaptive filter: {initial_rows} → {final_rows} rows ({retained_pct:.1f}% retained)")
        
        return filtered_df
    
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
    
    def save_motifs_to_database(
        self,
        df: pd.DataFrame,
        mill_number: int,
        table_suffix: str = 'MOTIFS',
        if_exists: str = 'replace'
    ) -> bool:
        """
        Save segmented motifs data to database table.
        
        Creates a table named MOTIFS_XX where XX is the mill number (e.g., MOTIFS_06, MOTIFS_08).
        The table is created in the 'mills' schema.
        
        Args:
            df: DataFrame containing segmented motifs data
            mill_number: Mill number (6, 7, or 8)
            table_suffix: Prefix for the table name (default: 'MOTIFS')
            if_exists: How to behave if table exists: 'fail', 'replace', or 'append' (default: 'replace')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.use_database or self.connector is None:
            logger.warning("Database connector not available")
            return False
        
        try:
            logger.info(f"Saving motifs data to database table: mills.{table_suffix}_{mill_number:02d}")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            
            success = self.connector.save_motifs_to_database(
                df=df,
                mill_number=mill_number,
                table_suffix=table_suffix,
                if_exists=if_exists
            )
            
            if success:
                logger.info(f"✅ Successfully saved {len(df)} rows to mills.{table_suffix}_{mill_number:02d}")
            else:
                logger.warning(f"⚠ Failed to save to database")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error saving motifs to database: {e}")
            return False
