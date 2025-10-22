# Segmented Motifs Database Storage

## Overview

This feature enables automatic storage of segmented motifs data into PostgreSQL database tables. The segmented motifs (generated from the data preparation pipeline) are saved to dedicated tables for each mill, making them easily accessible for further analysis and model training.

## Table Structure

### Table Naming Convention
- **Schema**: `mills`
- **Table Name Format**: `MOTIFS_XX` where XX is the mill number (e.g., `MOTIFS_06`, `MOTIFS_08`)
- **Behavior**: Tables are **recreated** on each pipeline run (using `if_exists='replace'`)

### Table Contents
The tables store the complete segmented motifs dataset (`segmented_motifs_all_XX.csv`), which includes:
- **TimeStamp**: Timestamp of each data point
- **MV Features**: Manipulated variables (Ore, WaterMill, WaterZumpf, etc.)
- **CV Features**: Controlled variables (DensityHC, PulpHC, PressureHC, CirculativeLoad)
- **DV Features**: Disturbance variables (Class_15, Daiki, FE)
- **Target**: PSI200 (particle size distribution)
- **Motif Metadata**: instance_id, segment_position, M_solid_to_cyclone, etc.

### Excluded Columns
The following columns are **filtered out** from both CSV files and database tables:
- `id`, `Date`, `Shift`, `Original_Sheet`, `mill_id`
- `C_v`, `C_m` (intermediate circulative load calculations)
- `motif_id`, `segment_start`, `segment_end`, `motif_distance`

## Implementation

### 1. Database Connector (`db/db_connector.py`)

Added method `save_motifs_to_database()` to the `MillsDataConnector` class:

```python
def save_motifs_to_database(self, df, mill_number, table_suffix='MOTIF', if_exists='replace'):
    """
    Save segmented motifs data to database table.
    
    Args:
        df: DataFrame containing segmented motifs data
        mill_number: Mill number (6, 7, or 8)
        table_suffix: Prefix for the table name (default: 'MOTIF')
        if_exists: How to behave if table exists: 'fail', 'replace', or 'append'
    
    Returns:
        bool: True if successful, False otherwise
    """
```

**Features**:
- Automatic table creation if it doesn't exist
- Handles TimeStamp column formatting
- Batch insertion with chunking (1000 rows per batch)
- Verification of saved data
- Comprehensive error handling and logging

### 2. Data Loader (`modeling/database.py`)

Added wrapper method to the `DataLoader` class:

```python
def save_motifs_to_database(self, df, mill_number, table_suffix='MOTIF', if_exists='replace'):
    """
    Save segmented motifs data to database.
    
    Args:
        df: DataFrame containing segmented motifs data
        mill_number: Mill number (6, 7, or 8)
        table_suffix: Prefix for the table name (default: 'MOTIF')
        if_exists: How to behave if table exists: 'fail', 'replace', or 'append'
    
    Returns:
        bool: True if successful, False otherwise
    """
```

### 3. Data Preparation Pipeline (`modeling/prepare_data.py`)

Integrated database save as **Step 6** in the pipeline:

```python
def save_to_database(self):
    """Save segmented motifs data to database."""
    # Saves self.segmented_df to database table MOTIF_XX
```

**Pipeline Flow**:
1. Load data
2. Discover MV motifs
3. Discover density motifs (optional)
4. Merge motifs
5. Create segmented dataset
6. **Save segmented data to database** ← NEW STEP
7. Generate analysis outputs
8. Create visualizations

## Usage

### Automatic Save (via Pipeline)

When running the data preparation pipeline, the segmented data is automatically saved to the database if `use_database=True`:

```python
from config import PipelineConfig
from prepare_data import DataPreparationPipeline

# Create configuration
config = PipelineConfig.create_default(
    mill_number=6,
    start_date="2025-01-01",
    end_date="2025-10-19"
)

# Ensure database is enabled
config.use_database = True

# Run pipeline (includes automatic database save)
pipeline = DataPreparationPipeline(config)
pipeline.run()
```

### Manual Save (Standalone)

You can also save existing segmented data manually:

```python
from database import DataLoader
import pandas as pd

# Load segmented data (note the mill suffix in filename)
df = pd.read_csv('output/segmented_motifs_all_06.csv', parse_dates=['TimeStamp'])

# Initialize data loader
loader = DataLoader(use_database=True)

# Save to database
success = loader.save_motifs_to_database(
    df=df,
    mill_number=6,
    table_suffix='MOTIFS',
    if_exists='replace'  # Table is always recreated
)

if success:
    print("✅ Data saved successfully!")
```

### Testing

Run the test script to verify functionality:

```bash
python test_save_motifs.py
```

## Configuration Options

### `if_exists` Parameter

Controls behavior when table already exists:

- **`'replace'`** (default and recommended): Drop existing table and create new one
  - **This is the default behavior in the pipeline** to ensure fresh data on each run
- **`'append'`**: Add data to existing table (use with caution)
- **`'fail'`**: Raise error if table exists

### CSV File Naming

CSV files are automatically named with mill suffix:
- `segmented_motifs_all_06.csv` for Mill 6
- `segmented_motifs_all_08.csv` for Mill 8
- `segmented_motifsMV_06.csv` for MV-only motifs

This prevents file conflicts when processing multiple mills.

## Database Schema

### Example Table: `mills.MOTIFS_06`

| Column | Type | Description |
|--------|------|-------------|
| TimeStamp | timestamp | Data point timestamp |
| Ore | float | Ore feed rate (t/h) |
| WaterMill | float | Water to mill (m³/h) |
| WaterZumpf | float | Water to sump (m³/h) |
| DensityHC | float | Hydrocyclone density (kg/m³) |
| PulpHC | float | Hydrocyclone pulp flow (m³/h) |
| PressureHC | float | Hydrocyclone pressure (bar) |
| CirculativeLoad | float | Circulative load ratio |
| PSI200 | float | Particle size target (%) |
| instance_id | int | Instance identifier |
| segment_position | int | Position within segment |
| M_solid_to_cyclone | float | Mass flow of solids to cyclone |
| ... | ... | Additional columns |

**Note**: Columns like `motif_id`, `segment_start`, `segment_end`, `C_v`, `C_m`, etc. are excluded from the final table.

## Benefits

1. **Centralized Storage**: All segmented motifs data in one database
2. **Easy Access**: Query data directly from database for analysis
3. **Clean Data**: Unnecessary columns automatically filtered out
4. **Fresh Data**: Tables recreated on each run to ensure data consistency
5. **Multi-Mill Support**: Separate tables and CSV files for each mill (MOTIFS_06, MOTIFS_08, etc.)
6. **Integration Ready**: Data readily available for model training and validation
7. **Scalability**: Database handles large datasets efficiently

## Querying Saved Data

### SQL Examples

```sql
-- Get all motifs for Mill 6
SELECT * FROM mills."MOTIFS_06";

-- Count rows per instance
SELECT instance_id, COUNT(*) as count
FROM mills."MOTIFS_06"
GROUP BY instance_id
ORDER BY instance_id;

-- Get data for specific time range
SELECT * FROM mills."MOTIFS_06"
WHERE "TimeStamp" BETWEEN '2025-01-01' AND '2025-01-31';

-- Get average values per instance
SELECT 
    instance_id,
    AVG("Ore") as avg_ore,
    AVG("DensityHC") as avg_density,
    AVG("PSI200") as avg_psi200
FROM mills."MOTIFS_06"
GROUP BY instance_id;

-- Get total row count
SELECT COUNT(*) as total_rows FROM mills."MOTIFS_06";
```

### Python Examples

```python
from db.db_connector import MillsDataConnector
from db.settings import settings
import pandas as pd

# Connect to database
connector = MillsDataConnector(
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    dbname=settings.DB_NAME,
    user=settings.DB_USER,
    password=settings.DB_PASSWORD
)

# Query motifs data
query = 'SELECT * FROM mills."MOTIFS_06" LIMIT 1000'
df = pd.read_sql_query(query, connector.engine)

print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(df.head())
```

## Troubleshooting

### Issue: "Database not enabled"
**Solution**: Ensure `use_database=True` in pipeline configuration

### Issue: "Database connector not initialized"
**Solution**: Check database connection settings in `db/settings.py`

### Issue: "Permission denied"
**Solution**: Verify database user has CREATE TABLE permissions on `mills` schema

### Issue: "Table already exists"
**Solution**: Use `if_exists='replace'` or `if_exists='append'` parameter

## Files Modified

1. **`db/db_connector.py`**: Added `save_motifs_to_database()` method
2. **`modeling/database.py`**: Added wrapper method in `DataLoader` class
3. **`modeling/prepare_data.py`**: Integrated database save into pipeline

## Files Created

1. **`modeling/test_save_motifs.py`**: Test script for database save functionality
2. **`modeling/MOTIFS_DATABASE_SAVE_README.md`**: This documentation file

## Future Enhancements

Potential improvements:
- Add indexing on TimeStamp and motif_id columns for faster queries
- Implement incremental updates (only save new motifs)
- Add data validation before saving
- Support for saving to different schemas
- Automatic backup before replace operations
