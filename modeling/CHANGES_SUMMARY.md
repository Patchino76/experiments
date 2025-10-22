# Changes Summary - Motifs Database Storage

## Overview
Enhanced the data preparation pipeline to filter unwanted columns, add mill-specific suffixes to CSV files, and ensure database tables are recreated on each run.

## Changes Made

### 1. Column Filtering (`prepare_data.py`)

**Excluded Columns** (removed from both CSV and database):
- `id`, `Date`, `Shift`, `Original_Sheet`, `mill_id`
- `C_v`, `C_m` (intermediate circulative load calculations)
- `motif_id`, `segment_start`, `segment_end`, `motif_distance`

**Implementation**:
```python
columns_to_exclude = [
    'id', 'Date', 'Shift', 'Original_Sheet', 'mill_id',
    'C_v', 'C_m', 'motif_id', 'segment_start', 'segment_end', 'motif_distance'
]

# Filter before saving
segmented_all_filtered = self.segmented_df.drop(
    columns=[col for col in columns_to_exclude if col in self.segmented_df.columns],
    errors='ignore'
)
```

### 2. Mill-Specific CSV Filenames

**Old Naming**:
- `segmented_motifs_all.csv`
- `segmented_motifsMV.csv`

**New Naming**:
- `segmented_motifs_all_06.csv` (for Mill 6)
- `segmented_motifs_all_08.csv` (for Mill 8)
- `segmented_motifsMV_06.csv` (for Mill 6)
- `segmented_motifsMV_08.csv` (for Mill 8)

**Benefits**:
- Prevents file conflicts when processing multiple mills
- Clear identification of which mill the data belongs to
- Easier file management and organization

### 3. Database Table Recreation

**Table Naming**:
- Changed from `MOTIF_XX` to `MOTIFS_XX` (plural)
- Schema: `mills`
- Examples: `mills.MOTIFS_06`, `mills.MOTIFS_08`

**Behavior**:
- Tables are **always recreated** on each pipeline run
- Uses `if_exists='replace'` parameter
- Ensures fresh, consistent data
- Prevents data accumulation from multiple runs

**Log Output**:
```
STEP 6: SAVING SEGMENTED DATA TO DATABASE
  Saving segmented motifs data for Mill 6...
  Table will be recreated (if_exists='replace')
✓ Segmented data saved to database table: MOTIFS_06
```

## Files Modified

### 1. `modeling/prepare_data.py`
- Added column filtering in `create_segments()` method
- Updated CSV filenames to include mill suffix
- Updated `save_to_database()` to use filtered data
- Changed table suffix from 'MOTIF' to 'MOTIFS'
- Added explicit logging about table recreation

### 2. `modeling/test_save_motifs.py`
- Updated to use new CSV filename format with mill suffix
- Changed table name references from MOTIF to MOTIFS

### 3. `modeling/MOTIFS_DATABASE_SAVE_README.md`
- Updated table naming convention (MOTIFS instead of MOTIF)
- Added section on excluded columns
- Updated CSV file naming examples
- Updated SQL query examples
- Added note about table recreation behavior

## Usage Examples

### Running the Pipeline

```python
from config import PipelineConfig
from prepare_data import DataPreparationPipeline

# Create configuration for Mill 6
config = PipelineConfig.create_default(
    mill_number=6,
    start_date="2025-09-01",
    end_date="2025-10-19"
)

# Run pipeline
pipeline = DataPreparationPipeline(config)
pipeline.run()
```

**Output Files**:
- `output/segmented_motifs_all_06.csv` (filtered data)
- `output/segmented_motifsMV_06.csv` (filtered MV-only data)

**Database Table**:
- `mills.MOTIFS_06` (recreated with filtered data)

### Testing

```bash
# Run test script
python test_save_motifs.py
```

The test script will:
1. Load `segmented_motifs_all_06.csv`
2. Save to database table `mills.MOTIFS_06`
3. Verify the save operation

## SQL Query Examples

```sql
-- View all data for Mill 6
SELECT * FROM mills."MOTIFS_06";

-- Count total rows
SELECT COUNT(*) FROM mills."MOTIFS_06";

-- Check columns (verify excluded columns are not present)
SELECT column_name 
FROM information_schema.columns 
WHERE table_schema = 'mills' 
  AND table_name = 'MOTIFS_06'
ORDER BY ordinal_position;

-- Verify no excluded columns exist
SELECT column_name 
FROM information_schema.columns 
WHERE table_schema = 'mills' 
  AND table_name = 'MOTIFS_06'
  AND column_name IN ('id', 'Date', 'Shift', 'motif_id', 'C_v', 'C_m');
-- Should return 0 rows
```

## Benefits

1. **Cleaner Data**: Removed unnecessary metadata and intermediate calculation columns
2. **Better Organization**: Mill-specific filenames prevent confusion
3. **Data Consistency**: Table recreation ensures no stale data
4. **Easier Debugging**: Clear file naming makes it easy to identify data sources
5. **Production Ready**: Filtered data ready for model training without preprocessing

## Migration Notes

If you have existing code that references the old filenames or table names:

**Old Code**:
```python
df = pd.read_csv('output/segmented_motifs_all.csv')
query = 'SELECT * FROM mills."MOTIF_06"'
```

**New Code**:
```python
df = pd.read_csv('output/segmented_motifs_all_06.csv')
query = 'SELECT * FROM mills."MOTIFS_06"'
```

## Verification Checklist

After running the pipeline, verify:

- [ ] CSV files have mill suffix (e.g., `_06`)
- [ ] Excluded columns are not in CSV files
- [ ] Database table uses MOTIFS (plural) naming
- [ ] Excluded columns are not in database table
- [ ] Table is recreated on each run (check timestamps)
- [ ] Row counts match between CSV and database
- [ ] Column counts are reduced (excluded columns removed)

## Example Log Output

```
STEP 5: CREATING SEGMENTED DATASET
  Creating MV motifs dataset (20 motifs)...
  Creating complete dataset (20 motifs)...
  ✓ MV motifs data saved to segmented_motifsMV_06.csv (12000 rows, 15 columns)
  ✓ Complete data saved to segmented_motifs_all_06.csv (12000 rows, 18 columns)

STEP 6: SAVING SEGMENTED DATA TO DATABASE
  Saving segmented motifs data for Mill 6...
  Table will be recreated (if_exists='replace')
✓ Segmented data saved to database table: MOTIFS_06
   Verification: Table mills.MOTIFS_06 now contains 12000 rows
```

## Troubleshooting

### Issue: Old CSV files still exist
**Solution**: Old files are not automatically deleted. You can manually remove them or they will be ignored.

### Issue: Old database tables still exist
**Solution**: Old `MOTIF_XX` tables will remain in the database. You can drop them manually if needed:
```sql
DROP TABLE IF EXISTS mills."MOTIF_06";
DROP TABLE IF EXISTS mills."MOTIF_08";
```

### Issue: Columns still appear in output
**Solution**: Check that the column names match exactly (case-sensitive). The filtering uses exact string matching.
