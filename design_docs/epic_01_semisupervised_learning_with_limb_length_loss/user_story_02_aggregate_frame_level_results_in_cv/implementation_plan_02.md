# Implementation Plan: Aggregate Frame-Level Validation Results in Cross-Validation

## Executive Summary

**Goal**: Enhance `run_cv.py` to automatically aggregate frame-level validation results across all cross-validation folds and seeds.

**Scope**: 4 targeted changes to a single file (`run_cv.py`), adding ~37 lines of code.

**Key Features**:
- ✅ Optional feature controlled by `SAVE_FRAME_LEVEL_RESULTS` constant (default: `False`)
- ✅ Separate aggregate CSV per landmark set (e.g., `cv_frame_level_results_ll_0d025_all.csv`)
- ✅ Incremental appending after each fold/seed evaluation
- ✅ Metadata columns added: `fold`, `seed`, `experiment_id`, `landmark_set_name`, `shuffle_num`
- ✅ Graceful error handling for missing files
- ✅ Compatible with both TensorFlow and PyTorch backends

**Impact**: Enables downstream analysis of frame-level predictions across entire CV experiments without manual file aggregation.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Overview](#overview)
3. [Files to Modify](#files-to-modify)
4. [Implementation Changes](#implementation-changes)
   - [Change 1: Add Configuration Constant](#change-1-add-configuration-constant)
   - [Change 2: Add Frame-Level Aggregation Function](#change-2-add-frame-level-aggregation-function)
   - [Change 3: Modify evaluate_network Call](#change-3-modify-deeplabutevaluate_network-call)
   - [Change 4: Add Frame-Level CSV Parsing and Aggregation Logic](#change-4-add-frame-level-csv-parsing-and-aggregation-logic)
5. [Data Flow](#data-flow)
6. [File Naming and Location](#file-naming-and-location)
7. [Variables Available in Context](#variables-available-in-context)
8. [Error Handling](#error-handling)
9. [Testing Considerations](#testing-considerations)
10. [Dependencies](#dependencies)
11. [Summary of Changes](#summary-of-changes)
12. [Acceptance Criteria Mapping](#acceptance-criteria-mapping)
13. [Concrete Example](#concrete-example)
14. [Code Diff Preview](#code-diff-preview)
15. [Next Steps](#next-steps)

---

## Overview
This document provides a detailed implementation plan for adding frame-level result aggregation to `run_cv.py`. The implementation will enable automatic collection and aggregation of frame-level validation outputs across all cross-validation folds and seeds, creating separate aggregate CSV files for each landmark set.

---

## Files to Modify

### Primary File: `run_cv.py`
**Location**: `/home/alek/projects/dlc-dev2/run_cv.py`

**No new files needed** - all changes are contained within this single file.

---

## Implementation Changes

### Change 1: Add Configuration Constant
**Location**: After line 18 (after `TRAIN_BATCH_SIZE = 48`)

**Action**: Add a new constant to enable/disable frame-level result aggregation

**Code to Add**:
```python
SAVE_FRAME_LEVEL_RESULTS = False  # Set to True to aggregate frame-level validation results
```

**Rationale**: 
- Provides a simple on/off switch for the feature
- Default `False` maintains backward compatibility
- Prevents generating large files unless explicitly requested
- Follows existing pattern of configuration constants at top of file

---

### Change 2: Add Frame-Level Aggregation Function
**Location**: After line 204 (after `save_results_incrementally()` function)

**Action**: Create a new function to handle incremental appending of frame-level results

**Code to Add**:
```python
def save_frame_level_results_incrementally(results_df, results_file):
    """
    Save frame-level results to CSV file incrementally.
    
    Appends new frame-level results to an existing aggregate file, or creates
    a new file if it doesn't exist. Each landmark set has its own aggregate file.
    
    Args:
        results_df: DataFrame containing frame-level results with metadata columns
                   (fold, seed, experiment_id, landmark_set_name, shuffle_num)
        results_file: Path to the aggregate results file
    """
    # Load existing data if file exists
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        combined_df = results_df
    
    combined_df.to_csv(results_file, index=False)
    print(f"Frame-level results appended to: {results_file} ({len(results_df)} new rows)")
```

**Rationale**:
- Mirrors the pattern of `save_results_incrementally()` for consistency
- Handles both new file creation and appending to existing files
- Provides informative output about number of rows added
- Uses `pd.concat()` for efficient DataFrame combination

---

### Change 3: Modify `deeplabcut.evaluate_network()` Call
**Location**: Line 148 in `run_single_fold()` function

**Current Code**:
```python
deeplabcut.evaluate_network(config_path, Shuffles=[shuffle_num], plotting=False, comparisonbodyparts=landmark_set)
```

**Modified Code**:
```python
deeplabcut.evaluate_network(
    config_path, 
    Shuffles=[shuffle_num], 
    plotting=False, 
    comparisonbodyparts=landmark_set,
    save_frame_level_results=SAVE_FRAME_LEVEL_RESULTS
)
```

**Rationale**:
- Passes the global constant to control frame-level CSV generation
- Both TensorFlow and PyTorch backends support this parameter (default=True in DLC)
- No need to pass through function signatures - uses global constant directly

---

### Change 4: Add Frame-Level CSV Parsing and Aggregation Logic
**Location**: After line 148 (immediately after `deeplabcut.evaluate_network()` call, before line 150)

**Action**: Parse the generated frame-level CSV and append to landmark-set-specific aggregate file

**Code to Add**:
```python
            # Parse and aggregate frame-level results (if enabled)
            if SAVE_FRAME_LEVEL_RESULTS:
                # Find frame-level CSV file
                frame_level_csv_files = list(evaluation_folder.glob('*-frame-level-results.csv'))
                
                if frame_level_csv_files:
                    # Read the frame-level CSV
                    df_frame_level = pd.read_csv(frame_level_csv_files[0])
                    
                    # Add metadata columns at the beginning
                    df_frame_level.insert(0, 'shuffle_num', shuffle_num)
                    df_frame_level.insert(0, 'landmark_set_name', landmark_set_name)
                    df_frame_level.insert(0, 'experiment_id', experiment_id)
                    df_frame_level.insert(0, 'seed', seed_idx)
                    df_frame_level.insert(0, 'fold', fold_idx)
                    
                    # Append to landmark-set-specific aggregate file
                    aggregate_file = f'cv_frame_level_results_{experiment_id}_{landmark_set_name}.csv'
                    save_frame_level_results_incrementally(df_frame_level, aggregate_file)
                else:
                    print(f"Warning: No frame-level CSV found in {evaluation_folder}")
```

**Rationale**:
- Only executes when `SAVE_FRAME_LEVEL_RESULTS` is `True`
- Uses same pattern as existing summary CSV parsing (lines 154-161)
- Adds metadata columns at the beginning for easy filtering/grouping
- Uses `DataFrame.insert(0, ...)` to prepend columns in desired order
- Handles missing files gracefully with warning (doesn't crash execution)
- Creates separate aggregate file per landmark set for easier analysis

---

## Data Flow

### Frame-Level CSV Structure
Generated by `deeplabcut.evaluate_network()` when `save_frame_level_results=True`:

**Filename**: `{DLCscorer}-frame-level-results.csv`

**Columns** (example with 3 bodyparts: nose, leftear, rightear):
```
frame_index, image_path, gt_nose_x, gt_nose_y, pred_nose_x, pred_nose_y, conf_nose,
gt_leftear_x, gt_leftear_y, pred_leftear_x, pred_leftear_y, conf_leftear,
gt_rightear_x, gt_rightear_y, pred_rightear_x, pred_rightear_y, conf_rightear
```

**Rows**: One row per test frame (typically 20% of total frames per fold)

---

### Aggregate CSV Structure
Created by `save_frame_level_results_incrementally()`:

**Filename**: `cv_frame_level_results_{experiment_id}_{landmark_set_name}.csv`

**Columns** (metadata + original columns):
```
fold, seed, experiment_id, landmark_set_name, shuffle_num, frame_index, image_path,
gt_nose_x, gt_nose_y, pred_nose_x, pred_nose_y, conf_nose, ...
```

**Rows**: Accumulated across all folds/seeds for a specific landmark set
- Example: 5 folds × 2 seeds × 100 test frames/fold = 1000 rows per landmark set

---

## File Naming and Location

### Aggregate Files
- **Pattern**: `cv_frame_level_results_{experiment_id}_{landmark_set_name}.csv`
- **Location**: Same directory as `cv_results_{experiment_id}.csv` (project root or working directory)
- **Examples**:
  - `cv_frame_level_results_ll_0d025_all.csv`
  - `cv_frame_level_results_ll_0d025_truncated.csv`

### Per-Fold Frame-Level Files
- **Pattern**: `{DLCscorer}-frame-level-results.csv`
- **Location**: `{project_path}/evaluation-results-{engine}/iteration-{iteration}/{trainingset_identifier}/`
- **Note**: These are temporary - deleted before each evaluation (lines 140-145)

---

## Variables Available in Context

At the point of implementation (inside `run_single_fold()`, within the landmark set loop):

| Variable | Type | Description | Example Value |
|----------|------|-------------|---------------|
| `fold_idx` | int | Current fold index (0-based) | 2 |
| `seed_idx` | int | Current seed index (0-based) | 1 |
| `experiment_id` | str | Experiment identifier | 'll_0d025' |
| `landmark_set_name` | str | Name of current landmark set | 'all' or 'truncated' |
| `landmark_set` | str or list | Bodyparts for evaluation | 'all' or ['nose', 'leftear'] |
| `shuffle_num` | int | Unique shuffle number | 107 |
| `evaluation_folder` | Path | Path to evaluation results | Path('.../evaluation-results-pytorch/...') |
| `config_path` | str | Path to fold-specific config | '.../config_full_seed1_fold2.yaml' |

---

## Error Handling

### Missing Frame-Level CSV
**Scenario**: `deeplabcut.evaluate_network()` completes but no frame-level CSV is found

**Handling**:
```python
if frame_level_csv_files:
    # Process file
else:
    print(f"Warning: No frame-level CSV found in {evaluation_folder}")
```

**Behavior**: Logs warning and continues execution without crashing

### File I/O Errors
**Scenario**: Permission errors, disk full, etc.

**Handling**: Let exceptions propagate to outer try-except in `run_experiment()` (lines 334-348)
- Saves progress up to that point
- Provides clear error message
- Supports resume functionality

---

## Testing Considerations

### Manual Testing Checklist
1. **Constant Off**: Verify `SAVE_FRAME_LEVEL_RESULTS=False` doesn't generate aggregate files
2. **Constant On**: Verify `SAVE_FRAME_LEVEL_RESULTS=True` generates correct aggregate files
3. **Multiple Landmark Sets**: Verify separate files created for each landmark set
4. **Metadata Accuracy**: Verify fold, seed, experiment_id, etc. are correct in aggregate
5. **Column Order**: Verify metadata columns appear first
6. **Incremental Appending**: Verify new rows append correctly to existing files
7. **Resume Functionality**: Verify resume works correctly with frame-level aggregation

### Expected File Sizes
- **Per-fold CSV**: ~10-100 KB (depends on test set size and number of bodyparts)
- **Aggregate CSV**: ~100 KB - 10 MB (depends on total folds/seeds and bodyparts)

---

## Dependencies

### Existing Imports (No New Imports Needed)
All required libraries are already imported in `run_cv.py`:
- `pandas` (line 4): For DataFrame operations
- `os` (line 5): For file existence checks
- `pathlib.Path` (line 6): For path operations

### External Dependencies
- `deeplabcut.evaluate_network()`: Must support `save_frame_level_results` parameter
  - ✅ Supported in both TensorFlow backend (line 615 in `evaluate.py`)
  - ✅ Supported in both PyTorch backend (line 796 in `evaluation.py`)

---

## Summary of Changes

| Change # | Location | Type | Lines Added | Complexity |
|----------|----------|------|-------------|------------|
| 1 | After line 18 | Add constant | 1 | Low |
| 2 | After line 204 | Add function | ~15 | Low |
| 3 | Line 148 | Modify call | 1 | Low |
| 4 | After line 148 | Add logic block | ~20 | Medium |

**Total Lines Added**: ~37 lines
**Total Files Modified**: 1 file (`run_cv.py`)
**Total New Files**: 0

### Quick Reference: Line Numbers in `run_cv.py`

| Section | Current Line | Description |
|---------|--------------|-------------|
| Constants section | 14-18 | Where to add `SAVE_FRAME_LEVEL_RESULTS` |
| `save_results_incrementally()` | 195-204 | Where to add new function after |
| `run_single_fold()` starts | 51 | Function containing evaluation loop |
| Evaluation loop starts | 134 | Loop over landmark sets |
| `evaluate_network()` call | 148 | Where to modify and add logic after |
| Summary CSV parsing | 154-161 | Pattern to follow for frame-level parsing |

---

## Acceptance Criteria Mapping

| AC # | Description | Implementation |
|------|-------------|----------------|
| AC1 | Constant to enable frame-level results | Change 1: Add `SAVE_FRAME_LEVEL_RESULTS` constant |
| AC2 | Parse frame-level CSV after evaluation | Change 4: Parse CSV and add metadata |
| AC3 | Separate aggregate files per landmark set | Change 4: Use `{landmark_set_name}` in filename |
| AC4 | Incremental appending | Change 2: `save_frame_level_results_incrementally()` |
| AC5 | Metadata accuracy | Change 4: Insert metadata columns with correct values |
| AC6 | Handle missing frame-level CSV | Change 4: Check `if frame_level_csv_files` with warning |
| AC7 | Column order preservation | Change 4: Use `insert(0, ...)` to prepend metadata |
| AC8 | Integration with existing CV workflow | All changes: Non-invasive, uses global constant |

---

## Concrete Example

### Scenario
- **Experiment ID**: `ll_0d025`
- **Folds**: 5
- **Seeds**: 2
- **Landmark Sets**: `{'all': 'all', 'truncated': ['nose', 'leftear', 'rightear']}`
- **Test Frames per Fold**: ~100 frames

### Generated Files

#### Aggregate Files (in project root)
```
cv_results_ll_0d025.csv                          # Summary results (existing)
cv_frame_level_results_ll_0d025_all.csv          # Frame-level for 'all' landmark set
cv_frame_level_results_ll_0d025_truncated.csv    # Frame-level for 'truncated' landmark set
```

#### Aggregate File Contents (example: `cv_frame_level_results_ll_0d025_all.csv`)
```csv
fold,seed,experiment_id,landmark_set_name,shuffle_num,frame_index,image_path,gt_nose_x,gt_nose_y,pred_nose_x,pred_nose_y,conf_nose,...
0,0,ll_0d025,all,100,0,/path/to/img1.png,245.3,189.2,244.8,188.9,0.98,...
0,0,ll_0d025,all,100,1,/path/to/img2.png,251.1,192.4,250.9,192.1,0.97,...
...
0,1,ll_0d025,all,105,0,/path/to/img1.png,245.3,189.2,245.1,189.0,0.99,...
...
4,1,ll_0d025,all,109,99,/path/to/img100.png,198.7,156.3,198.5,156.2,0.96,...
```

**Total Rows**: 5 folds × 2 seeds × 100 test frames = 1000 rows per landmark set

### Execution Timeline

```
Task 1/10: Seed 1/2, Fold 1/5
  ├─ Train model (shuffle 100)
  ├─ Evaluate with landmark_set='all'
  │   ├─ Generate: DLCscorer-frame-level-results.csv (100 rows)
  │   ├─ Add metadata: fold=0, seed=0, experiment_id='ll_0d025', landmark_set_name='all', shuffle_num=100
  │   └─ Append to: cv_frame_level_results_ll_0d025_all.csv (100 rows total)
  ├─ Evaluate with landmark_set='truncated'
  │   ├─ Generate: DLCscorer-frame-level-results.csv (100 rows)
  │   ├─ Add metadata: fold=0, seed=0, experiment_id='ll_0d025', landmark_set_name='truncated', shuffle_num=100
  │   └─ Append to: cv_frame_level_results_ll_0d025_truncated.csv (100 rows total)
  └─ Save summary to: cv_results_ll_0d025.csv (1 row total)

Task 2/10: Seed 1/2, Fold 2/5
  ├─ Train model (shuffle 101)
  ├─ Evaluate with landmark_set='all'
  │   └─ Append to: cv_frame_level_results_ll_0d025_all.csv (200 rows total)
  ├─ Evaluate with landmark_set='truncated'
  │   └─ Append to: cv_frame_level_results_ll_0d025_truncated.csv (200 rows total)
  └─ Save summary to: cv_results_ll_0d025.csv (2 rows total)

...

Task 10/10: Seed 2/2, Fold 5/5
  ├─ Train model (shuffle 109)
  ├─ Evaluate with landmark_set='all'
  │   └─ Append to: cv_frame_level_results_ll_0d025_all.csv (1000 rows total)
  ├─ Evaluate with landmark_set='truncated'
  │   └─ Append to: cv_frame_level_results_ll_0d025_truncated.csv (1000 rows total)
  └─ Save summary to: cv_results_ll_0d025.csv (10 rows total)
```

---

## Code Diff Preview

### Before (lines 18-19)
```python
TRAIN_BATCH_SIZE = 48
# CUSTOM_WEIGHTS = '/home/alek/projects/cdl-test1/resnet50_unet_encoder_tuned.pth'
```

### After (lines 18-20)
```python
TRAIN_BATCH_SIZE = 48
SAVE_FRAME_LEVEL_RESULTS = False  # Set to True to aggregate frame-level validation results
# CUSTOM_WEIGHTS = '/home/alek/projects/cdl-test1/resnet50_unet_encoder_tuned.pth'
```

---

### Before (line 148)
```python
            deeplabcut.evaluate_network(config_path, Shuffles=[shuffle_num], plotting=False, comparisonbodyparts=landmark_set)
```

### After (lines 148-153)
```python
            deeplabcut.evaluate_network(
                config_path,
                Shuffles=[shuffle_num],
                plotting=False,
                comparisonbodyparts=landmark_set,
                save_frame_level_results=SAVE_FRAME_LEVEL_RESULTS
            )
```

---

### Before (line 150)
```python
            # e. Parse evaluation results and store them
```

### After (lines 155-173)
```python
            # Parse and aggregate frame-level results (if enabled)
            if SAVE_FRAME_LEVEL_RESULTS:
                # Find frame-level CSV file
                frame_level_csv_files = list(evaluation_folder.glob('*-frame-level-results.csv'))

                if frame_level_csv_files:
                    # Read the frame-level CSV
                    df_frame_level = pd.read_csv(frame_level_csv_files[0])

                    # Add metadata columns at the beginning
                    df_frame_level.insert(0, 'shuffle_num', shuffle_num)
                    df_frame_level.insert(0, 'landmark_set_name', landmark_set_name)
                    df_frame_level.insert(0, 'experiment_id', experiment_id)
                    df_frame_level.insert(0, 'seed', seed_idx)
                    df_frame_level.insert(0, 'fold', fold_idx)

                    # Append to landmark-set-specific aggregate file
                    aggregate_file = f'cv_frame_level_results_{experiment_id}_{landmark_set_name}.csv'
                    save_frame_level_results_incrementally(df_frame_level, aggregate_file)
                else:
                    print(f"Warning: No frame-level CSV found in {evaluation_folder}")

            # e. Parse evaluation results and store them
```

---

## Next Steps

1. **Implement Changes**: Apply the 4 changes to `run_cv.py` in order
2. **Test with Small Dataset**: Run with `SAVE_FRAME_LEVEL_RESULTS=True` on 2 folds, 1 seed
3. **Verify Output**: Check aggregate CSV structure and metadata
4. **Test Resume**: Interrupt and resume to verify incremental appending works
5. **Full Test**: Run complete CV experiment with multiple landmark sets
6. **Documentation**: Update any user-facing documentation if needed


