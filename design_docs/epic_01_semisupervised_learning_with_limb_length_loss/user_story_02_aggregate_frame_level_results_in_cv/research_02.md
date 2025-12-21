# Research: Aggregate Frame-Level Validation Results in Cross-Validation

## Overview
This document identifies the relevant files, functions, and specific lines of code needed to implement frame-level result aggregation in `run_cv.py`.

---

## Primary File: `run_cv.py`

### File Structure
- **Location**: `/home/alek/projects/dlc-dev2/run_cv.py`
- **Total Lines**: 410
- **Purpose**: Orchestrates cross-validation experiments with multiple folds, seeds, and landmark sets

---

## Key Functions and Code Sections

### 1. `run_single_fold()` Function
**Lines**: 51-191

**Purpose**: Executes a single fold+seed combination, including training and evaluation

**Function Signature** (Lines 51-55):
```python
def run_single_fold(args):
    """Run a single fold+seed combination."""
    (seed_idx, fold_idx, train_indices, test_indices, config_path_template,
     experiment_id, group_by_video, train_overrides, landmark_sets,
     n_folds, n_seeds, num_frames, timestamp) = args
```

**Key Variables Available**:
- `seed_idx`: Current seed index (0-based)
- `fold_idx`: Current fold index (0-based)
- `experiment_id`: Experiment identifier string
- `landmark_sets`: Dictionary mapping landmark set names to bodypart lists
- `shuffle_num`: Computed at line 72 as `seed_idx * n_folds + fold_idx + SHUFFLE_OFFSET`

---

### 2. Evaluation Loop for Multiple Landmark Sets
**Lines**: 134-185

**Critical Section** (Lines 134-148):
```python
evaluation_results = {}
for l_idx, (landmark_set_name, landmark_set) in enumerate(landmark_sets.items()):
    iteration = cfg['iteration']
    engine_name = deeplabcut.compat.get_project_engine(cfg).aliases[0]
    trainingset_identifier = f"{cfg['Task']}{cfg['date']}-trainset{train_fraction_percent}shuffle{shuffle_num}"
    evaluation_folder = Path(project_path) / f"evaluation-results-{engine_name}" / f"iteration-{iteration}" / trainingset_identifier
    # recursively delete evaluation folder contents, but not the folder itself
    if evaluation_folder.exists():
        for child in evaluation_folder.glob('*'):
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)

    deeplabcut.evaluate_network(config_path, Shuffles=[shuffle_num], plotting=False, comparisonbodyparts=landmark_set)
```

**Key Variables**:
- `landmark_set_name`: Name of the current landmark set (e.g., 'all', 'truncated')
- `landmark_set`: List of bodyparts or 'all'
- `evaluation_folder`: Path object pointing to the evaluation results directory
- `engine_name`: Either 'pytorch' or 'tensorflow' (determines folder name)
- `trainingset_identifier`: String identifying the specific training set/shuffle

**Line 148**: Call to `deeplabcut.evaluate_network()`
- **MODIFICATION POINT**: Add `save_frame_level_results=True` parameter here

---

### 3. Parsing Evaluation Results (Summary CSV)
**Lines**: 150-185

**Current Implementation** (Lines 154-161):
```python
# Find the results CSV file
csv_files = list(evaluation_folder.glob('*-results.csv'))
if not csv_files:
    raise FileNotFoundError(f"No evaluation CSV file found in {evaluation_folder}")

# Read the CSV and clean column names
eval_df = pd.read_csv(csv_files[0])
eval_df.columns = eval_df.columns.str.strip().str.replace('%', '') # Clean '%Training...'
```

**Pattern to Follow**: Similar logic needed for frame-level CSV
- Use `evaluation_folder.glob('*-frame-level-results.csv')` to find the file
- Read with `pd.read_csv()`
- Add metadata columns
- Append to aggregate file

**Lines 163-185**: Metadata addition and aggregation for summary results
```python
prefix_columns = ['test rmse', 'test rmse_pcutoff', 'test mAP', 'test mAR']

if not eval_df.empty:
    # Convert the first row to a dictionary to get all columns
    summary_dict = eval_df.iloc[0].to_dict()
    summary_dict['fold'] = fold_idx # Add our custom fold number
    summary_dict['seed'] = seed_idx
    summary_dict['experiment'] = experiment_id
    # summary_dict['params'] = params_str
    summary_dict['group_by_video'] = group_by_video
    summary_dict['timestamp'] = timestamp
    for key, value in train_overrides.items():
        summary_dict[f'override__{key}'] = value
    for col in prefix_columns:
        summary_dict[f'{landmark_set_name}__{col}'] = summary_dict.pop(col)
    if l_idx == 0:
        evaluation_results = summary_dict
    else:
        for col in prefix_columns:
            evaluation_results[f'{landmark_set_name}__{col}'] = summary_dict[f'{landmark_set_name}__{col}']
```

**Pattern to Adapt**: Add similar metadata to frame-level results
- Add `fold`, `seed`, `experiment_id`, `landmark_set_name`, `shuffle_num` columns
- But for frame-level, we append ALL rows (not just first row)

---

### 4. `save_results_incrementally()` Function
**Lines**: 195-204

**Implementation**:
```python
def save_results_incrementally(results_df, results_file):
    """
    Save results to CSV file incrementally.

    Args:
        results_df: DataFrame containing results
        results_file: Path to the results file
    """
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
```

**Usage Pattern**: This function is called after each fold completes (line 340)

**New Function Needed**: `save_frame_level_results_incrementally()`
- Similar signature and implementation
- Will be called after parsing each frame-level CSV
- Separate file per landmark set

---

### 5. `load_existing_results()` Function
**Lines**: 206-241

**Purpose**: Loads existing CV results to support resume functionality

**Implementation Pattern**:
```python
def load_existing_results(results_file):
    """
    Load existing results from CSV file if it exists.

    Args:
        results_file: Path to the results file

    Returns:
        DataFrame with existing results, or None if file doesn't exist
    """
    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file)
            return df
        except Exception as e:
            print(f"Warning: Could not load existing results from {results_file}: {e}")
            return None
    return None
```

**Consideration**: May need similar function for frame-level results if implementing resume logic
- For initial implementation, can skip this (just overwrite/append)

---

### 6. `run_experiment()` Function
**Lines**: 244-355

**Purpose**: Main orchestration function that runs all folds and seeds

**Key Parameters** (Lines 244):
```python
def run_experiment(config_path, n_folds, n_seeds, experiment_id='experiment_1', 
                   group_by_video=False, train_overrides={}, landmark_sets={'all': 'all'}):
```

**MODIFICATION POINT**: Add `save_frame_level_results=False` parameter here

**Results File Setup** (Lines 258-268):
```python
# Prepare results file (without timestamp for recovery)
results_file = f'cv_results_{experiment_id}.csv'

# Load existing results
existing_results = load_existing_results(results_file)

# Initialize results list with existing results
all_results = []
if existing_results is not None:
    all_results = existing_results.to_dict('records')
    print(f"\nRetaining {len(existing_results)} existing results")
```

**Pattern to Follow**: Initialize frame-level aggregate files similarly
- One file per landmark set: `cv_frame_level_results_{experiment_id}_{landmark_set_name}.csv`

**Task Execution Loop** (Lines 328-348):
```python
# Run tasks sequentially with incremental saving
for task_idx, task in enumerate(all_tasks):
    seed_idx, fold_idx = task[0], task[1]
    print(f"\n{'='*60}")
    print(f"Task {task_idx + 1}/{len(all_tasks)}: Seed {seed_idx+1}/{n_seeds}, Fold {fold_idx+1}/{n_folds}")
    print(f"{'='*60}")

    try:
        result = run_single_fold(task)
        all_results.append(result)

        # Save results incrementally after each task
        results_df = pd.DataFrame(all_results)
        save_results_incrementally(results_df, results_file)

        print(f"\n✓ Task {task_idx + 1}/{len(all_tasks)} completed successfully")

    except Exception as e:
        print(f"\n✗ Task {task_idx + 1}/{len(all_tasks)} failed with error: {e}")
        print(f"Results up to this point have been saved to: {results_file}")
        print(f"To resume, simply run the script again - it will skip completed tasks")
        raise
```

**Note**: Frame-level aggregation happens INSIDE `run_single_fold()`, not here
- This loop only handles summary results aggregation

---

## Implementation Strategy

### Location of Changes

#### 1. Add Constant (After line 18)
**Location**: After `TRAIN_BATCH_SIZE = 48` in the constants section at the top of the file

**Code to Add**:
```python
SAVE_FRAME_LEVEL_RESULTS = False  # Set to True to aggregate frame-level validation results
```

#### 2. Add New Function (After line 204)
**Function**: `save_frame_level_results_incrementally()`
```python
def save_frame_level_results_incrementally(results_df, results_file):
    """
    Save frame-level results to CSV file incrementally.
    
    Args:
        results_df: DataFrame containing frame-level results with metadata
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

#### 3. Modify `run_single_fold()` Function
**Location**: Line 148 and after

**Pseudocode**:
```python
# Line 148: Modify evaluate_network call to use constant
deeplabcut.evaluate_network(
    config_path,
    Shuffles=[shuffle_num],
    plotting=False,
    comparisonbodyparts=landmark_set,
    save_frame_level_results=SAVE_FRAME_LEVEL_RESULTS  # Use global constant
)

# NEW CODE: Parse and aggregate frame-level results (only if constant is True)
if SAVE_FRAME_LEVEL_RESULTS:
    # Find frame-level CSV
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

---

## Related Code Patterns

### Pattern: Incremental CSV Saving
**Reference**: Lines 195-204, 338-340

**Key Insight**: Simple overwrite pattern, not true append
- Load existing results into memory
- Append new results
- Save entire DataFrame

### Pattern: Metadata Addition
**Reference**: Lines 168-175

**Key Insight**: Add metadata columns to each row
- Use dictionary for summary results (single row)
- Use DataFrame.insert() for frame-level results (multiple rows)

### Pattern: File Naming
**Reference**: Line 259

**Key Insight**: Use experiment_id in filename
- Summary: `cv_results_{experiment_id}.csv`
- Frame-level: `cv_frame_level_results_{experiment_id}_{landmark_set_name}.csv`

---

## Dependencies and Imports

**Current Imports** (Lines 1-12):
```python
import deeplabcut
import deeplabcut.compat
import numpy as np
import pandas as pd
import os
from pathlib import Path
import torch
import yaml
from sklearn.model_selection import GroupKFold, KFold
from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
import shutil
import datetime
```

**No New Imports Needed**: All required libraries already imported

---

## Summary of Modification Points

1. **After Line 18** (after `TRAIN_BATCH_SIZE`): Add constant `SAVE_FRAME_LEVEL_RESULTS = False`
2. **Line 148**: Modify `deeplabcut.evaluate_network()` call to pass `save_frame_level_results=SAVE_FRAME_LEVEL_RESULTS`
3. **After Line 148**: Add frame-level CSV parsing and aggregation logic (wrapped in `if SAVE_FRAME_LEVEL_RESULTS:`)
4. **After Line 204**: Add `save_frame_level_results_incrementally()` function

**Note**: No need to modify function signatures or pass parameters through the call chain - just use the global constant directly.

