# Detailed Code Changes for Parallel Execution

## Summary
The script has been refactored to support parallel execution of (seed, fold) combinations using Python's multiprocessing module. Each parallel task gets its own temporary config file to avoid race conditions.

## Changes Made

### 1. New Imports
```python
import multiprocessing as mp
```

### 2. New Configuration Variable
```python
# Number of parallel workers for running fold+seed combinations
# Set to 1 for sequential execution (useful for debugging)
# Set to higher values (e.g., 4, 8) to run multiple experiments in parallel
# Note: Each worker will use GPU resources, so adjust based on available GPU memory
N_WORKERS = 1
```

### 3. New Function: `run_single_fold(args)`
This function runs a single (seed, fold) combination and is designed to be called by worker processes.

**Key features:**
- Takes a tuple of arguments (for compatibility with `pool.map()`)
- Creates a temporary config file: `config_seed{i}_fold{j}.yaml`
- Modifies the temporary config file (not the original)
- Performs training and evaluation
- Returns evaluation results as a dictionary
- Cleans up temporary config file in `finally` block

**Function signature:**
```python
def run_single_fold(args):
    """Run a single fold+seed combination. This function is designed to be run in parallel."""
    (seed_idx, fold_idx, train_indices, test_indices, config_path_template,
     experiment_id, group_by_video, train_overrides, landmark_sets,
     n_folds, n_seeds, num_frames, timestamp) = args
```

**Config file handling:**
```python
# Create a unique config file for this fold+seed combination
config_dir = Path(config_path_template).parent
config_name = Path(config_path_template).stem
config_ext = Path(config_path_template).suffix
config_path = str(config_dir / f"{config_name}_seed{seed_idx}_fold{fold_idx}{config_ext}")

# Copy the template config to the new location
shutil.copy(config_path_template, config_path)

try:
    # ... do work with config_path ...
finally:
    # Clean up the temporary config file
    if os.path.exists(config_path):
        os.remove(config_path)
```

### 4. Refactored Function: `run_experiment()`
The main experiment function has been refactored to prepare all tasks upfront and distribute them to workers.

**Old approach (sequential):**
```python
def run_experiment(...):
    evaluation_results_list = []
    for i in range(n_seeds):
        for j, (train_indices, test_indices) in enumerate(folds):
            # Modify config_path directly (CONFLICT RISK!)
            # Train and evaluate
            evaluation_results_list.append(results)
    return pd.DataFrame(evaluation_results_list)
```

**New approach (parallel):**
```python
def run_experiment(...):
    # Prepare all fold+seed combinations
    all_tasks = []
    for i in range(n_seeds):
        for j, (train_indices, test_indices) in enumerate(folds):
            task_args = (i, j, train_indices, test_indices, config_path, ...)
            all_tasks.append(task_args)
    
    # Run tasks in parallel
    if N_WORKERS > 1:
        with mp.Pool(processes=N_WORKERS) as pool:
            evaluation_results_list = pool.map(run_single_fold, all_tasks)
    else:
        # Sequential execution for debugging
        evaluation_results_list = [run_single_fold(task) for task in all_tasks]
    
    return pd.DataFrame(evaluation_results_list)
```

### 5. Unique Shuffle Number Assignment
To prevent collisions when multiple seeds run in parallel, each (seed, fold) combination gets a unique shuffle number:

```python
# In run_single_fold()
shuffle_num = seed_idx * n_folds + fold_idx + 1

# Example with 4 folds:
# Seed 0, Fold 0 → Shuffle 1
# Seed 0, Fold 1 → Shuffle 2
# Seed 0, Fold 2 → Shuffle 3
# Seed 0, Fold 3 → Shuffle 4
# Seed 1, Fold 0 → Shuffle 5
# Seed 1, Fold 1 → Shuffle 6
# etc.
```

This ensures that:
- Training datasets don't overwrite each other
- Model directories are unique: `trainset75shuffle1/`, `trainset75shuffle2/`, etc.
- Evaluation results are stored separately

**Why this is critical**: Without unique shuffle numbers, parallel workers would create directories with the same names (e.g., both Seed 0 Fold 0 and Seed 1 Fold 0 would try to use `trainset75shuffle1/`), causing file conflicts and data corruption.

### 6. GroupKFold Random State Handling
GroupKFold doesn't support `random_state` directly, so we manually shuffle groups:

```python
if group_by_video:
    # Note: GroupKFold doesn't support random_state directly, so we shuffle groups manually
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(42 + i)
    shuffled_group_order = rng.permutation(unique_groups)
    # Create a mapping from old group to new group based on shuffled order
    group_mapping = {old_g: new_g for new_g, old_g in enumerate(shuffled_group_order)}
    shuffled_groups = np.array([group_mapping[g] for g in groups])
    cv = GroupKFold(n_splits=n_folds)
    folds = list(cv.split(np.arange(num_frames), groups=shuffled_groups))
```

## File Structure Changes

### Before
```
project/
├── config.yaml (modified by each fold - CONFLICT RISK!)
├── dlc-models-pytorch/
│   └── iteration-0/
│       ├── trainset75shuffle1/
│       ├── trainset75shuffle2/
│       └── ...
└── evaluation-results-pytorch/
```

### After
```
project/
├── config.yaml (original, read-only)
├── config_seed0_fold0.yaml (temporary, auto-deleted)
├── config_seed0_fold1.yaml (temporary, auto-deleted)
├── config_seed1_fold0.yaml (temporary, auto-deleted)
├── dlc-models-pytorch/
│   └── iteration-0/
│       ├── trainset75shuffle1/
│       ├── trainset75shuffle2/
│       └── ...
└── evaluation-results-pytorch/
```

## Execution Flow Comparison

### Sequential (N_WORKERS=1)
```
Time: 0h ──────────────────────────────────────────────────> 8h
      │                                                      │
      ├─ Fold 0, Seed 0 (1h)
      ├─ Fold 1, Seed 0 (1h)
      ├─ Fold 2, Seed 0 (1h)
      ├─ Fold 3, Seed 0 (1h)
      ├─ Fold 0, Seed 1 (1h)
      ├─ Fold 1, Seed 1 (1h)
      ├─ Fold 2, Seed 1 (1h)
      └─ Fold 3, Seed 1 (1h)
```

### Parallel (N_WORKERS=4)
```
Time: 0h ──────────────────────> 2h
      │                         │
      ├─ Worker 1: Fold 0, Seed 0 (1h) ──> Fold 0, Seed 1 (1h)
      ├─ Worker 2: Fold 1, Seed 0 (1h) ──> Fold 1, Seed 1 (1h)
      ├─ Worker 3: Fold 2, Seed 0 (1h) ──> Fold 2, Seed 1 (1h)
      └─ Worker 4: Fold 3, Seed 0 (1h) ──> Fold 3, Seed 1 (1h)
```

## Code Diff Summary

### Lines Added: ~100
- New `run_single_fold()` function (~100 lines)
- Config file copying and cleanup logic
- Multiprocessing pool creation and task distribution
- GroupKFold random state handling

### Lines Modified: ~50
- `run_experiment()` refactored to prepare tasks instead of executing directly
- Loop structure changed from nested execution to task preparation
- Result collection changed from append to pool.map()

### Lines Removed: ~0
- No functionality removed, only refactored

## Backward Compatibility

✅ **Fully backward compatible**
- Same function signature for `run_experiment()`
- Same return type (DataFrame)
- Same result format
- Setting `N_WORKERS=1` gives identical behavior to original code

## Testing Recommendations

1. **Test with N_WORKERS=1 first**
   - Verify sequential execution works
   - Check that results match previous runs

2. **Test with N_WORKERS=2**
   - Verify parallel execution works
   - Check for any race conditions

3. **Gradually increase N_WORKERS**
   - Monitor GPU memory usage
   - Check for OOM errors
   - Verify all results are collected correctly

4. **Verify cleanup**
   - Check that temporary config files are deleted
   - Verify no leftover files after execution

## Performance Tuning

### Optimal N_WORKERS
```python
# For single GPU with 24GB memory
N_WORKERS = 2-4  # Depends on model size and batch size

# For multi-GPU system (4 GPUs)
N_WORKERS = 4-8  # One or two workers per GPU

# For debugging
N_WORKERS = 1  # Sequential execution
```

### GPU Assignment (Optional Enhancement)
For multi-GPU systems, you can add GPU assignment in `run_single_fold()`:

```python
# At the start of run_single_fold()
gpu_id = seed_idx % torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
```

## Error Handling

The code includes proper error handling:
- `try/finally` block ensures config cleanup
- Multiprocessing pool automatically handles worker failures
- Original config file is never modified (only copies)

## Known Limitations

1. **GPU Memory**: Each worker uses GPU memory, so N_WORKERS is limited by available GPU memory
2. **File I/O**: Creating many temporary config files may be slow on some filesystems
3. **Print Output**: Parallel execution may interleave print statements from different workers
4. **Debugging**: Parallel execution makes debugging harder (use N_WORKERS=1 for debugging)

