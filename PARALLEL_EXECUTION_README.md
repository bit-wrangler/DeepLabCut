# Parallel Execution Updates

## Overview
The `run_experiments_with_cv.py` script has been updated to support parallel execution of cross-validation experiments using multiple worker processes. This allows multiple (seed, fold) combinations to run simultaneously, significantly reducing total execution time.

## Key Changes

### 1. Parallel Processing Architecture
- **Worker Function**: `run_single_fold()` - Executes a single (seed, fold) combination
- **Orchestrator Function**: `run_experiment()` - Prepares tasks and distributes them to workers
- **Parallelization**: Uses Python's `multiprocessing.Pool` to run tasks in parallel

### 2. Configuration File Handling
Each (seed, fold) combination now gets its own temporary config file to avoid conflicts:
- **Template**: Original config file at `config_path`
- **Per-task copies**: `config_seed{i}_fold{j}.yaml`
- **Automatic cleanup**: Temporary config files are deleted after each task completes

This is critical because the script modifies `cfg_raw['TrainingFraction']` for each fold, and parallel workers would otherwise overwrite each other's changes.

### 3. N_WORKERS Configuration
Control parallelism with the `N_WORKERS` variable:
```python
N_WORKERS = 1  # Sequential execution (default, safe for debugging)
N_WORKERS = 4  # Run 4 experiments in parallel
N_WORKERS = 8  # Run 8 experiments in parallel
```

**Important considerations:**
- Each worker uses GPU resources during training
- Adjust `N_WORKERS` based on available GPU memory
- For multi-GPU systems, you may need to add GPU assignment logic
- Start with `N_WORKERS = 1` to verify everything works, then increase

### 4. Task Distribution
All (seed, fold) combinations are prepared upfront:
- For `n_seeds=2` and `k_folds=4`: Creates 8 tasks total
- Tasks are distributed to workers via `pool.map()`
- Results are collected and merged into a single DataFrame

### 5. File-Specific Handling
The following files are now handled per (seed, fold) combination:
- **Config file**: Temporary copy created for each task
- **Training dataset**: Created with unique shuffle number (see below)
- **Model config**: Modified with experiment-specific overrides
- **Evaluation results**: Parsed and returned from each worker

### 6. Shuffle Number Scheme
To prevent collisions between parallel workers, each (seed, fold) combination gets a unique shuffle number:

**Formula**: `shuffle_num = seed_idx * n_folds + fold_idx + 1`

**Example** (4 folds, 2 seeds):
- Seed 0, Fold 0 → Shuffle 1
- Seed 0, Fold 1 → Shuffle 2
- Seed 0, Fold 2 → Shuffle 3
- Seed 0, Fold 3 → Shuffle 4
- Seed 1, Fold 0 → Shuffle 5
- Seed 1, Fold 1 → Shuffle 6
- Seed 1, Fold 2 → Shuffle 7
- Seed 1, Fold 3 → Shuffle 8

This ensures that:
- Training datasets don't overwrite each other: `trainset75shuffle1`, `trainset75shuffle2`, etc.
- Model directories are unique: `dlc-models-pytorch/iteration-0/trainset75shuffle1/`, etc.
- Evaluation results are stored separately

## Usage

### Basic Usage (Sequential)
```python
# Set N_WORKERS = 1 for sequential execution
N_WORKERS = 1

# Run experiments as before
results = run_experiment(
    config_path, 
    k_folds=4, 
    n_seeds=2, 
    experiment_id='my_experiment',
    train_overrides={...},
    landmark_sets={...}
)
```

### Parallel Execution
```python
# Set N_WORKERS to desired parallelism level
N_WORKERS = 4  # Run 4 tasks in parallel

# Run experiments - same API
results = run_experiment(
    config_path, 
    k_folds=4, 
    n_seeds=2, 
    experiment_id='my_experiment',
    train_overrides={...},
    landmark_sets={...}
)
```

## Benefits

1. **Faster Execution**: Run multiple experiments simultaneously
2. **Better Resource Utilization**: Maximize GPU usage across experiments
3. **Same API**: No changes needed to calling code
4. **Safe Parallelization**: Each task has isolated config files
5. **Flexible**: Easy to switch between sequential and parallel modes

## Potential Issues & Solutions

### Issue: Out of GPU Memory
**Solution**: Reduce `N_WORKERS` or adjust batch size

### Issue: File conflicts
**Solution**: The script now creates temporary config files per task, avoiding conflicts

### Issue: Debugging parallel execution
**Solution**: Set `N_WORKERS = 1` to run sequentially and see full error traces

### Issue: Multi-GPU systems
**Solution**: You may need to add GPU assignment logic in `run_single_fold()`:
```python
# Example: Assign GPU based on worker ID
gpu_id = seed_idx % torch.cuda.device_count()
device = torch.device(f"cuda:{gpu_id}")
```

## Performance Expectations

For a typical experiment with:
- 4 folds
- 2 seeds
- 8 total tasks (shuffles 1-8)

**Sequential (N_WORKERS=1)**:
- If each task takes 1 hour → Total: 8 hours

**Parallel (N_WORKERS=4)**:
- If each task takes 1 hour → Total: ~2 hours (4x speedup)

**Parallel (N_WORKERS=8)**:
- If each task takes 1 hour → Total: ~1 hour (8x speedup)
- Requires sufficient GPU memory for 8 concurrent training jobs

**Note**: Each task creates a unique shuffle directory (shuffle1, shuffle2, ..., shuffle8), so there are no file collisions.

## Technical Details

### Multiprocessing Pool
- Uses `multiprocessing.Pool` for process-based parallelism
- Each worker is a separate Python process
- Workers don't share memory (safe for parallel execution)
- Results are serialized and returned to main process

### Config File Management
```python
# Original config (template, never modified)
config_path = '/path/to/config.yaml'

# Per-task temporary config files
config_path = '/path/to/config_seed0_fold0.yaml'  # Seed 0, Fold 0 → Shuffle 1
config_path = '/path/to/config_seed0_fold1.yaml'  # Seed 0, Fold 1 → Shuffle 2
config_path = '/path/to/config_seed1_fold0.yaml'  # Seed 1, Fold 0 → Shuffle 5
# ... etc
```

### Shuffle Number Assignment
Each (seed, fold) combination gets a globally unique shuffle number to prevent directory collisions:
```python
shuffle_num = seed_idx * n_folds + fold_idx + 1

# Example with 4 folds:
# Seed 0: shuffles 1, 2, 3, 4
# Seed 1: shuffles 5, 6, 7, 8
# Seed 2: shuffles 9, 10, 11, 12
```

This creates unique directories:
- `dlc-models-pytorch/iteration-0/trainset75shuffle1/`
- `dlc-models-pytorch/iteration-0/trainset75shuffle2/`
- `dlc-models-pytorch/iteration-0/trainset75shuffle5/`
- etc.

### Result Aggregation
- Each worker returns a dictionary with evaluation results
- Main process collects all results into a list
- List is converted to DataFrame for analysis
- All results include seed, fold, and experiment metadata

