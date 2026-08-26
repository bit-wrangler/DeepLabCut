# Quick Start Guide: Parallel Execution

## TL;DR
The script now supports parallel execution. Set `N_WORKERS` to control how many experiments run simultaneously.

## Quick Setup

### 1. Edit the script
```python
# In run_experiments_with_cv.py, line ~21
N_WORKERS = 4  # Change from 1 to desired number of parallel workers
```

### 2. Run the script
```bash
python run_experiments_with_cv.py
```

That's it! The script will now run 4 experiments in parallel.

## Choosing N_WORKERS

| GPU Memory | Recommended N_WORKERS |
|------------|----------------------|
| 8 GB       | 1-2                  |
| 12 GB      | 2-3                  |
| 16 GB      | 2-4                  |
| 24 GB      | 4-6                  |
| 32 GB      | 6-8                  |
| 48 GB      | 8-12                 |

**Rule of thumb:** Start with `N_WORKERS = 2`, then increase gradually while monitoring GPU memory.

## Monitoring GPU Usage

```bash
# Watch GPU memory usage in real-time
watch -n 1 nvidia-smi

# Or use this one-liner
nvidia-smi -l 1
```

## Expected Speedup

For 4 folds × 2 seeds = 8 total experiments:

| N_WORKERS | Time (if each takes 1h) | Speedup |
|-----------|-------------------------|---------|
| 1         | 8 hours                 | 1x      |
| 2         | 4 hours                 | 2x      |
| 4         | 2 hours                 | 4x      |
| 8         | 1 hour                  | 8x      |

## Troubleshooting

### Problem: Out of GPU memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce `N_WORKERS` by half

### Problem: Script seems stuck
**Cause:** Parallel output is less visible
**Solution:** Check GPU usage with `nvidia-smi` - if GPUs are active, it's working

### Problem: Want to debug an error
**Solution:** Set `N_WORKERS = 1` to see full error traces

## What Changed?

### Config Files
- **Before:** Script modified `config.yaml` directly (caused conflicts)
- **After:** Each worker gets its own temporary config file (no conflicts)

### Execution
- **Before:** Experiments ran one after another
- **After:** Multiple experiments run simultaneously

### Results
- **Same format:** Results are identical, just computed faster

## Advanced: Multi-GPU Systems

If you have multiple GPUs, you can assign workers to specific GPUs:

```python
# Add this at the start of run_single_fold() function
gpu_id = seed_idx % torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
```

## Files Created

### Temporary (auto-deleted)
- `config_seed0_fold0.yaml`
- `config_seed0_fold1.yaml`
- etc.

These are automatically deleted after each experiment completes.

### Permanent (same as before)
- `results_{timestamp}_{uuid}.csv`
- `all_results_{timestamp}.csv`
- Training models in `dlc-models-pytorch/`
- Evaluation results in `evaluation-results-pytorch/`

## Safety Features

✅ Original `config.yaml` is never modified
✅ Temporary config files are auto-deleted
✅ Can switch back to sequential mode anytime (`N_WORKERS = 1`)
✅ Results format unchanged
✅ No changes to training or evaluation logic

## Example Output

```
==================== Running 8 tasks with 4 workers ====================

==================== FOLD 1/4 SEED 1/2 ====================
Train ratio: 0.75
  Shuffle 1: Training with 150 frames, testing with 50 frames.
  Creating training dataset for shuffle 1...

==================== FOLD 2/4 SEED 1/2 ====================
Train ratio: 0.75
  Shuffle 2: Training with 150 frames, testing with 50 frames.
  Creating training dataset for shuffle 2...

[... 4 experiments running in parallel ...]

==================== Cross-Validation Summary ====================
```

## Comparison: Before vs After

### Before (Sequential)
```python
N_WORKERS = 1  # (or not set)

# Runs experiments one by one
# Total time: 8 hours (for 8 experiments)
# GPU utilization: 100% on 1 GPU
```

### After (Parallel)
```python
N_WORKERS = 4

# Runs 4 experiments simultaneously
# Total time: 2 hours (for 8 experiments)
# GPU utilization: 100% on 1 GPU (shared by 4 workers)
```

## Best Practices

1. **Start small:** Begin with `N_WORKERS = 1` to verify everything works
2. **Increase gradually:** Try 2, then 4, then 8
3. **Monitor resources:** Watch GPU memory with `nvidia-smi`
4. **Adjust as needed:** If you see OOM errors, reduce N_WORKERS
5. **For debugging:** Always use `N_WORKERS = 1`

## Need Help?

See detailed documentation:
- `PARALLEL_EXECUTION_README.md` - Full documentation
- `PARALLEL_EXECUTION_SUMMARY.md` - Visual overview
- `CHANGES_DETAILED.md` - Code-level changes

## Rollback

To go back to sequential execution:
```python
N_WORKERS = 1
```

That's it! The script will behave exactly as before.

