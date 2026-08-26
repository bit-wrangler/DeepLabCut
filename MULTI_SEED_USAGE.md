# Multi-Seed Learning Curve Analysis

## Overview

The learning curve script now supports running multiple seeds with automatic result aggregation. All results for a given `experiment_id` are stored in a single CSV file, making it easy to analyze variance across different random seeds.

## Key Features

### 1. Consistent Results File
- Results are saved to: `lc_results_{experiment_id}.csv` (no timestamp)
- All seeds for the same experiment share this file
- Each row is uniquely identified by `(experiment_id, seed, step)`

### 2. Automatic Resume
- When you run with `start_from_step: 0` (default), the script automatically detects existing results for the current seed
- If results exist, it resumes from where that seed left off
- No manual intervention needed!

### 3. Seed Isolation
- Results from different seeds don't interfere with each other
- You can run seeds in any order
- Interrupted runs can be resumed without affecting other seeds

### 4. Manual Resume Control
- Set `start_from_step: N` to manually control where to start
- If results already exist for that seed/step, they will be overwritten
- Useful for re-running specific steps

### 5. Force Re-run
- Set `force_rerun: true` to completely overwrite existing results for a seed
- Useful when you want to re-run a seed from scratch
- Only affects the current seed, not others

## Usage Examples

### Example 1: Run Multiple Seeds Sequentially

**Step 1:** Run with seed 42
```yaml
# lc_config.yaml
experiment_id: 'my_experiment'
seed: 42
n_steps: 5
start_from_step: 0  # or omit, defaults to 0
```
```bash
python run_learning_curve.py lc_config.yaml
```
**Result:** Creates `lc_results_my_experiment.csv` with 5 rows (seed 42, steps 0-4)

---

**Step 2:** Run with seed 123 (just change the seed in config)
```yaml
# lc_config.yaml
experiment_id: 'my_experiment'  # Same experiment_id!
seed: 123  # Different seed
n_steps: 5
start_from_step: 0
```
```bash
python run_learning_curve.py lc_config.yaml
```
**Result:** Updates `lc_results_my_experiment.csv` with 10 rows total:
- 5 rows for seed 42 (steps 0-4)
- 5 rows for seed 123 (steps 0-4)

---

**Step 3:** Run with seed 999
```yaml
experiment_id: 'my_experiment'
seed: 999
n_steps: 5
```
```bash
python run_learning_curve.py lc_config.yaml
```
**Result:** Updates `lc_results_my_experiment.csv` with 15 rows total (3 seeds × 5 steps)

### Example 2: Resume After Interruption

**Scenario:** Seed 42 was interrupted after step 2

**What's in the file:**
```
experiment_id,seed,step,...
my_experiment,42,0,...
my_experiment,42,1,...
my_experiment,42,2,...
```

**Run again with same config:**
```yaml
experiment_id: 'my_experiment'
seed: 42
n_steps: 5
start_from_step: 0  # Auto-resume!
```
```bash
python run_learning_curve.py lc_config.yaml
```

**Output:**
```
Found existing results for seed 42 up to step 2
Auto-resuming from step 3
```

**Result:** Completes steps 3-4, file now has all 5 steps for seed 42

### Example 3: Re-run a Complete Seed

**Scenario:** Seed 42 is complete, but you want to re-run it

**First attempt (without force_rerun):**
```yaml
experiment_id: 'my_experiment'
seed: 42
n_steps: 5
start_from_step: 0
force_rerun: false  # or omit
```
```bash
python run_learning_curve.py lc_config.yaml
```

**Output:**
```
WARNING: Seed 42 already has complete results (all 5 steps)
To re-run this seed, set 'force_rerun: true' in your config file
Skipping execution to avoid duplicate results.
```

**Result:** Nothing runs, existing results preserved

---

**Second attempt (with force_rerun):**
```yaml
experiment_id: 'my_experiment'
seed: 42
n_steps: 5
force_rerun: true  # Force overwrite
```
```bash
python run_learning_curve.py lc_config.yaml
```

**Output:**
```
Force re-run enabled: Removing all existing results for seed 42
```

**Result:** All 5 steps are re-run and overwritten for seed 42

### Example 4: Re-run Specific Steps

**Force re-run from step 2 onwards:**
```yaml
experiment_id: 'my_experiment'
seed: 42
n_steps: 5
start_from_step: 2  # Manual override
```
```bash
python run_learning_curve.py lc_config.yaml
```

**Output:**
```
Warning: Seed 42 already has results up to step 4
Starting from step 2 will overwrite existing results for this seed
```

**Result:** Steps 2-4 are re-run and overwritten for seed 42

## Analyzing Multi-Seed Results

### Load and Filter by Seed
```python
import pandas as pd

df = pd.read_csv('lc_results_my_experiment.csv')

# Get results for specific seed
seed_42 = df[df['seed'] == 42]

# Get results for specific step across all seeds
step_0 = df[df['step'] == 0]

# Calculate mean and std across seeds for each step
summary = df.groupby('step')['all__test rmse'].agg(['mean', 'std', 'min', 'max'])
print(summary)
```

### Plot with Error Bars
```python
import matplotlib.pyplot as plt

# Group by train_size and calculate statistics
stats = df.groupby('train_size')['all__test rmse'].agg(['mean', 'std'])

plt.errorbar(stats.index, stats['mean'], yerr=stats['std'], 
             marker='o', capsize=5, label='Mean ± Std')
plt.xscale('log')
plt.xlabel('Training Size')
plt.ylabel('Test RMSE')
plt.legend()
plt.show()
```

## Best Practices

1. **Use descriptive experiment_id**: Makes it easy to identify results files
   - Good: `experiment_id: 'resnet50_baseline'`
   - Bad: `experiment_id: 'test1'`

2. **Run seeds sequentially**: Easier to track progress
   ```bash
   # Edit config to set seed: 42
   python run_learning_curve.py lc_config.yaml
   # Edit config to set seed: 123
   python run_learning_curve.py lc_config.yaml
   # etc.
   ```

3. **Use consistent n_steps**: All seeds should use the same number of steps for fair comparison

4. **Backup results file**: Before re-running with `start_from_step`, consider backing up:
   ```bash
   cp lc_results_my_experiment.csv lc_results_my_experiment_backup.csv
   ```

5. **Check existing seeds**: Before running a new seed, check what's already in the file:
   ```bash
   python -c "import pandas as pd; df=pd.read_csv('lc_results_my_experiment.csv'); print(df.groupby('seed')['step'].count())"
   ```

## Troubleshooting

**Q: I want to start fresh with a new experiment**
- Change the `experiment_id` in your config, or
- Delete/rename the existing results file

**Q: I accidentally ran the same seed twice**
- The second run will have overwritten the first
- Restore from backup if you have one

**Q: Can I run multiple seeds in parallel?**
- Not recommended! File writes may conflict
- Run seeds sequentially instead

**Q: How do I know which seeds are in my results file?**
```bash
python -c "import pandas as pd; print(pd.read_csv('lc_results_my_experiment.csv')['seed'].unique())"
```

