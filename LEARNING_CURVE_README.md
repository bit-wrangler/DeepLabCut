# Learning Curve Analysis Script

## Overview
The `run_learning_curve.py` script performs learning curve analysis on DeepLabCut training data. It evaluates model performance as training data size increases, using a fixed test set and progressively larger training sets.

## Features
- **Fixed Test Set**: Sets aside a configurable fraction of data as a constant test set
- **Progressive Training**: Runs multiple steps with increasing amounts of training data
- **Incremental Saving**: Saves results after each step to prevent data loss
- **Resume Capability**: Can resume from a specific step if interrupted
- **Group-by-Video**: Optional grouping to keep entire videos together in splits
- **Reproducible**: Configurable seed for reproducibility

## Installation
No additional installation required beyond DeepLabCut and its dependencies.

## Usage

### 1. Create a Configuration File
Copy and modify `lc_config_example.yaml`:

```bash
cp lc_config_example.yaml my_lc_config.yaml
```

Edit `my_lc_config.yaml` to set:
- `config_path`: Path to your DeepLabCut project's config.yaml
- `experiment_id`: Unique identifier for this experiment
- `test_fraction`: Fraction of data for test set (e.g., 0.2 = 20%)
- `n_steps`: Number of learning curve steps (e.g., 5)
- `seed`: Random seed for reproducibility
- `group_by_video`: Whether to group frames by video
- `epochs`: Number of training epochs per step
- `train_overrides`: Any training parameter overrides

### 2. Run the Script
```bash
python run_learning_curve.py my_lc_config.yaml
```

### 3. Monitor Progress
The script will:
1. Load and split your data into fixed train/test sets
2. For each learning curve step:
   - Create a training dataset with progressively more data
   - Train a model
   - Evaluate on the fixed test set
   - Save results incrementally

### 4. Resume if Interrupted
If the script is interrupted, you can resume from a specific step:

1. Edit your config file and set `start_from_step: N` (where N is the step to resume from)
2. Run the script again with the same config file

The script will automatically find and continue using the existing results file.

## Output

### Results File
Results are saved to: `lc_results_{experiment_id}_{timestamp}.csv`

Each row contains:
- `step`: Learning curve step index (0-based)
- `train_fraction`: Fraction of total data used for training
- `train_size`: Number of training frames
- `test_size`: Number of test frames
- `experiment_id`: Experiment identifier
- `seed`: Random seed used
- `group_by_video`: Whether grouping was used
- `timestamp`: Timestamp of the run
- `all__test rmse`: Test RMSE for all landmarks
- `all__test rmse_pcutoff`: Test RMSE with p-cutoff
- `all__test mAP`: Test mean Average Precision
- `all__test mAR`: Test mean Average Recall
- `override__*`: All training overrides as separate columns

### Console Output
The script provides detailed progress information including:
- Data split statistics
- Training progress for each step
- Evaluation results
- Final learning curve summary

## Example Configuration

```yaml
config_path: '/workspace/workdir/config.yaml'
experiment_id: 'lc_baseline'
test_fraction: 0.2
n_steps: 5
seed: 42
group_by_video: false
epochs: 50
start_from_step: 0

train_overrides:
  runner.key_metric: 'test.rmse'
  runner.key_metric_asc: false
```

## Learning Curve Steps Example
With `n_steps: 5` and `test_fraction: 0.2`:
- **Step 1**: 20% of training data (16% of total data)
- **Step 2**: 40% of training data (32% of total data)
- **Step 3**: 60% of training data (48% of total data)
- **Step 4**: 80% of training data (64% of total data)
- **Step 5**: 100% of training data (80% of total data)

The test set (20% of total data) remains constant across all steps.

## Differences from Cross-Validation Script
- **No Cross-Validation**: Uses simple train/test split instead of k-fold CV
- **Fixed Test Set**: Test set is constant across all steps
- **Progressive Training**: Training set size increases with each step
- **No Skeletal Parameters**: Simplified configuration without skeletal-related parameters
- **Fixed Landmarks**: Always evaluates on all landmarks (not configurable)

## Troubleshooting

### Script Interrupted
If the script is interrupted, check the console output for the last completed step. Set `start_from_step` to the next step and run again.

### Out of Memory
If you run out of GPU memory:
- Reduce `epochs`
- Reduce batch size in training overrides
- Use a smaller model

### Results File Not Found
If resuming and the results file is not found, ensure:
- The `experiment_id` matches the previous run
- You're running from the same directory
- The results file hasn't been moved or deleted

## Support
For issues or questions, refer to the DeepLabCut documentation or the user story document.

