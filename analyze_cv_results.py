import pandas as pd
import os

# Configuration
EXPERIMENT_ID = 'tht_gt_1d05_fix' # 'll_0d05'
LANDMARK_SET_NAMES = ['all', 'truncated', 'non_truncated']
MAX_LIST_IDX = 40

# Metric names to analyze
METRIC_NAMES = ['test rmse', 'test rmse_pcutoff', 'test mAP', 'test mAR']

def main():
    # Construct results file path
    results_file = f'cv_results_{EXPERIMENT_ID}.csv'
    
    # Check if file exists
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            f"Please check that the experiment ID '{EXPERIMENT_ID}' is correct."
        )
    
    # Load results
    print(f"Loading results from: {results_file}")
    results_df = pd.read_csv(results_file)[:MAX_LIST_IDX]
    print(f"Loaded {len(results_df)} results\n")
    
    # Build column names based on landmark sets and metrics
    columns_to_analyze = []
    for landmark_set in LANDMARK_SET_NAMES:
        for metric in METRIC_NAMES:
            col_name = f'{landmark_set}__{metric}'
            if col_name in results_df.columns:
                columns_to_analyze.append(col_name)
            else:
                print(f"Warning: Column '{col_name}' not found in results")
    
    if not columns_to_analyze:
        raise ValueError("No valid columns found to analyze")
    
    # Calculate means
    print(f"{'='*60}")
    print(f"Cross-Validation Results Summary for Experiment: {EXPERIMENT_ID}")
    print(f"{'='*60}\n")
    
    means = results_df[columns_to_analyze].mean()
    stds = results_df[columns_to_analyze].std()

    # Print results organized by landmark set
    for landmark_set in LANDMARK_SET_NAMES:
        print(f"\n{landmark_set.upper()} Landmarks:")
        print(f"{'-'*40}")
        for metric in METRIC_NAMES:
            col_name = f'{landmark_set}__{metric}'
            if col_name in means.index:
                print(f"  {metric:25s}: {means[col_name]:.4f} ± {stds[col_name]:.4f}")
    
    # Print overall summary table
    print(f"\n{'='*60}")
    print(f"Complete Results Table:")
    print(f"{'='*60}\n")
    
    # Create a formatted table
    summary_data = []
    for landmark_set in LANDMARK_SET_NAMES:
        row = {'Landmark Set': landmark_set}
        for metric in METRIC_NAMES:
            col_name = f'{landmark_set}__{metric}'
            if col_name in means.index:
                row[metric] = f"{means[col_name]:.4f} ± {stds[col_name]:.4f}"
            else:
                row[metric] = "N/A"
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Print additional statistics
    print(f"\n{'='*60}")
    print(f"Additional Statistics:")
    print(f"{'='*60}")
    print(f"Number of folds: {results_df['fold'].nunique() if 'fold' in results_df.columns else 'N/A'}")
    print(f"Number of seeds: {results_df['seed'].nunique() if 'seed' in results_df.columns else 'N/A'}")
    print(f"Total evaluations: {len(results_df)}")
    
    if 'fold' in results_df.columns and 'seed' in results_df.columns:
        print(f"\nFolds: {sorted(results_df['fold'].unique())}")
        print(f"Seeds: {sorted(results_df['seed'].unique())}")


if __name__ == "__main__":
    main()

