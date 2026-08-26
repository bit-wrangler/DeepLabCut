import pandas as pd

RESULTS_PATH = 'all_results_20250930-154315.csv'

PREFIXES = [
    'all',
    'truncated',
    'non_truncated',
]

cols = [
    'test rmse',
    # 'test rmse_pcutoff',
    # 'test mAP',
    # 'test mAR',
]

results_df = pd.read_csv(RESULTS_PATH)

# group by experiment_id and average
grouped = results_df.groupby('experiment').mean(numeric_only=True)

select_cols =  [f'{prefix}__{col}' for prefix in PREFIXES for col in cols]

print(grouped[select_cols])