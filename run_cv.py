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

N_EPOCHS = 200
MODEL = 'resnet_50'
OUTPUT_STRIDE = 16
KEY_METRIC = 'test.rmse' #'test.mAP'
TRAIN_BATCH_SIZE = 48
# CUSTOM_WEIGHTS = '/home/alek/projects/cdl-test1/resnet50_unet_encoder_tuned.pth'

# 1. DEFINE CONFIGURATION AND PARAMETERS
# ---------------------------------------
# Set the full path to the project's config.yaml file
# IMPORTANT: Use an absolute path to avoid issues.
config_path = '/home/alek/projects/cdl-test1/data/cdl-projects/test1-haag-2025-05-21/config_full.yaml'

# Check if the config file exists
if not os.path.exists(config_path):
    raise FileNotFoundError(
        f"The specified config file does not exist: {config_path}\n"
        "Please update the 'config_path' variable with the correct path to your config.yaml file."
    )

# Number of folds for cross-validation
N_FOLDS = 5
N_SEEDS = 8
SHUFFLE_OFFSET = 100

# Note: Shuffle numbers are automatically assigned to prevent collisions
# Formula: shuffle_num = seed_idx * N_FOLDS + fold_idx + 1
# Example (4 folds, 2 seeds): Seed 0 uses shuffles 1-4, Seed 1 uses shuffles 5-8

# 2. MERGE DATA AND PREPARE FOR SPLITTING
# ------------------------------------------
# The `mergeandsplit` function ensures all labeled data is in one file.
# print("Merging annotated datasets...")
# deeplabcut.mergeandsplit(config_path, uniform=True)

# Read the merged data to get the total number of labeled frames.

def run_single_fold(args):
    """Run a single fold+seed combination."""
    (seed_idx, fold_idx, train_indices, test_indices, config_path_template,
     experiment_id, group_by_video, train_overrides, landmark_sets,
     n_folds, n_seeds, num_frames, timestamp) = args

    # Create a unique config file for this fold+seed combination
    config_dir = Path(config_path_template).parent
    config_name = Path(config_path_template).stem
    config_ext = Path(config_path_template).suffix
    config_path = str(config_dir / f"{config_name}_seed{seed_idx}_fold{fold_idx}{config_ext}")

    # Copy the template config to the new location
    shutil.copy(config_path_template, config_path)

    try:
        # Use a unique shuffle number that incorporates both seed and fold
        # This prevents collisions when multiple seeds run in parallel
        # Formula: shuffle_num = seed_idx * n_folds + fold_idx + 1
        # Example: seed=0,fold=0 → shuffle=1; seed=0,fold=1 → shuffle=2
        #          seed=1,fold=0 → shuffle=5; seed=1,fold=1 → shuffle=6
        shuffle_num = seed_idx * n_folds + fold_idx + SHUFFLE_OFFSET
        print(f"\n\n{'='*20} FOLD {fold_idx+1}/{n_folds} SEED {seed_idx+1}/{n_seeds} (Shuffle {shuffle_num}) {'='*20}")
        train_fraction = round(len(train_indices) / num_frames, 2)
        print(f"Train ratio: {train_fraction:.2f}")

        train_fraction_percent = int(train_fraction * 100)

        # Read and update config file with new training fraction
        cfg = deeplabcut.auxiliaryfunctions.read_config(config_path)
        with open(config_path, 'r') as f:
            cfg_raw = yaml.safe_load(f)
        cfg_raw['TrainingFraction'] = [train_fraction]
        with open(config_path, 'w') as f:
            yaml.dump(cfg_raw, f)

        print(f"  Shuffle {shuffle_num}: Training with {len(train_indices)} frames, testing with {len(test_indices)} frames.")

        # b. Create the training dataset for this specific split
        print(f"  Creating training dataset for shuffle {shuffle_num}...")
        deeplabcut.create_training_dataset(
            config_path,
            Shuffles=[shuffle_num],
            trainIndices=[list(train_indices)],
            testIndices=[list(test_indices)],
            userfeedback=False,
            net_type=MODEL,
            augmenter_type='albumentations',
            # weight_init=WeightInitialization(CUSTOM_WEIGHTS) if CUSTOM_WEIGHTS else None
        )

        # override output_striFalsede (model.backbone.output_stride) and key_metric (runner.key_metric)
        # sample path: data/cdl-projects/test1-haag-2025-05-21/dlc-models-pytorch/iteration-0/test1May21-trainset75shuffle1/train/pytorch_config.yaml
        project_path = cfg['project_path']
        trainingset_identifier = f"{cfg['Task']}{cfg['date']}-trainset{train_fraction_percent}shuffle{shuffle_num}"
        model_config_path = Path(project_path) / 'dlc-models-pytorch' / f'iteration-{cfg["iteration"]}' / trainingset_identifier / 'train' / 'pytorch_config.yaml'
        with open(model_config_path, 'r') as f:
            model_cfg = yaml.safe_load(f)
        for key, value in train_overrides.items():
            key_parts = key.split('.')
            current = model_cfg
            for part in key_parts[:-1]:
                current = current[part]
            current[key_parts[-1]] = value
        with open(model_config_path, 'w') as f:
            yaml.dump(model_cfg, f)

        # c. Train the network for this fold
        print(f"  Training network for shuffle {shuffle_num}...")
        # Adjust training parameters (e.g., maxiters) as needed.
        deeplabcut.train_network(
            config_path,
            shuffle=shuffle_num,
            max_snapshots_to_keep=2,
            autotune=False,
            displayiters=100,
            saveiters=5000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

        # d. Evaluate the trained network on the held-out test set
        print(f"  Evaluating network for shuffle {shuffle_num}...")
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

            # e. Parse evaluation results and store them
            print(f"  Parsing evaluation results for shuffle {shuffle_num}...")
            # Construct the path to the evaluation folder

            # Find the results CSV file
            csv_files = list(evaluation_folder.glob('*-results.csv'))
            if not csv_files:
                raise FileNotFoundError(f"No evaluation CSV file found in {evaluation_folder}")

            # Read the CSV and clean column names
            eval_df = pd.read_csv(csv_files[0])
            eval_df.columns = eval_df.columns.str.strip().str.replace('%', '') # Clean '%Training...'

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
                    evaluation_results.update(summary_dict)
            else:
                raise ValueError("Evaluation CSV file is empty.")

        return evaluation_results

    finally:
        # Clean up the temporary config file
        if os.path.exists(config_path):
            os.remove(config_path)




def save_results_incrementally(results_df, results_file):
    """
    Save results to CSV file incrementally.

    Args:
        results_df: DataFrame containing results
        results_file: Path to the results file
    """
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")


def load_existing_results(results_file):
    """
    Load existing results from CSV file if it exists.

    Args:
        results_file: Path to the results file

    Returns:
        pd.DataFrame or None: Existing results or None if file doesn't exist
    """
    if os.path.exists(results_file):
        print(f"Loading existing results from: {results_file}")
        existing_results = pd.read_csv(results_file)
        print(f"  Found {len(existing_results)} existing results")
        return existing_results
    return None


def is_task_completed(existing_results, seed_idx, fold_idx):
    """
    Check if a specific fold+seed combination has already been completed.

    Args:
        existing_results: DataFrame with existing results
        seed_idx: Seed index
        fold_idx: Fold index

    Returns:
        bool: True if task is already completed
    """
    if existing_results is None or len(existing_results) == 0:
        return False

    mask = (existing_results['seed'] == seed_idx) & (existing_results['fold'] == fold_idx)
    return mask.any()


def run_experiment(config_path, n_folds, n_seeds, experiment_id='experiment_1', group_by_video=False, train_overrides={}, landmark_sets={'all': 'all'}):
    """Run cross-validation experiment with sequential processing of folds and seeds."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg = deeplabcut.auxiliaryfunctions.read_config(config_path)
    project_path = cfg['project_path']
    trainingsetfolder = deeplabcut.auxiliaryfunctions.get_training_set_folder(cfg)
    Data = merge_annotateddatasets(
                cfg,
                Path(os.path.join(project_path, trainingsetfolder)),
            )
    groups = np.array(list(map(lambda x: x[1], Data.axes[0])))
    num_frames = len(Data)
    print(f"Total number of labeled frames: {num_frames}")

    # Prepare results file (without timestamp for recovery)
    results_file = f'cv_results_{experiment_id}.csv'

    # Load existing results
    existing_results = load_existing_results(results_file)

    # Initialize results list with existing results
    all_results = []
    if existing_results is not None:
        all_results = existing_results.to_dict('records')
        print(f"\nRetaining {len(existing_results)} existing results")

    # Prepare all fold+seed combinations
    all_tasks = []
    skipped_tasks = 0
    for i in range(n_seeds):
        print(f"\n\n{'='*20} Preparing SEED {i+1}/{n_seeds} {'='*20}")

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
        else:
            cv = KFold(n_splits=n_folds, random_state=42+i, shuffle=True)
            folds = list(cv.split(np.arange(num_frames)))

        for j, (train_indices, test_indices) in enumerate(folds):
            # Check if this task is already completed
            if is_task_completed(existing_results, i, j):
                print(f"  Skipping Seed {i+1} Fold {j+1} (already completed)")
                skipped_tasks += 1
                continue

            task_args = (
                i,  # seed_idx
                j,  # fold_idx
                train_indices,
                test_indices,
                config_path,  # config_path_template
                experiment_id,
                group_by_video,
                train_overrides,
                landmark_sets,
                n_folds,
                n_seeds,
                num_frames,
                timestamp
            )
            all_tasks.append(task_args)

    if skipped_tasks > 0:
        print(f"\n{'='*60}")
        print(f"Skipped {skipped_tasks} already completed tasks")
        print(f"{'='*60}")

    if len(all_tasks) == 0:
        print(f"\n{'='*60}")
        print(f"All tasks already completed!")
        print(f"{'='*60}")
        return pd.DataFrame(all_results)

    print(f"\n\n{'='*20} Running {len(all_tasks)} tasks sequentially {'='*20}")

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

    # 5. AGGREGATE AND REPORT FINAL RESULTS
    # ------------------------------------
    print(f"\n\n{'='*20} Cross-Validation Summary {'='*20}")

    results_df = pd.DataFrame(all_results)
    return results_df


if __name__ == "__main__":
    # skeletal_loss_weight: 0.0
    # skeletal_radius_multiplier_start: 1.15
    # skeletal_radius_multiplier_end: 1.15
    # union_intersect_adjacent_skeletal_mask_alpha_start: 0.0
    # union_intersect_adjacent_skeletal_mask_alpha_end: 0.0
    # union_intersect_adjacent_skeletal_mask_start_epoch: 0
    # union_intersect_adjacent_skeletal_mask_end_epoch: 1
    # use_skeletal_reference: true
    # truncate_targets: true
    # model.heads.bodypart.predictor.locref_std: 7.2801
    # model.heads.bodypart.target_generator.locref_std: 7.2801
    # model.heads.bodypart.target_generator.pos_dist_thresh: 17\
    landmark_sets={
        'all':'all',
        'truncated': ['left_elbow', 'left_wrist', 'right_elbow', 'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'],
        'non_truncated': ['snout', 'base_of_head', 'left_shoulder', 'right_shoulder', 'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5', 'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3', 'tail4', 'tail5'],
        }
    experiment = {
            'train_overrides': {
            'skeletal_loss_weight': 0.0,
            'skeletal_loss_radius_multiplier': 1.0,
            'skeletal_radius_multiplier_start': 1.05,
            'skeletal_radius_multiplier_end': 1.05,
            'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
            'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
            'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
            'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
            'use_skeletal_reference': False,
            'truncate_targets': False,
            'model.heads.bodypart.predictor.locref_std': 7.2801,
            'model.heads.bodypart.target_generator.locref_std': 7.2801,
            'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
            'runner.key_metric': 'test.rmse',
            'runner.key_metric_asc': False,
            'train_settings.batch_size': TRAIN_BATCH_SIZE,
          },
          'experiment_id': 'control',
          'group_by_video': True,
        }

    results: pd.DataFrame = run_experiment(
        config_path,
        N_FOLDS,
        N_SEEDS,
        experiment['experiment_id'],
        group_by_video=experiment['group_by_video'],
        train_overrides=experiment['train_overrides'],
        landmark_sets=landmark_sets
    )

    print("\n✓ Cross-validation completed successfully!")
    print(f"Final results saved to: cv_results_{experiment['experiment_id']}.csv")


#     Starting pose model training...
# --------------------------------------------------
# Epoch 1/200 (lr=0.0005), train loss 0.01404
# Epoch 2/200 (lr=0.0005), train loss 0.00876
# Epoch 3/200 (lr=0.0005), train loss 0.00645
# Epoch 4/200 (lr=0.0005), train loss 0.00517
# Epoch 5/200 (lr=0.0005), train loss 0.00441
# Epoch 6/200 (lr=0.0005), train loss 0.00406
# Epoch 7/200 (lr=0.0005), train loss 0.00363
# Epoch 8/200 (lr=0.0005), train loss 0.00347
# Epoch 9/200 (lr=0.0005), train loss 0.00326
# Training for epoch 10 done, starting evaluation
# Epoch 10/200 (lr=0.0005), train loss 0.00300, valid loss 0.00497
# Model performance:
#   metrics/test.rmse:          18.06
#   metrics/test.rmse_pcutoff:   5.82
#   metrics/test.mAP:           77.99
#   metrics/test.mAR:           80.96
