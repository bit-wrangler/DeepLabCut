import deeplabcut
import deeplabcut.compat
import numpy as np
import pandas as pd
import os
from pathlib import Path
import torch
import yaml
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
import shutil
import datetime
import sys
from ruamel.yaml import YAML

# Configuration constants
SHUFFLE_OFFSET = 100
MODEL = 'resnet_50'

def load_and_split_data(config_path, test_fraction, seed, group_by_video):
    """
    Load data and create a fixed test/train split.

    Args:
        config_path: Path to the DeepLabCut config.yaml file
        test_fraction: Fraction of data to use for test set
        seed: Random seed for reproducibility
        group_by_video: Whether to group frames by video

    Returns:
        tuple: (Data, train_indices, test_indices, groups)
    """
    cfg = deeplabcut.auxiliaryfunctions.read_config(config_path)
    project_path = cfg['project_path']
    trainingsetfolder = deeplabcut.auxiliaryfunctions.get_training_set_folder(cfg)
    Data = merge_annotateddatasets(
        cfg,
        Path(os.path.join(project_path, trainingsetfolder)),
    )

    num_frames = len(Data)
    groups = np.array(list(map(lambda x: x[1], Data.axes[0])))

    print(f"Total number of labeled frames: {num_frames}")
    print(f"Test fraction: {test_fraction}")
    print(f"Group by video: {group_by_video}")

    if group_by_video:
        # Use GroupShuffleSplit to ensure entire videos stay together
        gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        train_indices, test_indices = next(gss.split(np.arange(num_frames), groups=groups))
    else:
        # Simple random split at frame level
        train_indices, test_indices = train_test_split(
            np.arange(num_frames),
            test_size=test_fraction,
            random_state=seed
        )

    print(f"Train set size: {len(train_indices)} frames")
    print(f"Test set size: {len(test_indices)} frames")

    return Data, train_indices, test_indices, groups


def get_training_subset_for_step(train_indices, step_idx, n_steps, seed, groups, group_by_video):
    """
    Get a subset of training data for a specific learning curve step.

    Args:
        train_indices: Full set of training indices
        step_idx: Current step index (0-based)
        n_steps: Total number of steps
        seed: Random seed
        groups: Group labels for each frame
        group_by_video: Whether to respect video grouping

    Returns:
        np.array: Subset of training indices for this step
    """
    # Calculate fraction of training data to use for this step
    # Step 0: 1/n_steps, Step 1: 2/n_steps, ..., Step n_steps-1: n_steps/n_steps
    step_fraction = (step_idx + 1) / n_steps
    target_size = int(len(train_indices) * step_fraction)

    if step_idx == n_steps - 1:
        # Last step uses all training data
        return train_indices

    if group_by_video:
        # Sample groups, not individual frames
        train_groups = groups[train_indices]
        unique_groups = np.unique(train_groups)

        # Shuffle groups with seed
        rng = np.random.RandomState(seed + step_idx)
        shuffled_groups = rng.permutation(unique_groups)

        # Select groups until we reach target size
        selected_groups = []
        current_size = 0
        for group in shuffled_groups:
            group_mask = train_groups == group
            group_size = np.sum(group_mask)
            if current_size + group_size <= target_size or len(selected_groups) == 0:
                selected_groups.append(group)
                current_size += group_size
            if current_size >= target_size:
                break

        # Get indices for selected groups
        mask = np.isin(train_groups, selected_groups)
        subset_indices = train_indices[mask]
    else:
        # Simple random sampling
        rng = np.random.RandomState(seed + step_idx)
        subset_indices = rng.choice(train_indices, size=target_size, replace=False)

    return subset_indices


def run_single_step(args):
    """
    Run a single learning curve step: create dataset, train, and evaluate.

    Args:
        args: Tuple containing all necessary parameters

    Returns:
        dict: Evaluation results for this step
    """
    (step_idx, train_subset_indices, test_indices, config_path_template,
     experiment_id, group_by_video, train_overrides, n_steps, num_frames,
     timestamp, epochs, seed, total_split_size) = args

    # Create a unique config file for this step
    config_dir = Path(config_path_template).parent
    config_name = Path(config_path_template).stem
    config_ext = Path(config_path_template).suffix
    config_path = str(config_dir / f"{config_name}_step{step_idx}{config_ext}")

    # Copy the template config to the new location
    shutil.copy(config_path_template, config_path)

    try:
        shuffle_num = step_idx + SHUFFLE_OFFSET
        print(f"\n\n{'='*20} STEP {step_idx+1}/{n_steps} (Shuffle {shuffle_num}) {'='*20}")

        # Calculate fraction based on the actual split size (what DLC will see)
        # DLC rounds to 2 decimal places, so we must do the same
        train_fraction = round(len(train_subset_indices) / total_split_size, 2)
        train_fraction_of_total = round(len(train_subset_indices) / num_frames, 4)
        print(f"Train fraction (of split): {train_fraction:.4f}")
        print(f"Train fraction (of total dataset): {train_fraction_of_total:.4f}")
        print(f"Train size: {len(train_subset_indices)} frames")
        print(f"Test size: {len(test_indices)} frames")
        print(f"Total split size: {total_split_size} frames (train+test)")
        print(f"Total dataset size: {num_frames} frames")

        # DLC uses int(trainFraction * 100) for folder names
        train_fraction_percent = int(train_fraction * 100)

        # Read and update config file with new training fraction
        cfg = deeplabcut.auxiliaryfunctions.read_config(config_path)
        with open(config_path, 'r') as f:
            cfg_raw = yaml.safe_load(f)
        cfg_raw['TrainingFraction'] = [train_fraction_percent / 100]
        with open(config_path, 'w') as f:
            yaml.dump(cfg_raw, f)

        # Create the training dataset for this specific split
        print(f"  Creating training dataset for shuffle {shuffle_num}...")
        deeplabcut.create_training_dataset(
            config_path,
            Shuffles=[shuffle_num],
            trainIndices=[list(train_subset_indices)],
            testIndices=[list(test_indices)],
            userfeedback=False,
            net_type=MODEL,
            augmenter_type='albumentations',
        )

        # Override training parameters in pytorch_config.yaml
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

        # Train the network
        print(f"  Training network for shuffle {shuffle_num}...")
        deeplabcut.train_network(
            config_path,
            shuffle=shuffle_num,
            max_snapshots_to_keep=2,
            autotune=False,
            displayiters=100,
            saveiters=5000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            epochs=epochs
        )

        # Evaluate the trained network on the held-out test set
        print(f"  Evaluating network for shuffle {shuffle_num}...")

        iteration = cfg['iteration']
        engine_name = deeplabcut.compat.get_project_engine(cfg).aliases[0]
        trainingset_identifier = f"{cfg['Task']}{cfg['date']}-trainset{train_fraction_percent}shuffle{shuffle_num}"
        evaluation_folder = Path(project_path) / f"evaluation-results-{engine_name}" / f"iteration-{iteration}" / trainingset_identifier

        print(f"The evaluation folder is: {evaluation_folder}")

        # Clear evaluation folder if it exists
        if evaluation_folder.exists():
            for child in evaluation_folder.glob('*'):
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)

        # Evaluate with all landmarks
        deeplabcut.evaluate_network(config_path, Shuffles=[shuffle_num], plotting=False, comparisonbodyparts='all')

        # Parse evaluation results
        print(f"  Parsing evaluation results for shuffle {shuffle_num}...")

        # Find the results CSV file
        csv_files = list(evaluation_folder.glob('*-results.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No evaluation CSV file found in {evaluation_folder}")

        # Read the CSV and clean column names
        eval_df = pd.read_csv(csv_files[0])
        eval_df.columns = eval_df.columns.str.strip().str.replace('%', '')

        if not eval_df.empty:
            # Convert the first row to a dictionary
            summary_dict = eval_df.iloc[0].to_dict()

            # Add metadata
            summary_dict['step'] = step_idx
            summary_dict['train_fraction'] = train_fraction  # Fraction of split (matches DLC)
            summary_dict['train_fraction_of_total'] = train_fraction_of_total  # Fraction of full dataset
            summary_dict['train_size'] = len(train_subset_indices)
            summary_dict['test_size'] = len(test_indices)
            summary_dict['total_split_size'] = total_split_size
            summary_dict['total_dataset_size'] = num_frames
            summary_dict['experiment_id'] = experiment_id
            summary_dict['seed'] = seed
            summary_dict['group_by_video'] = group_by_video
            summary_dict['timestamp'] = timestamp

            # Add train overrides
            for key, value in train_overrides.items():
                summary_dict[f'override__{key}'] = value

            # Rename metric columns to include 'all__' prefix
            metric_columns = ['test rmse', 'test rmse_pcutoff', 'test mAP', 'test mAR']
            for col in metric_columns:
                if col in summary_dict:
                    summary_dict[f'all__{col}'] = summary_dict.pop(col)

            return summary_dict
        else:
            raise ValueError("Evaluation CSV file is empty.")

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


def load_existing_results(results_file, seed=None):
    """
    Load existing results from CSV file if it exists.

    Args:
        results_file: Path to the results file
        seed: Optional seed to filter results for

    Returns:
        tuple: (all_results_df, seed_results_df) where:
            - all_results_df: All results from the file (or None)
            - seed_results_df: Results filtered by seed (or None)
    """
    if os.path.exists(results_file):
        print(f"Loading existing results from: {results_file}")
        all_results = pd.read_csv(results_file)

        if seed is not None and 'seed' in all_results.columns:
            seed_results = all_results[all_results['seed'] == seed]
            if len(seed_results) > 0:
                print(f"  Found {len(seed_results)} existing results for seed {seed}")
                return all_results, seed_results
            else:
                print(f"  No existing results found for seed {seed}")
                return all_results, None

        return all_results, None
    return None, None


def run_learning_curve(config_path, lc_cfg):
    """
    Run learning curve analysis with progressively increasing training data.

    Args:
        config_path: Path to DeepLabCut config.yaml
        lc_cfg: Learning curve configuration dictionary
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Extract configuration parameters
    test_fraction = lc_cfg.get('test_fraction', 0.2)
    n_steps = lc_cfg.get('n_steps', 5)
    seed = lc_cfg.get('seed', 42)
    group_by_video = lc_cfg.get('group_by_video', False)
    experiment_id = lc_cfg.get('experiment_id', 'lc_experiment')
    epochs = lc_cfg.get('epochs', 50)
    start_from_step = lc_cfg.get('start_from_step', 0)
    force_rerun = lc_cfg.get('force_rerun', False)
    train_overrides = lc_cfg.get('train_overrides', {})

    print(f"\n{'='*60}")
    print(f"Learning Curve Analysis")
    print(f"{'='*60}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Test fraction: {test_fraction}")
    print(f"Number of steps: {n_steps}")
    print(f"Seed: {seed}")
    print(f"Group by video: {group_by_video}")
    print(f"Epochs: {epochs}")
    print(f"Start from step: {start_from_step}")
    print(f"{'='*60}\n")

    # Load and split data
    print("Loading and splitting data...")
    Data, train_indices, test_indices, groups = load_and_split_data(
        config_path, test_fraction, seed, group_by_video
    )

    num_frames = len(Data)

    # Prepare results file (use consistent name without timestamp for aggregation)
    results_file = f'lc_results_{experiment_id}.csv'

    # Load existing results (always check, not just when resuming)
    all_existing_results, seed_existing_results = load_existing_results(results_file, seed)

    # Determine starting step based on existing results for this seed
    actual_start_step = start_from_step
    if seed_existing_results is not None and len(seed_existing_results) > 0:
        completed_steps = seed_existing_results['step'].values
        max_completed_step = completed_steps.max()

        if force_rerun:
            # Force re-run: remove all existing results for this seed
            print(f"\nForce re-run enabled: Removing all existing results for seed {seed}")
            if all_existing_results is not None:
                all_existing_results = all_existing_results[all_existing_results['seed'] != seed]
            actual_start_step = 0
        elif start_from_step == 0:
            # Auto-resume from where this seed left off
            actual_start_step = max_completed_step + 1
            if actual_start_step < n_steps:
                print(f"\nFound existing results for seed {seed} up to step {max_completed_step}")
                print(f"Auto-resuming from step {actual_start_step}")
            elif actual_start_step >= n_steps:
                # All steps already complete for this seed
                print(f"\n{'='*60}")
                print(f"WARNING: Seed {seed} already has complete results (all {n_steps} steps)")
                print(f"{'='*60}")
                print(f"To re-run this seed, set 'force_rerun: true' in your config file")
                print(f"Skipping execution to avoid duplicate results.")
                print(f"{'='*60}\n")
                # Return existing results without running anything
                return pd.DataFrame(all_existing_results.to_dict('records'))
        else:
            # User specified start_from_step, validate it
            if start_from_step <= max_completed_step:
                print(f"\nWarning: Seed {seed} already has results up to step {max_completed_step}")
                print(f"Starting from step {start_from_step} will overwrite existing results for this seed")
                # Remove existing results for this seed from this step onwards
                if all_existing_results is not None:
                    all_existing_results = all_existing_results[
                        ~((all_existing_results['seed'] == seed) &
                          (all_existing_results['step'] >= start_from_step))
                    ]

    # Initialize results list with existing results from OTHER seeds
    all_results = []
    if all_existing_results is not None:
        # Keep results from other seeds, and results from this seed before actual_start_step
        other_results = all_existing_results[
            (all_existing_results['seed'] != seed) |
            (all_existing_results['step'] < actual_start_step)
        ]
        all_results = other_results.to_dict('records')
        print(f"\nRetaining {len(other_results)} existing results from other seeds/steps")

    # Run learning curve steps (use actual_start_step which may be auto-adjusted)
    for step_idx in range(actual_start_step, n_steps):
        print(f"\n{'='*60}")
        print(f"Preparing Step {step_idx + 1}/{n_steps}")
        print(f"{'='*60}")

        # Get training subset for this step
        train_subset_indices = get_training_subset_for_step(
            train_indices, step_idx, n_steps, seed, groups, group_by_video
        )

        # Calculate total split size for this step (this is what DLC will see)
        total_split_size = len(train_subset_indices) + len(test_indices)

        # Prepare arguments for this step
        task_args = (
            step_idx,
            train_subset_indices,
            test_indices,
            config_path,
            experiment_id,
            group_by_video,
            train_overrides,
            n_steps,
            num_frames,
            timestamp,
            epochs,
            seed,
            total_split_size
        )

        # Run the step
        try:
            step_results = run_single_step(task_args)
            all_results.append(step_results)

            # Save results incrementally
            results_df = pd.DataFrame(all_results)
            save_results_incrementally(results_df, results_file)

            print(f"\n✓ Step {step_idx + 1}/{n_steps} completed successfully")

        except Exception as e:
            print(f"\n✗ Step {step_idx + 1}/{n_steps} failed with error: {e}")
            print(f"Results up to step {step_idx} have been saved to: {results_file}")
            print(f"To resume, set 'start_from_step: {step_idx + 1}' in your config file")
            raise

    # Final summary
    print(f"\n\n{'='*60}")
    print(f"Learning Curve Analysis Complete")
    print(f"{'='*60}")
    print(f"Total steps completed for seed {seed}: {n_steps}")
    print(f"Results saved to: {results_file}")

    # Display summary statistics for current seed
    results_df = pd.DataFrame(all_results)
    current_seed_results = results_df[results_df['seed'] == seed].sort_values('step')

    print(f"\nLearning Curve Summary (Seed {seed}):")
    summary_cols = ['step', 'train_fraction', 'train_fraction_of_total', 'train_size', 'test_size', 'all__test rmse']
    print(current_seed_results[summary_cols].to_string(index=False))

    # Show info about other seeds if present
    other_seeds = results_df[results_df['seed'] != seed]['seed'].unique()
    if len(other_seeds) > 0:
        print(f"\nNote: Results file also contains data for seeds: {sorted(other_seeds)}")
        print(f"Total results in file: {len(results_df)} rows across {len(results_df['seed'].unique())} seeds")

    return results_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # print("Usage: python run_learning_curve.py <lc_config.yaml>")
        # sys.exit(1)
        lc_config_filename = 'lc_config_example.yaml'

    # Load learning curve configuration
    else:
        lc_config_filename = sys.argv[1]

    if not os.path.exists(lc_config_filename):
        raise FileNotFoundError(f"Configuration file not found: {lc_config_filename}")

    with open(lc_config_filename, "r") as f:
        lc_cfg = YAML(typ="safe", pure=True).load(f)

    # Get DeepLabCut config path from learning curve config
    config_path = lc_cfg.get('config_path', '/workspace/workdir/config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"The specified DeepLabCut config file does not exist: {config_path}\n"
            "Please update the 'config_path' in your learning curve config file."
        )

    # Run learning curve analysis
    results = run_learning_curve(config_path, lc_cfg)

    print("\n✓ Learning curve analysis completed successfully!")


