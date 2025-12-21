#!/usr/bin/env python3
"""
Predict body length relative error using Random Forest Regressor.

This script uses all landmark confidences and predicted body length as features
to predict the relative error in body length predictions. It trains a Random Forest
regressor from scikit-learn and evaluates its performance.

Usage:
    1. Update the configuration constants at the top of this file
    2. Run: python predict_body_length_error_rf.py

The script will:
    - Load frame-level aggregate CSV file (from User Story 02)
    - Calculate body length (SVL) from snout and tail1 positions
    - Extract all confidence scores as features
    - Train/test split the data
    - Train a Random Forest regressor to predict relative error
    - Evaluate model performance (R², MAE, RMSE)
    - Generate feature importance plot
    - Generate prediction vs actual scatter plot
    - Optionally save the trained model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ============================================================================
# Configuration
# ============================================================================

EXPERIMENT_ID = 'control'          # Experiment to analyze
LANDMARK_SET_NAME = 'all'           # Landmark set ('all', 'truncated', etc.)

# Body length calculation settings
BODYPART_1 = 'snout'                # First bodypart for body length
BODYPART_2 = 'tail1'                # Second bodypart for body length

# Filtering thresholds
MIN_BODY_LENGTH_PIXELS = 10.0       # Minimum valid body length (avoid division by zero)
MAX_RELATIVE_ERROR = 1.0            # Cap relative error at 100% (1.0)

# Random Forest settings
TEST_SIZE = 0.2                     # Fraction of data for testing
RANDOM_STATE = 42                   # Random seed for reproducibility
N_ESTIMATORS = 100                  # Number of trees in the forest
MAX_DEPTH = 20                      # Maximum depth of trees (None = unlimited)
MIN_SAMPLES_SPLIT = 5               # Minimum samples required to split a node
MIN_SAMPLES_LEAF = 2                # Minimum samples required at a leaf node
N_JOBS = -1                         # Number of parallel jobs (-1 = use all cores)

# Visualization settings
PLOT_DPI = 300                      # Plot resolution
PLOT_FIGSIZE = (10, 8)              # Figure size in inches

# Output settings
SAVE_MODEL = True                   # Whether to save the trained model
OUTPUT_DIR = '.'                    # Directory for output files

# ============================================================================
# Core Functions
# ============================================================================

def load_and_validate_data(input_file, bodypart_1, bodypart_2):
    """
    Load frame-level aggregate CSV and validate required columns exist.

    Args:
        input_file: Path to the frame-level aggregate CSV file
        bodypart_1: Name of first bodypart (e.g., 'snout')
        bodypart_2: Name of second bodypart (e.g., 'tail1')

    Returns:
        pd.DataFrame: Loaded and validated DataFrame

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns are missing
    """
    # Check if file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Frame-level results file not found: {input_file}\n"
            f"Please check that EXPERIMENT_ID='{EXPERIMENT_ID}' and "
            f"LANDMARK_SET_NAME='{LANDMARK_SET_NAME}' are correct."
        )

    # Load CSV file
    df = pd.read_csv(input_file)

    # Build list of required columns for body length calculation
    required_cols = [
        f'gt_{bodypart_1}_x', f'gt_{bodypart_1}_y',
        f'gt_{bodypart_2}_x', f'gt_{bodypart_2}_y',
        f'pred_{bodypart_1}_x', f'pred_{bodypart_1}_y',
        f'pred_{bodypart_2}_x', f'pred_{bodypart_2}_y',
    ]

    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        available_cols = [col for col in df.columns if col.startswith(('gt_', 'pred_', 'conf_'))]
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available bodypart columns: {available_cols}\n"
            f"Please check that BODYPART_1='{bodypart_1}' and "
            f"BODYPART_2='{bodypart_2}' are correct."
        )

    return df


def calculate_body_length(df, bodypart_1, bodypart_2):
    """
    Calculate true and predicted body length using Euclidean distance.

    Args:
        df: DataFrame with ground truth and prediction columns
        bodypart_1: Name of first bodypart
        bodypart_2: Name of second bodypart

    Returns:
        pd.DataFrame: DataFrame with added columns:
            - true_body_length: Ground truth body length
            - pred_body_length: Predicted body length
    """
    # Calculate true body length (ground truth)
    df['true_body_length'] = np.sqrt(
        (df[f'gt_{bodypart_1}_x'] - df[f'gt_{bodypart_2}_x'])**2 +
        (df[f'gt_{bodypart_1}_y'] - df[f'gt_{bodypart_2}_y'])**2
    )

    # Calculate predicted body length
    df['pred_body_length'] = np.sqrt(
        (df[f'pred_{bodypart_1}_x'] - df[f'pred_{bodypart_2}_x'])**2 +
        (df[f'pred_{bodypart_1}_y'] - df[f'pred_{bodypart_2}_y'])**2
    )

    return df


def calculate_errors(df):
    """
    Calculate absolute and relative errors.

    Args:
        df: DataFrame with true_body_length and pred_body_length columns

    Returns:
        pd.DataFrame: DataFrame with added columns:
            - absolute_error: Absolute error in pixels
            - relative_error: Relative error (capped at MAX_RELATIVE_ERROR)
    """
    # Calculate absolute error
    df['absolute_error'] = np.abs(df['pred_body_length'] - df['true_body_length'])

    # Calculate relative error
    df['relative_error'] = df['absolute_error'] / df['true_body_length']

    # Cap relative error at MAX_RELATIVE_ERROR
    df['relative_error'] = df['relative_error'].clip(upper=MAX_RELATIVE_ERROR)

    return df


def extract_confidence_features(df):
    """
    Extract all confidence scores as features for the model.

    Args:
        df: DataFrame with confidence columns (conf_{bodypart})

    Returns:
        tuple: (feature_names, feature_matrix)
            - feature_names: List of feature column names
            - feature_matrix: numpy array of shape (n_samples, n_features)
    """
    # Find all confidence columns
    conf_cols = [col for col in df.columns if col.startswith('conf_')]

    if len(conf_cols) == 0:
        raise ValueError("No confidence columns found in the DataFrame!")

    # Add predicted body length as a feature
    feature_cols = conf_cols + ['pred_body_length']

    # Extract features
    feature_names = feature_cols
    feature_matrix = df[feature_cols].values

    return feature_names, feature_matrix


def filter_valid_data(df, min_body_length):
    """
    Filter out invalid data points.

    Args:
        df: DataFrame with computed metrics
        min_body_length: Minimum valid body length threshold

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    initial_count = len(df)

    # Filter by minimum body length
    df_filtered = df[df['true_body_length'] >= min_body_length].copy()
    removed_body_length = initial_count - len(df_filtered)

    # Remove rows with NaN values in key columns
    key_cols = ['relative_error', 'pred_body_length'] + [col for col in df.columns if col.startswith('conf_')]
    df_filtered = df_filtered.dropna(subset=key_cols)
    removed_nan = len(df) - removed_body_length - len(df_filtered)

    # Print filtering statistics
    print(f"\nFiltering statistics:")
    print(f"  Initial frames: {initial_count}")
    print(f"  Removed (body length < {min_body_length} px): {removed_body_length}")
    print(f"  Removed (NaN values): {removed_nan}")
    print(f"  Valid frames remaining: {len(df_filtered)}")
    print(f"  Percentage retained: {100 * len(df_filtered) / initial_count:.1f}%")

    return df_filtered


def train_random_forest(X_train, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf, n_jobs, random_state):
    """
    Train a Random Forest regressor.

    Args:
        X_train: Training features
        y_train: Training targets (relative errors)
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples at leaf
        n_jobs: Number of parallel jobs
        random_state: Random seed

    Returns:
        RandomForestRegressor: Trained model
    """
    print(f"\nTraining Random Forest with {n_estimators} trees...")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0
    )

    model.fit(X_train, y_train)

    print("Training complete!")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets

    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'y_pred': y_pred
    }

    return metrics


def plot_feature_importance(model, feature_names, experiment_id, landmark_set_name, output_dir, plot_dpi, plot_figsize):
    """
    Plot feature importance from the trained Random Forest.

    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        experiment_id: Experiment identifier
        landmark_set_name: Landmark set name
        output_dir: Directory to save plot
        plot_dpi: Plot resolution
        plot_figsize: Figure size
    """
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create figure
    fig, ax = plt.subplots(figsize=plot_figsize)

    # Plot top 20 features (or all if less than 20)
    n_features_to_plot = min(20, len(feature_names))
    y_pos = np.arange(n_features_to_plot)

    ax.barh(y_pos, importances[indices[:n_features_to_plot]], align='center', color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices[:n_features_to_plot]])
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Random Forest Feature Importance: {experiment_id} ({landmark_set_name})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Save plot
    output_filename = f'rf_feature_importance_{experiment_id}_{landmark_set_name}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close()

    print(f"Feature importance plot saved to: {output_filename}")


def plot_predictions(y_test, y_pred, experiment_id, landmark_set_name, output_dir, plot_dpi, plot_figsize, metrics):
    """
    Plot predicted vs actual relative errors.

    Args:
        y_test: Actual relative errors
        y_pred: Predicted relative errors
        experiment_id: Experiment identifier
        landmark_set_name: Landmark set name
        output_dir: Directory to save plot
        plot_dpi: Plot resolution
        plot_figsize: Figure size
        metrics: Dictionary with evaluation metrics
    """
    # Create figure
    fig, ax = plt.subplots(figsize=plot_figsize)

    # Scatter plot
    ax.scatter(y_test * 100, y_pred * 100, alpha=0.5, s=20, color='steelblue', edgecolor='black', linewidth=0.5)

    # Perfect prediction line
    max_val = max(y_test.max(), y_pred.max()) * 100
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Add metrics text
    textstr = f"R² = {metrics['r2']:.3f}\nMAE = {metrics['mae']*100:.2f}%\nRMSE = {metrics['rmse']*100:.2f}%"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # Labels and title
    ax.set_xlabel('Actual Relative Error (%)', fontsize=12)
    ax.set_ylabel('Predicted Relative Error (%)', fontsize=12)
    ax.set_title(f'Random Forest Predictions: {experiment_id} ({landmark_set_name})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save plot
    output_filename = f'rf_predictions_{experiment_id}_{landmark_set_name}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close()

    print(f"Predictions plot saved to: {output_filename}")


def save_model_to_disk(model, experiment_id, landmark_set_name, output_dir):
    """
    Save the trained model to disk using joblib.

    Args:
        model: Trained Random Forest model
        experiment_id: Experiment identifier
        landmark_set_name: Landmark set name
        output_dir: Directory to save model
    """
    output_filename = f'rf_model_{experiment_id}_{landmark_set_name}.joblib'
    output_path = os.path.join(output_dir, output_filename)

    joblib.dump(model, output_path)

    print(f"Model saved to: {output_filename}")


def print_summary_statistics(df_filtered, feature_names, metrics, y_test):
    """
    Print informative summary statistics to console.

    Args:
        df_filtered: Filtered DataFrame
        feature_names: List of feature names
        metrics: Dictionary with evaluation metrics
        y_test: Test targets
    """
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")

    print(f"\nDataset:")
    print(f"  Total samples: {len(df_filtered)}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Features: {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}")

    print(f"\nRelative Error Statistics (all data):")
    print(f"  Mean: {df_filtered['relative_error'].mean()*100:.2f}%")
    print(f"  Std:  {df_filtered['relative_error'].std()*100:.2f}%")
    print(f"  Min:  {df_filtered['relative_error'].min()*100:.2f}%")
    print(f"  Max:  {df_filtered['relative_error'].max()*100:.2f}%")

    print(f"\nModel Performance (test set):")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  Mean Absolute Error (MAE): {metrics['mae']*100:.2f}%")
    print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']*100:.2f}%")
    print(f"  Test set size: {len(y_test)} samples")

    # Baseline comparison (always predict mean)
    baseline_mae = np.mean(np.abs(y_test - y_test.mean()))
    improvement = (baseline_mae - metrics['mae']) / baseline_mae * 100
    print(f"\nBaseline Comparison:")
    print(f"  Baseline MAE (predict mean): {baseline_mae*100:.2f}%")
    print(f"  Improvement over baseline: {improvement:.1f}%")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main execution function."""

    print(f"{'='*60}")
    print("Random Forest Body Length Error Prediction")
    print(f"{'='*60}")
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Landmark Set: {LANDMARK_SET_NAME}")
    print(f"Body parts: {BODYPART_1} <-> {BODYPART_2}")
    print(f"{'='*60}")

    # 1. Load data
    input_file = f'cv_frame_level_results_{EXPERIMENT_ID}_{LANDMARK_SET_NAME}.csv'
    print(f"\nLoading frame-level results from: {input_file}")
    df = load_and_validate_data(input_file, BODYPART_1, BODYPART_2)
    print(f"Loaded {len(df)} frames")

    # 2. Calculate body length
    print("\nCalculating body length metrics...")
    df = calculate_body_length(df, BODYPART_1, BODYPART_2)

    # 3. Calculate errors
    print("Calculating errors...")
    df = calculate_errors(df)

    # 4. Filter valid data
    print("\nFiltering valid data...")
    df_filtered = filter_valid_data(df, MIN_BODY_LENGTH_PIXELS)

    if len(df_filtered) == 0:
        print("\nError: No valid data remaining after filtering!")
        sys.exit(1)

    # 5. Extract features
    print("\nExtracting features...")
    feature_names, X = extract_confidence_features(df_filtered)
    y = df_filtered['relative_error'].values
    print(f"Extracted {len(feature_names)} features from {len(X)} samples")

    # 6. Train/test split
    print(f"\nSplitting data (test size = {TEST_SIZE*100:.0f}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # 7. Train model
    model = train_random_forest(
        X_train, y_train,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE
    )

    # 8. Evaluate model
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, X_test, y_test)

    # 9. Print summary statistics
    print_summary_statistics(df_filtered, feature_names, metrics, y_test)

    # 10. Generate plots
    print(f"\n{'='*60}")
    print("Generating plots...")

    print("\n1. Generating feature importance plot...")
    plot_feature_importance(model, feature_names, EXPERIMENT_ID, LANDMARK_SET_NAME,
                           OUTPUT_DIR, PLOT_DPI, PLOT_FIGSIZE)

    print("2. Generating predictions plot...")
    plot_predictions(y_test, metrics['y_pred'], EXPERIMENT_ID, LANDMARK_SET_NAME,
                    OUTPUT_DIR, PLOT_DPI, PLOT_FIGSIZE, metrics)

    # 11. Save model
    if SAVE_MODEL:
        print("\n3. Saving trained model...")
        save_model_to_disk(model, EXPERIMENT_ID, LANDMARK_SET_NAME, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

