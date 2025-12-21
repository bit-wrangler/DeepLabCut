#!/usr/bin/env python3
"""
Analyze body length prediction error vs confidence scores.

This script loads frame-level aggregate CSV files from cross-validation experiments,
calculates body length metrics (true, predicted, error), and generates visualizations
showing the relationship between prediction error and confidence scores.

Usage:
    1. Update the configuration constants at the top of this file
    2. Run: python analyze_body_length_vs_confidence.py

The script will:
    - Load frame-level aggregate CSV file (from User Story 02)
    - Calculate body length (SVL) from snout and tail1 positions
    - Calculate prediction errors (absolute and relative)
    - Calculate confidence metrics (arithmetic mean, harmonic mean, and min)
    - Generate two scatter plots:
        1. Error vs arithmetic mean confidence
        2. Error vs harmonic mean confidence
    - Optionally save computed metrics to CSV

Note: Harmonic mean is more sensitive to low confidence values than arithmetic mean,
      making it useful for identifying cases where one bodypart has low confidence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ============================================================================
# Configuration
# ============================================================================

EXPERIMENT_ID = 'll_0d025'          # Experiment to analyze
LANDMARK_SET_NAME = 'all'           # Landmark set ('all', 'truncated', etc.)

# Body length calculation settings
BODYPART_1 = 'snout'                # First bodypart for body length
BODYPART_2 = 'tail1'                # Second bodypart for body length

# Filtering thresholds
MIN_BODY_LENGTH_PIXELS = 10.0       # Minimum valid body length (avoid division by zero)
CONFIDENCE_THRESHOLD = 0.0          # Minimum confidence to include (0.0 = include all)

# Visualization settings
PLOT_DPI = 300                      # Plot resolution
PLOT_FIGSIZE = (10, 8)              # Figure size in inches
PLOT_ALPHA = 0.5                    # Point transparency
MAX_RELATIVE_ERROR_DISPLAY = 1.0   # Cap relative error at 100% for visualization

# Output settings
SAVE_COMPUTED_METRICS = True        # Whether to save computed metrics to CSV
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

    # Build list of required columns
    required_cols = [
        f'gt_{bodypart_1}_x', f'gt_{bodypart_1}_y',
        f'gt_{bodypart_2}_x', f'gt_{bodypart_2}_y',
        f'pred_{bodypart_1}_x', f'pred_{bodypart_1}_y',
        f'pred_{bodypart_2}_x', f'pred_{bodypart_2}_y',
        f'conf_{bodypart_1}', f'conf_{bodypart_2}',
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


def calculate_body_length_metrics(df, bodypart_1, bodypart_2):
    """
    Calculate body length and error metrics using vectorized operations.

    Args:
        df: DataFrame with ground truth and predicted positions
        bodypart_1: Name of first bodypart
        bodypart_2: Name of second bodypart

    Returns:
        pd.DataFrame: DataFrame with added columns:
            - true_body_length: Ground truth body length in pixels
            - pred_body_length: Predicted body length in pixels
            - absolute_error: Absolute error in pixels
            - relative_error: Relative error as decimal (0.05 = 5%)
    """
    # Calculate true body length (SVL) using Euclidean distance
    # Formula: sqrt((x1 - x2)^2 + (y1 - y2)^2)
    df['true_body_length'] = np.sqrt(
        (df[f'gt_{bodypart_1}_x'] - df[f'gt_{bodypart_2}_x'])**2 +
        (df[f'gt_{bodypart_1}_y'] - df[f'gt_{bodypart_2}_y'])**2
    )

    # Calculate predicted body length using Euclidean distance
    df['pred_body_length'] = np.sqrt(
        (df[f'pred_{bodypart_1}_x'] - df[f'pred_{bodypart_2}_x'])**2 +
        (df[f'pred_{bodypart_1}_y'] - df[f'pred_{bodypart_2}_y'])**2
    )

    # Calculate absolute error
    df['absolute_error'] = np.abs(df['pred_body_length'] - df['true_body_length'])

    # Calculate relative error (as decimal, e.g., 0.05 = 5%)
    # Avoid division by zero - will result in NaN which we'll filter later
    df['relative_error'] = df['absolute_error'] / df['true_body_length']

    return df


def calculate_confidence_metrics(df, bodypart_1, bodypart_2):
    """
    Calculate confidence score aggregations.

    Args:
        df: DataFrame with confidence scores
        bodypart_1: Name of first bodypart
        bodypart_2: Name of second bodypart

    Returns:
        pd.DataFrame: DataFrame with added columns:
            - mean_confidence: Mean of bodypart_1 and bodypart_2 confidence
            - min_confidence: Minimum of bodypart_1 and bodypart_2 confidence
            - harmonic_mean_confidence: Harmonic mean of bodypart_1 and bodypart_2 confidence
    """
    # Calculate mean confidence (arithmetic mean)
    df['mean_confidence'] = (df[f'conf_{bodypart_1}'] + df[f'conf_{bodypart_2}']) / 2

    # Calculate minimum confidence
    df['min_confidence'] = df[[f'conf_{bodypart_1}', f'conf_{bodypart_2}']].min(axis=1)

    # Calculate harmonic mean confidence
    # Formula: 2 / (1/c1 + 1/c2) = 2 * c1 * c2 / (c1 + c2)
    # Harmonic mean is more sensitive to low values than arithmetic mean
    conf_1 = df[f'conf_{bodypart_1}']
    conf_2 = df[f'conf_{bodypart_2}']
    df['harmonic_mean_confidence'] = 2 * conf_1 * conf_2 / (conf_1 + conf_2)

    return df


def filter_valid_data(df, min_body_length, confidence_threshold):
    """
    Filter out invalid or low-quality data.

    Args:
        df: DataFrame with computed metrics
        min_body_length: Minimum body length threshold (pixels)
        confidence_threshold: Minimum confidence threshold

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    initial_count = len(df)

    # Filter out frames where true_body_length is too small (avoid division by zero)
    df_filtered = df[df['true_body_length'] >= min_body_length].copy()
    small_body_length_count = initial_count - len(df_filtered)

    # Filter out frames where confidence is below threshold
    df_filtered = df_filtered[df_filtered['mean_confidence'] >= confidence_threshold].copy()
    low_confidence_count = initial_count - small_body_length_count - len(df_filtered)

    # Drop rows with NaN values in critical columns
    critical_cols = ['true_body_length', 'pred_body_length', 'relative_error',
                     'mean_confidence', 'min_confidence', 'harmonic_mean_confidence']
    df_filtered = df_filtered.dropna(subset=critical_cols)
    nan_count = initial_count - small_body_length_count - low_confidence_count - len(df_filtered)

    # Print filtering statistics
    print(f"Filtering statistics:")
    print(f"  Initial frames: {initial_count}")
    print(f"  Removed (body length < {min_body_length} px): {small_body_length_count}")
    print(f"  Removed (confidence < {confidence_threshold}): {low_confidence_count}")
    print(f"  Removed (NaN values): {nan_count}")
    print(f"  Valid frames remaining: {len(df_filtered)}")
    print(f"  Percentage retained: {100 * len(df_filtered) / initial_count:.1f}%")

    if len(df_filtered) == 0:
        print("\nWarning: All data was filtered out! Consider relaxing filter thresholds.")

    return df_filtered


def generate_scatter_plot(df, experiment_id, landmark_set_name, output_dir, plot_settings):
    """
    Create and save scatter plot visualization.

    Args:
        df: DataFrame with computed metrics
        experiment_id: Experiment identifier
        landmark_set_name: Landmark set name
        output_dir: Directory to save plot
        plot_settings: Dictionary with plot configuration
    """
    if len(df) == 0:
        print("Warning: No data to plot. Skipping plot generation.")
        return

    # Create figure
    plt.figure(figsize=plot_settings['figsize'])

    # Cap relative error for visualization if specified
    relative_error_display = df['relative_error'].copy()
    if plot_settings['max_error'] is not None:
        relative_error_display = relative_error_display.clip(upper=plot_settings['max_error'])

    # Create scatter plot (convert relative error to percentage)
    plt.scatter(df['mean_confidence'],
                relative_error_display * 100,
                alpha=plot_settings['alpha'],
                s=20,
                edgecolors='none')

    # Set axis labels
    plt.xlabel('Mean Confidence Score', fontsize=12)
    plt.ylabel('Relative Error (%)', fontsize=12)

    # Set title
    plt.title(f'Body Length Error vs Confidence: {experiment_id} ({landmark_set_name})',
              fontsize=14, fontweight='bold')

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add reference lines
    # Horizontal lines for error thresholds
    for error_threshold in [5, 10, 20]:
        plt.axhline(y=error_threshold, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Vertical lines for confidence thresholds
    for conf_threshold in [0.5, 0.7, 0.9]:
        plt.axvline(x=conf_threshold, color='blue', linestyle='--', alpha=0.3, linewidth=1)

    # Set axis limits
    plt.xlim(0, 1.0)
    plt.ylim(0, max(100, relative_error_display.max() * 100 * 1.1))

    # Tight layout
    plt.tight_layout()

    # Save plot
    output_filename = f'body_length_error_vs_confidence_{experiment_id}_{landmark_set_name}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=plot_settings['dpi'], bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_filename}")


def generate_harmonic_mean_plot(df, experiment_id, landmark_set_name, output_dir, plot_settings):
    """
    Create and save scatter plot with harmonic mean confidence.

    Args:
        df: DataFrame with computed metrics
        experiment_id: Experiment identifier
        landmark_set_name: Landmark set name
        output_dir: Directory to save plot
        plot_settings: Dictionary with plot configuration
    """
    if len(df) == 0:
        print("Warning: No data to plot. Skipping harmonic mean plot generation.")
        return

    # Create figure
    plt.figure(figsize=plot_settings['figsize'])

    # Cap relative error for visualization if specified
    relative_error_display = df['relative_error'].copy()
    if plot_settings['max_error'] is not None:
        relative_error_display = relative_error_display.clip(upper=plot_settings['max_error'])

    # Create scatter plot (convert relative error to percentage)
    plt.scatter(df['harmonic_mean_confidence'],
                relative_error_display * 100,
                alpha=plot_settings['alpha'],
                s=20,
                edgecolors='none')

    # Set axis labels
    plt.xlabel('Harmonic Mean Confidence Score', fontsize=12)
    plt.ylabel('Relative Error (%)', fontsize=12)

    # Set title
    plt.title(f'Body Length Error vs Harmonic Mean Confidence: {experiment_id} ({landmark_set_name})',
              fontsize=14, fontweight='bold')

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add reference lines
    # Horizontal lines for error thresholds
    for error_threshold in [5, 10, 20]:
        plt.axhline(y=error_threshold, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Vertical lines for confidence thresholds
    for conf_threshold in [0.5, 0.7, 0.9]:
        plt.axvline(x=conf_threshold, color='blue', linestyle='--', alpha=0.3, linewidth=1)

    # Set axis limits
    plt.xlim(0, 1.0)
    plt.ylim(0, max(100, relative_error_display.max() * 100 * 1.1))

    # Tight layout
    plt.tight_layout()

    # Save plot
    output_filename = f'body_length_error_vs_harmonic_confidence_{experiment_id}_{landmark_set_name}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=plot_settings['dpi'], bbox_inches='tight')
    plt.close()

    print(f"Harmonic mean plot saved to: {output_filename}")


def save_computed_metrics(df, experiment_id, landmark_set_name, output_dir):
    """
    Save DataFrame with computed metrics to CSV file.

    Args:
        df: DataFrame with computed metrics
        experiment_id: Experiment identifier
        landmark_set_name: Landmark set name
        output_dir: Directory to save CSV
    """
    output_filename = f'body_length_analysis_{experiment_id}_{landmark_set_name}.csv'
    output_path = os.path.join(output_dir, output_filename)

    df.to_csv(output_path, index=False)
    print(f"Computed metrics saved to: {output_filename}")


def print_summary_statistics(df, df_filtered):
    """
    Print informative summary statistics to console.

    Args:
        df: Original DataFrame (before filtering)
        df_filtered: Filtered DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}\n")

    # Body length statistics
    print("Body Length Statistics (pixels):")
    print(f"  True body length:")
    print(f"    Mean: {df_filtered['true_body_length'].mean():.2f}")
    print(f"    Std:  {df_filtered['true_body_length'].std():.2f}")
    print(f"    Min:  {df_filtered['true_body_length'].min():.2f}")
    print(f"    Max:  {df_filtered['true_body_length'].max():.2f}")
    print(f"  Predicted body length:")
    print(f"    Mean: {df_filtered['pred_body_length'].mean():.2f}")
    print(f"    Std:  {df_filtered['pred_body_length'].std():.2f}")
    print(f"    Min:  {df_filtered['pred_body_length'].min():.2f}")
    print(f"    Max:  {df_filtered['pred_body_length'].max():.2f}")

    # Error statistics
    print(f"\nError Statistics:")
    print(f"  Absolute error (pixels):")
    print(f"    Mean: {df_filtered['absolute_error'].mean():.2f}")
    print(f"    Std:  {df_filtered['absolute_error'].std():.2f}")
    print(f"    Min:  {df_filtered['absolute_error'].min():.2f}")
    print(f"    Max:  {df_filtered['absolute_error'].max():.2f}")
    print(f"  Relative error (%):")
    print(f"    Mean: {df_filtered['relative_error'].mean() * 100:.2f}%")
    print(f"    Std:  {df_filtered['relative_error'].std() * 100:.2f}%")
    print(f"    Min:  {df_filtered['relative_error'].min() * 100:.2f}%")
    print(f"    Max:  {df_filtered['relative_error'].max() * 100:.2f}%")

    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence (arithmetic):")
    print(f"    Mean: {df_filtered['mean_confidence'].mean():.4f}")
    print(f"    Std:  {df_filtered['mean_confidence'].std():.4f}")
    print(f"    Min:  {df_filtered['mean_confidence'].min():.4f}")
    print(f"    Max:  {df_filtered['mean_confidence'].max():.4f}")
    print(f"  Harmonic mean confidence:")
    print(f"    Mean: {df_filtered['harmonic_mean_confidence'].mean():.4f}")
    print(f"    Std:  {df_filtered['harmonic_mean_confidence'].std():.4f}")
    print(f"    Min:  {df_filtered['harmonic_mean_confidence'].min():.4f}")
    print(f"    Max:  {df_filtered['harmonic_mean_confidence'].max():.4f}")
    print(f"  Min confidence:")
    print(f"    Mean: {df_filtered['min_confidence'].mean():.4f}")
    print(f"    Std:  {df_filtered['min_confidence'].std():.4f}")
    print(f"    Min:  {df_filtered['min_confidence'].min():.4f}")
    print(f"    Max:  {df_filtered['min_confidence'].max():.4f}")

    # Correlation between confidence and error
    correlation_mean = df_filtered['mean_confidence'].corr(df_filtered['relative_error'])
    correlation_harmonic = df_filtered['harmonic_mean_confidence'].corr(df_filtered['relative_error'])
    print(f"\nCorrelation with relative_error:")
    print(f"  Mean confidence (arithmetic): {correlation_mean:.4f}")
    print(f"  Harmonic mean confidence:     {correlation_harmonic:.4f}")

    # Error distribution by confidence bins
    print(f"\nError by Confidence Bins:")
    bins = [0.0, 0.5, 0.7, 0.9, 1.0]
    bin_labels = ['0.0-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
    df_filtered['conf_bin'] = pd.cut(df_filtered['mean_confidence'], bins=bins, labels=bin_labels)

    for bin_label in bin_labels:
        bin_data = df_filtered[df_filtered['conf_bin'] == bin_label]
        if len(bin_data) > 0:
            mean_error = bin_data['relative_error'].mean() * 100
            count = len(bin_data)
            print(f"  Confidence {bin_label}: {mean_error:.2f}% mean error (n={count})")
        else:
            print(f"  Confidence {bin_label}: No data")


def main():
    """Main function to run the analysis."""
    # 1. Construct input file path from configuration
    input_file = f'cv_frame_level_results_{EXPERIMENT_ID}_{LANDMARK_SET_NAME}.csv'
    input_path = os.path.join(OUTPUT_DIR, input_file)

    # 2. Print header
    print(f"\n{'='*60}")
    print(f"Body Length Error vs Confidence Analysis")
    print(f"{'='*60}")
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Landmark Set: {LANDMARK_SET_NAME}")
    print(f"Body parts: {BODYPART_1} <-> {BODYPART_2}")
    print(f"{'='*60}\n")

    # 3. Load and validate data
    print(f"Loading frame-level results from: {input_file}")
    try:
        df = load_and_validate_data(input_path, BODYPART_1, BODYPART_2)
        print(f"Loaded {len(df)} frames\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 4. Calculate body length metrics
    print("Calculating body length metrics...")
    df = calculate_body_length_metrics(df, BODYPART_1, BODYPART_2)

    # 5. Calculate confidence metrics
    print("Calculating confidence metrics...")
    df = calculate_confidence_metrics(df, BODYPART_1, BODYPART_2)

    # 6. Filter valid data
    print("\nFiltering valid data...")
    df_filtered = filter_valid_data(df, MIN_BODY_LENGTH_PIXELS, CONFIDENCE_THRESHOLD)

    if len(df_filtered) == 0:
        print("\nError: No valid data remaining after filtering. Exiting.")
        sys.exit(1)

    # 7. Print summary statistics
    print_summary_statistics(df, df_filtered)

    # 8. Generate scatter plots
    print(f"\n{'='*60}")
    print("Generating scatter plots...")
    plot_settings = {
        'figsize': PLOT_FIGSIZE,
        'alpha': PLOT_ALPHA,
        'dpi': PLOT_DPI,
        'max_error': MAX_RELATIVE_ERROR_DISPLAY
    }

    # Plot 1: Mean confidence (arithmetic mean)
    print("\n1. Generating plot with arithmetic mean confidence...")
    generate_scatter_plot(df_filtered, EXPERIMENT_ID, LANDMARK_SET_NAME,
                         OUTPUT_DIR, plot_settings)

    # Plot 2: Harmonic mean confidence
    print("2. Generating plot with harmonic mean confidence...")
    generate_harmonic_mean_plot(df_filtered, EXPERIMENT_ID, LANDMARK_SET_NAME,
                                OUTPUT_DIR, plot_settings)

    # 9. Optionally save computed metrics
    if SAVE_COMPUTED_METRICS:
        print("\nSaving computed metrics to CSV...")
        save_computed_metrics(df_filtered, EXPERIMENT_ID, LANDMARK_SET_NAME, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

