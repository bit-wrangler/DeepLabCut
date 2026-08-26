"""
Plot learning curve results from a CSV file.

This script reads learning curve results and plots test RMSE vs training size
with a logarithmic x-axis scale.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Hardcoded results file name
FILENAME = 'lc_results_lc_baseline.csv'
OUTPUT_DIR = os.path.expanduser("~/projects/lizard-pub")

# Matplotlib publication style (match plot_loss_curves.py)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
})


def plot_learning_curve(filename):
    """
    Plot learning curve: test RMSE vs training size.

    Args:
        filename: Path to the CSV file containing learning curve results
    """
    # Load results
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        print("Please update the FILENAME constant at the top of this script.")
        sys.exit(1)

    # Check required columns exist
    required_cols = ['train_size', 'all__test rmse']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Sort by train_size for proper line plotting
    df = df.sort_values('train_size')

    # Create the plot — single-column width to match loss curve figures
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.plot(df['train_size'], df['all__test rmse'],
            marker='o', markersize=3, color="#2171b5")

    # Set logarithmic scale on x-axis
    ax.set_xscale('log')

    # Labels
    ax.set_xlabel('Training Size (frames)')
    ax.set_ylabel('Test RMSE (pixels)')

    # Grid for better readability
    ax.grid(True, which='both')

    # Display summary statistics
    print(f"\nLearning Curve Summary:")
    print(f"{'='*60}")
    print(f"Results file: {filename}")
    print(f"Number of steps: {len(df)}")
    print(f"Training size range: {df['train_size'].min()} - {df['train_size'].max()} frames")
    print(f"Test RMSE range: {df['all__test rmse'].min():.2f} - {df['all__test rmse'].max():.2f} pixels")
    print(f"{'='*60}\n")

    # Show detailed results
    print("Detailed Results:")
    display_cols = ['step', 'train_size', 'all__test rmse']
    if 'train_fraction' in df.columns:
        display_cols.insert(1, 'train_fraction')
    print(df[display_cols].to_string(index=False))

    # Save to paper repo
    for ext in ("pdf", "png"):
        out = os.path.join(OUTPUT_DIR, f"fig_learning_curve.{ext}")
        fig.savefig(out)
        print(f"  Saved {out}")
    plt.close(fig)


if __name__ == '__main__':
    print(f"Plotting learning curve from: {FILENAME}\n")
    plot_learning_curve(FILENAME)

