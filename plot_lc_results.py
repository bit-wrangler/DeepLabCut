"""
Plot learning curve results from a CSV file.

This script reads learning curve results and plots test RMSE vs training size
with a logarithmic x-axis scale.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys

# Hardcoded results file name
FILENAME = 'lc_results_lc_baseline_20251205-022006.csv'


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
    
    # Create the plot
    plt.figure(figsize=(4, 3))
    plt.plot(df['train_size'], df['all__test rmse'], 
             marker='o', label='Test RMSE')
    
    # Set logarithmic scale on x-axis
    plt.xscale('log')
    
    # Labels and title
    plt.xlabel('Training Size (frames)', fontsize=12)
    plt.ylabel('Test RMSE (pixels)', fontsize=12)
    # plt.title('Learning Curve: Test RMSE vs Training Size', fontsize=14, fontweight='bold')
    
    # Grid for better readability
    plt.grid(True, alpha=0.3, which='both')
    
    # Legend
    # plt.legend(fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
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
    
    # Show the plot
    plt.savefig('learning_curve.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    print(f"Plotting learning curve from: {FILENAME}\n")
    plot_learning_curve(FILENAME)

