import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.style as style
import argparse
from pathlib import Path

# Set scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18
})

# Methods to try (will skip if data doesn't exist)
METHODS = ["rwalks", "hnsw-inline", "stf", "acorn-1", "acorn-g"]

# Color scheme for methods
COLORS = {
    "rwalks": "#1f77b4",
    "hnsw-inline": "#ff7f0e",
    "stf": "#2ca02c",
    "acorn-1": "#d62728",
    "acorn-g": "#9467bd"
}

# Marker styles for methods
MARKERS = {
    "rwalks": "o",
    "hnsw-inline": "s",
    "stf": "^",
    "acorn-1": "D",
    "acorn-g": "D"
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate QPS vs Recall plots for unfiltered search experiments')

    parser.add_argument('--dataset', type=str, default='sift_1m',
                        help='Dataset name (default: sift_1m)')

    return parser.parse_args()


def load_and_process_data(dataset):
    """Load data for each method."""
    results = {}

    # Construct data directory path relative to parent of this file
    data_dir = str(Path(__file__).parent.parent / "data")

    for method in METHODS:
        try:
            filepath = f"{data_dir}/unf_search_experiment_{dataset}_unf_{method}.csv"
            df = pd.read_csv(filepath)

            # Check if this is acorn data with specificity column
            if 'specificity' in df.columns:
                # For acorn data, filter to specificity closest to 1.0 or just remove the column
                # Since it's unfiltered search, we expect specificity to be very high
                if not df.empty:
                    results[method] = df
            else:
                # For other methods (like rwalks), use data as-is
                results[method] = df

            print(f"Loaded data for {method}: {len(df)} points")

        except FileNotFoundError:
            print(f"File not found for method: {method}")
            continue
        except Exception as e:
            print(f"Error loading {method}: {e}")
            continue

    return results


def create_plot(results, dataset):
    """Create scientific plot showing QPS vs Recall for unfiltered search."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    fig.suptitle(f'QPS vs Recall - Unfiltered Search\n{dataset.upper()} Dataset',
                 fontsize=18, y=0.98)

    # Plot each method
    for method in METHODS:
        if method in results:
            data = results[method]
            if not data.empty:
                # Sort by recall for proper line plotting
                sorted_data = data.sort_values('recall')
                ax.plot(sorted_data['recall'], sorted_data['qps'],
                        color=COLORS[method], marker=MARKERS[method],
                        linewidth=2, markersize=8, alpha=0.8,
                        label=method, markerfacecolor=COLORS[method],
                        markeredgecolor='white', markeredgewidth=0.5)

    # Format the plot
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('QPS (Queries Per Second)', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)

    # Format tick labels
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Add minor grid
    ax.grid(True, which='minor', alpha=0.2, linestyle='--')

    # Add legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    return fig


def main():
    """Main function to load data and create plots."""
    args = parse_args()
    dataset = args.dataset

    print(f"Loading unfiltered search data for dataset: {dataset}...")
    results = load_and_process_data(dataset)

    if not results:
        print("No data found! Please check that CSV files exist in the data directory.")
        return

    print("Creating plot...")
    fig = create_plot(results, dataset)

    # Save the plot
    data_dir = str(Path(__file__).parent.parent / "data")
    output_filename = f"{data_dir}/qps_vs_recall_unf_{dataset}.png"
    fig.savefig(output_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Plot saved as {output_filename}")

    plt.show()


if __name__ == "__main__":
    main()
