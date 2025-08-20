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
        description='Generate QPS vs Recall plots for specificity experiments')

    parser.add_argument('--data_src_path', type=str, default='/Users/mac/Downloads/sift_1m_old_dist.h5',
                        help='Dataset name (default: /Users/mac/Downloads/sift_1m_old_dist.h5)')
    parser.add_argument('--specificity', type=float, nargs='+',
                        default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
                        help='Specificity values to plot (default: 0.01 0.05 0.1 0.2 0.3 0.5)')

    return parser.parse_args()


def calculate_pareto_frontier(df):
    """
    Calculate Pareto optimal points for QPS vs Recall.
    A point is Pareto optimal if no other point has both higher QPS and higher recall.
    """
    # Sort by recall first, then by QPS (descending)
    df_sorted = df.sort_values(['recall', 'qps'], ascending=[False, False])

    pareto_points = []
    max_recall_so_far = float('-inf')

    for idx, row in df_sorted.iterrows():
        if abs(row['recall'] - max_recall_so_far) > 0.01:
            pareto_points.append(idx)
            max_recall_so_far = row['recall']

    return df_sorted.loc[pareto_points].sort_values('recall')


def load_and_process_data(data_hash, specificity_set):
    """Load data and calculate Pareto frontiers for each method and specificity."""
    results = {}
    pareto_results = {}

    # Construct data directory path relative to parent of this file
    data_dir = str(Path(__file__).parent.parent / "data")

    for method in METHODS:
        try:
            df = pd.read_csv(
                f"{data_dir}/specificity_experiment_{data_hash}_{method}.csv")
            results[method] = df
            pareto_results[method] = {}

            # Calculate Pareto frontier for each specificity
            for spec in specificity_set:
                spec_data = df[df['specificity'] == spec]
                if not spec_data.empty:
                    pareto_results[method][spec] = spec_data  # calculate_pareto_frontier(
                    # spec_data)

        except FileNotFoundError:
            print(f"File not found for method: {method}")
            continue

    return results, pareto_results


def create_plots(pareto_results, data_hash, specificity_set):
    """Create scientific plots showing QPS vs Recall for each specificity."""
    n_specs = len(specificity_set)

    # Create subplot grid - one row with multiple columns
    fig, axes = plt.subplots(1, n_specs, figsize=(4*n_specs, 6), sharey=True)
    if n_specs == 1:
        axes = [axes]

    fig.suptitle(f'QPS vs Recall \n{data_hash.upper()} Dataset',
                 fontsize=18, y=0.98)

    for i, specificity in enumerate(specificity_set):
        ax = axes[i]

        # Plot each method
        for method in METHODS:
            print(f"Method: {method}")
            if method in pareto_results and specificity in pareto_results[method]:
                data = pareto_results[method][specificity]
                if not data.empty:
                    # Plot with lines and markers in one call
                    sorted_data = data  # .sort_values('recall')
                    ax.plot(sorted_data['recall'], sorted_data['qps'],
                            color=COLORS[method], marker=MARKERS[method],
                            linewidth=2, markersize=8, alpha=0.8,
                            label=method, markerfacecolor=COLORS[method],
                            markeredgecolor='white', markeredgewidth=0.5)

        # Format the subplot
        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_yscale('log')
        ax.set_title(f'Specificity = {specificity}', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.0)

        # Set ylabel only for the first subplot
        if i == 0:
            ax.set_ylabel('QPS (Queries Per Second)', fontweight='bold')

        # Format tick labels
        ax.tick_params(axis='both', which='major', labelsize=11)

        # Add minor grid
        ax.grid(True, which='minor', alpha=0.2, linestyle='--')

        # Add figure-level legend below the plots
    # Collect unique handles and labels from all subplots
    all_handles, all_labels = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)

    fig.legend(all_handles, all_labels, loc='lower center', ncol=len(METHODS),
               frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.02))

    # Adjust layout to accommodate legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.15)

    return fig


def main():
    """Main function to load data and create plots."""
    args = parse_args()
    data_hash = args.data_src_path.split(
        "/")[-1].split(".")[0]

    print("Loading data and calculating Pareto frontiers...")
    results, pareto_results = load_and_process_data(
        data_hash, args.specificity)

    print("Creating plots...")
    fig = create_plots(pareto_results, data_hash, args.specificity)

    # Save the plot
    data_dir = str(Path(__file__).parent.parent / "data")
    output_filename = f"{data_dir}/qps_vs_recall_pareto_{data_hash}.png"
    fig.savefig(output_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Plot saved as {output_filename}")

    plt.show()


if __name__ == "__main__":
    main()
