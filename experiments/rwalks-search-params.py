#!/usr/bin/env python3
"""
Random walks search parameter experiment script that tests different pruning factor values.

Usage:
    python rwalks-search-params.py --data_src_path /data/anas.aitaomar/sift_1m_old_dist.h5 \
                                   --prun_factor_values -10,0.0,0.01,0.05 \
                                   --ef_values 10,50,100,150,200,250,300,350,400
"""

from pathlib import Path
import pandas as pd
import time
import argparse
import hnswlib
from utils import load_dataset, compute_recall
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv


load_dotenv()

# set OMP_NUM_THREADS to NUM_THREADS
os.environ['OMP_NUM_THREADS'] = os.getenv('NUM_THREADS')

print(f"PARAMS ")
print(f"RWALKS_EF_CONSTRUCTION: {os.getenv('RWALKS_EF_CONSTRUCTION')}")
print(f"RWALKS_M: {os.getenv('RWALKS_M')}")
print(f"NUM_THREADS: {os.getenv('NUM_THREADS')}")
print(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS')}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run random walks search parameter experiments')
    parser.add_argument('--data_src_path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--prun_factor_values', type=str, required=True,
                        help='Comma-separated pruning factor values to test (e.g., -10,0.0,0.01,0.05)')
    parser.add_argument('--ef_values', type=str, default='10,50,100,150,200,250,300,350,400',
                        help='Comma-separated EF values to test (default: 10,50,100,150,200,250,300,350,400)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of neighbors to retrieve (default: 10)')
    parser.add_argument('--attr_depth', type=int, default=3,
                        help='Attribute depth for index building (default: 3)')
    parser.add_argument('--attr_steps', type=int, default=10,
                        help='Attribute steps for index building (default: 10)')
    return parser.parse_args()


def build_index(dataset, attr_depth, attr_steps, num_threads=None):
    """Build HNSW index with the given dataset and random walk parameters."""
    if num_threads is None:
        num_threads = int(os.getenv('NUM_THREADS', -1))

    print(f"Building index with {dataset.train_vectors.shape[0]} vectors...")
    print(
        f"  Random walk params: depth={attr_depth}, steps={attr_steps}")

    index = hnswlib.Index(
        space='l2',
        dim=dataset.train_vectors.shape[1],
        dim_attr=dataset.train_labels.shape[1]
    )

    index.init_index(
        max_elements=dataset.train_vectors.shape[0],
        ef_construction=int(os.getenv('RWALKS_EF_CONSTRUCTION', 100)),
        M=int(os.getenv('RWALKS_M', 16))
    )

    index.set_num_threads(num_threads)

    # Add items with specified random walk parameters
    index.add_items(
        dataset.train_vectors,
        dataset.train_labels,
        attr_depth=attr_depth,
        attr_steps=attr_steps
    )

    print("Index built successfully!")
    return index


def run_prun_factor_experiment(dataset, index, prun_factor_values, ef_values, k, num_threads=None):
    """Run experiments with varying pruning factor values."""

    if num_threads is None:
        num_threads = int(os.getenv('NUM_THREADS', -1))

    # Select 1% specificity query chunk (same as specificity.py)
    specificities = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    specificity_index = 0  # Index for 0.01 (1%)
    query_range = (
        specificity_index *
        int(dataset.test_vectors.shape[0] / len(specificities)),
        (specificity_index + 1) *
        int(dataset.test_vectors.shape[0] / len(specificities))
    )

    queries_vecs = dataset.test_vectors[query_range[0]:query_range[1], :]
    queries_labels = dataset.test_labels[query_range[0]:query_range[1], :]
    queries_neighbors = dataset.neighbors[query_range[0]:query_range[1], :]

    print(
        f"Using 1% specificity query chunk: {query_range[0]}:{query_range[1]} ({queries_vecs.shape[0]} queries)")

    results = []

    print(f"\n{'='*80}")
    print(f"Running PRUNING FACTOR experiments")
    print(f"  Pruning factor values: {prun_factor_values}")
    print(f"  EF values: {ef_values}")
    print(f"{'='*80}\n")

    # Set search mode to rwalks
    index.set_search_mode(search_mode=0)
    index.set_num_threads(num_threads)

    for prun_factor in prun_factor_values:
        print(f"\nTesting pruning factor={prun_factor}...")

        # Set pruning factor
        index.set_pron_factor(float(prun_factor))

        # Test different EF values
        for ef in ef_values:
            print(f"  Testing EF={ef}...")
            index.set_ef(ef)

            # Run queries 4 times and average QPS
            qps_values = []
            neighbors = None
            distances = None

            for iteration in range(4):
                t0 = time.time()
                neighbors, distances = index.knn_query(
                    queries_vecs, queries_labels, k=k)
                t1 = time.time()

                query_time = t1 - t0
                qps = queries_vecs.shape[0] / query_time
                qps_values.append(qps)

            # Calculate average QPS and total time
            avg_qps = sum(qps_values) / len(qps_values)
            avg_query_time = queries_vecs.shape[0] / avg_qps

            # Compute recall (use last iteration's results)
            recall = compute_recall(neighbors, queries_neighbors)

            result = {
                'prun_factor': prun_factor,
                'ef': ef,
                'query_time': avg_query_time,
                'qps': avg_qps,
                'recall': recall,
                'num_queries': queries_vecs.shape[0],
                'k': k
            }

            results.append(result)
            print(
                f"    QPS (avg over 4 runs): {avg_qps:.2f}, Recall: {recall:.4f}")

    return results


def plot_results(results_df, data_hash, output_dir):
    """Create QPS-Recall curve plot for different pruning factor values."""

    # Get unique pruning factor values
    prun_factor_values = sorted(results_df['prun_factor'].unique())

    # Create figure
    plt.figure(figsize=(12, 8))

    # Color map for different pruning factor values
    colors = plt.cm.viridis(np.linspace(0, 1, len(prun_factor_values)))

    # Plot each pruning factor value as a separate curve
    for idx, prun_factor in enumerate(prun_factor_values):
        prun_df = results_df[results_df['prun_factor']
                             == prun_factor].sort_values('recall')

        if prun_factor == -10:
            label = f'prun_factor=disabled'
        else:
            label = f'prun_factor={prun_factor}'

        plt.plot(prun_df['recall'], prun_df['qps'],
                 marker='o', label=label, color=colors[idx], linewidth=2, markersize=6)

    # Formatting
    plt.xlabel('Recall@10', fontsize=14, fontweight='bold')
    plt.ylabel('QPS (Queries Per Second)', fontsize=14, fontweight='bold')

    plt.title(f'QPS-Recall Curve: Varying Pruning Factor\nDataset: {data_hash}',
              fontsize=16, fontweight='bold', pad=20)

    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save plot
    filename = f"rwalks_search_params_experiment_{data_hash}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")

    plt.close()

    return filepath


def save_results(results, data_hash, output_dir):
    """Save results to CSV file."""

    # Create meaningful filename
    filename = f"rwalks_search_params_experiment_{data_hash}.csv"
    filepath = output_dir / filename

    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)

    print(f"\nResults saved to: {filepath}")

    # Print summary statistics
    print("\nExperiment Summary:")
    print("=" * 80)
    print(f"  Total runs: {len(df)}")
    print(f"  Average QPS: {df['qps'].mean():.2f}")
    print(f"  Average Recall: {df['recall'].mean():.4f}")
    print(
        f"  QPS Range: {df['qps'].min():.2f} - {df['qps'].max():.2f}")
    print(
        f"  Recall Range: {df['recall'].min():.4f} - {df['recall'].max():.4f}")

    print("\n  Best configurations by pruning factor:")
    for prun_factor in sorted(df['prun_factor'].unique()):
        prun_subset = df[df['prun_factor'] == prun_factor]
        best_row = prun_subset.loc[prun_subset['recall'].idxmax()]
        pf_label = 'disabled' if prun_factor == -10 else str(prun_factor)
        print(
            f"    Pruning factor {pf_label}: Recall {best_row['recall']:.4f} (EF={best_row['ef']}, QPS={best_row['qps']:.2f})")

    return filepath, df


def main():
    """Main function."""
    args = parse_arguments()

    # Extract dataset name from path
    data_hash = args.data_src_path.split("/")[-1].split(".")[0]

    print("Random Walks Search Parameter Experiment Runner")
    print("=" * 80)
    print(f"Dataset: {data_hash}")
    print(f"Dataset path: {args.data_src_path}")
    print()

    # Parse parameter values
    try:
        prun_factor_values = [float(x.strip())
                              for x in args.prun_factor_values.split(',')]
        ef_values = [int(x.strip()) for x in args.ef_values.split(',')]
    except ValueError as e:
        print(f"Error parsing parameter values: {e}")
        print("Please provide comma-separated values.")
        sys.exit(1)

    print(f"Pruning factor values to test: {prun_factor_values}")
    print(f"EF values to test: {ef_values}")
    print(f"K: {args.k}")
    print(
        f"Index build params: depth={args.attr_depth}, steps={args.attr_steps}")
    print()

    # Create output directory
    output_dir = Path(__file__).resolve().parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    try:
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset(args.data_src_path)
        print(f"Dataset loaded: {dataset.train_vectors.shape[0]} training vectors, "
              f"{dataset.test_vectors.shape[0]} test vectors")
        print()

        # Build index once with default parameters
        index = build_index(
            dataset,
            attr_depth=args.attr_depth,
            attr_steps=args.attr_steps
        )

        # Run pruning factor experiments
        results = run_prun_factor_experiment(
            dataset=dataset,
            index=index,
            prun_factor_values=prun_factor_values,
            ef_values=ef_values,
            k=args.k
        )

        # Save results
        csv_filepath, results_df = save_results(
            results, data_hash, output_dir)

        # Create plot
        print("\nGenerating plot...")
        plot_filepath = plot_results(results_df, data_hash, output_dir)

        print("\n" + "=" * 80)
        print("Experiment completed successfully!")
        print("=" * 80)
        print("\nGenerated files:")
        print(f"  CSV Results: {csv_filepath}")
        print(f"\nGenerated plot:")
        print(f"  Search params plot: {plot_filepath}")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
