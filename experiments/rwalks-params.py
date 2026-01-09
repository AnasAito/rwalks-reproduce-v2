#!/usr/bin/env python3
"""
Random walks parameter experiment script that tests different depth and walk count values.

Usage:
    python rwalks-params.py --data_src_path /data/anas.aitaomar/sift_1m_old_dist.h5 \
                            --depth_values 2,3,5,7,10 \
                            --walk_values 5,10,20,50,100 \
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
print(f"RWALKS_PRUN_FACTOR: {os.getenv('RWALKS_PRUN_FACTOR')}")
print(f"NUM_THREADS: {os.getenv('NUM_THREADS')}")
print(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS')}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run random walks parameter experiments')
    parser.add_argument('--data_src_path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--depth_values', type=str, required=True,
                        help='Comma-separated depth values to test (e.g., 2,3,5,7,10)')
    parser.add_argument('--walk_values', type=str, required=True,
                        help='Comma-separated walk count values to test (e.g., 5,10,20,50,100)')
    parser.add_argument('--ef_values', type=str, default='10,50,100,150,200,250,300,350,400',
                        help='Comma-separated EF values to test (default: 10,50,100,150,200,250,300,350,400)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of neighbors to retrieve (default: 10)')
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


def run_depth_experiment(dataset, depth_values, ef_values, fixed_walks, k, num_threads=None):
    """Run experiments with varying depth values."""

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
    print(f"Running DEPTH experiments")
    print(f"  Depth values: {depth_values}")
    print(f"  Fixed walks: {fixed_walks}")
    print(f"  EF values: {ef_values}")
    print(f"{'='*80}\n")

    for depth in depth_values:
        print(f"\nTesting depth={depth} (walks={fixed_walks})...")

        # Build index with this depth value
        index = build_index(dataset, attr_depth=depth, attr_steps=fixed_walks,
                            num_threads=num_threads)

        # Set search mode to rwalks
        index.set_search_mode(search_mode=0)
        index.set_pron_factor(float(os.getenv('RWALKS_PRUN_FACTOR', 0.0)))
        index.set_num_threads(num_threads)

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
                'param_type': 'depth',
                'param_value': depth,
                'fixed_param': fixed_walks,
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


def run_walks_experiment(dataset, walk_values, ef_values, fixed_depth, k, num_threads=None):
    """Run experiments with varying walk count values."""

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
    print(f"Running WALKS experiments")
    print(f"  Walk values: {walk_values}")
    print(f"  Fixed depth: {fixed_depth}")
    print(f"  EF values: {ef_values}")
    print(f"{'='*80}\n")

    for walks in walk_values:
        print(f"\nTesting walks={walks} (depth={fixed_depth})...")

        # Build index with this walk value
        index = build_index(dataset, attr_depth=fixed_depth, attr_steps=walks,
                            num_threads=num_threads)

        # Set search mode to rwalks
        index.set_search_mode(search_mode=0)
        index.set_pron_factor(float(os.getenv('RWALKS_PRUN_FACTOR', 0.0)))
        index.set_num_threads(num_threads)

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
                'param_type': 'walks',
                'param_value': walks,
                'fixed_param': fixed_depth,
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


def plot_results(results_df, param_type, data_hash, output_dir):
    """Create QPS-Recall curve plot for the given parameter type."""

    # Filter results for this parameter type
    df = results_df[results_df['param_type'] == param_type].copy()

    # Get unique parameter values
    param_values = sorted(df['param_value'].unique())

    # Create figure
    plt.figure(figsize=(12, 8))

    # Color map for different parameter values
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))

    # Plot each parameter value as a separate curve
    for idx, param_val in enumerate(param_values):
        param_df = df[df['param_value'] == param_val].sort_values('recall')

        if param_type == 'depth':
            label = f'depth={param_val} (walks={param_df.iloc[0]["fixed_param"]})'
        else:
            label = f'walks={param_val} (depth={param_df.iloc[0]["fixed_param"]})'

        plt.plot(param_df['recall'], param_df['qps'],
                 marker='o', label=label, color=colors[idx], linewidth=2, markersize=6)

    # Formatting
    plt.xlabel('Recall@10', fontsize=14, fontweight='bold')
    plt.ylabel('QPS (Queries Per Second)', fontsize=14, fontweight='bold')

    title_param = 'Depth' if param_type == 'depth' else 'Walk Count'
    plt.title(f'QPS-Recall Curve: Varying {title_param}\nDataset: {data_hash}',
              fontsize=16, fontweight='bold', pad=20)

    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save plot
    filename = f"rwalks_{param_type}_experiment_{data_hash}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")

    plt.close()

    return filepath


def save_results(results, data_hash, output_dir):
    """Save results to CSV file."""

    # Create meaningful filename
    filename = f"rwalks_params_experiment_{data_hash}.csv"
    filepath = output_dir / filename

    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)

    print(f"\nResults saved to: {filepath}")

    # Print summary statistics
    print("\nExperiment Summary:")
    print("=" * 80)

    # Depth experiments summary
    depth_df = df[df['param_type'] == 'depth']
    if not depth_df.empty:
        print("\nDEPTH Experiments:")
        print(f"  Total runs: {len(depth_df)}")
        print(f"  Average QPS: {depth_df['qps'].mean():.2f}")
        print(f"  Average Recall: {depth_df['recall'].mean():.4f}")
        print(
            f"  QPS Range: {depth_df['qps'].min():.2f} - {depth_df['qps'].max():.2f}")
        print(
            f"  Recall Range: {depth_df['recall'].min():.4f} - {depth_df['recall'].max():.4f}")

        print("\n  Best configurations by depth:")
        for depth in sorted(depth_df['param_value'].unique()):
            depth_subset = depth_df[depth_df['param_value'] == depth]
            best_row = depth_subset.loc[depth_subset['recall'].idxmax()]
            print(
                f"    Depth {depth}: Recall {best_row['recall']:.4f} (EF={best_row['ef']}, QPS={best_row['qps']:.2f})")

    # Walks experiments summary
    walks_df = df[df['param_type'] == 'walks']
    if not walks_df.empty:
        print("\nWALKS Experiments:")
        print(f"  Total runs: {len(walks_df)}")
        print(f"  Average QPS: {walks_df['qps'].mean():.2f}")
        print(f"  Average Recall: {walks_df['recall'].mean():.4f}")
        print(
            f"  QPS Range: {walks_df['qps'].min():.2f} - {walks_df['qps'].max():.2f}")
        print(
            f"  Recall Range: {walks_df['recall'].min():.4f} - {walks_df['recall'].max():.4f}")

        print("\n  Best configurations by walk count:")
        for walks in sorted(walks_df['param_value'].unique()):
            walks_subset = walks_df[walks_df['param_value'] == walks]
            best_row = walks_subset.loc[walks_subset['recall'].idxmax()]
            print(
                f"    Walks {walks}: Recall {best_row['recall']:.4f} (EF={best_row['ef']}, QPS={best_row['qps']:.2f})")

    return filepath, df


def main():
    """Main function."""
    args = parse_arguments()

    # Extract dataset name from path
    data_hash = args.data_src_path.split("/")[-1].split(".")[0]

    print("Random Walks Parameter Experiment Runner")
    print("=" * 80)
    print(f"Dataset: {data_hash}")
    print(f"Dataset path: {args.data_src_path}")
    print()

    # Parse parameter values
    try:
        depth_values = [int(x.strip()) for x in args.depth_values.split(',')]
        walk_values = [int(x.strip()) for x in args.walk_values.split(',')]
        ef_values = [int(x.strip()) for x in args.ef_values.split(',')]
    except ValueError as e:
        print(f"Error parsing parameter values: {e}")
        print("Please provide comma-separated integer values.")
        sys.exit(1)

    print(f"Depth values to test: {depth_values}")
    print(f"Walk values to test: {walk_values}")
    print(f"EF values to test: {ef_values}")
    print(f"K: {args.k}")
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

        # Use default values as fixed parameters
        fixed_walks = 10  # Default walk count
        fixed_depth = 3   # Default depth

        print(f"Fixed parameters (defaults):")
        print(f"  For depth experiments: walks={fixed_walks}")
        print(f"  For walks experiments: depth={fixed_depth}")
        print()

        all_results = []

        # Run depth experiments
        depth_results = run_depth_experiment(
            dataset=dataset,
            depth_values=depth_values,
            ef_values=ef_values,
            fixed_walks=fixed_walks,
            k=args.k
        )
        all_results.extend(depth_results)

        # Run walks experiments
        walks_results = run_walks_experiment(
            dataset=dataset,
            walk_values=walk_values,
            ef_values=ef_values,
            fixed_depth=fixed_depth,
            k=args.k
        )
        all_results.extend(walks_results)

        # Save results
        csv_filepath, results_df = save_results(
            all_results, data_hash, output_dir)

        # Create plots
        print("\nGenerating plots...")
        depth_plot = plot_results(results_df, 'depth', data_hash, output_dir)
        walks_plot = plot_results(results_df, 'walks', data_hash, output_dir)

        print("\n" + "=" * 80)
        print("Experiment completed successfully!")
        print("=" * 80)
        print("\nGenerated files:")
        print(f"  CSV Results: {csv_filepath}")
        print(f"\nGenerated plots:")
        print(f"  Depth plot:  {depth_plot}")
        print(f"  Walks plot:  {walks_plot}")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
