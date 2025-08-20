#!/usr/bin/env python3
"""
Specificity experiment script that runs comprehensive tests across different specificities and EF values.

Usage:
    python specificity.py --data_src_path /data/anas.aitaomar/sift_1m_old_dist.h5 --search_mode rwalks
"""

from pathlib import Path
import pandas as pd
import time
import argparse
import hnswlib
from utils import load_dataset, compute_recall, compute_acorn_results
import sys
import os
from acorn_prep_data import create_acorn_data
from dotenv import load_dotenv


load_dotenv()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run specificity experiments')
    parser.add_argument('--search_mode', type=str, required=True,
                        choices=['rwalks', 'hnsw-inline',
                                 'stf', 'acorn-1', 'acorn-g'],
                        help='Search mode to use')
    parser.add_argument('--data_src_path', type=str, required=True,
                        help='Path to the dataset')
    return parser.parse_args()


def build_index(dataset, num_threads=None):
    """Build HNSW index with the given dataset."""
    if num_threads is None:
        num_threads = int(os.getenv('NUM_THREADS', 48))

    print(f"Building index with {dataset.train_vectors.shape[0]} vectors...")

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
    index.add_items(dataset.train_vectors, dataset.train_labels)

    print("Index built successfully!")
    return index


def run_experiments(dataset, index, search_mode, num_threads=None):
    """Run experiments across all specificities and EF values."""

    # Configuration
    search_modes = {
        "rwalks": 0,
        "hnsw-inline": 1,
        "stf": 3,
    }

    specificities = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    ef_values = range(
        10, 500, 50) if search_mode != "hnsw-inline" else range(10, 50, 10)
    k = 10  # Number of neighbors to retrieve

    # Set search mode
    if num_threads is None:
        num_threads = int(os.getenv('NUM_THREADS', 48))

    index.set_search_mode(search_mode=search_modes[search_mode])
    index.set_pron_factor(float(os.getenv('RWALKS_PRUN_FACTOR', 0.0)))
    index.set_num_threads(num_threads)

    results = []

    print(f"Running experiments with search mode: {search_mode}")
    print(f"Testing specificities: {specificities}")
    print(f"Testing EF values: {ef_values}")

    for specificity in specificities:
        print(f"\nProcessing specificity: {specificity}")

        # Get query range for this specificity
        query_range = (
            specificities.index(
                specificity) * int(dataset.test_vectors.shape[0] / len(specificities)),
            (specificities.index(specificity) + 1) *
            int(dataset.test_vectors.shape[0] / len(specificities))
        )

        queries_vecs = dataset.test_vectors[query_range[0]:query_range[1], :]
        queries_labels = dataset.test_labels[query_range[0]:query_range[1], :]
        queries_neighbors = dataset.neighbors[query_range[0]:query_range[1], :]

        print(
            f"  Query range: {query_range[0]}:{query_range[1]} ({queries_vecs.shape[0]} queries)")

        for ef in ef_values:
            print(f"  Testing EF: {ef}")

            # Set EF
            index.set_ef(ef)

            # Run queries and measure time
            t0 = time.time()
            neighbors, distances = index.knn_query(
                queries_vecs, queries_labels, k=k)
            t1 = time.time()

            query_time = t1 - t0
            qps = queries_vecs.shape[0] / query_time

            # Compute recall
            recall_start = time.time()
            recall = compute_recall(neighbors, queries_neighbors)
            recall_time = time.time() - recall_start

            # Store results
            result = {
                'specificity': specificity,
                'ef': ef,
                'query_time': query_time,
                'qps': qps,
                'recall': recall,
                'recall_computation_time': recall_time,
                'num_queries': queries_vecs.shape[0],
                'k': k
            }

            results.append(result)

            print(f"    QPS: {qps:.2f}, Recall: {recall:.4f}")

    return results


def save_results(results, data_hash, search_mode):
    """Save results to CSV file."""

    # Create data directory if it doesn't exist
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Create meaningful filename
    filename = f"specificity_experiment_{data_hash}_{search_mode}.csv"
    filepath = data_dir / filename

    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)

    print(f"\nResults saved to: {filepath}")

    # Print summary
    print("\nExperiment Summary:")
    print("=" * 80)
    print(f"Search Mode: {search_mode}")
    print(f"Total experiments: {len(results)}")
    print("\nOverall Statistics:")
    print(f"  Average QPS: {df['qps'].mean():.2f}")
    print(f"  Average Recall: {df['recall'].mean():.4f}")
    print(f"  QPS Range: {df['qps'].min():.2f} - {df['qps'].max():.2f}")
    print(
        f"  Recall Range: {df['recall'].min():.4f} - {df['recall'].max():.4f}")

    # Print per-specificity summary
    print("\nPer-Specificity Summary (best recall):")
    specificity_summary = df.loc[df.groupby('specificity')['recall'].idxmax()]
    for _, row in specificity_summary.iterrows():
        print(
            f"  Specificity {row['specificity']}: Recall {row['recall']:.4f} (EF={row['ef']}, QPS={row['qps']:.2f})")

    return filepath


def main():
    """Main function."""
    args = parse_arguments()
    data_hash = args.data_src_path.split(
        "/")[-1].split(".")[0]
    print("Specificity Experiment Runner")
    print("=" * 50)
    print(f"Dataset: {data_hash}")
    print(f"Search Mode: {args.search_mode}")
    print()

    try:
        if "acorn" in args.search_mode:
            #  prep data
            data_root_dir = str(Path(__file__).parent.parent / "data")
            data_meta = create_acorn_data(
                args.data_src_path,
                data_root_dir + "/acorn_data"
            )
            #  run acorn on data_dst
            import subprocess

            # Use the parameterized shell script
            # Assume this script is in a folder (e.g., 'experiments') that is a sibling to 'acorn'
            acorn_path = str(
                (Path(__file__).parent.parent / "acorn").resolve())
            print(f"ACORN path: {acorn_path}")
            script_path = acorn_path + "/run_acorn.sh"
            num_vecs = str(data_meta["num_vecs"])
            gamma = "1" if "acorn-1" in args.search_mode else str(
                os.getenv('ACORN_GAMMA', "10"))
            dataset_path = "acorn_data"
            M = str(os.getenv('ACORN_M', "16"))
            M_beta = "16" if "acorn-1" in args.search_mode else str(
                os.getenv('ACORN_MB', "32"))

            cmd = [
                script_path,
                num_vecs,
                gamma,
                dataset_path,
                M,
                M_beta
            ]

            print(f"Running ACORN script: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=False, cwd=acorn_path)
            print("ACORN run completed.")
            #  save results
            results = compute_acorn_results(
                data_root_dir + "/acorn_data")
            output_file = save_results(
                results, data_hash, args.search_mode)
        else:
            # Load dataset
            print("Loading dataset...")
            dataset = load_dataset(
                args.data_src_path)
            print(f"Dataset loaded: {dataset.train_vectors.shape[0]} training vectors, "
                  f"{dataset.test_vectors.shape[0]} test vectors")

            # Build index
            index = build_index(dataset)

            # Run experiments
            results = run_experiments(dataset, index, args.search_mode)

            # Save results

            output_file = save_results(
                results, data_hash, args.search_mode)

            print(f"\nExperiment completed successfully!")
            print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
