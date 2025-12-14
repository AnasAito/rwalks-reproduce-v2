#!/usr/bin/env python3
"""
Unfiltered search experiment script that runs tests on the full workload without specificity segmentation.

Usage:
    python unf_search.py --data_src_path /data/anas.aitaomar/unfiltered_dataset.h5 --search_mode rwalks
"""

from pathlib import Path
import pandas as pd
import time
import argparse
import hnswlib
from utils import load_dataset, compute_recall, compute_acorn_results
import sys
import os
import multiprocessing
from acorn_prep_unf import create_acorn_data
from dotenv import load_dotenv
import shutil


load_dotenv()

# set OMP_NUM_THREADS to NUM_THREADS
os.environ['OMP_NUM_THREADS'] = os.getenv('NUM_THREADS')

print(f"PARAMS ")
print(f"RWALKS_EF_CONSTRUCTION: {os.getenv('RWALKS_EF_CONSTRUCTION')}")
print(f"RWALKS_M: {os.getenv('RWALKS_M')}")
print(f"RWALKS_PRUN_FACTOR: {os.getenv('RWALKS_PRUN_FACTOR')}")
print(f"ACORN_GAMMA: {os.getenv('ACORN_GAMMA')}")
print(f"ACORN_M: {os.getenv('ACORN_M')}")
print(f"ACORN_MB: {os.getenv('ACORN_MB')}")
print(f"NUM_THREADS: {os.getenv('NUM_THREADS')}")
print(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS')}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run unfiltered search experiments')
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
        num_threads = int(os.getenv('NUM_THREADS', -1))

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
    index.add_items(dataset.train_vectors, dataset.train_labels,
                    attr_depth=3, attr_steps=20)

    print("Index built successfully!")
    return index


def run_experiments(dataset, index, search_mode, num_threads=None):
    """Run experiments on the full workload with different EF values."""

    # Configuration
    search_modes = {
        "rwalks": 0,
        "hnsw-inline": 1,
        "stf": 3,
    }

    ef_values = range(
        10, 500, 50) if search_mode != "hnsw-inline" else range(10, 50, 10)
    k = 10  # Number of neighbors to retrieve

    # Set search mode
    if num_threads is None:
        num_threads = int(os.getenv('NUM_THREADS', -1))

    index.set_search_mode(search_mode=search_modes[search_mode])
    index.set_pron_factor(-10)
    index.set_num_threads(num_threads)

    results = []

    print(f"Running experiments with search mode: {search_mode}")
    print(f"Testing EF values: {ef_values}")
    print(f"Full workload: {dataset.test_vectors.shape[0]} queries")

    # Use all queries (no specificity segmentation)
    queries_vecs = dataset.test_vectors
    queries_labels = dataset.test_labels
    queries_neighbors = dataset.neighbors

    print(f"Number of queries: {queries_vecs.shape[0]}")
    print(
        f"Attribute shape (10 columns: all 0s except last column is 1): {queries_labels.shape}")

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
    filename = f"unf_search_experiment_{data_hash}_{search_mode}.csv"
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

    # Print best recall configuration
    print("\nBest Configuration:")
    best_recall_idx = df['recall'].idxmax()
    best_row = df.loc[best_recall_idx]
    print(
        f"  Recall {best_row['recall']:.4f} (EF={best_row['ef']}, QPS={best_row['qps']:.2f})")

    return filepath


def empty_acorn_data_directory(acorn_data_path):
    """Empty acorn data directory."""
    if os.path.exists(acorn_data_path):
        shutil.rmtree(acorn_data_path)


def main():
    """Main function."""
    args = parse_arguments()
    data_hash = args.data_src_path.split(
        "/")[-1].split(".")[0]
    print("Unfiltered Search Experiment Runner")
    print("=" * 50)
    print(f"Dataset: {data_hash}")
    print(f"Search Mode: {args.search_mode}")
    print()

    try:
        if "acorn" in args.search_mode:
            #  prep data
            data_root_dir = str(Path(__file__).parent.parent / "data")
            # empty acorn data directory
            empty_acorn_data_directory(data_root_dir + "/acorn_data")
            data_meta = create_acorn_data(
                args.data_src_path,
                data_root_dir + "/acorn_data"
            )
            #  run acorn directly using test_acorn executable
            import subprocess

            # Get acorn directory path
            acorn_path = str(
                (Path(__file__).parent.parent / "acorn").resolve())
            print(f"ACORN path: {acorn_path}")

            # Prepare parameters
            num_vecs = str(data_meta["num_vecs"])
            gamma = "1" if "acorn-1" in args.search_mode else str(
                os.getenv('ACORN_GAMMA', "10"))
            dataset_path = "acorn_data"
            M = str(os.getenv('ACORN_M', "32"))
            M_beta = "16" if "acorn-1" in args.search_mode else str(
                os.getenv('ACORN_MB', "32"))

            # Call test_acorn directly (same as: cd acorn && ./build/demos/test_acorn ...)
            test_acorn_path = acorn_path + "/build/demos/test_acorn"
            cmd = [
                test_acorn_path,
                num_vecs,
                gamma,
                dataset_path,
                M,
                M_beta
            ]

            # Get number of threads from environment, default to CPU count
            num_threads = os.getenv('NUM_THREADS')
            if num_threads is None or num_threads == '':
                num_threads = str(multiprocessing.cpu_count())
            else:
                num_threads = str(num_threads)

            # Set OMP_NUM_THREADS for ACORN (OpenMP uses this)
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = num_threads

            print(f"Running ACORN: {' '.join(cmd)}")
            print(f"Working directory: {acorn_path}")
            print(f"OMP_NUM_THREADS: {num_threads}")

            # Run with real-time output streaming
            process = subprocess.Popen(
                cmd,
                cwd=acorn_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                env=env  # Pass environment with OMP_NUM_THREADS set
            )

            # Stream output in real-time
            print("ACORN process started. Streaming output:")
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    line_stripped = line.rstrip()
                    output_lines.append(line_stripped)
                    # Print all output for visibility
                    print(f"  {line_stripped}")
                    sys.stdout.flush()

            # Wait for completion and check return code
            return_code = process.wait()
            if return_code != 0:
                print(f"ERROR: ACORN failed with return code {return_code}")
                if output_lines:
                    print("\nLast 20 lines of output:")
                    for line in output_lines[-20:]:
                        print(f"  {line}")
                sys.exit(1)

            print("ACORN run completed successfully.")

            # Wait a moment for file I/O to complete
            import time
            time.sleep(1)

            # Verify output file exists before trying to read results
            acorn_data_path = data_root_dir + "/acorn_data"
            qps_file = acorn_data_path + "/all_qps.bin"
            if not os.path.exists(qps_file):
                print(f"ERROR: Expected output file not found: {qps_file}")
                print("ACORN may have failed silently. Check output above for details.")
                sys.exit(1)

            #  save results
            results = compute_acorn_results(acorn_data_path)
            output_file = save_results(
                results, data_hash, args.search_mode)
        else:
            # Load dataset
            print("Loading dataset...")
            dataset = load_dataset(
                args.data_src_path)
            print(f"Dataset loaded: {dataset.train_vectors.shape[0]} training vectors, "
                  f"{dataset.test_vectors.shape[0]} test vectors")
            print(
                f"Attribute dimensions (10 columns: all 0s except last is 1): {dataset.train_labels.shape[1]}")

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
