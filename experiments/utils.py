import struct
from dataclasses import dataclass
import numpy as np
import h5py


@dataclass
class Dataset:
    train_vectors: np.ndarray
    train_labels: np.ndarray
    test_vectors: np.ndarray
    test_labels: np.ndarray
    neighbors: np.ndarray


def load_dataset(raw_data_path):
    with h5py.File(raw_data_path, "r") as f:
        train_vectors = f["train_vectors"][()]
        train_labels = f["train_attr_vectors"][()]
        test_vectors = f["test_vectors"][()]
        test_labels = f["test_attr_vectors_0"][()]
        neighbors = f["neighbors_0"][()]
    return Dataset(train_vectors, train_labels, test_vectors, test_labels, neighbors)


def compute_recall(gt_neighbors, queries_neighbors, k=10):
    recalls = []
    for neighbors, true_neighbors in zip(queries_neighbors, gt_neighbors):
        recall_at_k = len(np.intersect1d(
            true_neighbors[:k], neighbors)) / min(k, len(true_neighbors))
        recalls.append(recall_at_k)
    return np.mean(recalls)


def read_neighbors_from_binary(file_path):
    with open(file_path, 'rb') as f:
        # Read nq and k
        nq = np.fromfile(f, dtype=np.int32, count=1)[0]
        k = np.fromfile(f, dtype=np.int32, count=1)[0]

        # Read the neighbors (nns2)
        neighbors = np.fromfile(f, dtype=np.int64, count=nq * k)

    return neighbors.reshape(nq, k)


def read_qps_from_binary(filepath="/Users/mac/dev/rwalks-reproduce/data/sift50k/all_qps.bin"):
    """
    Read QPS values from binary file and return as a Python list.

    Args:
        filepath (str): Path to the binary QPS file

    Returns:
        list: List of QPS values as floats
    """
    try:
        with open(filepath, 'rb') as file:
            # Read the number of QPS values (size_t, typically 8 bytes on 64-bit systems)
            num_qps_bytes = file.read(8)
            # 'Q' for unsigned long long (size_t)
            num_qps = struct.unpack('Q', num_qps_bytes)[0]

            # Read all QPS values as floats
            qps_bytes = file.read(num_qps * 4)  # 4 bytes per float
            qps_values = struct.unpack(
                f'{num_qps}f', qps_bytes)  # 'f' for float

            print(f"Successfully read {num_qps} QPS values from {filepath}")
            return list(qps_values)

    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return []
    except Exception as e:
        print(f"Error reading QPS file: {e}")
        return []


def read_distances_from_binary(file_path):
    with open(file_path, 'rb') as f:
        # Read nq and k
        nq = np.fromfile(f, dtype=np.int32, count=1)[0]
        k = np.fromfile(f, dtype=np.int32, count=1)[0]

        # Read the neighbors (nns2)
        neighbors = np.fromfile(f, dtype=np.float32, count=nq * k)

    return neighbors.reshape(nq, k)

# Usage


def compute_acorn_results(acorn_path):

    all_qps = read_qps_from_binary(
        acorn_path + "/all_qps.bin")
    efs = [10,  20,  30,  40,  45,  50,  55,  60,  70,   80,   90,
           120, 130, 140, 200, 300, 400, 600, 900, 1200, 1500, 2000]
    specificities = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    qps_per_spec = {}
    # Divide all_qps into lists by number of efs (per spec)
    num_efs = len(efs)
    num_specs = len(all_qps) // num_efs
    qps_per_spec = {}
    for spec_idx in range(num_specs):
        start = spec_idx * num_efs
        end = (spec_idx + 1) * num_efs
        qps_per_spec[spec_idx] = all_qps[start:end]

    results = []
    for spec_idx in range(num_specs):
        _neighbors = np.load(
            f"{acorn_path}/gt_{spec_idx}.npy")
        for idx, i in enumerate(efs):
            acorn_nn = read_neighbors_from_binary(
                f"{acorn_path}/01_nn_{i}_{spec_idx}.bin")
            acorn_dist = read_distances_from_binary(
                f"{acorn_path}/01_nn_dist{i}_{spec_idx}.bin")

            recalls = {'top10': []}
            k = 10
            for neighbors, true_neighbors in zip(acorn_nn, _neighbors):
                recall_at_10 = len(np.intersect1d(
                    true_neighbors[:k], neighbors)) / min(k, len(true_neighbors))
                recalls['top10'].append(recall_at_10)
            results.append(
                {
                    'specificity': specificities[spec_idx],
                    'ef': i,
                    'query_time': -1,
                    'qps': qps_per_spec[spec_idx][idx],
                    'recall':  round(np.mean(recalls['top10']), 3),
                    'recall_computation_time': -1,
                    'num_queries': -1,
                    'k': k
                }
            )
    return results
