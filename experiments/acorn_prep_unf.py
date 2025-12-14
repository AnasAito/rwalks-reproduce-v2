
from tqdm import tqdm
from dataclasses import dataclass
import h5py
import os
import json
import numpy as np


@dataclass
class DataSet:
    train: np.ndarray
    test: np.ndarray
    neighbors: np.ndarray
    train_attr: np.ndarray
    test_attr: np.ndarray


def h5_to_memory(exp_env_name):
    """Load unfiltered dataset with 10 attribute columns (all 0s except last column is 1)."""
    with h5py.File(exp_env_name, 'r') as dataset:
        # Load datasets into memory
        train = np.array(dataset["train_vectors"])
        test = np.array(dataset["test_vectors"])
        test_attr = np.array(dataset["test_attr_vectors_0"])
        neighbors = np.array(dataset["neighbors_0"])
        train_attr = np.array(dataset["train_attr_vectors"])

        return DataSet(
            train=train,
            test=test,
            train_attr=train_attr,
            test_attr=test_attr,
            neighbors=neighbors,
        )


def write_fvecs(file_path, vectors):
    """Write vectors in fvecs format."""
    # Ensure the array is of type float32
    vectors = vectors.astype(np.float32)

    # Open the file in binary write mode
    with open(file_path, 'wb') as f:
        # For each vector
        for vector in vectors:
            # Write the dimension (as int32)
            dim = np.int32(len(vector))
            f.write(dim.tobytes())

            # Write the vector data as float32
            f.write(vector.tobytes())


def create_permitted_ids_optimized(train_attrs, test_attrs):
    """
    Create permitted IDs for unfiltered search.
    Since all train and test vectors have 1 in the last column (column 9), all training vectors match all queries.
    """
    # For unfiltered search, all train vectors are permitted for all test queries
    # Since both train_attr and test_attr have 1 in the last column (and 0s elsewhere)
    # all vectors match on the last attribute
    permitted_ids = np.ones(
        (test_attrs.shape[0], train_attrs.shape[0]), dtype=np.uint8)

    return permitted_ids


def create_acorn_data(data_src, data_dst):
    """
    Create ACORN data files for unfiltered search.
    No specificity segmentation - process the full workload.
    """
    # create data_dst if it doesn't exist
    if not os.path.exists(data_dst):
        os.makedirs(data_dst)

    # load data
    print("Loading unfiltered dataset...")
    data = h5_to_memory(data_src)

    print(f"Train vectors: {data.train.shape}")
    print(f"Test vectors: {data.test.shape}")
    print(f"Train attributes: {data.train_attr.shape}")
    print(f"Test attributes: {data.test_attr.shape}")
    print(f"Neighbors: {data.neighbors.shape}")

    # write train vectors
    print("Writing base vectors...")
    write_fvecs(f'{data_dst}/base.fvecs', data.train)

    # For unfiltered search, create single query file (no specificity segmentation)
    print("Creating query and filter data...")

    # Create permitted IDs (all 1s for unfiltered search)
    print("Creating permitted IDs...")
    permited_ids = create_permitted_ids_optimized(
        data.train_attr, data.test_attr)

    # Write single query file
    permited_ids.tofile(f'{data_dst}/filter_ids_map_0.bin')
    write_fvecs(f'{data_dst}/query_0.fvecs', data.test)
    np.save(f'{data_dst}/gt_0.npy', data.neighbors)

    print(f"Permitted IDs shape: {permited_ids.shape}")
    print(f"Data preparation complete!")
    print(f"Output directory: {data_dst}")

    return {
        "num_vecs": data.train.shape[0],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare ACORN data for unfiltered search')
    parser.add_argument('--data_src', type=str, required=True,
                        help='Path to source HDF5 dataset')
    parser.add_argument('--data_dst', type=str, required=True,
                        help='Path to destination directory for ACORN data')

    args = parser.parse_args()

    create_acorn_data(args.data_src, args.data_dst)
