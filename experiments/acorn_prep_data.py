
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


def h5_to_memory(exp_env_name, selectivity=None):

    with h5py.File(exp_env_name, 'r') as dataset:
        # for k in dataset.keys():
        #      print(k,dataset[k].shape,np.array(dataset[k]).sum(axis=1) if k == "test_attr_vectors_0" else "" )
        # Load datasets into memory
        train = np.array(dataset["train_vectors"])
        test = np.array(dataset["test_vectors"])  # [random_indices]
        # [random_indices]
        test_attr = np.array(dataset["test_attr_vectors_0"])
        neighbors = np.array(dataset["neighbors_0"])  # [random_indices]
        train_attr = np.array(dataset["train_attr_vectors"])

        if selectivity:
            # Mask test attributes based on selectivity
            column_idx_start, column_idx_end = selectivity

            # if not only_neighbors:
            train_attr = train_attr[:, column_idx_start:column_idx_end]
            test_attr = test_attr[:, column_idx_start:column_idx_end]

            # select valid queries
            valid_queries = np.argwhere(test_attr.sum(axis=1) > 0).ravel()
            # print(valid_queries[:10])
            test_attr = test_attr[valid_queries]
            test = test[valid_queries]
            neighbors = neighbors[valid_queries]

        return DataSet(
            train=train,
            test=test,
            train_attr=train_attr,
            test_attr=test_attr,
            neighbors=neighbors,
        )


# write test


def write_fvecs(file_path, vectors):
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
    # Find the positions of the active attribute (1) for each test point
    test_active_positions = np.argmax(test_attrs, axis=1)  # Shape (N_test,)

    # Initialize the permitted_ids array as np.bool_ to reduce memory usage
    permitted_ids = np.zeros(
        (test_attrs.shape[0], train_attrs.shape[0]), dtype=np.bool_)

    # Iterate through each test point and check for matching active attributes
    for i, pos in tqdm(enumerate(test_active_positions)):
        # Check if the training vectors have a 1 at the same position as the test point
        # Extract the column corresponding to the active test position
        permitted_ids[i] = train_attrs[:, pos]

    # Optionally convert the boolean array to uint8 (0 and 1) to save space
    return permitted_ids.astype(np.uint8)


def create_acorn_data(data_src, data_dst):
    # create data_dst if it doesn't exist
    if not os.path.exists(data_dst):
        os.makedirs(data_dst)
    # load data
    data = h5_to_memory(data_src, selectivity=None)
    # write train
    write_fvecs(f'{data_dst}/base.fvecs', data.train)
    # write queries + attrb
    for spec_idx, spec in enumerate([
        [0, 100],
        [100, 120],
        [120, 130],
        [130, 135],
        [135, 138],
        [138, 140]
    ]):
        data = h5_to_memory(data_src, selectivity=spec)
        permited_ids = create_permitted_ids_optimized(
            data.train_attr, data.test_attr)
        permited_ids.tofile(f'{data_dst}/filter_ids_map_{spec_idx}.bin')
        write_fvecs(f'{data_dst}/query_{spec_idx}.fvecs', data.test)
        np.save(f'{data_dst}/gt_{spec_idx}.npy', data.neighbors)

    return {
        "num_vecs": data.train.shape[0],
    }


if __name__ == "__main__":
    data_dir = "/Users/mac/Downloads/sift_1m_old_dist.h5"
    data_dst = "/Users/mac/dev/rwalks-reproduce/data/sift50k"
    create_acorn_data(data_dir, data_dst)
