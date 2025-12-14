#!/usr/bin/env python3

import argparse
import os
import h5py
import numpy as np
import faiss


def ensure_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        return x.astype(np.float32, copy=False)
    return x


def build_unfiltered_dataset(
    src_path: str,
    dst_path: str,
    k: int = 100,
    query_batch: int = 1000,
):
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)

    if os.path.exists(dst_path):
        raise FileExistsError(f"Destination already exists: {dst_path}")

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # --- Load vectors ---
        train = ensure_float32(src["train_vectors"][:])
        test = ensure_float32(src["test_vectors"][:])

        n_train, dim = train.shape
        n_test = test.shape[0]

        # --- Copy vectors ---
        dst.create_dataset(
            "train_vectors",
            data=train,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        dst.create_dataset(
            "test_vectors",
            data=test,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        # --- Create unfiltered attribute vectors (10 columns: all 0s except last column is 1) ---
        num_attr_cols = 10
        train_attr = dst.create_dataset(
            "train_attr_vectors",
            shape=(n_train, num_attr_cols),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(min(200_000, n_train), num_attr_cols),
        )

        test_attr = dst.create_dataset(
            "test_attr_vectors_0",
            shape=(n_test, num_attr_cols),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(min(10_000, n_test), num_attr_cols),
        )

        # Fill train attrs in chunks (all 0s except last column is 1)
        chunk = min(200_000, n_train)
        for start in range(0, n_train, chunk):
            end = min(n_train, start + chunk)
            train_attr[start:end, :] = 0.0
            train_attr[start:end, -1] = 1.0

        test_attr[:, :] = 0.0
        test_attr[:, -1] = 1.0

        # --- Build FAISS index ---
        index = faiss.IndexFlatL2(dim)
        index.add(train)

        # --- Output datasets ---
        neighbors_ds = dst.create_dataset(
            "neighbors_0",
            shape=(n_test, k),
            dtype=np.int64,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(min(query_batch, n_test), k),
        )

        distances_ds = dst.create_dataset(
            "distances_0",
            shape=(n_test, k),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(min(query_batch, n_test), k),
        )

        # --- Search queries in batches ---
        for start in range(0, n_test, query_batch):
            end = min(n_test, start + query_batch)
            q = test[start:end]

            D, I = index.search(q, k)
            distances_ds[start:end, :] = D
            neighbors_ds[start:end, :] = I

            print(f"[INFO] Processed queries {start}:{end} / {n_test}")

    print(f"[DONE] Wrote unfiltered dataset to: {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create an unfiltered SIFT HDF5 dataset and recompute FAISS neighbors"
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Path to source HDF5 file",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Path to destination HDF5 file",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=100,
        help="Number of nearest neighbors (default: 100)",
    )
    parser.add_argument(
        "--query-batch",
        type=int,
        default=1000,
        help="Query batch size for FAISS search (default: 1000)",
    )

    args = parser.parse_args()

    build_unfiltered_dataset(
        src_path=args.src,
        dst_path=args.dst,
        k=args.k,
        query_batch=args.query_batch,
    )


if __name__ == "__main__":
    main()
