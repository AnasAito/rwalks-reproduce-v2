import os
import time
from typing import List, Tuple

import numpy as np

from acornpy import ACORNIndex


def make_blob(N: int, d: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xb = rng.normal(0, 1.0, size=(N, d)).astype(np.float32)
    return xb


def make_queries(nq: int, d: int, seed: int = 456) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xq = rng.normal(0, 1.0, size=(nq, d)).astype(np.float32)
    return xq


def make_filter_bitmap(nq: int, ntotal: int, allow_frac: float, ensure_k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bitmap = (rng.random((nq, ntotal)) < allow_frac).astype(np.uint8)
    if ensure_k > 0:
        for i in range(nq):
            cnt = int(bitmap[i].sum())
            if cnt < ensure_k:
                need = ensure_k - cnt
                idx = rng.choice(ntotal, size=need, replace=False)
                bitmap[i, idx] = 1
    return bitmap


def brute_force_knn_filtered(xq: np.ndarray, xb: np.ndarray, k: int, bitmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    qn = (xq * xq).sum(axis=1, keepdims=True)
    bn = (xb * xb).sum(axis=1, keepdims=True).T
    D = qn + bn - 2.0 * (xq @ xb.T)
    D_masked = np.where(bitmap.astype(bool), D, np.inf)
    idx_part = np.argpartition(D_masked, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(xq.shape[0])[:, None]
    topk_vals = D_masked[row_idx, idx_part]
    order = np.argsort(topk_vals, axis=1)
    I = idx_part[row_idx, order]
    Dk = topk_vals[row_idx, order]
    return Dk.astype(np.float32, copy=False), I.astype(np.int64, copy=False)


def recall_at_k(I_true: np.ndarray, I_pred: np.ndarray) -> float:
    k = I_true.shape[1]
    inter = 0
    for a, b in zip(I_true, I_pred):
        inter += len(set(a.tolist()).intersection(b.tolist()))
    return inter / (I_true.shape[0] * k)


def main():
    # Config (override via env vars)
    N0 = 100_000
    N_ADD = 60_000
    CYCLES = 5
    nq = 200
    k = 10
    allow_frac_base = 0.05
    threads = 72
    ef = 100
    M = 32
    M_beta = 64
    gamma = 20

    # Optional dataset path: if present, load via memmap; otherwise fall back to synthetic
    data_path = os.environ.get(
        "ACORN_DATA_PATH",
        "/data/anas.aitaomar/yfcc/yfcc_data.npy",
    )

    # Operation sequence per cycle: list of 'add' or 'update'
    ops_env = os.environ.get(
        "ACORN_CYCLES_OPS", "add,update,add,update,update")
    ops: List[str] = [op.strip() for op in ops_env.split(",") if op.strip()]
    if len(ops) < CYCLES:
        # pad with 'update'
        ops += ["update"] * (CYCLES - len(ops))

    # Data source selection
    xb_all: np.ndarray
    if os.path.exists(data_path):
        print(f"Loading dataset via memmap: {data_path}")
        xb_all = np.load(data_path, mmap_mode="r")  # shape: (N, d)
        if xb_all.ndim != 2:
            raise ValueError("Loaded npy must be 2D (N, d)")
        d = int(xb_all.shape[1])
        total_capacity = int(xb_all.shape[0])
        # Queries: take a sequential slice following the initial base block
        start_q = min(N0, total_capacity - 1)
        end_q = min(start_q + nq, total_capacity)
        if end_q - start_q < nq:
            # If not enough tail space, just take the first nq
            start_q, end_q = 0, min(nq, total_capacity)
        xq = np.array(xb_all[start_q:end_q], dtype=np.float32, copy=False)
    else:
        print("No dataset found at ACORN_DATA_PATH; using synthetic data")
        # Pre-generate a large blob to draw from for adds
        d = 128
        total_capacity = N0 + CYCLES * N_ADD
        xb_all = make_blob(total_capacity, d, seed=123)
        xq = make_queries(nq, d, seed=456)

    # Build index and add initial block
    # Preallocate capacity to the dataset size to allow sequential adds
    index = ACORNIndex(d, M=M, gamma=gamma, M_beta=M_beta,
                       metric="l2", capacity=int(total_capacity))
    index.set_num_threads(threads)

    cur_N = N0
    t0 = time.perf_counter()
    index.add(np.array(xb_all[:cur_N], dtype=np.float32, copy=False))
    t1 = time.perf_counter()
    print(f"Initial add: N0={N0} in {t1 - t0:.3f}s, ntotal={index.ntotal}")

    # Initial filter and truth
    bitmap = make_filter_bitmap(
        nq, cur_N, allow_frac=allow_frac_base, ensure_k=k, seed=789)
    D_true, I_true = brute_force_knn_filtered(xq, np.array(
        xb_all[:cur_N], dtype=np.float32, copy=False), k, bitmap)

    # Run a baseline search
    tb0 = time.perf_counter()
    D, I = index.search(xq, k, filter_ids=bitmap, ef_search=ef)
    tb1 = time.perf_counter()
    print(
        f"Cycle -1 (baseline): ef={ef} recall@{k}={recall_at_k(I_true, I):.4f} QPS={nq / (tb1 - tb0):.1f}")

    # Cycles: apply op, recompute ground truth, measure search
    for c in range(CYCLES):
        print(f"Cycle {c}: {ops[c]}")
        op = ops[c]
        if op == "add":
            next_end = min(cur_N + N_ADD, int(total_capacity))
            block = np.array(xb_all[cur_N:next_end],
                             dtype=np.float32, copy=False)
            tadd0 = time.perf_counter()
            index.add(block)
            tadd1 = time.perf_counter()
            cur_N = next_end
            print(
                f"Cycle {c}: add {block.shape[0]} -> N={cur_N} ({tadd1 - tadd0:.3f}s)")
            # expand bitmap for new ntotal
            bitmap = make_filter_bitmap(
                nq, cur_N, allow_frac=allow_frac_base, ensure_k=k, seed=789 + c + 1)
        else:
            # update permitted ids: regenerate bitmap with slightly different allow_frac
            allow_frac = min(
                0.95, max(0.05, allow_frac_base * (1.0 + 0.1 * ((-1) ** c))))
            bitmap = make_filter_bitmap(
                nq, cur_N, allow_frac=allow_frac, ensure_k=k, seed=789 + c + 1)
            print("bitmap:")
            print((bitmap.sum(axis=1) / bitmap.shape[1]).mean())
            print(f"Cycle {c}: update labels (allow_frac={allow_frac:.2f})")

        # Recompute ground truth for current state
        gt0 = time.perf_counter()
        D_true, I_true = brute_force_knn_filtered(
            xq, np.array(xb_all[:cur_N], dtype=np.float32, copy=False), k, bitmap)
        gt1 = time.perf_counter()

        # Search
        ts0 = time.perf_counter()
        D, I = index.search(xq, k, filter_ids=bitmap, ef_search=ef)
        ts1 = time.perf_counter()

        rec = recall_at_k(I_true, I)
        qps = nq / (ts1 - ts0)
        print(
            f"Cycle {c}: ef={ef} recall@{k}={rec:.4f} QPS={qps:.1f} (gt {gt1 - gt0:.3f}s)"
        )


if __name__ == "__main__":
    main()
