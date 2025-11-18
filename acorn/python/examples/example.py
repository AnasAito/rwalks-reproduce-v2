import os
import time
import numpy as np

from acornpy import ACORNIndex


def make_synthetic_data(N: int, d: int, nq: int, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Two-cluster Gaussian blobs to make recall behavior meaningful
    centers = np.stack([
        np.zeros(d, dtype=np.float32),
        np.full(d, 2.0, dtype=np.float32),
    ])
    xb = np.empty((N, d), dtype=np.float32)
    xb[: N // 2] = centers[0] + \
        rng.normal(0, 1.0, size=(N // 2, d)).astype(np.float32)
    xb[N // 2:] = centers[1] + \
        rng.normal(0, 1.0, size=(N - N // 2, d)).astype(np.float32)

    # Query around both clusters
    xq = np.empty((nq, d), dtype=np.float32)
    q_half = nq // 2
    xq[:q_half] = centers[0] + \
        rng.normal(0, 1.0, size=(q_half, d)).astype(np.float32)
    xq[q_half:] = centers[1] + \
        rng.normal(0, 1.0, size=(nq - q_half, d)).astype(np.float32)

    return xb, xq


def make_filter_bitmap(nq: int, ntotal: int, allow_frac: float, ensure_k: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    bitmap = (rng.random((nq, ntotal)) < allow_frac).astype(np.uint8)
    # Ensure each query has at least ensure_k permitted ids
    for i in range(nq):
        if bitmap[i].sum() < ensure_k:
            # randomly flip on some ids
            need = ensure_k - int(bitmap[i].sum())
            idx = rng.choice(ntotal, size=need, replace=False)
            bitmap[i, idx] = 1
    return bitmap


def brute_force_knn_filtered(xq: np.ndarray, xb: np.ndarray, k: int, bitmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Compute squared L2 distances with masking
    # D = ||xq||^2 + ||xb||^2 - 2 xq xb^T
    qn = (xq * xq).sum(axis=1, keepdims=True)
    bn = (xb * xb).sum(axis=1, keepdims=True).T
    D = qn + bn - 2.0 * (xq @ xb.T)
    # Mask disallowed ids to +inf
    D_masked = np.where(bitmap.astype(bool), D, np.inf)
    # Top-k per row
    idx_part = np.argpartition(D_masked, kth=k - 1, axis=1)[:, :k]
    # Gather and sort within top-k
    row_idx = np.arange(xq.shape[0])[:, None]
    topk_vals = D_masked[row_idx, idx_part]
    order = np.argsort(topk_vals, axis=1)
    I = idx_part[row_idx, order]
    Dk = topk_vals[row_idx, order]
    return Dk.astype(np.float32, copy=False), I.astype(np.int64, copy=False)


def recall_at_k(I_true: np.ndarray, I_pred: np.ndarray) -> float:
    # Average recall@k across queries: |intersection|/k
    k = I_true.shape[1]
    inter = 0
    for a, b in zip(I_true, I_pred):
        inter += len(set(a.tolist()).intersection(b.tolist()))
    return inter / (I_true.shape[0] * k)


def main():
    # Config (override via env vars)
    N = 100_000
    d = 128
    nq = 200
    k = 10
    allow_frac = 0.3
    threads = 8

    xb, xq = make_synthetic_data(N, d, nq)
    index = ACORNIndex(d, M=16, gamma=1, M_beta=16, metric="l2")
    index.set_num_threads(threads)

    print(f"Adding N={N}, d={d}...")
    t0 = time.perf_counter()
    index.add(xb)
    t1 = time.perf_counter()
    print(f"Added in {t1 - t0:.3f}s, ntotal={index.ntotal}")

    # Filter bitmap and brute-force ground truth
    bitmap = make_filter_bitmap(
        nq, index.ntotal, allow_frac=allow_frac, ensure_k=k)
    print("Computing brute-force filtered ground truth...")
    t2 = time.perf_counter()
    D_true, I_true = brute_force_knn_filtered(xq, xb, k, bitmap)
    t3 = time.perf_counter()
    print(f"Brute-force time: {t3 - t2:.3f}s")

    # Sweep ef values and report recall/QPS
    ef_list = [10, 20, 30, 50, 100, 150, 200, 300, 400]
    results = []
    for ef in ef_list:
        tq0 = time.perf_counter()
        D_pred, I_pred = index.search(xq, k, filter_ids=bitmap, ef_search=ef)
        tq1 = time.perf_counter()
        qps = nq / (tq1 - tq0)
        rec = recall_at_k(I_true, I_pred)
        results.append((ef, rec, qps))
        print(f"ef={ef:<4d}  recall@{k}={rec:.4f}  QPS={qps:.1f}")

    # Optional plot if matplotlib present
    try:
        import matplotlib.pyplot as plt  # type: ignore

        efs = [r[0] for r in results]
        recalls = [r[1] for r in results]
        qps_vals = [r[2] for r in results]

        fig, ax1 = plt.subplots()
        color = "tab:blue"
        ax1.set_xlabel("ef_search")
        ax1.set_ylabel("Recall@k", color=color)
        ax1.plot(efs, recalls, marker="o", color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("QPS", color=color)
        ax2.plot(efs, qps_vals, marker="x", color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()


# steps to run the example:
# build
# cd /home/anas.aitaomar/rwalks-reproduce-v2/acorn
# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build
# make -C build -j faiss
# cd python
# pip install -e .
# cd ..
# python examples/example.py
