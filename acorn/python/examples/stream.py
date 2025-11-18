import os
import time
import json
import pickle
from typing import List, Optional, Tuple

import numpy as np

from acornpy import ACORNIndex


# --------------------
# Configuration (constants)
DATA_PATH = "/data/anas.aitaomar/foodvec_v2/vectors.npy"
# DATA_PATH = "/data/anas.aitaomar/yfcc/yfcc_data.npy"
NQ = 1000
D = 0  # 0 means infer from data
M = 32
M_BETA = 32
GAMMA = 1
THREADS = 72
EF_MAX = 2000
TOPK = 10
METRIC = "l2"


SELECTIVITY = 0.02

BASE_RECALL = 0.90
USE_FIRST_EF = True


# Cache
USE_CACHE = True
# CACHE_PATH = "/home/anas.aitaomar/GWAD/stream-logs/data_cache/data_0.02_random_5000000_80_50000_10_c551fd096d995b1cf269c5abcb281aa534f5c32c34c29f774a39b51f492af4df.pkl"
CACHE_PATH = "/home/anas.aitaomar/GWAD/stream-logs/data_cache/data_0.02_random_200000_30_50000_10_7898713e2460a2441fc14531e60aa3725fb4cf7fd7dbcc0d7fce57312fe1670d.pkl"
LOG_PATH = f"/home/anas.aitaomar/GWAD/stream-logs/stream_k{TOPK}_m{M}_m_beta{M_BETA}_gamma{GAMMA}_data_{hash(CACHE_PATH)}_debug.json"
# --------------------


# Optional rebuild comparison
# Define cycles at which to rebuild an index from scratch using active nodes
# and compare against the dynamic index with the fixed ef value.
REBUILD_CYCLES: List[int] = []  # e.g., [10, 50, 100]
REBUILD_USE_FIXED_EF = True


def recall_at_k_variable(I_true: np.ndarray, I_pred: np.ndarray, k_per_query: np.ndarray) -> float:
    inter = 0.0
    denom = 0.0
    for a, b, k_i in zip(I_true, I_pred, k_per_query):
        if k_i <= 0:
            # skip queries with no permitted ids
            continue
        a_set = set(a[:k_i].tolist())
        b_set = set(b[:k_i].tolist())
        inter += len(a_set.intersection(b_set))
        denom += float(k_i)
    if denom == 0.0:
        return 0.0
    return inter / denom


def recall_at_k_fixed(I_true: np.ndarray, I_pred: np.ndarray, k: int) -> float:
    inter = 0
    for a, b in zip(I_true, I_pred):
        inter += len(set(a[:k].tolist()).intersection(b[:k].tolist()))
    return inter / (I_true.shape[0] * k)


def brute_force_knn_filtered(xq: np.ndarray, xb: np.ndarray, k: int, bitmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Deprecated: not used in cached-only flow
    raise NotImplementedError


def make_permitted_from_attrs(train_attrs: np.ndarray, test_attrs: np.ndarray) -> Tuple[np.ndarray, float]:
    # permitted if training vector has 1s at all positions where test has 1
    nq = test_attrs.shape[0]
    nt = train_attrs.shape[0]
    permitted = np.zeros((nq, nt), dtype=np.uint8)
    avg_frac = 0.0
    # precompute positions of ones per test row
    test_positions: List[np.ndarray] = [
        np.where(test_attrs[i] == 1)[0] for i in range(nq)]
    for i, pos in enumerate(test_positions):
        if pos.size == 0:
            continue
        permitted[i] = np.all(train_attrs[:, pos] == 1,
                              axis=1).astype(np.uint8)
        avg_frac += permitted[i].sum() / float(nt)
    avg_frac = avg_frac / float(nq) if nq > 0 else 0.0
    return permitted, avg_frac


def get_attr_one_hot(vec_count: int, selectivity: float, *, seed: Optional[int] = None) -> np.ndarray:
    if not (0 < selectivity <= 1.0):
        raise ValueError("selectivity must be in (0, 1]")
    if seed is not None:
        np.random.seed(seed)
    num_classes = int(round(1.0 / selectivity))
    num_classes = max(1, num_classes)
    out = np.zeros((vec_count, num_classes), dtype=np.uint8)
    cols = np.random.randint(0, num_classes, size=vec_count)
    out[np.arange(vec_count), cols] = 1
    return out


def update_one_hot_rows(one_hot_vectors: np.ndarray, fraction: float, *, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    updated = one_hot_vectors.copy()
    n = updated.shape[0]
    m = int(max(0, min(1.0, fraction)) * n)
    if m == 0:
        return updated
    idx = np.random.choice(n, size=m, replace=False)
    shuf = idx.copy()
    np.random.shuffle(shuf)
    updated[idx] = updated[shuf]
    return updated


def try_bfindex_truth(xq: np.ndarray, xb: np.ndarray, k: int, bitmap: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # Deprecated: not used in cached-only flow
    return None


def find_ef_for_target(
    index: ACORNIndex,
    xq: np.ndarray,
    k: int,
    bitmap: np.ndarray,
    target_recall: float,
    ef_min: int,
    ef_max: int,
    *,
    I_true_external: np.ndarray,
    warmup_skip: int = 1,
    avg_runs: int = 5,
) -> Tuple[int, Optional[float], float]:
    # Binary search on ef to reach target recall using cached ground truth
    I_true = I_true_external
    left, right = max(k, ef_min), max(k, ef_max)
    best_ef = right
    best_rec = 0.0
    while left <= right:
        mid = (left + right) // 2
        index_ef = int(mid)
        t0 = time.perf_counter()
        _D, I_pred = index.search(xq, k, filter_ids=bitmap, ef_search=index_ef)
        t1 = time.perf_counter()
        rec = recall_at_k_fixed(I_true, I_pred, k)
        best_ef = index_ef
        best_rec = rec
        if rec < target_recall:
            left = mid + 1
        else:
            right = mid - 1

    # stabilize qps with repeated runs (skip warmup)
    qps_sum = 0.0
    for i in range(max(1, avg_runs)):
        t0 = time.perf_counter()
        _D, _I = index.search(xq, k, filter_ids=bitmap, ef_search=best_ef)
        t1 = time.perf_counter()
        if i >= warmup_skip:
            qps_sum += xq.shape[0] / (t1 - t0)
    denom = max(1, avg_runs - warmup_skip)
    qps = qps_sum / denom
    return best_ef, float(best_rec), qps


def report_rebuild_stats(
    d: int,
    m: int,
    m_beta: int,
    gamma: int,
    metric: str,
    threads: int,
    xq: np.ndarray,
    k: int,
    bitmap: np.ndarray,
    ef_search: int,
    train_vecs: np.ndarray,
    I_true: Optional[np.ndarray],
    runs: int = 5,
) -> Tuple[float, Optional[float], float]:
    """
    Build a fresh index from the provided train vectors and report:
      - qps: stabilized queries-per-second over `runs` (skips first run)
      - recall: if I_true provided, recall@k for the last run, else None
      - build_time: time to add all vectors to the fresh index
    """
    t0 = time.perf_counter()
    rebuild_index = ACORNIndex(d, M=m, gamma=gamma, M_beta=m_beta,
                               metric=metric, capacity=max(train_vecs.shape[0], 1))
    rebuild_index.set_num_threads(threads)
    if train_vecs.size:
        rebuild_index.add(train_vecs.astype(np.float32, copy=False))
    t1 = time.perf_counter()
    build_time = t1 - t0

    qps_sum = 0.0
    I_pred_last = None
    for i in range(max(1, runs)):
        ts0 = time.perf_counter()
        _D_r, I_pred_r = rebuild_index.search(
            xq, k, filter_ids=bitmap, ef_search=int(ef_search))
        ts1 = time.perf_counter()
        if i > 0:
            qps_sum += xq.shape[0] / (ts1 - ts0)
        I_pred_last = I_pred_r
    denom = max(1, runs - 1)
    qps = qps_sum / denom
    rec = recall_at_k_fixed(I_true, I_pred_last, k) if (
        I_true is not None and I_pred_last is not None) else None
    return qps, rec, build_time


def main():
    # Data
    data_path = DATA_PATH
    test_count = NQ
    # Index and search
    d = D  # if 0, infer from data
    m = M
    m_beta = M_BETA
    gamma = GAMMA
    threads = THREADS
    ef_max = EF_MAX
    k = TOPK
    metric = METRIC
    selectivity = SELECTIVITY
    # Recall targeting
    base_recall = BASE_RECALL
    use_first_ef = USE_FIRST_EF
    # cached-only flow; no BFIndex truth usage
    # Logging
    log_path = LOG_PATH
    rebuild_cycles = set(REBUILD_CYCLES)

    # data = h5_to_memory(DATA_PATH, selectivity=SELECTIVITY)
    vectors = np.load(DATA_PATH, mmap_mode='r')
    print("vectors.shape = ", vectors.shape)

    def get_attr_vector(vec_count, selectivity, seed=42):
        np.random.seed(seed)
        attr_init = np.zeros((vec_count, int(1/selectivity)), dtype=int)
        for i in range(vec_count):
            col = np.random.randint(0, attr_init.shape[1])
            attr_init[i, col] = 1
        return attr_init

    # VECS_COUNT = 7_000_000
    VECS_COUNT = 999_000
    xq = vectors[VECS_COUNT:VECS_COUNT+1000]

    # Test attributes (deterministic)
    test_attrs = get_attr_vector(1000, SELECTIVITY)
    d = xq.shape[1]

    # Cached flow: load events and use their GT/labels without regenerating
    if USE_CACHE and os.path.exists(CACHE_PATH):
        print(f"Loading cached data from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            all_data = pickle.load(f)

        # Capacity from add events
        capacity = 0
        for ev in all_data:
            if ev.get("event_type") == "add" and ev.get("train_vecs") is not None:
                capacity += int(np.array(ev["train_vecs"]).shape[0])

        index = ACORNIndex(d, M=m, gamma=gamma, M_beta=m_beta,
                           metric=metric, capacity=max(capacity, 1))
        index.set_num_threads(threads)

        cur_N = 0
        ef_90: Optional[int] = None
        log: List[dict] = []
        # Track full active dataset for rebuilds
        current_vecs = np.zeros((0, d), dtype=np.float32)
        current_attrbs = np.zeros(
            (0, int(round(1.0 / selectivity))), dtype=np.uint8)

        for c, ev in enumerate(all_data):
            ev_type = ev.get("event_type")
            print(f"Cycle {c}: {ev_type}")
            add_time = 0.0

            if ev_type == "add":
                train_vecs = np.asarray(
                    ev["train_vecs"], dtype=np.float32)
                attrb_to_use = np.asarray(
                    ev["labels"], dtype=np.uint8)
                t0 = time.perf_counter()
                index.add(train_vecs)
                t1 = time.perf_counter()
                add_time = t1 - t0
                cur_N += train_vecs.shape[0]
                # Maintain active vectors and attributes
                if current_vecs.size == 0:
                    current_vecs = train_vecs.copy()
                else:
                    current_vecs = np.concatenate(
                        [current_vecs, train_vecs], axis=0)
                if current_attrbs.size == 0:
                    current_attrbs = attrb_to_use.copy()
                else:
                    current_attrbs = np.concatenate(
                        [current_attrbs, attrb_to_use], axis=0)
                print(
                    f"  add {train_vecs.shape[0]} -> N={cur_N} ({add_time:.3f}s)")
            else:
                # replace labels wholesale from cache
                labels = np.asarray(ev["labels"], dtype=np.uint8)
                current_attrbs = labels
                print("  update labels from cache")

            # Build permitted bitmap and run search
            bitmap, spec = make_permitted_from_attrs(
                current_attrbs, test_attrs)
            I_true = np.asarray(ev["gt"], dtype=np.int64)

            if use_first_ef and ef_90 is not None:
                ef_use = int(ef_90)
                # stabilize qps
                qps_sum = 0.0
                runs = 5
                I_pred_last = None
                for i in range(runs):
                    ts0 = time.perf_counter()
                    _D, I_pred = index.search(
                        xq, k, filter_ids=bitmap, ef_search=ef_use)
                    ts1 = time.perf_counter()
                    if i > 0:
                        qps_sum += xq.shape[0] / (ts1 - ts0)
                    I_pred_last = I_pred
                qps = qps_sum / max(1, runs - 1)
                rec = recall_at_k_fixed(
                    I_true, I_pred_last, k) if I_pred_last is not None else None
            elif ev_type == "add":
                # Only determine ef on the first add, then reuse
                ef_use, rec, qps = find_ef_for_target(
                    index=index,
                    xq=xq,
                    k=k,
                    bitmap=bitmap,
                    target_recall=base_recall,
                    ef_min=max(k, 10),
                    ef_max=ef_max,
                    I_true_external=I_true,
                )
                if use_first_ef and ef_90 is None and rec is not None:
                    ef_90 = int(ef_use)
                    print(f"  saved ef_90={ef_90}")
            else:
                # Update before ef_90 exists: use ef_max as fallback
                ef_use = int(ef_max)
                qps_sum = 0.0
                runs = 5
                I_pred_last = None
                for i in range(runs):
                    ts0 = time.perf_counter()
                    _D, I_pred = index.search(
                        xq, k, filter_ids=bitmap, ef_search=ef_use)
                    ts1 = time.perf_counter()
                    if i > 0:
                        qps_sum += xq.shape[0] / (ts1 - ts0)
                    I_pred_last = I_pred
                qps = qps_sum / max(1, runs - 1)
                rec = recall_at_k_fixed(
                    I_true, I_pred_last, k) if I_pred_last is not None else None

            entry = {
                "ef": int(ef_use),
                "recall": None if rec is None else float(rec),
                "qps": float(qps),
                "itter_time": float(add_time),
                "itter_type": str(ev_type),
                "itter_idx": int(c),
            }

            # Optional rebuild-from-scratch comparison on configured cycles
            # Uses only currently active nodes (all adds up to this cycle)
            # and runs search with the fixed ef value for fairness.
            if c in rebuild_cycles:
                can_use_fixed_ef = (use_first_ef is False) or (
                    ef_90 is not None)
                if can_use_fixed_ef:
                    ef_rebuild = int(ef_90) if (
                        use_first_ef and ef_90 is not None and REBUILD_USE_FIXED_EF) else int(ef_use)
                    rebuild_qps, rebuild_rec, rebuild_build_time = report_rebuild_stats(
                        d=d,
                        m=m,
                        m_beta=m_beta,
                        gamma=gamma,
                        metric=metric,
                        threads=threads,
                        xq=xq,
                        k=k,
                        bitmap=bitmap,
                        ef_search=ef_rebuild,
                        train_vecs=current_vecs,
                        I_true=I_true,
                    )
                    entry.update({
                        "rebuild_cycle": True,
                        "rebuild_ef": int(ef_rebuild),
                        "rebuild_recall": None if rebuild_rec is None else float(rebuild_rec),
                        "rebuild_qps": float(rebuild_qps),
                        "rebuild_build_time": float(rebuild_build_time),
                    })
                else:
                    # Skip rebuild until fixed ef is established (if requested)
                    entry.update({
                        "rebuild_cycle": True,
                        "rebuild_ef": None,
                        "rebuild_recall": None,
                        "rebuild_qps": None,
                        "rebuild_build_time": None,
                    })
            else:
                entry.update({
                    "rebuild_cycle": False,
                    "rebuild_ef": None,
                    "rebuild_recall": None,
                    "rebuild_qps": None,
                    "rebuild_build_time": None,
                })
            print(entry)
            log.append(entry)

        # Save log and exit
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Saved log to {log_path}")
        return

    # Cache required
    raise FileNotFoundError(f"CACHE_PATH not found: {CACHE_PATH}")


if __name__ == "__main__":
    main()
