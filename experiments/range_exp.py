import h5py
import hnswlib
import faiss
import hashlib
from pathlib import Path
import numpy as np
import time
import pandas as pd
BASE_DIR = "/data/anas.aitaomar/foodvec_v2"
DATA_LIMIT = 1_001_000
QUERY_SAMPLE_SIZE = 1000
LABEL_MIN = 1
LABEL_MAX = 10_000
SELECTIVITY_LOW = 0.01
SELECTIVITY_HIGH = 0.5

# Caching and configuration constants
USE_CACHE = False
BASE_PATH = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_PATH / ".cache"

# Encoding choice (single toggle)
USE_BINARY_BUCKETING = True

# HNSW index/search params (rwalks)
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 500
HNSW_NUM_THREADS = 48
RWALKS_SEARCH_MODE = "rwalks"  # {'rwalks','hnsw-inline','stf'}
RWALKS_PRUN_FACTOR = 0.0
RWALKS_EF_SEARCH = 10

# Ground-truth (FAISS) params
GROUND_TRUTH_FAISS_METRIC = 'l2'  # {'l2','ip'}
FINAL_K = 10
CANDIDATE_K = 1200


def load_data(data_limit: int, query_sample_size: int):
    data = np.load(f"{BASE_DIR}/vectors.npy", mmap_mode="r")
    data = data[:data_limit]
    query_indices = np.random.choice(
        data.shape[0], size=query_sample_size, replace=False)
    train_indices = np.setdiff1d(np.arange(data.shape[0]), query_indices)
    query_data = data[query_indices]
    train_data = data[train_indices]
    return train_data, query_data


def generate_labels(train_data_size: int, label_min: float = 1, label_max: float = 10_000):
    return np.random.uniform(low=label_min, high=label_max, size=train_data_size)


def generate_ranges(query_data_size: int, s_low: float, s_high: float, label_min: float = 1, label_max: float = 10_000):
    """
    Generate (low, high) range tuples for query_data_size queries,
    with range width (selectivity) randomly sampled between s_low and s_high.
    Selectivity is interpreted as a fraction of the label space [label_min, label_max].
    Returned as a (query_data_size, 2) ndarray.
    """
    label_range = label_max - label_min
    # Generate normalized positions and widths
    lows_normalized = np.random.uniform(
        low=0, high=1 - s_high, size=query_data_size)
    widths_normalized = np.random.uniform(
        low=s_low, high=s_high, size=query_data_size)
    highs_normalized = lows_normalized + widths_normalized

    # Scale to actual label range [label_min, label_max]
    lows = lows_normalized * label_range + label_min
    highs = highs_normalized * label_range + label_min
    ranges = np.stack([lows, highs], axis=1)
    return ranges


def get_naive(
    data_labels: np.ndarray,
    query_ranges: np.ndarray,
    label_min: float,
    label_max: float,
    smallest_specificity: float,
):
    """
    Bucketize labels and ranges into one-hot 2D arrays.

    - Data labels: exactly one 1 per row (point bucket)
    - Query ranges: contiguous 1s covering the buckets overlapped by [low, high]

    smallest_specificity is a fraction in (0, 1], interpreted as the minimum
    resolvable fraction of the label space. The bucket size is
    (label_max - label_min) * smallest_specificity.

    Returns (data_one_hot, query_one_hot).
    """
    label_range = label_max - label_min
    if smallest_specificity <= 0 or smallest_specificity > 1:
        raise ValueError("smallest_specificity must be in (0, 1]")

    bucket_size = label_range * smallest_specificity
    # Ensure at least one bucket and cover the full range
    num_buckets = int(np.ceil(label_range / bucket_size))
    bucket_size = label_range / num_buckets  # adjust to evenly partition

    # ----- Data labels → one-hot -----
    # Map each label to a bucket index in [0, num_buckets-1]
    label_indices = np.floor(
        (data_labels - label_min) / bucket_size).astype(int)
    label_indices = np.clip(label_indices, 0, num_buckets - 1)

    data_one_hot = np.zeros(
        (data_labels.shape[0], num_buckets), dtype=np.uint8)
    data_one_hot[np.arange(data_labels.shape[0]), label_indices] = 1

    # ----- Query ranges → contiguous one-hot -----
    lows = query_ranges[:, 0]
    highs = query_ranges[:, 1]
    # Inclusive coverage of the high end: use ceil(high) - 1
    start_idx = np.floor((lows - label_min) / bucket_size).astype(int)
    end_idx = np.ceil((highs - label_min) / bucket_size).astype(int) - 1

    start_idx = np.clip(start_idx, 0, num_buckets - 1)
    end_idx = np.clip(end_idx, 0, num_buckets - 1)

    query_one_hot = np.zeros(
        (query_ranges.shape[0], num_buckets), dtype=np.uint8)
    for i in range(query_ranges.shape[0]):
        s = start_idx[i]
        e = end_idx[i]
        if e >= s:
            query_one_hot[i, s: e + 1] = 1

    return data_one_hot, query_one_hot


def get_dyadic_bucketing(
    data_labels: np.ndarray,
    query_ranges: np.ndarray,
    label_min: float,
    label_max: float,
    smallest_specificity: float,
):
    """
    Dyadic (binary) bucketing using a full binary partition up to depth D,
    where 2^D buckets approximate the smallest specificity.

    - For each label: activate the path from root to leaf (one node per depth).
    - For each range: activate the minimal set of dyadic nodes whose union
      exactly covers the leaf-index range (canonical segment-tree cover).
    Returns (data_one_hot, query_one_hot).
    """
    if smallest_specificity <= 0 or smallest_specificity > 1:
        raise ValueError("smallest_specificity must be in (0, 1]")

    label_range = label_max - label_min
    # Depth so that 1/2^D <= smallest_specificity
    depth = int(np.ceil(np.log2(1.0 / smallest_specificity))
                ) if smallest_specificity < 1 else 0
    num_leaf = 1 << depth
    bucket_size = label_range / num_leaf if num_leaf > 0 else label_range
    total_nodes = 2 * num_leaf - 1 if num_leaf > 0 else 1

    def value_to_leaf_idx(v: float) -> int:
        if num_leaf == 0:
            return 0
        idx = int(np.floor((v - label_min) / bucket_size))
        return int(np.clip(idx, 0, num_leaf - 1))

    def leaf_path_indices(leaf_idx: int):
        # Depth-major flattening: offset(d) = 2^d - 1, position i = floor(leaf / 2^{D-d})
        for d in range(depth + 1):
            block = 1 << (depth - d) if depth - d >= 0 else 1
            i = leaf_idx // block if block > 0 else 0
            yield (1 << d) - 1 + i

    def range_to_nodes(l_idx: int, r_idx: int):
        # Canonical segment-tree cover over [l_idx, r_idx] inclusive
        if num_leaf == 0:
            return [0]
        N = num_leaf
        L = l_idx + N
        R = r_idx + 1 + N
        nodes = []
        while L < R:
            if L & 1:
                nodes.append(L)
                L += 1
            if R & 1:
                R -= 1
                nodes.append(R)
            L //= 2
            R //= 2
        # Map segment-tree node index to flattened (depth-major) index
        out = []
        for idx in nodes:
            d = int(np.floor(np.log2(idx)))  # depth with root at 1
            start_at_depth = 1 << d
            i = idx - start_at_depth
            flat = (1 << d) - 1 + i
            out.append(flat)
        return out

    data_one_hot = np.zeros(
        (data_labels.shape[0], total_nodes), dtype=np.uint8)
    for j, v in enumerate(data_labels):
        leaf = value_to_leaf_idx(float(v))
        for flat_idx in leaf_path_indices(leaf):
            data_one_hot[j, flat_idx] = 1

    query_one_hot = np.zeros(
        (query_ranges.shape[0], total_nodes), dtype=np.uint8)
    for q in range(query_ranges.shape[0]):
        low, high = float(query_ranges[q, 0]), float(query_ranges[q, 1])
        l_idx = value_to_leaf_idx(low)
        r_idx = value_to_leaf_idx(high)
        if r_idx < l_idx:
            l_idx, r_idx = r_idx, l_idx
        for flat_idx in range_to_nodes(l_idx, r_idx):
            query_one_hot[q, flat_idx] = 1

    return data_one_hot, query_one_hot


# Removed alias to simplify API; use get_dyadic_bucketing directly


def avg_active_ones_per_query(query_one_hot: np.ndarray) -> float:
    """Return the mean count of active 1s per query row."""
    if query_one_hot.size == 0:
        return 0.0
    return float(np.count_nonzero(query_one_hot, axis=1).mean())


def build_hnsw_index(
    train_vectors: np.ndarray,
    data_attr: np.ndarray,
    data_scalar_labels: np.ndarray,
    M: int = HNSW_M,
    ef_construction: int = HNSW_EF_CONSTRUCTION,
    num_threads: int | None = HNSW_NUM_THREADS,
):
    """
    Build an HNSW index with attribute dimension equal to data_attr.shape[1].
    Returns a configured index with items added.
    """
    if num_threads is None:
        num_threads = HNSW_NUM_THREADS
    index = hnswlib.Index(
        space='l2',
        dim=train_vectors.shape[1],
        dim_attr=data_attr.shape[1]
    )
    index.init_index(
        max_elements=train_vectors.shape[0],
        ef_construction=ef_construction,
        M=M,
    )
    index.set_num_threads(num_threads)
    index.add_items(data=train_vectors,
                    data_attr=data_attr,
                    data_scalar_labels=data_scalar_labels)
    return index


# Removed index caching to keep logic simple and explicit; index is always rebuilt


def search_hnsw_index(
    index: hnswlib.Index,
    query_vectors: np.ndarray,
    query_attr: np.ndarray,
    raw_query_ranges: np.ndarray,
    k: int = FINAL_K,
    ef_search: int | None = RWALKS_EF_SEARCH,
    prun_factor: float | None = RWALKS_PRUN_FACTOR,
    num_threads: int | None = HNSW_NUM_THREADS,
):
    """
    Retrieve candidate_k from HNSW using attribute-aware search, then post-filter
    by raw ranges on labels to return top final_k per query in original order.
    Returns (neighbors, distances) with shape (num_queries, final_k).
    Pads with -1/inf if fewer than final_k matches.
    """

    index.set_search_mode(search_mode=0)
    index.set_pron_factor(prun_factor)
    index.set_num_threads(num_threads)
    index.set_ef(ef_search)
    cand_idxs, cand_dists = index.knn_query(
        data=query_vectors,
        data_attr=query_attr,
        ranges=raw_query_ranges,
        k=k,
    )

    return cand_idxs, cand_dists


def build_faiss_index(
    train_vectors: np.ndarray,
    metric: str = GROUND_TRUTH_FAISS_METRIC,
):
    """
    Build a FAISS flat index for ground-truth candidate generation.
    metric in {'l2', 'ip'}.
    """
    dim = train_vectors.shape[1]
    if metric == 'l2':
        index = faiss.IndexFlatL2(dim)
    elif metric == 'ip':
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError("metric must be 'l2' or 'ip'")
    index.add(train_vectors.astype(np.float32))
    return index


def faiss_search_postfilter(
    index: faiss.Index,
    query_vectors: np.ndarray,
    raw_train_labels: np.ndarray,
    raw_query_ranges: np.ndarray,
    final_k: int = FINAL_K,
    candidate_k: int = CANDIDATE_K,
):
    """
    FAISS search to get candidate_k nearest neighbors by vector distance only,
    then post-filter by raw query ranges on labels to return top final_k.
    Returns (neighbors, distances); pads with -1/inf if needed.
    """
    cd, ci = index.search(query_vectors.astype(np.float32), candidate_k)
    num_q = query_vectors.shape[0]
    out_idxs = -np.ones((num_q, final_k), dtype=np.int64)
    out_dists = np.full((num_q, final_k), np.inf, dtype=np.float32)

    labels = raw_train_labels.astype(np.float64)
    lows = raw_query_ranges[:, 0].astype(np.float64)
    highs = raw_query_ranges[:, 1].astype(np.float64)

    for qi in range(num_q):
        low = lows[qi]
        high = highs[qi]
        for idx, dist in zip(ci[qi], cd[qi]):
            if idx < 0:
                continue
            val = labels[idx]
            if low <= val <= high:
                w = int(np.count_nonzero(out_idxs[qi] >= 0))
                if w < final_k:
                    out_idxs[qi, w] = idx
                    out_dists[qi, w] = dist
                if w + 1 == final_k:
                    break

    return out_idxs, out_dists


def _fingerprint_arrays(*arrays: np.ndarray) -> str:
    """Create a short fingerprint from the shapes and bytes of arrays."""
    h = hashlib.sha256()
    for arr in arrays:
        h.update(str(arr.shape).encode())
        h.update(arr.dtype.str.encode())
        # Use a small sample to avoid huge hashing cost
        sample = arr.reshape(-1)[: min(arr.size, 1000)].tobytes()
        h.update(sample)
    return h.hexdigest()[:16]


# Removed faiss_ground_truth_cached to centralize caching in main


def recall_at_k(retrieved_list, ground_truth: np.ndarray) -> float:
    recalls = []
    for r, g in zip(retrieved_list, ground_truth):
        g_valid = set(g[g >= 0])
        if not g_valid:
            recalls.append(1.0)
            continue
        inter = len(g_valid.intersection(set(r.tolist())))
        recalls.append(inter / len(g_valid))
    return float(np.mean(recalls)) if recalls else 0.0


def read_hdf5_dataset(filepath, keys):
    with h5py.File(filepath, "r") as f:
        ret = []
        for k in keys:
            ret.append(f[k][:])
    return ret


if __name__ == "__main__":
    # data_unify_path = "/data/anas.aitaomar/foodvec_v2/unify_store/foodvec_v2_with_scalar.hdf5"
    # data_unify_path = "/data/anas.aitaomar/sift/unify_store/sift-128-euclidean_with_scalar.hdf5"
    # data_unify_path = "/home/anas.aitaomar/sift/unify_store/sift50M_with_scalar.hdf5"
    # data_unify_path = "/home/anas.aitaomar/sift/unify_store/sift10M_with_scalar.hdf5"
    # data_unify_path = "/home/anas.aitaomar/sift/unify_store/sift-128-euclidean_with_scalar.hdf5"
    data_unify_path = "/home/anas.aitaomar/sift/unify_store/sift25M_with_scalar.hdf5"
    (
        train_data,
        train_labels,
        query_data,
        ranges,
        test_hybrid_knn,
    ) = read_hdf5_dataset(
        data_unify_path,
        ["base", "base_scalars", "test", "test_ranges", "test_hybrid_knn"],
    )
    # 1) Data generation
    # train_data, query_data = load_data(DATA_LIMIT, QUERY_SAMPLE_SIZE)

    # Centralized caching for labels, ranges, and encodings
    CACHE_DIR.mkdir(exist_ok=True)
    prep_key = (
        f"prep-trn={_fingerprint_arrays(train_data)}-"
        f"qry={_fingerprint_arrays(query_data)}-"
        f"lblrange={LABEL_MIN}-{LABEL_MAX}-sel={SELECTIVITY_LOW}-{SELECTIVITY_HIGH}"
    )
    labels_path = CACHE_DIR / f"{prep_key}.labels.npy"
    ranges_path = CACHE_DIR / f"{prep_key}.ranges.npy"
    naive_data_path = CACHE_DIR / f"{prep_key}.enc-naive.data.npy"
    naive_query_path = CACHE_DIR / f"{prep_key}.enc-naive.query.npy"
    bin_data_path = CACHE_DIR / f"{prep_key}.enc-dyadic.data.npy"
    bin_query_path = CACHE_DIR / f"{prep_key}.enc-dyadic.query.npy"

    # Labels and ranges
    if USE_CACHE and labels_path.exists() and ranges_path.exists():
        train_labels = np.load(labels_path)
        ranges = np.load(ranges_path)
    else:
        # train_labels = generate_labels(
        #     train_data.shape[0], LABEL_MIN, LABEL_MAX)
        # ranges = generate_ranges(
        #     query_data.shape[0], SELECTIVITY_LOW, SELECTIVITY_HIGH, LABEL_MIN, LABEL_MAX)
        train_labels = train_labels
        ranges = ranges
        if USE_CACHE:
            np.save(labels_path, train_labels)
            np.save(ranges_path, ranges)

    # Encodings
    have_naive = naive_data_path.exists() and naive_query_path.exists()
    have_bin = bin_data_path.exists() and bin_query_path.exists()

    if USE_CACHE and have_naive and have_bin:
        data_naive = np.load(naive_data_path)
        query_naive = np.load(naive_query_path)
        data_bin = np.load(bin_data_path)
        query_bin = np.load(bin_query_path)
    else:
        data_naive, query_naive = get_naive(
            data_labels=train_labels,
            query_ranges=ranges,
            label_min=min(train_labels),
            label_max=max(train_labels),
            smallest_specificity=min(
                ranges[:, 1] - ranges[:, 0]) / max(train_labels),
        )
        data_bin, query_bin = get_dyadic_bucketing(
            data_labels=train_labels,
            query_ranges=ranges,
            label_min=min(train_labels),
            label_max=max(train_labels),
            smallest_specificity=min(
                ranges[:, 1] - ranges[:, 0]) / max(train_labels),
        )
        if USE_CACHE:
            np.save(naive_data_path, data_naive)
            np.save(naive_query_path, query_naive)
            np.save(bin_data_path, data_bin)
            np.save(bin_query_path, query_bin)

    # Choose encoding for index/search
    if USE_BINARY_BUCKETING:
        data_attr, query_attr = data_bin, query_bin
        enc_name = "binary"
    else:
        data_attr, query_attr = data_naive, query_naive
        enc_name = "naive"

    # 3) Ground truth via FAISS (with centralized caching block)
    gt_start = time.time()
    CACHE_DIR.mkdir(exist_ok=True)
    gt_key = (
        f"faissgt-m={GROUND_TRUTH_FAISS_METRIC}-fk={FINAL_K}-ck={CANDIDATE_K}-"
        f"trn={_fingerprint_arrays(train_data)}-"
        f"qry={_fingerprint_arrays(query_data)}-"
        f"lab={_fingerprint_arrays(train_labels)}-"
        f"rng={_fingerprint_arrays(ranges)}"
    )
    gt_idx_path = CACHE_DIR / f"{gt_key}.idx.npy"
    gt_dst_path = CACHE_DIR / f"{gt_key}.dst.npy"

    if USE_CACHE and gt_idx_path.exists() and gt_dst_path.exists():
        gt_idxs = np.load(gt_idx_path)
        gt_dists = np.load(gt_dst_path)
    else:
        f_index = build_faiss_index(
            train_data, metric=GROUND_TRUTH_FAISS_METRIC)
        gt_idxs, gt_dists = faiss_search_postfilter(
            f_index,
            query_data,
            train_labels,
            ranges,
            final_k=FINAL_K,
            candidate_k=CANDIDATE_K,
        )
        if USE_CACHE:
            np.save(gt_idx_path, gt_idxs)
            np.save(gt_dst_path, gt_dists)
    gt_time = time.time() - gt_start

    print(test_hybrid_knn[:2])
    print(gt_idxs[:2])

    # 4) Build HNSW index (rwalks) using binary bucketing attributes
    idx_build_start = time.time()
    index = build_hnsw_index(
        train_vectors=train_data,
        data_attr=data_attr,
        data_scalar_labels=train_labels,
        M=HNSW_M,
        ef_construction=HNSW_EF_CONSTRUCTION,
        num_threads=HNSW_NUM_THREADS,
    )
    idx_build_time = time.time() - idx_build_start

    # Logs
    print(f"data loaded: train {train_data.shape}, query {query_data.shape}")
    print(f"labels: {train_labels.shape}, ranges: {ranges.shape}")
    print(
        f"naive enc: data {data_naive.shape}, query {query_naive.shape}, avg1sQ={avg_active_ones_per_query(query_naive):.2f}")
    print(
        f"binary enc: data {data_bin.shape}, query {query_bin.shape}, avg1sQ={avg_active_ones_per_query(query_bin):.2f}")
    print(
        f"ground truth built in {gt_time:.3f}s; idx build {idx_build_time:.3f}s;")
    print(f"used encoding: {enc_name}")

    # 5) Search (rwalks) with post-filter using raw ranges; query attributes from binary bucketing
    print("--------------------------------")
    print("recall@10 vs ef_search")
    print("--------------------------------")
    logs = []
    for ef_search in range(10, 2000, 20):
        for pron_factor in [-10.0, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]:
            srch_start = time.time()
            nn_idxs, nn_dists = search_hnsw_index(
                index=index,
                query_vectors=query_data,
                query_attr=query_attr,
                raw_query_ranges=ranges,
                k=FINAL_K,
                ef_search=ef_search,
                prun_factor=pron_factor,
                num_threads=1,
            )
            # print(nn_idxs[0])
            # print(gt_idxs[0])
            search_time = time.time() - srch_start
            qps = query_data.shape[0] / \
                search_time if search_time > 0 else float('inf')
            recall = recall_at_k(nn_idxs, gt_idxs)
            print(f"ef_search={ef_search}, qps={qps:.2f}, recall={recall:.4f}")
            logs.append({
                "ef": ef_search,
                "al": pron_factor,
                "recall": recall,
                "latency(ms)": search_time * 1000,
                "QPS": qps,
            })
    df = pd.DataFrame(logs)
    path = CACHE_DIR / \
        f"range_exp_{enc_name}_pf_{RWALKS_PRUN_FACTOR}_efc_{HNSW_EF_CONSTRUCTION}_bucketing_{'binary' if USE_BINARY_BUCKETING else 'naive'}_max_pron_M_{HNSW_M}_sift25M.csv"
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
