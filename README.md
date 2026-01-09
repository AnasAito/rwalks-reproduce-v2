

# RWalks Reproduce
Reproducibility package for **RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search**.

This repository reproduces the experimental comparison of filtered vector search methods across multiple filter specificity levels, and includes **end-to-end** scripts and a **fully runnable Kaggle notebook** (recommended) covering:
- Filtered search specificity experiments (RWalks, HNSW-inline, STF, ACORN-1, ACORN-G)
- Unfiltered (plain ANN) search experiments
- RWalks indexing-parameter sweep (walk depth, number of walks)
- RWalks search-parameter ablation (pruning factor)

## Zero-config, end-to-end reproduction (Recommended: Kaggle)

If you want to **run everything without local setup** (no dependency install, no manual dataset download/config), use the Kaggle notebook. It runs on a Kaggle machine with the dataset already available and contains **all experiments + plots** embedded directly after each experiment section.

**Kaggle notebook (full reproducibility, view or run):**  
https://www.kaggle.com/code/anasaitaomar/sigmod-25-ari-rwalks?scriptVersionId=286207306

**Repository branch used by the notebook:** `full-reproduce`

What you can do in Kaggle:
- Run all experiments end-to-end (or re-run individual sections)
- View generated recall–QPS plots immediately after each experiment block
- Export/download the produced CSVs and plots from the notebook output

## Contents
- [Paper information](#paper-information)
- [What’s included in this submission](#whats-included-in-this-submission)
- [Quickstart (local)](#quickstart-local)
- [Data download (local)](#data-download-local)
- [Installation (local)](#installation-local)
- [Reproducing the filtered specificity experiments](#reproducing-the-filtered-specificity-experiments)
- [Unfiltered search experiments](#unfiltered-search-experiments)
- [RWalks indexing-parameter sweep](#rwalks-indexing-parameter-sweep)
- [RWalks search-parameter ablation](#rwalks-search-parameter-ablation)
- [Visualization](#visualization)
- [Using custom datasets](#using-custom-datasets)
- [Configuration](#configuration)
- [Hardware notes](#hardware-notes)


## Paper information

**Title:** RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search  
**Authors:** Anas Ait Aomar, Karima Echihabi, Marco Arnaboldi, Ioannis Alagiannis, Damien Hilloulin, Manal Cherkaoui

**Abstract:**  
Analytical tasks in various domains increasingly encode complex information as dense vector data (e.g., embeddings), often requiring filtered vector search (i.e., vector search with attribute filtering). This search is challenging due to the volume and dimensionality of the data, the number and variety of filters, and the difference in distribution and/or update frequency between vectors and filters. Besides, many real applications require answers in a few milliseconds with high recall on large collections. Graph-based methods are considered the best choice for such applications, despite a lack of theoretical guarantees on query accuracy. Existing solutions for filtered vector search are either: 1) ad-hoc, using existing techniques with no or minor modifications; or 2) hybrid, providing specialized indexing and/or search algorithms. We show that neither is satisfactory and propose RWalks, an index-agnostic graph-based filtered vector search method that efficiently supports both filtered and unfiltered vector search. We demonstrate its scalability and robustness against the state-of-the-art with an exhaustive experimental evaluation on four real datasets (up to 100 million vectors), using query workloads with filters of different types (unique/composite), and varied specificity (proportion of points that satisfy a filter). The results show that RWalks can perform filtered search up to 2x faster than the second-best competitor (ACORN), while building the index 76x faster and answering unfiltered search 13x faster.

**Paper PDF:**  
https://github.com/AnasAito/rwalks-reproduce-v2/blob/master/RWalks_paper_public.pdf

## What’s included in this submission

### Experiments and scripts
- **Unfiltered search pipeline**
  - Script to prepare an unfiltered dataset variant (and recompute ground truth)
  - Script to run unfiltered search experiments across methods
  - Script to generate unfiltered search plots
- **RWalks indexing-parameter sweep**
  - Vary *walk depth* and *number of walks*, log recall–QPS, generate curves
- **RWalks search-parameter ablation**
  - Sweep pruning factor (including “disabled pruning”), report performance impact
- **Specificity experiments coverage**
  - RWalks vs baselines in the specificity experiment.

### Documentation
- README updated to clearly describe how to run all experiments (local + Kaggle)
- Guidance on running experiments on custom datasets (format expectations)

### Notebook update (Kaggle)
The Kaggle notebook includes:
- Unfiltered search: RWalks vs all baselines
- RWalks indexing-parameter sweep (depth / walks)
- RWalks search-parameter ablation (pruning factor)
- Specificity experiment with all baselines
- Dedicated plots shown immediately after each experiment section

## Quickstart (local)

If you prefer running locally, use the branch that corresponds to the full reproducibility package:

```bash
git clone -b full-reproduce https://github.com/AnasAito/rwalks-reproduce-v2.git
cd rwalks-reproduce-v2
```

### Data download (local)

Available datasets
	1.	SIFT-1M: sift_1m_old_dist.h5
	2.	YFCC-10M: yfcc10m_old_dist.h5

Download links
	•	SIFT-1M (MEGA): https://mega.nz/file/H1hnXDIK#i_F9chhKiLU3lABfyKXH22AKfK1cwX10k6pztu1jKv4
	•	YFCC-10M (MEGA): https://mega.nz/file/TsIDhACT#xbiaR659J2ec3P4KubmbRvtLub09TcLsdr-Eu5bomb0

Keep the full path to your .h5 file(s). You will pass it via --data_src_path.

### Installation (local)

1) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```
2) Install dependencies
```bash
pip install -r requirements.txt
```

### Reproducing the filtered specificity experiments

The main filtered evaluation runs queries across multiple specificity levels:
0.01, 0.05, 0.1, 0.2, 0.3, 0.5

Run a specificity experiment
```bash
python experiments/specificity.py \
  --data_src_path /path/to/sift_1m_old_dist.h5 \
  --search_mode <method>
```
Supported methods
- RWalks (our method)
- HNSW-inline
- STF (Search-Then-Filter)
- Acorn-1
- Acorn-g

Examples
```bash
python experiments/specificity.py --data_src_path /path/to/sift_1m_old_dist.h5 --search_mode rwalks
python experiments/specificity.py --data_src_path /path/to/sift_1m_old_dist.h5 --search_mode hnsw-inline
python experiments/specificity.py --data_src_path /path/to/sift_1m_old_dist.h5 --search_mode stf
python experiments/specificity.py --data_src_path /path/to/sift_1m_old_dist.h5 --search_mode acorn-1
python experiments/specificity.py --data_src_path /path/to/sift_1m_old_dist.h5 --search_mode acorn-g
```
Outputs

Each run writes a CSV: 
```bash
data/specificity_experiment_<dataset>_<method>.csv
```
The CSV contains recall and throughput (QPS) measured across EF values and specificity levels.


### Unfiltered search experiments

This reproduces vector search without attribute filtering.

Step 1: Prepare an unfiltered dataset
```bash
python experiments/prep-unf-dataset.py \
  --src /path/to/sift_1m_old_dist.h5 \
  --dst /path/to/sift_1m_unf.h5 \
  -k 100
```

What it does:
- Produces a new .h5 suitable for unfiltered experiments
- Recomputes ground truth neighbors using FAISS

Key parameters:
- --src source HDF5
- --dst output HDF5
- -k number of ground-truth neighbors (default: 100)
- --query-batch FAISS query batch size (default: 1000)

Step 2: Run unfiltered experiments
```bash
python experiments/unf_search.py --data_src_path /path/to/sift_1m_unf.h5 --search_mode rwalks
python experiments/unf_search.py --data_src_path /path/to/sift_1m_unf.h5 --search_mode hnsw-inline
python experiments/unf_search.py --data_src_path /path/to/sift_1m_unf.h5 --search_mode stf
python experiments/unf_search.py --data_src_path /path/to/sift_1m_unf.h5 --search_mode acorn-1
python experiments/unf_search.py --data_src_path /path/to/sift_1m_unf.h5 --search_mode acorn-g
```
Outputs:
	•	data/unf_search_experiment_<dataset>_unf_<method>.csv

Step 3: Plot unfiltered results

python experiments/unf_plot.py --dataset sift_1m

Output plot:
```bash
data/qps_vs_recall_unf_sift_1m.png
```

### RWalks indexing-parameter sweep

This experiment evaluates indexing-time parameters:
- walk depth
- number of walks

Run:
```bash
python experiments/rwalks-params.py \
  --data_src_path /path/to/sift_1m_old_dist.h5 \
  --depth_values 1,3,5 \
  --walk_values 10,20,50
```
What it produces:
- CSV: plots/rwalks_params_experiment_<dataset>.csv
- Depth plot: plots/rwalks_depth_experiment_<dataset>.png
- Walk plot: plots/rwalks_walks_experiment_<dataset>.png


### RWalks search-parameter ablation

This experiment studies the pruning factor effect during search.

Run:
```bash
python experiments/rwalks-search-params.py \
  --data_src_path /path/to/sift_1m_old_dist.h5 \
  --prun_factor_values -10,0.0,0.01,0.05
```
Interpretation:
- A negative pruning factor (e.g., -10) indicates pruning disabled.

Outputs:
- CSV: plots/rwalks_search_params_experiment_<dataset>.csv
- Plot: plots/rwalks_search_params_experiment_<dataset>.png


### Visualization

Filtered specificity plots:
```bash
python experiments/specificity_plot.py --data_src_path /path/to/sift_1m_old_dist.h5
```
(Additional plotting scripts are generated per experiment section as listed above.)

### Using custom datasets

RWalks expects:
- A vector array for the base dataset
- A binary metadata array (0/1) per point indicating attribute presence
- Query vectors and query metadata masks (1 for active attributes)

### Configuration

The experiments support environment variables for consistent configuration across runs.

- Threading
```bash
export NUM_THREADS=32
```
- RWalks (and baselines using HNSW) parameters
```bash
export RWALKS_EF_CONSTRUCTION=100
export RWALKS_M=16
export RWALKS_PRUN_FACTOR=0.0
```

- ACORN parameters
```bash
export ACORN_GAMMA=10
export ACORN_M=16
export ACORN_MB=32
```

### Hardware notes
- 1M-scale experiments were tested on a machine with 16GB RAM
- 10M-scale experiments were tested on a Linux machine with 128GB RAM



