# RWalks Reproduce: Experimental Comparison of Vector Search Methods


This repository contains code to reproduce experimental results comparing different ANN indices across various specificity levels (RWalks, HNSW-inline, STF, ACORN-1 and ACORN-G).


## Table of Contents
- [Paper Information](#-paper-information)
- [Data Download](#-data-download)
- [Installation](#-installation)
- [Running Experiments](#-running-experiments)
- [Visualization](#-visualization)
- [Unfiltered Search Experiments](#-unfiltered-search-experiments)
- [Search Methods](#-search-methods)
- [Configuration](#-configuration)

## Paper Information

### Title
RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search
### Authors
Anas Ait Aomar, Karima Echihabi, Marco Arnaboldi, Ioannis Alagiannis, Damien Hilloulin, Manal Cherkaoui
### Abstract
Analytical tasks in various domains increasingly encode complex information as dense vector data (e.g., embeddings), often requiring filtered vector search (i.e., vector search with attribute filtering). This search is challenging due to the volume and dimensionality of the data, the number and variety of filters, and the difference in distribution and/or update frequency between vectors and filters. Besides, many real applications require answers in a few milliseconds with high recall on large collections. Graph-based methods are considered the best choice for such applications, despite a lack of theoretical guarantees on query accuracy. Existing solutions for filtered vector search are either: 1) ad-hoc, using existing techniques with no or minor modifications; or 2) hybrid, providing specialized indexing and/or search algorithms. We show that neither is satisfactory and propose RWalks, an index-agnostic graph-based filtered vector search method that efficiently supports both filtered and unfiltered vector search. We demonstrate its scalability and robustness against the state-of-the-art with an exhaustive experimental evaluation on four real datasets (up to 100 million vectors), using query workloads with filters of different types (unique/composite), and varied specificity (proportion of points that satisfy a filter). The results show that RWalks can perform filtered search up to 2x faster than the second-best competitor (ACORN), while building the index 76x faster and answering unfiltered search 13x faster.

[Download paper ->](https://github.com/AnasAito/rwalks-reproduce-v2/blob/master/RWalks_paper_public.pdf)

## Data Download

Before running experiments, you need to download the required datasets:

### Available Datasets

1. **SIFT-1M**: `sift_1m_old_dist.h5` 
2. **YFCC-10M**: `yfcc10m_old_dist.h5`

### Download Instructions

1. Download the datasets:
   - **SIFT-1M**: [Download from MEGA](https://mega.nz/file/H1hnXDIK#i_F9chhKiLU3lABfyKXH22AKfK1cwX10k6pztu1jKv4)
   - **YFCC-10M**: [Download from MEGA](https://mega.nz/file/TsIDhACT#xbiaR659J2ec3P4KubmbRvtLub09TcLsdr-Eu5bomb0)
2. **Remember the full path** to your data files as you'll need it for running experiments.

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AnasAito/rwalks-reproduce
cd rwalks-reproduce
```

### 2. Create and Activate Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate 
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

Test that everything is installed correctly:
```bash
python -c "import hnswlib; print('RWalks installed successfully!')"
```

## 🧪 Running Experiments

### Basic Usage

Run experiments using the `specificity.py` script with your dataset and chosen search method:

```bash
python experiments/specificity.py --data_src_path /path/to/your/data/sift_1m_old_dist.h5 --search_mode <method>
```

### Methods supported

```bash
# Test with RWalks method
python experiments/specificity.py --data_src_path /data/anas.aitaomar/sift_1m_old_dist.h5 --search_mode rwalks

# Test with HNSW baseline
python experiments/specificity.py --data_src_path /data/anas.aitaomar/sift_1m_old_dist.h5 --search_mode hnsw-inline

# Test with STF method
python experiments/specificity.py --data_src_path /data/anas.aitaomar/sift_1m_old_dist.h5 --search_mode stf

# Test with ACORN methods
python experiments/specificity.py --data_src_path /data/anas.aitaomar/sift_1m_old_dist.h5 --search_mode acorn-1
python experiments/specificity.py --data_src_path /data/anas.aitaomar/sift_1m_old_dist.h5 --search_mode acorn-g
```

### What Happens During Experiments

Each experiment will:

1. **Load the dataset** from your specified `.h5` file
2. **Build search indices** 
3. **Run queries** across multiple specificity levels (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
4. **Test various EF values** (10-500 for most methods, 10-50 for HNSW-Inline as it saturates)
5. **Measure performance** (queries per second, recall)
6. **Save results** to `data/specificity_experiment_{dataset}_{method}.csv`



## 📊 Visualization

### Generate Performance Plots

After running experiments, visualize the results:

```bash
python experiments/specificity_plot.py --data_src_path /path/to/your/data/sift_1m_old_dist.h5
```



## 🔍 Search Methods

### Available Methods

| Method | Description |
|--------|-------------|
| `rwalks` | Random Walks method (our approach) |
| `hnsw-inline` | HNSW Baseline |
| `stf` | Search then Filter Method |
| `acorn-1` | ACORN (γ=1) |
| `acorn-g` | ACORN (γ=10) |

## 🔧 Configuration

### Environment Variables

You can customize experiments by setting environment variables.

**Note**: Make sure to choose a thread count appropriate for your machine. 

```bash
# Threading
export NUM_THREADS=32

# RWalks (and baselines using HNSW) parameters
export RWALKS_EF_CONSTRUCTION=100
export RWALKS_M=16
export RWALKS_PRUN_FACTOR=0.0

# ACORN parameters
export ACORN_GAMMA=10
export ACORN_M=16
export ACORN_MB=32
```


### RWalk Params

#### Indexing Parameters

Test the impact of different random walk indexing parameters (depth and walk count):

```bash
python experiments/rwalks-params.py \
    --data_src_path /path/to/your/data/sift_1m_old_dist.h5 \
    --depth_values 1,3,5 \
    --walk_values 10,20,50
```

**Example**:
```bash
python experiments/rwalks-params.py \
    --depth_values 1,3,5 \
    --walk_values 10,20,50 \
    --data_src_path /home/anas.aitaomar/rwalks-reproduce-v2/sift_1m_old_dist.h5
```

This will:
- Test varying **depth** values (1, 3, 5) while keeping walks fixed at the default (10)
- Test varying **walk count** values (10, 20, 50) while keeping depth fixed at the default (3)
- Run each search 4 times and average QPS for stable measurements
- Generate two plots: one for depth experiments, one for walk experiments
- Save results to `plots/rwalks_params_experiment_{dataset}.csv`
- Save plots to `plots/rwalks_depth_experiment_{dataset}.png` and `plots/rwalks_walks_experiment_{dataset}.png`

#### Search Parameters + Ablation

Test the impact of different pruning factor values during search:

```bash
python experiments/rwalks-search-params.py \
    --data_src_path /path/to/your/data/sift_1m_old_dist.h5 \
    --prun_factor_values -10,0.0,0.01,0.05
```
Note : a negative pron factor means that we are testing disabled prunning

This will:
- Test different **pruning factor** values (-10 for disabled, 0.0, 0.01, 0.05)
- Use the default indexing parameters (depth=3, walks=10)
- Generate a plot showing QPS-Recall curves for different pruning factors
- Save results to `plots/rwalks_search_params_experiment_{dataset}.csv`
- Save plot to `plots/rwalks_search_params_experiment_{dataset}.png`


### Hardware Requirements

- Experiments with 1M dataset were tested on a machine with 16GB RAM
- Experiments with 10M dataset were tested on a Linux machine with 128GB RAM


## 🔎 Unfiltered Search Experiments

This section describes how to run unfiltered search experiments (vector search without attribute filtering).

### Step 1: Prepare Unfiltered Dataset

First, create an unfiltered dataset from your original dataset:

```bash
python experiments/prep-unf-dataset.py \
    --src /path/to/your/data/sift_1m_old_dist.h5 \
    --dst /path/to/your/data/sift_1m_unf.h5 \
    -k 100
```

**What this does:**
- Creates a new HDF5 file with unfiltered attributes (10 columns: all zeros except last column is 1)
- Recomputes ground truth neighbors using FAISS.

**Parameters:**
- `--src`: Path to source HDF5 file (your original dataset)
- `--dst`: Path to destination HDF5 file (will be created)
- `-k`: Number of nearest neighbors to compute (default: 100)
- `--query-batch`: Query batch size for FAISS search (default: 1000)

**Example:**
```bash
python experiments/prep-unf-dataset.py \
    --src /data/anas.aitaomar/sift_1m_old_dist.h5 \
    --dst /data/anas.aitaomar/sift_1m_unf.h5 \
    -k 100
```

### Step 2: Run Unfiltered Search Experiments

Run experiments for each method on the unfiltered dataset:

```bash
# RWalks
python experiments/unf_search.py \
    --data_src_path /path/to/your/data/sift_1m_unf.h5 \
    --search_mode rwalks

# HNSW Baseline
python experiments/unf_search.py \
    --data_src_path /path/to/your/data/sift_1m_unf.h5 \
    --search_mode hnsw-inline

# STF Method
python experiments/unf_search.py \
    --data_src_path /path/to/your/data/sift_1m_unf.h5 \
    --search_mode stf

# ACORN Methods
python experiments/unf_search.py \
    --data_src_path /path/to/your/data/sift_1m_unf.h5 \
    --search_mode acorn-1

python experiments/unf_search.py \
    --data_src_path /path/to/your/data/sift_1m_unf.h5 \
    --search_mode acorn-g
```

**What this does:**
- Builds the search index for the chosen method
- Tests various EF values 
- Measures performance (queries per second, recall)
- Saves results to `data/unf_search_experiment_{dataset}_unf_{method}.csv`

### Step 3: Generate Performance Plot

After running experiments for all methods, visualize the results:

```bash
python experiments/unf_plot.py --dataset sift_1m
```




### Complete Example Workflow

Here's a complete workflow for running unfiltered search experiments on SIFT-1M:

```bash
# 1. Prepare unfiltered dataset
python experiments/prep-unf-dataset.py \
    --src /data/anas.aitaomar/sift_1m_old_dist.h5 \
    --dst /data/anas.aitaomar/sift_1m_unf.h5

# 2. Run experiments for each method
python experiments/unf_search.py --data_src_path /data/anas.aitaomar/sift_1m_unf.h5 --search_mode rwalks
python experiments/unf_search.py --data_src_path /data/anas.aitaomar/sift_1m_unf.h5 --search_mode hnsw-inline
python experiments/unf_search.py --data_src_path /data/anas.aitaomar/sift_1m_unf.h5 --search_mode stf
python experiments/unf_search.py --data_src_path /data/anas.aitaomar/sift_1m_unf.h5 --search_mode acorn-1
python experiments/unf_search.py --data_src_path /data/anas.aitaomar/sift_1m_unf.h5 --search_mode acorn-g

# 3. Generate plot
python experiments/unf_plot.py --dataset sift_1m
```

**Output:**
- CSV files in `data/` directory with detailed results for each method
- PNG plot comparing all methods: `data/qps_vs_recall_unf_sift_1m.png`

