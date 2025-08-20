# RWalks Reproduce: Experimental Comparison of Vector Search Methods

This repository contains code to reproduce experimental results comparing different ANN indices across various specificity levels (RWalks, HNSW-inline, STF, ACORN-1 and ACORN-G).

## 📋 Table of Contents

- [Data Download](#-data-download)
- [Installation](#-installation)
- [Running Experiments](#-running-experiments)
- [Visualization](#-visualization)
- [Search Methods](#-search-methods)
- [Configuration](#-configuration)

## 📥 Data Download

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

### Plot Features

The visualization script will:

- **Load all available results** from your experiments
- **Generate QPS vs Recall plots** for each specificity level
- **Show Pareto frontiers** comparing different methods
- **Save plots** to `data/qps_vs_recall_pareto_{dataset}.png`
You will get a plot similar to this:

![Example Plot](data/qps_vs_recall_pareto_sift_1m_old_dist.png)

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
### Hardware Requirements

- Experiments with 1M dataset were tested on a machine with 16GB RAM
- Experiments with 10M dataset were tested on a Linux machine with 128GB RAM

