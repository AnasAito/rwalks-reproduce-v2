#!/usr/bin/env bash

# run_acorn_macos.sh - macOS-only parameterized ACORN runner
# Usage: ./run_acorn_macos.sh <N> <gamma> <dataset> <M> <M_beta>
#
# Example:
#   ./run_acorn_macos.sh 1000000 1 sift50k 16 16

set -euo pipefail

############################################
# 1. OS CHECK (macOS only)
############################################
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script is intended for macOS (Darwin) only."
    exit 1
fi

############################################
# 2. ARGUMENTS
############################################
if [ $# -ne 5 ]; then
    echo "Usage: $0 <N> <gamma> <dataset> <M> <M_beta>"
    echo "Example: $0 1000000 1 sift50k 16 16"
    exit 1
fi

N=$1
gamma=$2
dataset=$3
M=$4
M_beta=$5

export debugSearchFlag=0

############################################
# 3. HOMEBREW & LIBOMP DETECTION
############################################

if ! command -v brew >/dev/null 2>&1; then
    echo "Error: Homebrew not found. Please install Homebrew from https://brew.sh first."
    exit 1
fi

BREW_PREFIX="$(brew --prefix)"

# Detect libomp installation (required for OpenMP with Apple Clang)
if [ -d "${BREW_PREFIX}/opt/libomp" ]; then
    OMP_PREFIX="${BREW_PREFIX}/opt/libomp"
else
    echo "Error: libomp not found at ${BREW_PREFIX}/opt/libomp."
    echo "Install it with: brew install libomp"
    exit 1
fi

############################################
# 4. OpenMP ENVIRONMENT VARIABLES (macOS)
############################################

export LDFLAGS="-L${OMP_PREFIX}/lib"
export CPPFLAGS="-I${OMP_PREFIX}/include"
export CXXFLAGS="-I${OMP_PREFIX}/include"

############################################
# 5. BUILD ACORN (macOS-specific CMake)
############################################

echo "Building ACORN with parameters: N=$N, gamma=$gamma, dataset=$dataset, M=$M, M_beta=$M_beta"

if [ ! -f "build/demos/test_acorn" ]; then
    echo "ACORN binary not found. Building..."

    cmake \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SKIP_INSTALL_RULES=ON \
        -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include" \
        -DOpenMP_C_LIB_NAMES="omp" \
        -DOpenMP_C_LIBRARIES="${OMP_PREFIX}/lib/libomp.dylib" \
        -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include" \
        -DOpenMP_CXX_LIB_NAMES="omp" \
        -DOpenMP_CXX_LIBRARIES="${OMP_PREFIX}/lib/libomp.dylib" \
        -DOpenMP_omp_LIBRARY="${OMP_PREFIX}/lib/libomp.dylib" \
        -DCMAKE_CXX_FLAGS="-I${OMP_PREFIX}/include" \
        -B build

    # Build Faiss & ACORN targets
    make -C build -j faiss
    make -C build test_acorn
else
    echo "ACORN binary already exists, skipping build..."
fi

############################################
# 6. OPENMP THREADS
############################################

export OMP_NUM_THREADS="${NUM_THREADS:-48}"
echo "ACORN THREADS: $OMP_NUM_THREADS"

############################################
# 7. OUTPUT DIRECTORY STRUCTURE
############################################

now="$(date +"%m-%d-%Y")"
parent_dir="${now}_${dataset}"
mkdir -p "${parent_dir}"

dir="${parent_dir}/MB${M_beta}"
mkdir -p "${dir}"

summary_file="${dir}/summary_sift_n=${N}_gamma=${gamma}.txt"

############################################
# 8. RUN ACORN
############################################

echo "Running ACORN test..."
TZ='America/Los_Angeles' date +"Start time: %H:%M" >> "${summary_file}" 2>&1

./build/demos/test_acorn "$N" "$gamma" "$dataset" "$M" "$M_beta" >> "${summary_file}" 2>&1

echo "ACORN test completed. Results saved to: ${summary_file}"