#!/bin/bash

# run_acorn.sh - Parameterized version of test.sh
# Usage: ./run_acorn.sh <N> <gamma> <dataset> <M> <M_beta>

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

# Set OpenMP environment variables for macOS
# export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
# export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
# export CXXFLAGS="-I/opt/homebrew/opt/libomp/include"

echo "Building ACORN with parameters: N=$N, gamma=$gamma, dataset=$dataset, M=$M, M_beta=$M_beta"

# Build if not already built
if [ ! -f "build/demos/test_acorn" ]; then
    echo "Building ACORN binary..."
    # cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
    #       -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    #       -DOpenMP_C_LIB_NAMES="omp" \
    #       -DOpenMP_C_LIBRARIES="/opt/homebrew/opt/libomp/lib/libomp.dylib" \
    #       -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    #       -DOpenMP_CXX_LIB_NAMES="omp" \
    #       -DOpenMP_CXX_LIBRARIES="/opt/homebrew/opt/libomp/lib/libomp.dylib" \
    #       -DOpenMP_omp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib" \
    #       -DCMAKE_CXX_FLAGS="-I/opt/homebrew/opt/libomp/include" \
    #       -B build
    cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build

    make -C build -j faiss
    make -C build utils
    make -C build test_acorn
else
    echo "ACORN binary already exists, skipping build..."
fi

export OMP_NUM_THREADS=${NUM_THREADS:-48}
echo "ACORN THREADS: $OMP_NUM_THREADS"

# Create output directory
now=$(date +"%m-%d-%Y")
parent_dir=${now}_${dataset}
mkdir -p ${parent_dir}
dir=${parent_dir}/MB${M_beta}
mkdir -p ${dir}

echo "Running ACORN test..."
TZ='America/Los_Angeles' date +"Start time: %H:%M" >> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt 2>&1

./build/demos/test_acorn $N $gamma $dataset $M $M_beta >> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt 2>&1

echo "ACORN test completed. Results saved to: ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt" 