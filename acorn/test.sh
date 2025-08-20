#!/bin/bash

export debugSearchFlag=0

# Set OpenMP environment variables for macOS
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export CXXFLAGS="-I/opt/homebrew/opt/libomp/include"

cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
      -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
      -DOpenMP_C_LIB_NAMES="omp" \
      -DOpenMP_C_LIBRARIES="/opt/homebrew/opt/libomp/lib/libomp.dylib" \
      -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
      -DOpenMP_CXX_LIB_NAMES="omp" \
      -DOpenMP_CXX_LIBRARIES="/opt/homebrew/opt/libomp/lib/libomp.dylib" \
      -DOpenMP_omp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib" \
      -DCMAKE_CXX_FLAGS="-I/opt/homebrew/opt/libomp/include" \
      -B build

make -C build -j faiss
make -C build utils
make -C build test_acorn

export OMP_NUM_THREADS=5

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

# run of sift1M test
N=1000000
gamma=1
dataset=sift50k 
M=16
M_beta=16

parent_dir=${now}_${dataset}
mkdir ${parent_dir}
dir=${parent_dir}/MB${M_beta}
mkdir ${dir}

TZ='America/Los_Angeles' date +"Start time: %H:%M" >> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt 2>&1

./build/demos/test_acorn $N $gamma $dataset $M $M_beta >> ${dir}/summary_sift_n=${N}_gamma=${gamma}.txt 2>&1

     





