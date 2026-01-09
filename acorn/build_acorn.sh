#!/bin/bash

# Build script for ACORN on macOS/Linux
# This script cleans any previous build and builds ACORN from scratch

set -e  # Exit on error

echo "=========================================="
echo "Building ACORN"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Clean previous build
echo "Cleaning previous build..."
rm -rf build

# Create build directory
echo "Creating build directory..."
mkdir build
cd build

# Detect OS and configure CMake accordingly
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS - using macOS-specific OpenMP configuration"
    
    # Check if Homebrew is available
    if ! command -v brew >/dev/null 2>&1; then
        echo "ERROR: Homebrew not found. Please install Homebrew from https://brew.sh first."
        exit 1
    fi
    
    BREW_PREFIX=$(brew --prefix)
    
    # Check if libomp is installed
    if [ ! -d "${BREW_PREFIX}/opt/libomp" ]; then
        echo "ERROR: libomp not found at ${BREW_PREFIX}/opt/libomp."
        echo "Install it with: brew install libomp"
        exit 1
    fi
    
    OMP_PREFIX="${BREW_PREFIX}/opt/libomp"
    echo "Using Homebrew prefix: $BREW_PREFIX"
    echo "Using OpenMP prefix: $OMP_PREFIX"
    
    # Configure CMake (matching run_acorn.sh)
    cmake .. \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SKIP_INSTALL_RULES=ON \
        -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include" \
        -DOpenMP_C_LIB_NAMES="omp" \
        -DOpenMP_C_LIBRARIES="${OMP_PREFIX}/lib/libomp.dylib" \
        -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include" \
        -DOpenMP_CXX_LIB_NAMES="omp" \
        -DOpenMP_CXX_LIBRARIES="${OMP_PREFIX}/lib/libomp.dylib" \
        -DOpenMP_omp_LIBRARY="${OMP_PREFIX}/lib/libomp.dylib" \
        -DCMAKE_CXX_FLAGS="-I${OMP_PREFIX}/include"
else
    echo "Detected Linux - using standard configuration"
    cmake .. \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBUILD_TESTING=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SKIP_INSTALL_RULES=ON
fi

# Build (matching run_acorn.sh)
echo "Building ACORN (this may take a few minutes)..."
if [[ "$(uname)" == "Darwin" ]]; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=$(nproc)
fi

# Build faiss library first (matching run_acorn.sh line 91)
echo "Building faiss library..."
make -j$CORES faiss

# Build test_acorn executable (matching run_acorn.sh line 92)
echo "Building test_acorn executable..."
make -j$CORES test_acorn

# Verify build (matching run_acorn.sh check)
if [[ -f "demos/test_acorn" ]]; then
    echo ""
    echo "=========================================="
    echo "✓ ACORN built successfully!"
    echo "=========================================="
    echo "Executable location: $(pwd)/demos/test_acorn"
    ls -lh demos/test_acorn
else
    echo ""
    echo "=========================================="
    echo "✗ Build failed - test_acorn not found"
    echo "=========================================="
    echo "Searched in: $(pwd)/demos/test_acorn"
    echo ""
    echo "Available files in demos/:"
    ls -la demos/ 2>/dev/null || echo "  (demos directory not found)"
    exit 1
fi

cd "$SCRIPT_DIR"

