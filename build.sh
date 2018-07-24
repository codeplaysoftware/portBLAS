#! /bin/bash
# Based heavily on build.sh from the SYCL Parallel STL

# Useless to go on when an error occurs
set -o errexit

# Minimal emergency case to display the help message whatever happens
trap display_help ERR

# Get the absolute path of the COMPUTECPP_PACKAGE_ROOT_DIR, from arg $1
if [ -z "$1"]
  then 
  echo "No ComputeCPP Package specified."
  exit 1
else
  CMAKE_ARGS="$CMAKE_ARGS -DCOMPUTECPP_PACKAGE_ROOT_DIR=$(readlink -f $1)"
  shift
fi

# Test to see if an OpenBLAS installation has been given
if [ -z "$1" ]
  then
  echo "Using CMake to find a BLAS installation"
  shift
else
  echo "Using user specified OpenBLAS from $1"
  CMAKE_ARGS="$CMAKE_ARGS -DOPENBLAS_ROOT=$(readlink -f $1)"
fi

CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release"

NPROC=$(nproc)

function configure {
    mkdir -p build && pushd build
    cmake .. $CMAKE_ARGS
    popd
}

function mak {
    pushd build && make -j$NPROC
    popd
}

function tst {
    pushd build/tests
    ctest -VV --timeout 60
    popd
}

function main {
    # install_gmock
    configure
    mak
    tst
}

main