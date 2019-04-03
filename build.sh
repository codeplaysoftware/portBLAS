#! /bin/bash
# Based heavily on build.sh from the SYCL Parallel STL

function display_help() {

cat <<EOT

To use build.sh to compile sycl-blas with ComputeCpp:

  ./build.sh "path/to/ComputeCpp"
  (the path to ComputeCpp can be relative)

  For example:
  ./build.sh /home/user/ComputeCpp


To use build.sh to compile sycl-blas with a specific blas installation (e.g. OpenBLAS):

  ./build.sh "path/to/ComputeCpp" "path/to/blas"

  For example:
  ./build.sh /home/user/ComputeCpp /tmp/OpenBLAS/build

EOT
}

# Useless to go on when an error occurs
set -o errexit

# Minimal emergency case to display the help message whatever happens
trap display_help ERR

# Get the absolute path of the ComputeCpp_DIR, from arg $1
if [ -z "$1" ]
  then
  echo "No ComputeCPP Package specified."
  exit 1
else
  CCPPPACKAGE=$(readlink -f $1)
  echo "ComputeCPP specified at: $CCPPPACKAGE"
  CMAKE_ARGS="$CMAKE_ARGS -DComputeCpp_DIR=$CCPPPACKAGE"
  shift
fi

# Test to see if an OpenBLAS installation has been given
if [ -z "$1" ]
  then
  echo "Using CMake to find a BLAS installation"
else
  OPENBLASROOT=$(readlink -f $1)
  echo "User specified OpenBLAS at: $OPENBLASROOT"
  CMAKE_ARGS="$CMAKE_ARGS -DSYSTEM_BLAS_ROOT=$OPENBLASROOT"
fi

echo "Making args"

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
    pushd build/test
    ctest -VV --timeout 1200
    popd
}

function main {
    configure
    mak
    tst
}

main
