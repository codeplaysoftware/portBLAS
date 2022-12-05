#! /bin/bash
# Based heavily on build.sh from the SYCL Parallel STL

function display_help() {

cat <<EOT

To use build.sh to compile sycl-blas with ComputeCpp:

  ./build.sh --compiler computecpp --computecppdir "path to computecpp"
  (the path to ComputeCpp can be relative)

  For example:
  ./build.sh --compiler computecpp --computecppdir /home/user/ComputeCpp


To use build.sh to compile sycl-blas with a specific blas installation (e.g. OpenBLAS):

  ./build.sh --openblas "path/to/blas"

  For example:
  ./build.sh --openblas /tmp/OpenBLAS/build

EOT
}

# Useless to go on when an error occurs
set -o errexit

# Minimal emergency case to display the help message whatever happens
trap display_help ERR

# Setting default values
COMPILER=computecpp
OPENBLASROOT=openblas
CCPPPACKAGE=/tmp/computecpp-latest

while [ $# -gt 0 ]; do
  echo $1
  if [[ $1 == *"--compiler"* ]]; then
     COMPILER=$2
     echo "Compiler specified: $COMPILER"
     CMAKE_ARGS="$CMAKE_ARGS -DSYCL_COMPILER=$COMPILER"
     shift
  elif [[ $1 == *"--openblas"* ]]; then
     OPENBLASROOT=$(readlink -f $2)
     echo "User specified OpenBLAS at: $OPENBLASROOT"
     CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_PREFIX_PATH=$OPENBLASROOT"
     shift
  elif [[ $1 == *"--computecppdir"* ]]; then
     CCPPPACKAGE=$(readlink -f $2)
     echo "ComputeCPP specified at: $CCPPPACKAGE"
     CMAKE_ARGS="$CMAKE_ARGS -DComputeCpp_DIR=$CCPPPACKAGE"
     shift
  else
     echo "Invalid argument passed" $1
     exit 1
  fi
  shift
done
echo "Making args"

if [[ $COMPILER == "dpcpp" ]]; then
  CMAKE_ARGS="$CMAKE_ARGS -DBLAS_ENABLE_CONST_INPUT=OFF"
fi

CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release"

echo $CMAKE_ARGS

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
