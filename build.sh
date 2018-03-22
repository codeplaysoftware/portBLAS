mkdir build
cd build
cmake .. -DCOMPUTECPP_PACKAGE_ROOT_DIR=$1 -DCMAKE_BUILD_TYPE=Release -DOPENBLAS_ROOT=/tmp/OpenBLAS/build
make
COMPUTECPP_TARGET="host" make test
