mkdir build
cd build
cmake .. -DCOMPUTECPP_PACKAGE_ROOT_DIR=$1 -DCMAKE_BUILD_TYPE=Release -DBLAS_DIR=/tmp/OpenBLAS/build
make
make test
