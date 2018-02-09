mkdir build
cd build
cmake .. -DCOMPUTECPP_PACKAGE_ROOT_DIR=$1 -DCMAKE_BUILD_TYPE=Release
make
COMPUTECPP_TARGET="host" make test
