mkdir build
cd build
cmake .. -DCOMPUTECPP_PACKAGE_ROOT_DIR=$1 -DCMAKE_BUILD_TYPE=Release -DVERBOSE=TRUE
make
ctest -VV
