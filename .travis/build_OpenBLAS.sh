##########################
# Get OpenBLAS
###########################
cd /tmp
git clone https://github.com/xianyi/OpenBLAS.git OpenBLAS
cd OpenBLAS
mkdir build
cd build
make -j8 -C ../
make PREFIX=/tmp/OpenBLAS/build install
