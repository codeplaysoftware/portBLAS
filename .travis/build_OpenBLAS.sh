##########################
# Get OpenBLAS
###########################
cd /tmp
git clone https://github.com/xianyi/OpenBLAS.git OpenBLAS
cd OpenBLAS
make 
mkdir build
make PREFIX=/tmp/OpenBLAS/build install
