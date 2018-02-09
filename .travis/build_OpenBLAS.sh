##########################
# Get OpenBLAS
###########################
wget https://github.com/xianyi/OpenBLAS/archive/develop.zip -O /tmp/OpenBLAS.zip
unzip /tmp/OpenBLAS.zip -d /tmp/OpenBLAS  &> /dev/null
cd /tmp/OpenBLAS
mkdir build
make && make PREFIX=/tmp/OpenBLAS/build install
