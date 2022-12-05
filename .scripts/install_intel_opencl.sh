#!/bin/bash
# Copyright (C) Codeplay Software Limited. All Rights Reserved.
#
# Install Intel CPU OpenCL runtime
# Instructions from https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html#apt
#
# Note: the FPGA emulator device is disabled to keep only the Intel CPU one.

set -e

wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list
apt update
apt install -y intel-oneapi-runtime-libs intel-oneapi-runtime-opencl
rm /etc/OpenCL/vendors/intel64-fpgaemu.icd
apt clean && rm -rf /var/lib/apt/lists/*
