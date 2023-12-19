#!/bin/bash

set -ev

###########################
# Get DPCPP
###########################
wget --no-verbose https://github.com/intel/llvm/releases/download/nightly-2023-12-06/sycl_linux.tar.gz -O dpcpp-compiler.tar.gz
rm -rf /tmp/dpcpp && mkdir /tmp/dpcpp/
tar -xzf dpcpp-compiler.tar.gz -C /tmp/dpcpp --strip-components 1
ls -R /tmp/dpcpp/
