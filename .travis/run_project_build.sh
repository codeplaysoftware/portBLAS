#!/bin/bash
sudo apt update

.travis/install_computecpp.sh
.travis/install_intel_opencl.sh

conan create . codeplay/testing
