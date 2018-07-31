#!/bin/bash

set -ev

CONAN_COMPUTECPP_COMMIT=fcec4a598934323101106369b9cd7bd2e2ca3934

wget -q https://github.com/mmha/conan-computecpp/archive/${CONAN_COMPUTECPP_COMMIT}.tar.gz
tar xf ${CONAN_COMPUTECPP_COMMIT}.tar.gz
cd conan-computecpp-${CONAN_COMPUTECPP_COMMIT}/

wget -q ${COMPUTECPP_X86_64_URL} -O ComputeCpp-CE-0.9.1-Ubuntu-14.04-64bit.tar.gz

conan create . codeplay/testing -osycl_language=False
cd ..