#!/bin/bash

set -ev

###########################
# Get Intel OpenCL Runtime
###########################

# PACKAGE_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/12556/opencl_runtime_16.1.2_x64_rh_6.4.0.37.tgz
# PACKAGE_NAME=opencl_runtime_16.1.2_x64_rh_6.4.0.37

# PACKAGE_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/9019/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz
# PACKAGE_NAME=opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25

PACKAGE_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/13793/l_opencl_p_18.1.0.013.tgz
PACKAGE_NAME=l_opencl_p_18.1.0.013

wget -q ${PACKAGE_URL} -O /tmp/opencl_runtime.tgz
tar -xzf /tmp/opencl_runtime.tgz -C /tmp
sed 's/decline/accept/g' -i /tmp/${PACKAGE_NAME}/silent.cfg
apt-get install -yq cpio
/tmp/${PACKAGE_NAME}/install.sh -s /tmp/${PACKAGE_NAME}/silent.cfg
