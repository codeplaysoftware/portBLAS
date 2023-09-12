#!/bin/bash

# This script runs the portBLAS tests using the provided Dockerfile.
# The intention is to provide a seamless alternative to .travis.yml, so that
# developers can locally test changes in a (somewhat) platform-agnostic manner
# without the usual delay that travis testing entails.
#
# By default, this script will compile the portBLAS with open source DPCPP
# implementation. Other compilers can be enabled by changing the `CXX_COMPILER`
# and `CC_COMPILER` environment variables.
# Git "slug" are also equally configurable. By default the git repository
# cloned is codeplay's portBLAS master.

export IMPL=DPCPP
export CXX_COMPILER="/tmp/dpcpp/bin/clang++"
export CC_COMPILER="/tmp/dpcpp/bin/clang"
export GIT_SLUG="codeplaysoftware/portBLAS"
export GIT_BRANCH="master"
export COMMAND="build-test"


docker build --build-arg c_compiler=${CC_COMPILER} \
    --build-arg cxx_compiler=${CXX_COMPILER} \
    --build-arg git_branch=${GIT_BRANCH} \
    --build-arg git_slug=${GIT_SLUG} \
    --build-arg impl=${IMPL} \
    --build-arg command=${COMMAND} \
    -t portblas .

docker run portblas
