#!/bin/bash

if [ -z $DOCKER_IMAGE ]; then
    # Run with native
    .travis/run_project_build.sh
else
    # Run with docker
    docker run -v$(pwd):/home/conan $DOCKER_IMAGE bash -c ".travis/run_project_build.sh"
fi