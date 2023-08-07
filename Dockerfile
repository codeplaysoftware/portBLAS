FROM ubuntu:focal

# Default values for the build
ARG command
ARG git_branch
ARG git_slug
ARG c_compiler
ARG cxx_compiler
ARG impl

# Timezone is required
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -yq update

# Utilities
RUN apt-get install -yq --allow-downgrades --allow-remove-essential            \
    --allow-change-held-packages git wget python3-pip apt-utils cmake unzip    \
    libboost-all-dev software-properties-common libtinfo5

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

RUN apt-get -yq update

RUN pip install enum34

# Clang 6.0
RUN if [ "${c_compiler}" = 'clang-6.0' ]; then apt-get install -yq             \
    --allow-downgrades --allow-remove-essential --allow-change-held-packages   \
     clang-6.0 libomp-dev; fi

# GCC 7
RUN if [ "${c_compiler}" = 'gcc-7' ]; then apt-get install -yq                 \
    --allow-downgrades --allow-remove-essential --allow-change-held-packages   \
    g++-7 gcc-7; fi

# OpenCL ICD Loader
RUN apt-get install -yq --allow-downgrades --allow-remove-essential               \
    --allow-change-held-packages ocl-icd-opencl-dev ocl-icd-dev opencl-headers

RUN git clone https://github.com/${git_slug}.git --recursive -b ${git_branch} /portBLAS

#OpenBLAS
RUN bash /portBLAS/.scripts/build_OpenBLAS.sh
# Intel OpenCL Runtime
RUN bash /portBLAS/.scripts/install_intel_opencl.sh

# SYCL
RUN if [ "${impl}" = 'COMPUTECPP' ]; then cd /portBLAS && bash /portBLAS/.scripts/build_computecpp.sh; fi
RUN if [ "${impl}" = 'DPCPP' ]; then cd /portBLAS && bash /portBLAS/.scripts/build_dpcpp.sh; fi

ENV COMMAND=${command}
ENV CC=${c_compiler}
ENV CXX=${cxx_compiler}
ENV SYCL_IMPL=${impl}

CMD cd /portBLAS && \
    if [ "${COMMAND}" = 'build-test' ]; then \
      if [ "${SYCL_IMPL}" = 'COMPUTECPP' ]; then \
        /tmp/ComputeCpp-latest/bin/computecpp_info && \
        export COMPUTECPP_TARGET="intel:cpu" && mkdir -p build && cd build && \
        cmake .. -DBLAS_ENABLE_STATIC_LIBRARY=ON -DGEMM_TALL_SKINNY_SUPPORT=OFF \
        -DSYCL_COMPILER=computecpp -DComputeCpp_DIR=/tmp/ComputeCpp-latest \
        -DCMAKE_PREFIX_PATH=/tmp/OpenBLAS/build -DCMAKE_BUILD_TYPE=Release && \
        make -j$(nproc) && cd test && ctest -VV --timeout 1200; \
      elif [ "${SYCL_IMPL}" = 'DPCPP' ]; then \
        export LD_LIBRARY_PATH="/tmp/dpcpp/lib" && mkdir -p build && cd build && \
        cmake .. -DBLAS_ENABLE_STATIC_LIBRARY=ON -DGEMM_TALL_SKINNY_SUPPORT=OFF \
        -DSYCL_COMPILER=dpcpp -DCMAKE_PREFIX_PATH=/tmp/OpenBLAS/build \
        -DBLAS_ENABLE_CONST_INPUT=OFF -DCMAKE_BUILD_TYPE=Release && \
        make -j$(nproc) && cd test && ctest -VV --timeout 1200; \
      else \
        echo "Unknown SYCL implementation ${SYCL_IMPL}"; return 1; \
      fi \
    elif [ "${COMMAND}" = 'auto-tuner' ]; then \
      export LD_LIBRARY_PATH="/tmp/dpcpp/lib" && mkdir -p tools/auto_tuner/build \
      && cd tools/auto_tuner/build && \
      cmake .. -DBLAS_ENABLE_STATIC_LIBRARY=ON -DGEMM_TALL_SKINNY_SUPPORT=OFF \
      -DSYCL_COMPILER=dpcpp -DCMAKE_PREFIX_PATH=/tmp/OpenBLAS/build \
      -DBLAS_ENABLE_CONST_INPUT=OFF -DCMAKE_BUILD_TYPE=Release && \
      make -j$(nproc); \
    else \
      echo "Unknown command ${COMMAND}"; return 1; \
    fi \
