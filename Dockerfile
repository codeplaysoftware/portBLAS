FROM ubuntu:focal

# Default values for the build
ARG git_branch
ARG git_slug
ARG c_compiler
ARG cxx_compiler
ARG impl
ARG target

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
RUN apt-get install -yq --allow-downgrades --allow-remove-essential           \
    --allow-change-held-packages ocl-icd-opencl-dev ocl-icd-dev opencl-headers

RUN git clone https://github.com/${git_slug}.git --recursive -b ${git_branch} /sycl-blas

#OpenBLAS
RUN bash /sycl-blas/.scripts/build_OpenBLAS.sh
# Intel OpenCL Runtime
RUN if [ "${target}" = 'opencl' ]; then bash /sycl-blas/.scripts/install_intel_opencl.sh; fi

# SYCL
RUN if [ "${impl}" = 'COMPUTECPP' ]; then cd /sycl-blas && bash /sycl-blas/.scripts/build_computecpp.sh; fi
RUN if [ "${impl}" = 'DPCPP' ]; then cd /sycl-blas && bash /sycl-blas/.scripts/build_dpcpp.sh; fi

ENV CC=${c_compiler}
ENV CXX=${cxx_compiler}
ENV SYCL_IMPL=${impl}
ENV TARGET=${target}

CMD cd /sycl-blas && \
    if [ "${SYCL_IMPL}" = 'COMPUTECPP' ]; then \
      if [ "${TARGET}" = 'host' ]; then \
        export COMPUTECPP_TARGET="host" && mkdir -p build && cd build && \
        cmake .. -DBLAS_ENABLE_STATIC_LIBRARY=ON -DGEMM_TALL_SKINNY_SUPPORT=OFF \
        -DSYCL_COMPILER=computecpp -DComputeCpp_DIR=/tmp/ComputeCpp-latest \
        -DCMAKE_PREFIX_PATH=/tmp/OpenBLAS/build -DCMAKE_BUILD_TYPE=Release && \
        make -j$(nproc) && cd test && ctest -VV --timeout 1200; \
      else \
        /tmp/ComputeCpp-latest/bin/computecpp_info && \
        export COMPUTECPP_TARGET="intel:cpu" && mkdir -p build && cd build && \
        cmake .. -DBLAS_ENABLE_STATIC_LIBRARY=ON -DGEMM_TALL_SKINNY_SUPPORT=OFF \
        -DSYCL_COMPILER=computecpp -DComputeCpp_DIR=/tmp/ComputeCpp-latest \
        -DCMAKE_PREFIX_PATH=/tmp/OpenBLAS/build -DCMAKE_BUILD_TYPE=Release && \
        make -j$(nproc) && cd test && ctest -VV --timeout 1200; \
      fi \
    elif [ "${SYCL_IMPL}" = 'DPCPP' ]; then \
      export LD_LIBRARY_PATH="/tmp/dpcpp/lib" && mkdir -p build && cd build && \
      cmake .. -DBLAS_ENABLE_STATIC_LIBRARY=ON -DGEMM_TALL_SKINNY_SUPPORT=OFF \
      -DSYCL_COMPILER=dpcpp -DCMAKE_PREFIX_PATH=/tmp/OpenBLAS/build \
      -DBLAS_ENABLE_CONST_INPUT=OFF -DCMAKE_BUILD_TYPE=Release && \
      make -j$(nproc) && cd test && ctest -VV --timeout 1200; \
    else \
      echo "Unknown SYCL implementation ${SYCL_IMPL}"; return 1; \
    fi
