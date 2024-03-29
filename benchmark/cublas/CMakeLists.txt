#/***************************************************************************
# *
# *  @license
# *  Copyright (C) Codeplay Software Limited
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# *  For your convenience, a copy of the License has been included in this
# *  repository.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *  portBLAS: BLAS implementation using SYCL
# *
# *  @filename CMakeLists.txt
# *
# **************************************************************************/
if(BLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

enable_language(CUDA) 
find_package(CUDAToolkit)

set(sources
  # Level 1 blas
  blas1/asum.cpp
  blas1/axpy.cpp
  blas1/dot.cpp
  blas1/iamax.cpp
  blas1/iamin.cpp
  blas1/nrm2.cpp
  blas1/rotg.cpp
  blas1/rotm.cpp
  blas1/rotmg.cpp
  blas1/scal.cpp
  #blas1/sdsdot.cpp -> not implemented in cuBLAS
  # Level 2 blas 
  blas2/gbmv.cpp 
  blas2/gemv.cpp
  blas2/ger.cpp
  blas2/sbmv.cpp 
  blas2/spmv.cpp 
  blas2/spr.cpp
  blas2/spr2.cpp
  blas2/symv.cpp
  blas2/syr.cpp
  blas2/syr2.cpp
  blas2/tbmv.cpp
  blas2/tbsv.cpp
  blas2/tpmv.cpp
  blas2/tpsv.cpp
  blas2/trmv.cpp
  blas2/trsv.cpp
  # Level 3 blas
  blas3/gemm.cpp
  blas3/gemm_batched.cpp
  blas3/gemm_batched_strided.cpp
  blas3/symm.cpp
  blas3/syrk.cpp
  blas3/syr2k.cpp
  blas3/trsm.cpp
  blas3/trsm_batched.cpp
  blas3/trmm.cpp
  # extension blas
  extension/omatcopy.cpp
  extension/omatadd.cpp
) 

# Operators supporting COMPLEX types benchmarking
set(CPLX_OPS "gemm" 
             "gemm_batched"
             "gemm_batched_strided")

# Operators supporting HALF type benchmarking
set(HALF_DATA_OPS "gemm"
                  "gemm_batched"
                  "gemm_batched_strided"
)

# Add individual benchmarks for each method
foreach(cublas_bench ${sources})
  get_filename_component(bench_cublas_exec ${cublas_bench} NAME_WE)
  add_executable(bench_cublas_${bench_cublas_exec} ${cublas_bench} main.cpp)
  target_link_libraries(bench_cublas_${bench_cublas_exec} PRIVATE benchmark CUDA::toolkit CUDA::cublas CUDA::cudart Clara::Clara bench_info)
  target_compile_definitions(bench_cublas_${bench_cublas_exec} PRIVATE -DBLAS_INDEX_T=${BLAS_BENCHMARK_INDEX_TYPE})
  if((${BLAS_ENABLE_COMPLEX}) AND ("${bench_cublas_exec}" IN_LIST CPLX_OPS))
    target_compile_definitions(bench_cublas_${bench_cublas_exec} PRIVATE BLAS_ENABLE_COMPLEX=1)
  endif()
  if((${BLAS_ENABLE_HALF}) AND ("${bench_cublas_exec}" IN_LIST HALF_DATA_OPS))
    target_compile_definitions(bench_cublas_${bench_cublas_exec} PRIVATE BLAS_ENABLE_HALF=1)
  endif()
  add_sycl_to_target(
    TARGET bench_cublas_${bench_cublas_exec}
    SOURCES ${cublas_bench}
  )
target_include_directories(bench_cublas_${bench_cublas_exec} PRIVATE $ENV{CUBLAS_INCLUDE} ${PORTBLAS_INCLUDE} ${PORTBLAS_COMMON_INCLUDE_DIR} ${CBLAS_INCLUDE})
  add_compile_definitions(BUILD_CUBLAS_BENCHMARKS)

  if(BLAS_VERIFY_BENCHMARK)
    target_compile_definitions(bench_cublas_${bench_cublas_exec} PRIVATE BLAS_VERIFY_BENCHMARK)
    target_link_libraries(bench_cublas_${bench_cublas_exec} PRIVATE blas::blas)
  endif()

  message(STATUS "Created benchmark: ${bench_cublas_exec}")
  install(TARGETS bench_cublas_${bench_cublas_exec}
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT benchmarks
  )
endforeach()
