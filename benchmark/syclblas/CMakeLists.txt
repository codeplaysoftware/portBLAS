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
# *  SYCL-BLAS: BLAS implementation using SYCL
# *
# *  @filename CMakeLists.txt
# *
# **************************************************************************/
if(BLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

set(sources
  # Level 1 blas
  blas1/axpy.cpp
  blas1/asum.cpp
  blas1/copy.cpp
  blas1/dot.cpp
  blas1/sdsdot.cpp
  blas1/iamax.cpp
  blas1/iamin.cpp
  blas1/nrm2.cpp
  blas1/scal.cpp
  blas1/rotg.cpp
  blas1/rotm.cpp
  blas1/rotmg.cpp
  # Level 2 blas
  blas2/spr.cpp
  blas2/spr2.cpp
  blas2/gbmv.cpp
  blas2/gemv.cpp
  blas2/ger.cpp
  blas2/sbmv.cpp
  blas2/spmv.cpp
  blas2/symv.cpp
  blas2/syr.cpp
  blas2/syr2.cpp
  blas2/tbmv.cpp
  blas2/tpmv.cpp
  blas2/trmv.cpp
  blas2/trsv.cpp
  blas2/tbsv.cpp
  blas2/tpsv.cpp
  # Level 3 blas
  blas3/gemm.cpp
  blas3/gemm_batched.cpp
  blas3/gemm_batched_strided.cpp
  blas3/trsm.cpp
  blas3/symm.cpp
)

if(${BLAS_ENABLE_EXTENSIONS})
  list(APPEND sources "extension/reduction.cpp")
endif()

# Add individual benchmarks for each method
foreach(syclblas_bench ${sources})
  get_filename_component(bench_exec ${syclblas_bench} NAME_WE)
  add_executable(bench_${bench_exec} ${syclblas_bench} main.cpp)
  target_link_libraries(bench_${bench_exec} PRIVATE benchmark Clara::Clara sycl_blas bench_info)
  target_compile_definitions(bench_${bench_exec} PRIVATE -DBLAS_INDEX_T=${BLAS_BENCHMARK_INDEX_TYPE})
  add_sycl_to_target(
    TARGET bench_${bench_exec}
    SOURCES ${syclblas_bench}
  )
  target_include_directories(bench_${bench_exec} PRIVATE ${SYCLBLAS_INCLUDE} ${CBLAS_INCLUDE} ${SYCLBLAS_COMMON_INCLUDE_DIR})

  if(BLAS_VERIFY_BENCHMARK)
    target_compile_definitions(bench_${bench_exec} PRIVATE BLAS_VERIFY_BENCHMARK)
    target_link_libraries(bench_${bench_exec} PRIVATE blas::blas)
  endif()

  message(STATUS "Created benchmark: ${bench_exec}")
  install(TARGETS bench_${bench_exec}
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT benchmarks
  )
endforeach()
