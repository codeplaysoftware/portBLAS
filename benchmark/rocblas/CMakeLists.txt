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

find_package(rocblas REQUIRED)

set(sources
  # Level 1 blas
  blas1/axpy.cpp
  blas1/asum.cpp
  blas1/dot.cpp
  blas1/scal.cpp
  blas1/iamin.cpp
  blas1/iamax.cpp
  blas1/nrm2.cpp
  blas1/rotg.cpp
  blas1/rotm.cpp
  blas1/rotmg.cpp

)

# Add individual benchmarks for each method
foreach(rocblas_benchmark ${sources})
  get_filename_component(rocblas_bench_exec ${rocblas_benchmark} NAME_WE)
  add_executable(bench_rocblas_${rocblas_bench_exec} ${rocblas_benchmark} main.cpp)
  target_link_libraries(bench_rocblas_${rocblas_bench_exec} PRIVATE benchmark Clara::Clara roc::rocblas) 
  target_compile_definitions(bench_rocblas_${rocblas_bench_exec} PRIVATE -DBLAS_INDEX_T=${BLAS_BENCHMARK_INDEX_TYPE})
  target_include_directories(bench_rocblas_${rocblas_bench_exec} PRIVATE  ${SYCLBLAS_INCLUDE} ${rocblas_INCLUDE_DIRS} ${CBLAS_INCLUDE} ${BLAS_BENCH} ${SYCLBLAS_COMMON_INCLUDE_DIR})

  # Even though rocblas does not use sycl, the common tools indirectly include sycl headers.
  add_sycl_to_target(
      TARGET bench_rocblas_${rocblas_bench_exec}
      SOURCES ${rocblas_benchmark}
  )

  if(BLAS_VERIFY_BENCHMARK)
    target_link_libraries(bench_rocblas_${rocblas_bench_exec} PRIVATE blas::blas)
    target_compile_definitions(bench_rocblas_${rocblas_bench_exec} PRIVATE BLAS_VERIFY_BENCHMARK)
  endif()

  message(STATUS "Created rocblas benchmark: ${rocblas_bench_exec}")
  install(TARGETS bench_rocblas_${rocblas_bench_exec}
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT benchmarks
  )
endforeach()
