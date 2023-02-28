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
  blas2/sbmv.cpp 
  blas2/gemv.cpp
) 

#if(${BLAS_ENABLE_EXTENSIONS})
#  list(APPEND sources "extension/reduction.cpp")
#endif()

# Add individual benchmarks for each method
foreach(cublas_bench ${sources})
  get_filename_component(bench_cublas_exec ${cublas_bench} NAME_WE)
  add_executable(bench_cublas_${bench_cublas_exec} ${cublas_bench} main.cpp)
  target_link_libraries(bench_cublas_${bench_cublas_exec} PRIVATE benchmark CUDA::toolkit CUDA::cublas CUDA::cudart sycl_blas Clara::Clara )
  target_compile_definitions(bench_cublas_${bench_cublas_exec} PRIVATE -DBLAS_INDEX_T=${BLAS_BENCHMARK_INDEX_TYPE})
  add_sycl_to_target(
    TARGET bench_cublas_${bench_cublas_exec}
    SOURCES ${cublas_bench}
  )
target_include_directories(bench_cublas_${bench_cublas_exec} PRIVATE $ENV{CUBLAS_INCLUDE} ${SYCLBLAS_INCLUDE} ${SYCLBLAS_COMMON_INCLUDE_DIR} ${CBLAS_INCLUDE})
  add_compile_definitions(BUILD_CUBLAS_BENCHMARKS)
  add_compile_options(-gdwarf-4)

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
