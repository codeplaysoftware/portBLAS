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

# Requires clBLAS_DIR to point to a valid clBLAS installation
find_package(clBLAS REQUIRED)

set(sources
  # Level 2 blas
  blas2/trsv.cpp
  blas2/tbsv.cpp
  # Level 3 blas
  blas3/trsm.cpp
)

foreach(clblas_benchmark ${sources})
  get_filename_component(clblas_bench_exec ${clblas_benchmark} NAME_WE)
  add_executable(bench_clblas_${clblas_bench_exec} ${clblas_benchmark} "main.cpp")
  target_link_libraries(bench_clblas_${clblas_bench_exec} PRIVATE benchmark Clara::Clara portblas)
  target_compile_definitions(bench_clblas_${clblas_bench_exec} PRIVATE -DBLAS_INDEX_T=${BLAS_BENCHMARK_INDEX_TYPE})
  target_include_directories(bench_clblas_${clblas_bench_exec} PRIVATE ${CLBLAS_INCLUDE_DIRS} ${BLAS_BENCH} ${PORTBLAS_INCLUDE} ${PORTBLAS_COMMON_INCLUDE_DIR})
  target_link_libraries(bench_clblas_${clblas_bench_exec} PRIVATE ${CLBLAS_LIBRARIES})

  if(BLAS_VERIFY_BENCHMARK)
    target_link_libraries(bench_clblas_${clblas_bench_exec} PRIVATE blas::blas)
    target_compile_definitions(bench_clblas_${clblas_bench_exec} PRIVATE BLAS_VERIFY_BENCHMARK)
  endif()

  message(STATUS "Created clBLAS benchmark: ${clblas_bench_exec}")
endforeach()
