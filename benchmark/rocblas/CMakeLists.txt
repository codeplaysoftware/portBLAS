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

  # Level 2 blas
  blas2/gemv.cpp
  blas2/gbmv.cpp
  blas2/sbmv.cpp
  blas2/tbmv.cpp
  blas2/trsv.cpp
  blas2/spr.cpp
  blas2/spr2.cpp
  blas2/symv.cpp
  blas2/syr.cpp
  blas2/syr2.cpp
  blas2/ger.cpp
  blas2/tbsv.cpp
  blas2/tpsv.cpp
  blas2/tpmv.cpp
  blas2/trmv.cpp
  blas2/spmv.cpp

  # Level 3 blas
  blas3/gemm.cpp
  blas3/symm.cpp
  blas3/syrk.cpp
  blas3/syr2k.cpp
  blas3/trsm.cpp
  blas3/trmm.cpp
  blas3/trsm_batched.cpp
  blas3/gemm_batched.cpp
  blas3/gemm_batched_strided.cpp

  # Extension blas
  extension/omatcopy.cpp
  extension/omatadd.cpp
  extension/axpy_batch.cpp
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

# HIP_CXX_COMPILER is defined equal to CMAKE_CXX_COMPILER while looking for rocblas (line 30)
# if it is passed by user it is supposed to be different from standard compiler, than it is up
# to the user provide the correct path with the correct library.
if("${HIP_CXX_COMPILER}" STREQUAL "${CMAKE_CXX_COMPILER}" )
# Verify that libclang_rt.builtins exists for current compiler
  if(EXISTS ${CLANGRT_BUILTINS} )
    message("libclang_rt found at " ${CLANGRT_BUILTINS})
  else()
    message(FATAL_ERROR "libclang_rt not found at " ${CLANGRT_BUILTINS}
      "\nCheck library availability for current CXX compiler. "
      "Using HIP_CXX_COMPILER option, set another compiler compatible with ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} that has libclang_rt.builtins for x86_64."
      "If not possible skip rocBLAS benchmarks compilation")
  endif()
else()
  message(WARNING "using CLANGRT_BUILTINS defined by user at " ${CLANGRT_BUILTINS})
endif()

# Add individual benchmarks for each method
foreach(rocblas_benchmark ${sources})
  get_filename_component(rocblas_bench_exec ${rocblas_benchmark} NAME_WE)
  add_executable(bench_rocblas_${rocblas_bench_exec} ${rocblas_benchmark} main.cpp)
  target_link_libraries(bench_rocblas_${rocblas_bench_exec} PRIVATE benchmark Clara::Clara roc::rocblas bench_info) 
  target_compile_definitions(bench_rocblas_${rocblas_bench_exec} PRIVATE -DBLAS_INDEX_T=${BLAS_BENCHMARK_INDEX_TYPE})
  target_include_directories(bench_rocblas_${rocblas_bench_exec} PRIVATE  ${PORTBLAS_INCLUDE} ${rocblas_INCLUDE_DIRS} ${CBLAS_INCLUDE} ${BLAS_BENCH} ${PORTBLAS_COMMON_INCLUDE_DIR})
  if((${BLAS_ENABLE_COMPLEX}) AND ("${rocblas_bench_exec}" IN_LIST CPLX_OPS))
    target_compile_definitions(bench_rocblas_${rocblas_bench_exec} PRIVATE BLAS_ENABLE_COMPLEX=1)
  endif()

  if((${BLAS_ENABLE_HALF}) AND ("${rocblas_bench_exec}" IN_LIST HALF_DATA_OPS))
    target_compile_definitions(bench_rocblas_${rocblas_bench_exec} PRIVATE BLAS_ENABLE_HALF=1)
  endif()
  # Even though rocblas does not use sycl, the common tools indirectly include sycl headers.
  add_sycl_to_target(
      TARGET bench_rocblas_${rocblas_bench_exec}
      SOURCES ${rocblas_benchmark}
  )

  target_compile_definitions(bench_rocblas_${rocblas_bench_exec} PRIVATE BUILD_ROCBLAS_BENCHMARKS)

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
