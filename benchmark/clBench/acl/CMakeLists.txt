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
find_package(ACL REQUIRED)

if(BLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

function(generate_acl_benchmark name source definition)
  add_executable(${name} ${source} main.cpp)
  target_link_libraries(${name} PRIVATE benchmark acl portblas Clara::Clara)
  target_compile_definitions(${name} PRIVATE -DBLAS_INDEX_T=${BLAS_BENCHMARK_INDEX_TYPE})
  target_include_directories(${name} PRIVATE ${CBLAS_INCLUDE} ${BLAS_BENCH} ${PORTBLAS_INCLUDE} ${PORTBLAS_COMMON_INCLUDE_DIR})
  target_compile_definitions(${name} PRIVATE ${definition})

  if(BLAS_VERIFY_BENCHMARK)
    target_compile_definitions(${name} PRIVATE BLAS_VERIFY_BENCHMARK)
    target_link_libraries(${name} PRIVATE blas::blas)
  endif()

  message(STATUS "Created benchmark: ${name}")
  install(TARGETS ${name}
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT benchmarks
  )
endfunction()

generate_acl_benchmark(bench_acl_gemm_neon blas3/gemm.cpp ACL_BACKEND_NEON)
generate_acl_benchmark(bench_acl_gemm_opencl blas3/gemm.cpp ACL_BACKEND_OPENCL)
