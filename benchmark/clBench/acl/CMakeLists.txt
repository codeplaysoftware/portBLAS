find_package(ACL REQUIRED)

if(BLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

function(generate_acl_benchmark name source definition)
  add_executable(${name} ${source} main.cpp)
  target_link_libraries(${name} PRIVATE benchmark acl sycl_blas Clara::Clara)
  target_include_directories(${name} PRIVATE ${CBLAS_INCLUDE} ${BLAS_BENCH} ${SYCLBLAS_INCLUDE} ${SYCLBLAS_COMMON_INCLUDE_DIR})
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
