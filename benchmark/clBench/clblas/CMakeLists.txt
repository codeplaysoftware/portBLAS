if(BLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

# Requires clBLAS_DIR to point to a valid clBLAS installation
find_package(clBLAS REQUIRED)

set(sources
  "blas3/trsm.cpp"
)

foreach(clblas_benchmark ${sources})
  get_filename_component(clblas_bench_exec ${clblas_benchmark} NAME_WE)
  add_executable(bench_clblas_${clblas_bench_exec} ${clblas_benchmark} "main.cpp")
  target_link_libraries(bench_clblas_${clblas_bench_exec} PRIVATE benchmark Clara::Clara sycl_blas)
  target_compile_definitions(bench_clblas_${clblas_bench_exec} PRIVATE -DBLAS_INDEX_T=${BLAS_BENCHMARK_INDEX_TYPE})
  target_include_directories(bench_clblas_${clblas_bench_exec} PRIVATE ${CLBLAS_INCLUDE_DIRS} ${BLAS_BENCH} ${SYCLBLAS_INCLUDE} ${SYCLBLAS_COMMON_INCLUDE_DIR})
  target_link_libraries(bench_clblas_${clblas_bench_exec} PRIVATE ${CLBLAS_LIBRARIES})

  if(BLAS_VERIFY_BENCHMARK)
    target_link_libraries(bench_clblas_${clblas_bench_exec} PRIVATE blas::blas)
    target_compile_definitions(bench_clblas_${clblas_bench_exec} PRIVATE BLAS_VERIFY_BENCHMARK)
  endif()

  message(STATUS "Created clBLAS benchmark: ${clblas_bench_exec}")
endforeach()
