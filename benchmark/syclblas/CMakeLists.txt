if(BLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

set(sources
  # Level 1 blas
  blas1/axpy.cpp
  blas1/asum.cpp
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
  blas2/gbmv.cpp
  blas2/gemv.cpp
  blas2/sbmv.cpp
  blas2/tbmv.cpp
  # Level 3 blas
  blas3/gemm.cpp
  blas3/gemm_batched.cpp
  blas3/trsm.cpp
)

if(${BLAS_ENABLE_EXTENSIONS})
  list(APPEND sources "extension/reduction.cpp")
endif()

# Add individual benchmarks for each method
foreach(syclblas_bench ${sources})
  get_filename_component(bench_exec ${syclblas_bench} NAME_WE)
  add_executable(bench_${bench_exec} ${syclblas_bench} main.cpp)
  target_link_libraries(bench_${bench_exec} PRIVATE benchmark Clara::Clara sycl_blas)
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
