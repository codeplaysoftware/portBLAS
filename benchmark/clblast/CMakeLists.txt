set(CLBLAST_BENCH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CLBLAST_BENCH})

if(${BLAS_VERIFY_BENCHMARK})
  add_definitions(-DBLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

find_package(SB_CLBlast REQUIRED)

set(CLBLAST_BENCH_SRCS
  # Level 1 blas
  ${CLBLAST_BENCH}/blas1/asum.cpp
  ${CLBLAST_BENCH}/blas1/axpy.cpp
  ${CLBLAST_BENCH}/blas1/dot.cpp
  ${CLBLAST_BENCH}/blas1/iamax.cpp
  ${CLBLAST_BENCH}/blas1/iamin.cpp
  ${CLBLAST_BENCH}/blas1/nrm2.cpp
  ${CLBLAST_BENCH}/blas1/scal.cpp
  # Level 2 blas
  ${CLBLAST_BENCH}/blas2/gemv.cpp
  # Level 3 blas
  ${CLBLAST_BENCH}/blas3/gemm.cpp
  ${CLBLAST_BENCH}/blas3/gemm_batched.cpp
)

# Add individual benchmarks for each method
foreach(clblast_benchmark ${CLBLAST_BENCH_SRCS})
  get_filename_component(clblast_bench_exec ${clblast_benchmark} NAME_WE)
  add_executable(bench_clblast_${clblast_bench_exec} ${clblast_benchmark} ${CLBLAST_BENCH}/main.cpp ${CLBLAST_BENCH}/clwrap.cpp)
  target_link_libraries(bench_clblast_${clblast_bench_exec} PRIVATE benchmark ${BLAS_LIBRARIES} clblast sycl_blas)
  target_include_directories(bench_clblast_${clblast_bench_exec} PRIVATE ${CLARA_INCLUDE} ${CBLAS_INCLUDE})

  message("-- Created benchmark: ${clblast_bench_exec}")
  install(TARGETS bench_clblast_${clblast_bench_exec}
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT benchmarks
  )
endforeach()
