set(sources
  blas1/axpy.cpp
  blas1/asum.cpp
  blas1/dot.cpp
  blas1/iamax.cpp
  blas1/iamin.cpp
  blas1/nrm2.cpp
  blas1/scal.cpp

  blas2/gemv.cpp

  blas3/gemm.cpp

  main.cpp
)
add_executable(bench ${sources})
add_sycl_to_target(TARGET bench SOURCES ${sources})
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench PRIVATE codeplay::sycl-blas benchmark_common)
