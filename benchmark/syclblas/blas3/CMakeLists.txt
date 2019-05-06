set(benchmarks
    gemm
)

foreach(b ${benchmarks})
    add_executable(benchmark-syclblas-blas3-${b} ${b}.cpp)
    target_link_libraries(benchmark-syclblas-blas3-${b} PRIVATE benchmark-syclblas-common)
endforeach()
