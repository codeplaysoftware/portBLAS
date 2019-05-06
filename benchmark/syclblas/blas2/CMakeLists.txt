set(benchmarks
    gemv
)

foreach(b ${benchmarks})
    add_executable(benchmark-syclblas-blas2-${b} ${b}.cpp)
    target_link_libraries(benchmark-syclblas-blas2-${b} PRIVATE benchmark-syclblas-common)
endforeach()
