set(benchmarks
    gemm
)

foreach(b ${benchmarks})
    add_executable(benchmark-clblast-blas3-${b} ${b}.cpp)
    target_link_libraries(benchmark-clblast-blas3-${b} PRIVATE benchmark-clblast-common)
endforeach()
