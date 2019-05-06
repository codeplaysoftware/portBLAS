set(benchmarks
    gemv
)

foreach(b ${benchmarks})
    add_executable(benchmark-clblast-blas2-${b} ${b}.cpp)
    target_link_libraries(benchmark-clblast-blas2-${b} PRIVATE benchmark-clblast-common)
endforeach()
