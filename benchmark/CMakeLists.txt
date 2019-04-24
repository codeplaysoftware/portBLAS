find_package(benchmark REQUIRED)
find_package(Clara REQUIRED)

add_subdirectory(common)
add_subdirectory(syclblas)
add_subdirectory(clblast)
