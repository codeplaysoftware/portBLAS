add_library(benchmark-syclblas-common OBJECT main.cpp)
target_link_libraries(benchmark-syclblas-common PUBLIC codeplay::sycl-blas benchmark-common)
target_include_directories(benchmark-syclblas-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(blas1)
add_subdirectory(blas2)
add_subdirectory(blas3)
