cmake_minimum_required(VERSION 3.4.3)
project(syclblas-test)

include("${CMAKE_BINARY_DIR}/conan_paths.cmake")

find_package(sycl-blas REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE sycl::sycl-blas)
add_sycl_to_target(TARGET app SOURCES main.cpp)