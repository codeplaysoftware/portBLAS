
cmake_minimum_required (VERSION 3.2.2)


project(syBLAS)

enable_testing()

message(STATUS "Path to the SYCL implementation ")


set(SYBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

include(FindOpenCL)

include(FindComputeCpp)

include_directories(${SYBLAS_INCLUDE} ${COMPUTECPP_INCLUDE_DIRECTORY})

add_subdirectory(tests)

