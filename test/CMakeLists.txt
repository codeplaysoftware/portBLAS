find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
if(NOT ${CMAKE_VERSION} VERSION_LESS "3.8")
  include(GoogleTest)
endif()

option(VERBOSE "Enable verbose output in the unit tests" OFF)
set(SYCL_DEVICE "" CACHE STRING "Select the SYCL Device to run on")

configure_file("blas_test_macros.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/blas_test_macros.hpp")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(VERBOSE)
  add_definitions(-DVERBOSE=VERBOSE)
endif(VERBOSE)

if(SYCL_DEVICE)
  add_definitions(-DSYCL_DEVICE=${SYCL_DEVICE})
endif(SYCL_DEVICE)

add_subdirectory(unittest)
add_subdirectory(exprtest)
