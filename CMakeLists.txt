cmake_minimum_required (VERSION 3.2.2)

project(sycl-blas)

enable_testing()

set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

include(FindOpenCL)
include(FindComputeCpp)
include(ConfigureSYCLBLAS)

include(FindBLAS)

if (DEFINED OPENBLAS_ROOT)
  # If OPENBLAS_ROOT is defined, then use it explicitly, and set the BLAS paths and
  # libraries based on the explicit path given 
  message(STATUS "Using explicit OpenBLAS installation path for unit tests")
  set(BLAS_LIBRARIES "${OPENBLAS_ROOT}/lib/libopenblas.so")
  set(BLAS_INCLUDE_DIRS "${OPENBLAS_ROOT}/include/")
else()
  message(STATUS "Using Cmake FindBLAS to locate a BLAS library for unit tests")
  set(BLA_STATIC on)
  # If we want to use a specific BLAS vendor, we could set it here:
  # by calling: set(BLA_VENDOR OpenBLAS) 
  find_package(BLAS REQUIRED) # We need BLAS for the tests - require it
  message(STATUS "Found BLAS library at: ${BLAS_LIBRARIES}")
endif()

include_directories(${SYCLBLAS_INCLUDE} ${ComputeCpp_INCLUDE_DIRS} ${BLAS_INCLUDE_DIRS})

add_subdirectory(test)