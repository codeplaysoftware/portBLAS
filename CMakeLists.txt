cmake_minimum_required (VERSION 3.2.2)

project(sycl-blas)

enable_testing()

message(STATUS "Path to the SYCL implementation ")

set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

include(FindOpenCL)

include(FindComputeCpp)

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

message(STATUS "${COMPUTECPP_DEVICE_COMPILER_FLAGS}")

# We add some flags to workaround OpenCL platform bugs, see ComputeCpp documentation
set(COMPUTECPP_DEVICE_COMPILER_FLAGS
    "${COMPUTECPP_DEVICE_COMPILER_FLAGS} -no-serial-memop -sycl-compress-name")
message(STATUS "${COMPUTECPP_DEVICE_COMPILER_FLAGS}")

# Check to see if we've disabled double support in the tests
option(NO_DOUBLE_SUPPORT "Disable double support when testing." off)
if(NO_DOUBLE_SUPPORT)
  # Define NO_DOUBLE_SUPPORT for the host cxx compiler
  add_definitions(-DNO_DOUBLE_SUPPORT)
  # Set the computecpp device compiler flags to also define NO_DOUBLE_SUPPORT 
  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -DNO_DOUBLE_SUPPORT)
endif()

# If the user has specified a specific workgroup size for tests, pass that on to the compiler
if(WGSIZE)
  add_definitions(-DWGSIZE=${WGSIZE})
  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -DWGSIZE=${WGSIZE})
endif()

# If the user has specified that we should use naive gemm, enable that
option(NAIVEGEMM "Default to naive GEMM implementations" off)
if(NAIVEGEMM)
  add_definitions(-DNAIVEGEMM)
  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -DNAIVEGEMM)
endif()

include_directories(${SYCLBLAS_INCLUDE} ${COMPUTECPP_INCLUDE_DIRECTORY} ${BLAS_INCLUDE_DIRS})



add_subdirectory(test)
