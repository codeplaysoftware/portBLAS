find_package(BLAS QUIET) # For some reason CMake does not set BLAS_FOUND even if the library is found
foreach(blas_impl
    ACML    ACML_GPU        ACML_MP
    Apple   "ATLAS PhiPACK"
    CXML    DXML            FLAME
    Generic Goto            IBMESSL
    Intel   Intel10_32      Intel10_64lp Intel10_64lp_seq
    NAS     OpenBLAS        SCSL SGIMATH SunPerf)
  if(BLAS_${blas_impl}_LIBRARY)
    set(BLAS_FOUND TRUE)
    link_libraries(${BLAS_${blas_impl}_LIBRARY})
    break()
  endif()
endforeach()

if(NOT BLAS_FOUND)
  find_package(OpenBLAS REQUIRED)
  link_libraries(OpenBLAS::OpenBLAS)
endif()

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
