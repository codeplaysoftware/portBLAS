cmake_minimum_required(VERSION 3.2.2)

# compile google test, google mock

include(ExternalProject)

set(SYCLBLAS_TEST ${CMAKE_CURRENT_SOURCE_DIR})
set(SYCLBLAS_TEST_INCLUDE "${SYCLBLAS_TEST}/include")

ExternalProject_Add(
  gtest
  URL https://github.com/google/googletest/archive/release-1.8.0.zip
  PREFIX ${SYCLBLAS_TEST_INCLUDE}/gtest
  INSTALL_COMMAND ""
)

add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)

ExternalProject_Get_Property(gtest source_dir binary_dir)
set_target_properties(libgtest PROPERTIES
  IMPORTED_LOCATION "${binary_dir}/googlemock/gtest/libgtest.a"
  IMPORTED_LINK_INTERFACE_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}"
)
set(GTEST_LIB "${source_dir}/googletest/include")

add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gmock)

set_target_properties(libgmock PROPERTIES
  IMPORTED_LOCATION "${binary_dir}/googlemock/libgmock.a"
  IMPORTED_LINK_INTERFACE_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}"
)
set(GMOCK_LIB "${source_dir}/googlemock/include")

if(VERBOSE)
  add_definitions(-DVERBOSE=VERBOSE)
endif(VERBOSE)

if(SYCL_DEVICE)
  add_definitions(-DSYCL_DEVICE=${SYCL_DEVICE})
endif(SYCL_DEVICE)

include_directories(
  ${GTEST_LIB}
  ${GMOCK_LIB}
)

add_subdirectory(${SYCLBLAS_TEST}/unittest)
add_subdirectory(${SYCLBLAS_TEST}/exprtest)
