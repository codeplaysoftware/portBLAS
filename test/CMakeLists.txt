find_package(Clara REQUIRED)

# compile googletest
include(ExternalProject)
find_package(SystemBLAS REQUIRED)

set(SYCLBLAS_TEST ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-shorten-64-to-32 -Wno-cast-align") 
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy-with-user-provided-copy -Wno-unused-variable")

# Download and build googletest
# The following operations are derived from the official advice on building googletest as a subproject,
# form the following documentation: https://github.com/google/googletest/blob/master/googletest/README.md
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/templates/GTest.txt.in
  ${CMAKE_CURRENT_BINARY_DIR}/googletest-download/CMakeLists.txt
)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# Set SYCL-BLAS options
if(CMAKE_CROSSCOMPILING)
    set(cmake_toolchain
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
       )
     message(STATUS "Crossing : ${cmake_toolchain}")
endif()

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

add_subdirectory(unittest)
if(ENABLE_EXPRESSION_TESTS)
  add_subdirectory(exprtest)
endif()
