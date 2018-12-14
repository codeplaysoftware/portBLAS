cmake_minimum_required(VERSION 3.2.2)

# compile google test, google mock

include(ExternalProject)

set(SYCLBLAS_TEST ${CMAKE_CURRENT_SOURCE_DIR})
set(SYCLBLAS_TEST_INCLUDE "${SYCLBLAS_TEST}/include")

# Download and build googletest
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/templates/CMakeLists.txt.in ${CMAKE_CURRENT_BINARY_DIR}/googletest-download/CMakeLists.txt)
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

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()
# Now simply link your own targets against gtest, gmock,
# etc. as appropriate

# ExternalProject_Add(
#   gtest
#   URL https://github.com/google/googletest/archive/release-1.8.0.zip
#   PREFIX ${SYCLBLAS_TEST_INCLUDE}/gtest
#   CMAKE_ARGS ${cmake_toolchain}
#   INSTALL_COMMAND ""
# )

# add_library(libgtest IMPORTED STATIC GLOBAL)
# add_dependencies(libgtest gtest)

# ExternalProject_Get_Property(gtest source_dir binary_dir)
# set_target_properties(libgtest PROPERTIES
#   IMPORTED_LOCATION "${binary_dir}/googlemock/gtest/libgtest.a"
#   IMPORTED_LINK_INTERFACE_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}"
# )
# set(GTEST_LIB "${source_dir}/googletest/include")

# add_library(libgmock IMPORTED STATIC GLOBAL)
# add_dependencies(libgmock gmock)

# set_target_properties(libgmock PROPERTIES
#   IMPORTED_LOCATION "${binary_dir}/googlemock/libgmock.a"
#   IMPORTED_LINK_INTERFACE_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}"
# )
# set(GMOCK_LIB "${source_dir}/googlemock/include")

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

add_subdirectory(${SYCLBLAS_TEST}/unittest)
add_subdirectory(${SYCLBLAS_TEST}/exprtest)
