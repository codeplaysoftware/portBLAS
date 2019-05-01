cmake_minimum_required(VERSION 3.2.2)

find_package(PkgConfig)

message(STATUS "Path to the SYCL implementation ")

set(COMPUTECPP_SDK_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../external/computecpp-sdk/include)
set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../external/computecpp-sdk/cmake/Modules)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}) # For the combinations

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)
find_package(PythonInterp REQUIRED)
include(ConfigureSYCLBLAS)

include(FindSystemBLAS)

set(GEN_COMB ${CMAKE_CURRENT_BINARY_DIR}/generate_combinations.inc.hpp)

# The generator's configuration file - add any new jsons to this
set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/default.json)
if(${TARGET} STREQUAL "RCAR")
set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/rcar.json)
endif()

# Set up the enum generation script
add_custom_target(
 generate_combinations ALL
 COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen/generate_combinations.py ${GEN_CONFIG} ${GEN_COMB}
 BYPRODUCTS $GEN_COMB
 COMMENT "Generating tuning configurations"
)

# Add syclblas benchmarks.
file(GLOB SYCL_AUTO_TUNNER_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tune_nn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tune_nt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tune_tn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tune_tt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tune_all.cpp
)
foreach(blas_tuner ${SYCL_AUTO_TUNNER_SRCS})
  get_filename_component(tuner_exec ${blas_tuner} NAME_WE)
  set(TARGET tuner_exec ${blas_tuner})
  add_executable(${tuner_exec} ${blas_tuner})
  set_property(TARGET ${tuner_exec} PROPERTY CXX_STANDARD 11)
  message(${BLAS_LIBRARIES})
  target_link_libraries(${tuner_exec} PUBLIC m PUBLIC ${BLAS_LIBRARIES})
  add_dependencies(${tuner_exec} generate_combinations)
  add_sycl_to_target(
    TARGET ${tuner_exec}
    SOURCES ${blas_tuner}
  )
  install(TARGETS ${tuner_exec} RUNTIME DESTINATION bin)
endforeach()
include_directories(${SYCLBLAS_INCLUDE} ${ComputeCpp_INCLUDE_DIRS} ${COMPUTECPP_SDK_INCLUDE} ${BLAS_INCLUDE_DIRS})
