set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}) # For the combinations

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(SystemBLAS REQUIRED)

set(GEN_COMB ${CMAKE_CURRENT_BINARY_DIR}/generate_combinations.inc.hpp)

# The generator's configuration file - add any new jsons to this
set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/default.json)
if(${TARGET} STREQUAL "RCAR")
  set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/rcar.json)
endif()

add_custom_command(OUTPUT ${GEN_COMB}
  COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/gen/generate_combinations.py
    ${GEN_CONFIG} ${GEN_COMB}
  MAIN_DEPENDENCY ${GEN_CONFIG}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen/generate_combinations.py
  COMMENT "Generating tuning configurations"
  VERBATIM
)
add_custom_target(generate_combinations DEPENDS ${GEN_COMB})

set(SYCL_AUTO_TUNNER_SRCS
  src/tune_nn.cpp
  src/tune_nt.cpp
  src/tune_tn.cpp
  src/tune_tt.cpp
  src/tune_all.cpp
)

foreach(blas_tuner ${SYCL_AUTO_TUNNER_SRCS})
  get_filename_component(tuner_exec ${blas_tuner} NAME_WE)
  set(TARGET tuner_exec ${blas_tuner})
  add_executable(${tuner_exec} ${blas_tuner})
  target_link_libraries(${tuner_exec} PRIVATE m blas::blas)
  target_include_directories(${tuner_exec} PRIVATE
    ${SYCLBLAS_INCLUDE}
    ${COMPUTECPP_SDK_INCLUDE}
  )
  add_dependencies(${tuner_exec} generate_combinations)
  add_sycl_to_target(
    TARGET ${tuner_exec}
    SOURCES ${blas_tuner}
  )
  install(TARGETS ${tuner_exec} RUNTIME DESTINATION bin)
endforeach()
