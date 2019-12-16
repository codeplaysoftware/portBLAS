set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)


find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(SystemBLAS REQUIRED)

set(tuner_def_file ${CMAKE_CURRENT_BINARY_DIR}/generated_combinations.def)

# The generator's configuration file - add any new jsons to this
set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/default.json)
if(${TARGET} STREQUAL "RCAR")
  set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/rcar.json)
endif()

add_custom_command(OUTPUT ${tuner_def_file}
  COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/gen/generate_combinations.py
    ${GEN_CONFIG} ${tuner_def_file}
  MAIN_DEPENDENCY ${GEN_CONFIG}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen/generate_combinations.py
  COMMENT "Generating tuning configurations"
  VERBATIM
)
add_custom_target(tuner_generate_def_file DEPENDS ${tuner_def_file})

set(_generator_script
  ${CMAKE_CURRENT_SOURCE_DIR}/gen/generate_combinations.py
)
set(_gen_src_dir ${CMAKE_CURRENT_BINARY_DIR}/gen)
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${_generator_script}
    ${GEN_CONFIG} --source_dir=${_gen_src_dir} --list_files
  OUTPUT_VARIABLE _gen_src_files
  RESULT_VARIABLE _gen_src_files_exitcode
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT ${_gen_src_files_exitcode} EQUAL 0)
  message(FATAL_ERROR "Failed to generate list of source files for tuner")
endif()

STRING(REGEX REPLACE "\n" ";" tuner_kernel_srcs "${_gen_src_files}")

add_custom_command(OUTPUT ${tuner_kernel_srcs}
  COMMAND ${PYTHON_EXECUTABLE} ${_generator_script}
    ${GEN_CONFIG} --source_dir=${_gen_src_dir}
  MAIN_DEPENDENCY ${GEN_CONFIG}
  DEPENDS ${_generator_script}
)
add_custom_target(tuner_generate_kernels DEPENDS ${tuner_kernel_srcs})

add_library(tuner_kernel_lib ${tuner_kernel_srcs})
target_include_directories(tuner_kernel_lib PRIVATE
  ${SYCLBLAS_INCLUDE}
  ${SYCLBLAS_SRC}
  ${COMPUTECPP_SDK_INCLUDE}
  include/
  ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(tuner_kernel_lib tuner_generate_kernels)
add_sycl_to_target(
  TARGET tuner_kernel_lib
  SOURCES ${tuner_kernel_srcs}
)

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
  target_link_libraries(${tuner_exec} PRIVATE m blas::blas tuner_kernel_lib)
  target_include_directories(${tuner_exec} PRIVATE
    ${SYCLBLAS_INCLUDE}
    ${SYCLBLAS_SRC}
    ${COMPUTECPP_SDK_INCLUDE}
    include/
    ${CMAKE_CURRENT_BINARY_DIR}
  )
  add_dependencies(${tuner_exec} tuner_generate_def_file)
  add_sycl_to_target(
    TARGET ${tuner_exec}
    SOURCES ${blas_tuner}
  )
  install(TARGETS ${tuner_exec} RUNTIME DESTINATION bin)
endforeach()
