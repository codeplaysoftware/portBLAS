cmake_minimum_required(VERSION 3.7)
project(auto_tuner VERSION 0.1.0 LANGUAGES CXX)

set(PORTBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
set(PORTBLAS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/Modules
)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-deprecated-declarations -Wno-shorten-64-to-32 -Wno-cast-align") 
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy-with-user-provided-copy -Wno-unused-variable")

# Setup datatypes, workgroup sizes and other options.
# NB: This has to be included before CmakeFunctionHelper as it declares various options.
include(ConfigurePORTBLAS)
include(SYCL)
find_package(PythonInterp 3 REQUIRED)
find_package(SystemBLAS REQUIRED)

# CmakeFunctionHelper has to be included after any options that it depends on are declared.
# These include:
# * TARGET
# * GEMM_TALL_SKINNY_SUPPORT
# * GEMM_VECTORIZATION_SUPPORT
# * BLAS_DATA_TYPES
# * NAIVE_GEMM
include(CmakeFunctionHelper)

set(tuner_def_file ${CMAKE_CURRENT_BINARY_DIR}/generated_combinations.def)

# The generator's configuration file - add any new jsons to this
set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/default.json)
if(${TUNING_TARGET} STREQUAL "INTEL_GPU")
  set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/intel_gpu.json)
endif()
if(${TUNING_TARGET} STREQUAL "NVIDIA_GPU")
  set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/nvidia_gpu.json)
endif()
if(${TUNING_TARGET} STREQUAL "AMD_GPU")
  set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/amd_gpu.json)
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
# older version of python cannot generate the dir when it does not exist
file(MAKE_DIRECTORY ${_gen_src_dir})
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
  ${PORTBLAS_INCLUDE}
  ${PORTBLAS_SRC}
  include/
  ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(tuner_kernel_lib tuner_generate_kernels)
if(is_dpcpp)
  target_link_libraries(tuner_kernel_lib PRIVATE DPCPP::DPCPP)
endif()

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
  target_link_libraries(${tuner_exec} PRIVATE blas::blas tuner_kernel_lib)
  target_include_directories(${tuner_exec} PRIVATE
    ${PORTBLAS_INCLUDE}
    ${PORTBLAS_SRC}
    include/
    ${CMAKE_CURRENT_BINARY_DIR}
  )
  add_dependencies(${tuner_exec} tuner_generate_def_file)
  if(is_dpcpp)
    target_link_libraries(${tuner_exec} PRIVATE DPCPP::DPCPP)
  endif()
  add_sycl_to_target(
    TARGET ${tuner_exec}
    SOURCES ${blas_tuner}
  )
  set_target_compile_def(${tuner_exec})
  install(TARGETS ${tuner_exec} RUNTIME DESTINATION bin)
endforeach()
