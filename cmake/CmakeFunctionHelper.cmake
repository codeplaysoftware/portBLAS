#/***************************************************************************
# *
# *  @license
# *  Copyright (C) Codeplay Software Limited
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# *  For your convenience, a copy of the License has been included in this
# *  repository.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *  portBLAS: BLAS implementation using SYCL
# *
# *  @filename CmakeFUnctionHelper.cmake
# *
# **************************************************************************/

# represent the list of supported index/increment types
set(index_list "${BLAS_INDEX_TYPES}" )

# BLAS_DATA_TYPES was provided by the user
#Each data type in a data list determines the container types.
#The container type for SYCLbackend is BufferIterator<${data}>
set(data_list "${BLAS_DATA_TYPES}")

# Converts a user specified type name into a C++ type
function(cpp_type output data)
  if (${data} STREQUAL "half")
    set(${output} "cl::sycl::half" PARENT_SCOPE)
    return()
  elseif(${data} STREQUAL "complex<float>")
    set(${output} "cl::sycl::ext::oneapi::experimental::complex<float>" PARENT_SCOPE)
    return()
  elseif(${data} STREQUAL "complex<double>")
    set(${output} "cl::sycl::ext::oneapi::experimental::complex<double>" PARENT_SCOPE)
    return()
  endif()
  set(${output} "${data}" PARENT_SCOPE)
endfunction()

function(set_complex_list output input append)
  set(output_temp "")
  if(${append} STREQUAL "true")
    foreach(data ${input})
      list(APPEND output_temp "${data};complex<${data}>")
    endforeach(data)
  else()
    foreach(data ${input})
      list(APPEND output_temp "complex<${data}>")
    endforeach(data)
  endif()
  set(${output} ${output_temp} PARENT_SCOPE)
endfunction(set_complex_list)

## represent the list of bolean options
set(boolean_list "true" "false")

# Cleans up the proposed file name so that it can be used in the file system
function(sanitize_file_name output file_name)
  string(REGEX REPLACE "(:|\\*|<| |,|>)" "_" file_name ${file_name})
  string(REGEX REPLACE "(_____|____|___|__)" "_" file_name ${file_name})
  if (PORTBLAS_USE_SHORT_NAMES)
    # Long paths are problematic on Windows and WSL so we hash the filename
    # to reduce its size
    string(MD5 file_name ${file_name})
    set(file_name ${file_name}.cpp)
  endif()
  set(${output} "${file_name}" PARENT_SCOPE)
endfunction()

#List of operators supporting Complex Data types
set(COMPLEX_OPS "gemm"
                "gemm_launcher"
                "scal")

#List of operators supporting Half Data types
set(HALF_DATA_OPS "axpy" 
                  "scal"
                  "gemm"
                  "gemm_launcher")

function(set_target_compile_def in_target)
  #setting compiler flag for backend
  if(${TUNING_TARGET} STREQUAL "INTEL_GPU")
    target_compile_definitions(${in_target} PUBLIC INTEL_GPU=1)
  elseif(${TUNING_TARGET} STREQUAL "AMD_GPU")
    target_compile_definitions(${in_target} PUBLIC AMD_GPU=1)
  elseif(${TUNING_TARGET} STREQUAL "POWER_VR")
    target_compile_definitions(${in_target} PUBLIC POWER_VR=1)
  elseif(${TUNING_TARGET} STREQUAL "NVIDIA_GPU")
    target_compile_definitions(${in_target} PUBLIC NVIDIA_GPU=1)
  else()
    if(NOT ${TUNING_TARGET} STREQUAL "DEFAULT")
      message(STATUS "${TUNING_TARGET} not supported. Switching to DEFAULT instead.")
      set(TUNING_TARGET "DEFAULT")
    endif()
    target_compile_definitions(${in_target} PUBLIC DEFAULT=1)
  endif()
  message(STATUS "Adding ${TUNING_TARGET} backend to target ${in_target}")
  #setting tall skinny support
  if(${GEMM_TALL_SKINNY_SUPPORT})
    message(STATUS "Tall and skinny Gemm support enabled for target ${in_target}")
    target_compile_definitions(${in_target} PUBLIC GEMM_TALL_SKINNY_SUPPORT=1)
  endif()
  #setting vectorization support
  if(${GEMM_VECTORIZATION_SUPPORT})
    message(STATUS "Gemm vectorization support enabled for target ${in_target}")
    target_compile_definitions(${in_target} PUBLIC GEMM_VECTORIZATION_SUPPORT=1)
  endif()
  #setting const data type support
  if(BLAS_ENABLE_CONST_INPUT)
    target_compile_definitions(${in_target} PUBLIC BLAS_ENABLE_CONST_INPUT=1)
  endif()
  #setting complex support
  if(${BLAS_ENABLE_COMPLEX})
    if("${in_target}" IN_LIST COMPLEX_OPS)
      message(STATUS "Complex Data type support enabled for target ${in_target}")
      target_compile_definitions(${in_target} PUBLIC BLAS_ENABLE_COMPLEX=1)
    endif()
  endif()
  if(${BLAS_ENABLE_HALF})
    if("${in_target}" IN_LIST HALF_DATA_OPS)
      message(STATUS "Half Data type support enabled for target ${in_target}")
      target_compile_definitions(${in_target} PUBLIC BLAS_ENABLE_HALF=1)
    endif()
  endif()
endfunction()

# blas unary function for generating source code
function(generate_blas_objects blas_level func)
  set(LOCATION "${PORTBLAS_GENERATED_SRC}/${blas_level}/${func}/")
  set(data_list_c ${data_list})
  # Extend data_list to complex<data> for each data in list
  # if target function is in COMPLEX_OPS
  if(BLAS_ENABLE_COMPLEX)
    if("${func}" IN_LIST COMPLEX_OPS)
      set_complex_list(data_list_c "${data_list}" "true")
    endif()
  endif()
  # Extend data_list with 'half' if target function is 
  # in HALF_DATA_OPS
  if(BLAS_ENABLE_HALF)
    if("${func}" IN_LIST HALF_DATA_OPS)
      list(APPEND data_list_c "half")
    endif()
  endif()
  foreach(data ${data_list_c})
    cpp_type(cpp_data ${data})
    foreach(index ${index_list})
      foreach(increment ${index_list})
        sanitize_file_name(file_name
                "${func}_${data}_${index}_${data}_${increment}.cpp")
        add_custom_command(OUTPUT "${LOCATION}/${file_name}"
                COMMAND ${PYTHON_EXECUTABLE} ${PORTBLAS_SRC_GENERATOR}/py_gen_blas_ops.py
                ${PROJECT_SOURCE_DIR}/external/
                ${PORTBLAS_SRC_GENERATOR}/gen
                ${blas_level}
                ${func}
                ${PORTBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                ${cpp_data}
                ${index}
                ${increment}
                ${file_name}
                MAIN_DEPENDENCY ${PORTBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                DEPENDS ${PORTBLAS_SRC_GENERATOR}/py_gen_blas_ops.py
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                VERBATIM
                )
        list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
      endforeach(increment)
    endforeach(index)
  endforeach(data)
  add_library(${func} OBJECT ${FUNC_SRC})
  set_target_compile_def(${func})
  target_include_directories(${func} PRIVATE ${PORTBLAS_SRC} ${PORTBLAS_INCLUDE}
          ${PORTBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
  message(STATUS "Adding SYCL to target ${func}")
  add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_objects)

# blas binary function for generating source code
function(generate_blas_reduction_objects blas_level func)
set(LOCATION "${PORTBLAS_GENERATED_SRC}/${blas_level}/${func}/")
set(operator_list "AddOperator" "MinOperator" "MaxOperator" "ProductOperator" "AbsoluteAddOperator" "MeanOperator")
foreach(data ${data_list})
  cpp_type(cpp_data ${data})
  foreach(index ${index_list})
    foreach(operator ${operator_list})
      foreach(increment ${index_list})
        sanitize_file_name(file_name
                "${func}_${operator}_${data}_${index}_${container0}_${container1}_${increment}.cpp")
        add_custom_command(OUTPUT "${LOCATION}/${file_name}"
                COMMAND ${PYTHON_EXECUTABLE} ${PORTBLAS_SRC_GENERATOR}/py_gen_blas_reduction.py
                ${PROJECT_SOURCE_DIR}/external/
                ${PORTBLAS_SRC_GENERATOR}/gen
                ${blas_level}
                ${func}
                ${PORTBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                ${cpp_data}
                ${index}
                ${increment}
                ${operator}
                ${file_name}
                MAIN_DEPENDENCY ${PORTBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                DEPENDS ${PORTBLAS_SRC_GENERATOR}/py_gen_blas_reduction.py
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                VERBATIM
                )
        list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
      endforeach(increment)
    endforeach(operator)
  endforeach(index)
endforeach(data)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${PORTBLAS_SRC} ${PORTBLAS_INCLUDE}
                           ${PORTBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_reduction_objects)

# blas function for generating source code for the rotg operator (asynchronous version with containers)
function(generate_blas_rotg_objects blas_level func)
  set(LOCATION "${PORTBLAS_GENERATED_SRC}/${blas_level}/${func}/")
  foreach (data ${data_list})
    cpp_type(cpp_data ${data})
    set(container_names "${data}_${data}_${data}_${data}")
    sanitize_file_name(file_name
            "${func}_${data}_${index}_${container_names}_${increment}.cpp")
    add_custom_command(OUTPUT "${LOCATION}/${file_name}"
            COMMAND ${PYTHON_EXECUTABLE} ${PORTBLAS_SRC_GENERATOR}/py_gen_blas_rotg.py
            ${PROJECT_SOURCE_DIR}/external/
            ${PORTBLAS_SRC_GENERATOR}/gen
            ${blas_level}
            ${func}
            ${PORTBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
            ${cpp_data}
            ${file_name}
            MAIN_DEPENDENCY ${PORTBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
            DEPENDS ${PORTBLAS_SRC_GENERATOR}/py_gen_blas_rotg.py
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            VERBATIM
            )
    list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
  endforeach (data)
  add_library(${func} OBJECT ${FUNC_SRC})
  set_target_compile_def(${func})
  target_include_directories(${func} PRIVATE ${PORTBLAS_SRC} ${PORTBLAS_INCLUDE}
          ${PORTBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
  message(STATUS "Adding SYCL to target ${func}")
  add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_rotg_objects)

# blas gemm function for generating source code
function(generate_blas_gemm_objects blas_level func)
set(LOCATION "${PORTBLAS_GENERATED_SRC}/${blas_level}/${func}/")
set(gemm_sources "")

# Generates a file for a new GEMM configuration
# Adds the file to gemm_sources
# If the configuration is not supported by the current settings
# (e.g. double type not enabled), it's ignored
function(add_gemm_configuration
  data
  wg_size
  double_buffer
  conflict_a
  conflict_b
  cache_line_size
  tir
  tic
  twr
  twc
  tsr
  tsc
  tlr
  tlc
  item_batch wg_batch
  jm_m jm_n jm_k 
  jm_in_type jm_out_type
  gemm_memory_type
  gemm_shape_type
  gemm_vectorize_type
  vector_size
  batch_type
  use_joint_matrix
)
  set(data_list_c ${data_list})
  if(BLAS_ENABLE_COMPLEX)
    set_complex_list(data_list_c "${data_list}" "true")
  endif()
  if(BLAS_ENABLE_HALF)
    list(APPEND data_list_c "half")
  endif()
  if(NOT ("${data}" IN_LIST data_list_c))
    # Data type not enabled, skip configuration
    return()
  endif()
  if(("${gemm_shape_type}" STREQUAL "tall_skinny") AND NOT GEMM_TALL_SKINNY_SUPPORT)
    # Tall/skinny configurations not enabled, skip
    return()
  endif()
  string(FIND ${func} "_const" const_pos)
  if(const_pos)
    string(REPLACE "_const" "" actualfunc ${func})
  endif()
  cpp_type(cpp_data ${data})
  foreach(symm_a ${boolean_list})
    foreach(symm_b ${boolean_list})
      if ((${data} MATCHES "half") AND (symm_a OR symm_b))
        continue()
      endif()
      if (symm_a AND symm_b)
        continue()
      endif()
      foreach(trans_a ${boolean_list})
        foreach(trans_b ${boolean_list})
          if ((symm_a AND trans_b) OR (symm_b AND trans_a))
            continue()
          endif()
          foreach(is_beta_zero ${boolean_list})
            foreach(index ${index_list})
              set(file_name "${func}_${double_buffer}_${conflict_a}_"
                      "${conflict_b}_${trans_a}_${trans_b}_"
                      "${is_beta_zero}_${gemm_memory_type}_"
                      "${gemm_shape_type}_${gemm_vectorize_type}_"
                      "${vector_size}_${batch_type}_${use_joint_matrix}_"
                      "${data}_${index}_${tir}_${tic}_${twr}_"
                      "${twc}_${tsr}_${tsc}_${tlr}_${tlc}_"
                      "${item_batch}_${wg_batch}_${symm_a}_${symm_b}_"
                      "${jm_m}_${jm_n}_${jm_k}_${jm_in_type}_${jm_out_type}_"
                      "${wg_size}_${cache_line_size}_${data}.cpp")
              sanitize_file_name(file_name "${file_name}")
              add_custom_command(OUTPUT "${LOCATION}/${file_name}"
                      COMMAND ${PYTHON_EXECUTABLE} ${PORTBLAS_SRC_GENERATOR}/py_gen_blas_gemm_launcher.py
                      ${PROJECT_SOURCE_DIR}/external/
                      ${PORTBLAS_SRC_GENERATOR}/gen
                      ${blas_level}
                      ${func}
                      ${PORTBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                      ${cpp_data}
                      ${index}
                      ${double_buffer}
                      ${conflict_a}
                      ${conflict_b}
                      ${trans_a}
                      ${trans_b}
                      ${is_beta_zero}
                      ${gemm_memory_type}
                      ${gemm_shape_type}
                      ${tir}
                      ${tic}
                      ${twr}
                      ${twc}
                      ${tsr}
                      ${tsc}
                      ${tlr}
                      ${tlc}
                      ${item_batch}
                      ${wg_batch}
                      ${jm_m}
                      ${jm_n}
                      ${jm_k}
                      ${jm_in_type}
                      ${jm_out_type}
                      ${wg_size}
                      ${cache_line_size}
                      ${file_name}
                      ${gemm_vectorize_type}
                      ${vector_size}
                      ${batch_type}
                      ${use_joint_matrix}
                      ${symm_a}
                      ${symm_b}
                      MAIN_DEPENDENCY ${PORTBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                      DEPENDS ${PORTBLAS_SRC_GENERATOR}/py_gen_blas_gemm_launcher.py
                      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                      VERBATIM
                      )
              list(APPEND gemm_sources "${LOCATION}/${file_name}")
              set(gemm_sources "${gemm_sources}" PARENT_SCOPE)
            endforeach(index)
          endforeach(is_beta_zero)
        endforeach(trans_b)
      endforeach(trans_a)
    endforeach(symm_b)
  endforeach(symm_a)
endfunction()
if(${TUNING_TARGET} STREQUAL "INTEL_GPU")
  set(supported_types
    "float"
    "double"
  )
  foreach(data ${supported_types})
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false")
    
    add_gemm_configuration(
      "${data}" 32 "true" "true" "true"
      64 2 1 8 4 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 16 "true" "false" "false"
      64 1 1 4 4 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 32 "true" "true" "true"
      64 2 2 8 4 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 16 "true" "false" "false"
      64 2 2 4 4 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "true" "true" "true"
      64 2 2 8 8 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "true" "false" "false"
      64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 4 "strided" "false")
    
    if (${data} STREQUAL "double")
      add_gemm_configuration(
        "${data}" 256 "true" "true" "true"
        64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    else()
      add_gemm_configuration(
        "${data}" 256 "true" "true" "true"
        64 4 4 16 16 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    endif()

    add_gemm_configuration(
      "${data}" 64 "true" "true" "true"
      64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 8 8 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 8 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 4 "strided" "false")
  endforeach()
  if(BLAS_ENABLE_HALF)
    add_gemm_configuration(
      "half" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false")
    add_gemm_configuration(
      "half" 16 "true" "false" "false"
      64 2 2 8 8 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    add_gemm_configuration(
      "half" 64 "false" "false" "false"
      64 4 8 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 4 "strided" "false")
  endif()

  if(BLAS_ENABLE_COMPLEX)
    # Extract list of complex<data> for each data in supported_types
    # list for complex<data> specific gemm configurations
    set(data_list_c)
    set_complex_list(data_list_c "${supported_types}" "false")
    foreach(data ${data_list_c})
      add_gemm_configuration(
        "${data}" 64 "true" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 8 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 8 8 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 1 "strided" "false")
      if (${data} STREQUAL "complex<double>")
        add_gemm_configuration(
          "${data}" 64 "true" "true" "true"
          64 4 4 4 4 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 1 "strided" "false")      
      else()
        add_gemm_configuration(
          "${data}" 64 "true" "true" "true"
          64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 1 "strided" "false")
      endif()
    endforeach()
  endif() # BLAS_ENABLE_COMPLEX
elseif(${TUNING_TARGET} STREQUAL "POWER_VR" AND NOT IMGDNN_DIR)
  set(supported_types
    "float"
    "half"
  )
  foreach(data ${supported_types})
    add_gemm_configuration(
      "${data}" 96 "true" "false" "false"
      16 4 6 12 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      128 1 1 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 128 "false" "false" "false"
      16 4 8 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      32 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false")
  endforeach()
elseif(${TUNING_TARGET} STREQUAL "AMD_GPU")  # need investigation
  set(supported_types
    "float"
    "double"
  )
  if(BLAS_ENABLE_HALF)
    list(APPEND supported_types "half")
  endif()
  set(workgroup_float 16)
  set(workgroup_double 8)
  set(workgroup_half 32)
  foreach(data ${supported_types})
    set(twr "${workgroup_${data}}")
    set(twc "${workgroup_${data}}")

    # General configuration
    add_gemm_configuration(
      "${data}" 256 "false" "false" "false"
      64 4 4 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 2 "strided" "false")

    # configuration for tall_skinny
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 1 1 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 2 "strided" "false")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 2 2 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 2 "strided" "false")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 4 4 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 2 "strided" "false")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 1 4 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 2 "strided" "false")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 4 1 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 2 "strided" "false")

    # configuration for batch
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false")

    # Configurations for gemm

    # low arithmetic intensity
    add_gemm_configuration(
      "${data}" 256 "false" "false" "true"
      128 1 1 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 256 "false" "false" "true"
      64 4 8 16 16 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    # highest arithmetic intensity
    add_gemm_configuration(
      "${data}" 256 "false" "false" "true"
      32 8 8 16 16 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    # high arithmetic intensity
    add_gemm_configuration(
      "${data}" 256 "false" "false" "true"
      64 4 4 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    # mid high 162 < a < 240 
    add_gemm_configuration(
      "${data}" 256 "false" "false" "true"
      128 4 4 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    # mid low 100 < a < 162
    add_gemm_configuration(
      "${data}" 256 "false" "true" "true"
      128 2 2 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")

  endforeach()
  if(BLAS_ENABLE_COMPLEX)
    # Extract list of complex<data> for each data in supported_types
    # list for complex<data> specific gemm configurations
    set(data_list_c)
    set_complex_list(data_list_c "${supported_types}" "false")
    foreach(data ${data_list_c})
      if (${data} STREQUAL "complex<double>")
        add_gemm_configuration(
          "${data}" 256 "true" "true" "true"
          64 1 4 4 4 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 1 "strided" "false")
        add_gemm_configuration(
          "${data}" 256 "false" "false" "false"
          64 1 1 4 4 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
        add_gemm_configuration(
          "${data}" 256 "false" "false" "false"
          64 4 4 4 4 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
      else()
        add_gemm_configuration(
          "${data}" 256 "true" "true" "true"
          64 1 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 1 "strided" "false")
        add_gemm_configuration(
          "${data}" 256 "false" "false" "false"
          64 1 1 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
        add_gemm_configuration(
          "${data}" 256 "false" "false" "false"
          64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
      endif()
    endforeach()
  endif() # BLAS_ENABLE_COMPLEX
elseif(${TUNING_TARGET} STREQUAL "NVIDIA_GPU")
  set(supported_types
    "float"
    "double"
    )
  if(is_dpcpp AND DEFINED DPCPP_SYCL_ARCH)
    string(FIND ${DPCPP_SYCL_ARCH} "_" start_idx)
    if(start_idx)
      MATH(EXPR start_idx "${start_idx} + 1")
      string(SUBSTRING ${DPCPP_SYCL_ARCH} ${start_idx} "2" sm_val)
    endif()
  endif()
  # Joint Matrix specific GEMM configurations (only for float)
  if(${start_idx} AND ${sm_val} GREATER_EQUAL "80")
    add_gemm_configuration(
        "float" 128 "false" "true" "true"
        128 2 4 16 8 16 2 1 1 1 1 16 16 16 cl::sycl::half float "local" "standard" "none" 1 "strided" "true")
    add_gemm_configuration(
        "float" 128 "false" "true" "true"
        128 4 8 16 8 16 2 1 1 1 1 16 16 16 cl::sycl::half float "local" "standard" "none" 1 "strided" "true")
    add_gemm_configuration(
        "float" 256 "false" "true" "true"
        128 8 8 16 16 16 2 1 1 1 1 16 16 16 cl::sycl::half float "local" "standard" "none" 1 "strided" "true")
  endif()
  foreach(data ${supported_types})
    # Non-Joint Matrix specific GEMM Configurations
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 2 2 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false")
    add_gemm_configuration(
        "${data}" 128 "false" "true" "true"
      128 2 2 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
        "${data}" 128 "false" "true" "true"
      128 4 4 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
     add_gemm_configuration(
        "${data}" 128 "false" "true" "true"
      128 8 8 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
        "${data}" 256 "false" "true" "true"
      128 8 8 16 16 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
        "${data}"  64 "false" "false" "true"
          64 8 8 8 8 1 1 2 2 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
  endforeach()
  if(BLAS_ENABLE_COMPLEX)
    # Extract list of complex<data> for each data in supported_types
    # list for complex<data> specific gemm configurations
    set(data_list_c)
    set_complex_list(data_list_c "${supported_types}" "false")
    foreach(data ${data_list_c})
      add_gemm_configuration(
        "${data}"  256 "false" "false" "true"
          64 2 2 16 16 1 1 2 2 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    endforeach()
  endif() # BLAS_ENABLE_COMPLEX

  if(BLAS_ENABLE_HALF)
    add_gemm_configuration(
      "half" 64 "false" "false" "false"
      64 2 2 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false")
    add_gemm_configuration(
      "half" 256 "false" "true" "true"
      128 4 4 16 16 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "half" 256 "false" "true" "true"
      128 8 8 16 16 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
  endif() # BLAS_ENABLE_HALF

else() # default cpu backend
  set(supported_types
    "float"
    "double"
  )
  foreach(data ${supported_types})
    if(NAIVE_GEMM)
      add_gemm_configuration(
        "${data}"  64 "false" "false" "false"
        64 8 8 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "naive" "none" 1 "strided" "false" "false")
    else()
      add_gemm_configuration(
        "${data}"  128 "false" "false" "false"
        64 2 2 2 2 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "full" 2 "strided" "false" "false")
      add_gemm_configuration(
        "${data}"  128 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "full" 1 "strided" "false" "false")
      add_gemm_configuration(
        "${data}"  128 "false" "false" "false"
        64 4 4 4 4 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 1 "strided" "false" "false")
      add_gemm_configuration(
        "${data}"  64 "false" "false" "false"
        64 2 2 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 2 "strided" "false" "false")

    endif()

    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 2 2 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false" "false")

    if(BLAS_ENABLE_HALF)
      add_gemm_configuration(
        "half"  128 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "full" 1 "strided" "false" "false")
      add_gemm_configuration(
        "half" 64 "false" "false" "false"
        64 2 2 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false" "false")
    endif()
  endforeach()

  if(BLAS_ENABLE_COMPLEX)
    # Extract list of complex<data> for each data in supported_types
    # list for complex<data> specific gemm configurations
    set(data_list_c)
    set_complex_list(data_list_c "${supported_types}" "false")
    foreach(data ${data_list_c})
      add_gemm_configuration(
        "${data}"  64 "false" "false" "false"
        64 2 2 4 4 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "full" 1 "strided" "false" "false")
      add_gemm_configuration(
        "${data}"  64 "false" "false" "false"
        64 8 8 4 4 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 1 "strided" "false" "false")
    endforeach()
  endif() # BLAS_ENABLE_COMPLEX
endif()
add_library(${func} OBJECT ${gemm_sources})
set_target_compile_def(${func})
# The blas library depends on FindComputeCpp
target_include_directories(${func} PRIVATE ${PORTBLAS_SRC} ${PORTBLAS_INCLUDE}
                           ${PORTBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${gemm_sources})
endfunction(generate_blas_gemm_objects)


function (build_library LIB_NAME ENABLE_EXTENSIONS)
  set(LIB_SRCS  $<TARGET_OBJECTS:axpy>
                $<TARGET_OBJECTS:asum>
                $<TARGET_OBJECTS:asum_return>
                $<TARGET_OBJECTS:copy>
                $<TARGET_OBJECTS:dot>
                $<TARGET_OBJECTS:dot_return>
                $<TARGET_OBJECTS:sdsdot>
                $<TARGET_OBJECTS:sdsdot_return>
                $<TARGET_OBJECTS:iamax>
                $<TARGET_OBJECTS:iamax_return>
                $<TARGET_OBJECTS:iamin>
                $<TARGET_OBJECTS:iamin_return>
                $<TARGET_OBJECTS:nrm2>
                $<TARGET_OBJECTS:nrm2_return>
                $<TARGET_OBJECTS:rot>
                $<TARGET_OBJECTS:rotm>
                $<TARGET_OBJECTS:rotmg>
                $<TARGET_OBJECTS:rotg>
                $<TARGET_OBJECTS:rotg_return>
                $<TARGET_OBJECTS:scal>
                $<TARGET_OBJECTS:swap>
                $<TARGET_OBJECTS:gbmv>
                $<TARGET_OBJECTS:gemv>
                $<TARGET_OBJECTS:ger>
                $<TARGET_OBJECTS:sbmv>
                $<TARGET_OBJECTS:spmv>
                $<TARGET_OBJECTS:symv>
                $<TARGET_OBJECTS:syr>
                $<TARGET_OBJECTS:spr>
                $<TARGET_OBJECTS:spr2>
                $<TARGET_OBJECTS:syr2>
                $<TARGET_OBJECTS:tbmv>
                $<TARGET_OBJECTS:tpmv>
                $<TARGET_OBJECTS:tbsv>
                $<TARGET_OBJECTS:tpsv>
                $<TARGET_OBJECTS:trmv>
                $<TARGET_OBJECTS:trsv>
                $<TARGET_OBJECTS:gemm_launcher>
                $<TARGET_OBJECTS:gemm>
                $<TARGET_OBJECTS:symm>
                $<TARGET_OBJECTS:trsm>
                $<TARGET_OBJECTS:matcopy>
                $<TARGET_OBJECTS:matcopy_batch>
                $<TARGET_OBJECTS:transpose>
                $<TARGET_OBJECTS:omatadd>
                $<TARGET_OBJECTS:omatadd_batch>
                $<TARGET_OBJECTS:axpy_batch>)

   if (${ENABLE_EXTENSIONS})
     list(APPEND LIB_SRCS $<TARGET_OBJECTS:reduction>)
   endif()

  add_library(${LIB_NAME} ${LIB_SRCS})

endfunction(build_library)
