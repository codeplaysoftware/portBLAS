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
# *  SYCL-BLAS: BLAS implementation using SYCL
# *
# *  @filename CmakeFUnctionHelper.cmake
# *
# **************************************************************************/
# represent the list of supported handler for executor
set(executor_list "PolicyHandler<codeplay_policy>")
#represent the list of supported index/increment type
set(index_list "int" )

# BLAS_DATA_TYPES represents the list of supported data type.
#Each data type in a data list determines the container types.
#The container type for SYCLbackend is BufferIterator<${data}, codeplay_policy>

## represent the list of bolean options
set(boolean_list "true" "false")

# Maps a user provided data type name to the C++ type
# See BLAS_DATA_TYPES
function(to_cpp_type output data)
  set(${output} "${data}" PARENT_SCOPE)
endfunction()

# Strips some C++ symbols from the string to make it suitable to be used
# as part of a filename
function(cpp_type_to_name output cpp_type)
  string(REGEX REPLACE "[<|>|,|:]+" "" temp_var "${cpp_type}")
  set(${output} "${temp_var}" PARENT_SCOPE)
endfunction()

# gemm_configuration(data, work_group_size, double_buffer, conflict_a, conflict_b,
#                    cache_line_size, tir, tic, twr, twc, tlr, tlc, item batch, wg batch, local_mem,
#                    gemm_type, vectorization_type, vector size, batch type)
set(gemm_configuration_lists "")

#intel GPU
if(${TARGET} STREQUAL "INTEL_GPU")
  set(gemm_configuration_0 "float" 64 "true" "false" "false" 64 4 4 8 8 1 1 1 1 "local" "standard" "full" 4 "strided")
  set(gemm_configuration_1 "float" 64 "false" "false" "false" 64 8 8 8 8 1 1 1 1 "local" "standard" "full" 4 "strided")
  set(gemm_configuration_2 "float" 64 "false" "false" "false" 64 8 8 8 8 1 1 1 1 "no_local" "standard" "partial" 4 "strided")

  set(gemm_configuration_3 "float" 16 "true" "false" "false" 64 1 1 4 4 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
  set(gemm_configuration_4 "float" 16 "true" "false" "false" 64 2 2 4 4 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
  set(gemm_configuration_5 "float" 64 "true" "true" "true" 64 2 2 8 8 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
  set(gemm_configuration_6 "float" 64 "true" "true" "true" 64 4 4 8 8 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
  set(gemm_configuration_7 "float" 256 "true" "true" "true" 64 4 4 16 16 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
  set(gemm_configuration_8 "float" 32 "true" "true" "true" 64 2 1 8 4 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
  set(gemm_configuration_9 "float" 32 "true" "true" "true" 64 2 2 8 4 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")

  set(gemm_configuration_10 "double" 64 "true" "false" "false" 64 4 4 8 8 1 1 1 1 "local" "standard" "full" 4 "strided")
  set(gemm_configuration_11 "double" 64 "true" "false" "false" 64 8 8 8 8 1 1 1 1 "local" "standard" "full" 4 "strided")
  set(gemm_configuration_12 "double" 64 "false" "false" "false" 64 8 8 8 8 1 1 1 1 "no_local" "standard" "partial" 4 "strided")

  set(gemm_configuration_13 "double" 16 "true" "false" "false" 64 1 1 4 4 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
  set(gemm_configuration_14 "double" 16 "true" "false" "false" 64 2 2 4 4 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
  set(gemm_configuration_15 "double" 64 "true" "true" "true" 64 2 2 8 8 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
  set(gemm_configuration_16 "double" 64 "true" "true" "true" 64 4 4 8 8 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
  set(gemm_configuration_17 "double" 32 "true" "true" "true" 64 2 1 8 4 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
  set(gemm_configuration_18 "double" 32 "true" "true" "true" 64 2 2 8 4 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")

  set(gemm_configuration_19 "float" 64 "false" "false" "false" 64 4 4 4 4 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")
  set(gemm_configuration_20 "double" 64 "false" "false" "false" 64 4 4 4 4 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")

  list(APPEND gemm_configuration_lists gemm_configuration_0 gemm_configuration_1
                                       gemm_configuration_2 gemm_configuration_19)

  if("double" IN_LIST BLAS_DATA_TYPES)
    list(APPEND gemm_configuration_lists
            gemm_configuration_10
            gemm_configuration_11
            gemm_configuration_12
            gemm_configuration_20)
  endif()

  if(GEMM_TALL_SKINNY_SUPPORT)
    list(APPEND gemm_configuration_lists gemm_configuration_3
                                         gemm_configuration_4
                                         gemm_configuration_5
                                         gemm_configuration_6
                                         gemm_configuration_7
                                         gemm_configuration_8
                                         gemm_configuration_9)

    if("double" IN_LIST BLAS_DATA_TYPES)
      list(APPEND gemm_configuration_lists
              gemm_configuration_13
              gemm_configuration_14
              gemm_configuration_15
              gemm_configuration_16
              gemm_configuration_17
              gemm_configuration_18)
    endif()
  endif()
elseif(${TARGET} STREQUAL "RCAR") # need investigation

  set(gemm_configuration_0 "float" 32 "false" "false" "false" 128 4 8 8 4 1 1 1 1 "local" "standard" "full" 4 "strided")
  set(gemm_configuration_1 "float" 32 "false" "false" "false" 128 8 4 4 8 1 1 1 1 "local" "standard" "full" 4 "strided")
  set(gemm_configuration_2 "float" 64 "false" "false" "false" 64 4 4 4 4 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")

  list(APPEND gemm_configuration_lists gemm_configuration_0 gemm_configuration_1 gemm_configuration_2)
elseif(${TARGET} STREQUAL "ARM_GPU")
  set(gemm_configuration_0 "float" 64 "false" "false" "false" 64 4 4 8 8 1 1 1 1 "no_local" "standard" "partial" 2 "strided")
  set(gemm_configuration_1 "float" 128 "false" "false" "false" 64 4 8 16 8 1 1 1 1 "no_local" "standard" "partial" 4 "strided")
  set(gemm_configuration_2 "float" 64 "false" "false" "false" 64 4 4 4 4 1 1 1 1 "no_local" "standard" "partial" 2 "strided")
  set(gemm_configuration_3 "float" 64 "false" "false" "false" 64 2 2 4 4 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")

  list(APPEND gemm_configuration_lists gemm_configuration_0 gemm_configuration_1
                                       gemm_configuration_2 gemm_configuration_3)
elseif(${TARGET} STREQUAL "POWER_VR")
  set(gemm_configuration_0 "float" 96 "true" "false" "false" 16 4 6 12 8 1 1 1 1 "local" "standard" "full" 1 "strided")
  set(gemm_configuration_1 "float" 64 "false" "false" "false" 128 1 1 8 8 1 1 1 1 "local" "standard" "full" 1 "strided")
  set(gemm_configuration_2 "float" 64 "false" "false" "false" 64 4 4 8 8 1 1 1 1 "no_local" "standard" "full" 1 "strided")
  set(gemm_configuration_3 "float" 128 "false" "false" "false" 16 4 8 16 8 1 1 1 1 "local" "standard" "full" 1 "strided")
  set(gemm_configuration_4 "float" 64 "false" "false" "false" 32 4 4 8 8 1 1 1 1 "local" "standard" "full" 1 "strided")
  set(gemm_configuration_5 "float" 64 "false" "false" "false" 64 4 4 4 4 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")
  list(APPEND gemm_configuration_lists gemm_configuration_0 gemm_configuration_1
                                    gemm_configuration_2 gemm_configuration_3
                                    gemm_configuration_4 gemm_configuration_5)
elseif(${TARGET} STREQUAL "AMD_GPU")  # need investigation
  set(gemm_configuration_0 "float" 256 "false" "false" "false" 64 1 1 16 16 1 1 1 1 "local" "standard" "full" 1 "strided")
  set(gemm_configuration_1 "float" 256 "false" "false" "false" 64 4 4 16 16 1 1 1 1 "local" "standard" "full" 2 "strided")

  set(gemm_configuration_2 "float" 256 "true" "true" "true" 64 1 1 16 16 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
  set(gemm_configuration_3 "float" 256 "true" "true" "true" 64 2 2 16 16 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
  set(gemm_configuration_4 "float" 256 "true" "true" "true" 64 4 4 16 16 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
  set(gemm_configuration_5 "float" 256 "true" "true" "true" 64 1 4 16 16 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
  set(gemm_configuration_6 "float" 256 "true" "true" "true" 64 4 1 16 16 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")

  set(gemm_configuration_7 "double" 256 "false" "false" "false" 64 1 1 8 8 1 1 1 1 "local" "standard" "full" 1 "strided")
  set(gemm_configuration_8 "double" 256 "false" "false" "false" 64 4 4 8 8 1 1 1 1 "local" "standard" "full" 2 "strided")

  set(gemm_configuration_9 "double" 256 "true" "true" "true" 64 1 1 8 8 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
  set(gemm_configuration_10 "double" 256 "true" "true" "true" 64 2 2 8 8 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
  set(gemm_configuration_11 "double" 256 "true" "true" "true" 64 4 4 8 8 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
  set(gemm_configuration_12 "double" 256 "true" "true" "true" 64 1 4 8 8 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
  set(gemm_configuration_13 "double" 256 "true" "true" "true" 64 4 1 8 8 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")

  set(gemm_configuration_14 "float" 64 "false" "false" "false" 64 4 4 4 4 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")
  set(gemm_configuration_15 "double" 64 "false" "false" "false" 64 4 4 4 4 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")

  list(APPEND gemm_configuration_lists
            gemm_configuration_0
            gemm_configuration_1
            gemm_configuration_14)

  if("double" IN_LIST BLAS_DATA_TYPES)
    list(APPEND gemm_configuration_lists
            gemm_configuration_7
            gemm_configuration_8
            gemm_configuration_15)
  endif()

  if(GEMM_TALL_SKINNY_SUPPORT)
    list(APPEND gemm_configuration_lists gemm_configuration_2
                                         gemm_configuration_3
                                         gemm_configuration_4
                                         gemm_configuration_5
                                         gemm_configuration_6
                                         )

    if("double" IN_LIST BLAS_DATA_TYPES)
      list(APPEND gemm_configuration_lists
              gemm_configuration_9
              gemm_configuration_10
              gemm_configuration_11
              gemm_configuration_12
              gemm_configuration_13)
    endif()
  endif()
else() # default cpu backend
  set(gemm_configuration_0 "float"  64 "false" "false" "false" 64 8 8 8 8 1 1 1 1 "no_local" "naive" "none" 1 "strided")
  set(gemm_configuration_1 "float"  64 "false" "false" "false" 64 2 2 8 8 1 1 1 1 "no_local" "standard" "full" 2 "strided")
  set(gemm_configuration_2 "float"  64 "false" "false" "false" 64 8 8 8 8 1 1 1 1 "no_local" "standard" "partial" 1 "strided")
  set(gemm_configuration_3 "double" 64 "false" "false" "false" 64 8 8 8 8 1 1 1 1 "no_local" "naive" "none" 1 "strided")
  set(gemm_configuration_4 "double" 64 "false" "false" "false" 64 2 2 8 8 1 1 1 1 "no_local" "standard" "full" 2 "strided")
  set(gemm_configuration_5 "double" 64 "false" "false" "false" 64 8 8 8 8 1 1 1 1 "no_local" "standard" "partial" 1 "strided")

  set(gemm_configuration_6 "float" 64 "false" "false" "false" 64 2 2 4 4 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")
  set(gemm_configuration_7 "double" 64 "false" "false" "false" 64 2 2 4 4 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")

  if(NAIVE_GEMM)
    list(APPEND gemm_configuration_lists gemm_configuration_0)
    if("double" IN_LIST BLAS_DATA_TYPES)
      list(APPEND gemm_configuration_lists gemm_configuration_3)
    endif()
  else()
    list(APPEND gemm_configuration_lists gemm_configuration_1 gemm_configuration_2)
    if("double" IN_LIST BLAS_DATA_TYPES)
      list(APPEND gemm_configuration_lists gemm_configuration_4 gemm_configuration_5)
    endif()
  endif()
  list(APPEND gemm_configuration_lists gemm_configuration_6)
  if("double" IN_LIST BLAS_DATA_TYPES)
    list(APPEND gemm_configuration_lists gemm_configuration_7)
  endif()
endif()


function(set_target_compile_def in_target)
  #setting compiler flag for backend
  if(${BACKEND_DEVICE} STREQUAL "INTEL_GPU")
    target_compile_definitions(${in_target} PUBLIC INTEL_GPU=1)
  elseif(${BACKEND_DEVICE} STREQUAL "AMD_GPU")
    target_compile_definitions(${in_target} PUBLIC AMD_GPU=1)
  elseif(${BACKEND_DEVICE} STREQUAL "ARM_GPU")
    target_compile_definitions(${in_target} PUBLIC ARM_GPU=1)
  elseif(${BACKEND_DEVICE} STREQUAL "RCAR")
    target_compile_definitions(${in_target} PUBLIC RCAR=1)
  elseif(${BACKEND_DEVICE} STREQUAL "ARM_GPU")
    target_compile_definitions(${in_target} PUBLIC ARM_GPU=1)
  elseif(${BACKEND_DEVICE} STREQUAL "POWER_VR")
    target_compile_definitions(${in_target} PUBLIC POWER_VR=1)
  else()
    target_compile_definitions(${in_target} PUBLIC DEFAULT_CPU=1)
  endif()
  #setting tall skinny support
  if(${GEMM_TALL_SKINNY_SUPPORT})
    target_compile_definitions(${in_target} PUBLIC GEMM_TALL_SKINNY_SUPPORT=1)
  endif()
  #setting vectorization support
  if(${GEMM_VECTORIZATION_SUPPORT})
    target_compile_definitions(${in_target} PUBLIC GEMM_VECTORIZATION_SUPPORT=1)
  endif()

endfunction()


# blas unary function for generating source code
function(generate_blas_unary_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${BLAS_DATA_TYPES})
    to_cpp_type(cpp_data "${data}")
    set(container_list "BufferIterator<${cpp_data},codeplay_policy>")
    foreach(index ${index_list})
      foreach(container0 ${container_list})
        cpp_type_to_name(container_name0 "${container0}")
        foreach(increment ${index_list})
          set(file_name
            "${func}_${executor}_${data}_${index}_${container_name0}_${increment}.cpp")
          STRING(REGEX REPLACE "(\\*|<| |,|>)" "_" file_name ${file_name})
          STRING(REGEX REPLACE "(___|__)" "_" file_name ${file_name})
          add_custom_command(OUTPUT "${LOCATION}/${file_name}"
            COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_unary.py
              ${PROJECT_SOURCE_DIR}/external/
              ${SYCLBLAS_SRC_GENERATOR}/gen
              ${blas_level}
              ${func}
              ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
              ${executor}
              ${cpp_data}
              ${index}
              ${increment}
              ${container0}
              ${file_name}
            MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
            DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_unary.py
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            VERBATIM
          )
          list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
        endforeach(increment)
      endforeach(container0)
    endforeach(index)
  endforeach(data)
endforeach(executor)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_unary_objects)


# blas binary function for generating source code
function(generate_blas_binary_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${BLAS_DATA_TYPES})
    to_cpp_type(cpp_data "${data}")
    set(container_list "BufferIterator<${cpp_data},codeplay_policy>")
    foreach(index ${index_list})
      foreach(container0 ${container_list})
        cpp_type_to_name(container_name0 "${container0}")
        foreach(container1 ${container_list})
          cpp_type_to_name(container_name1 "${container1}")
          set(container_names "${container_name0}_${container_name1}")
          foreach(increment ${index_list})
            set(file_name
              "${func}_${executor}_${data}_${index}_${container_names}_${increment}.cpp")
            STRING(REGEX REPLACE "(\\*|<| |,|>)" "_" file_name ${file_name})
            STRING(REGEX REPLACE "(___|__)" "_" file_name ${file_name})
            add_custom_command(OUTPUT "${LOCATION}/${file_name}"
              COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary.py
                ${PROJECT_SOURCE_DIR}/external/
                ${SYCLBLAS_SRC_GENERATOR}/gen
                ${blas_level}
                ${func}
                ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                ${executor}
                ${cpp_data}
                ${index}
                ${increment}
                ${container0}
                ${container1}
                ${file_name}
              MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
              DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary.py
              WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
              VERBATIM
            )
            list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
            endforeach(increment)
        endforeach(container1)
      endforeach(container0)
    endforeach(index)
  endforeach(data)
endforeach(executor)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_binary_objects)



# blas special binary function for generating source code
function(generate_blas_binary_special_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${BLAS_DATA_TYPES})
    to_cpp_type(cpp_data "${data}")
    set(container_list_in "BufferIterator<${cpp_data},codeplay_policy>")
    foreach(index ${index_list})
      set(container_list_out
        "BufferIterator<IndexValueTuple<${index},${cpp_data}>,codeplay_policy>")
      foreach(container0 ${container_list_in})
        cpp_type_to_name(container_name0 "${container0}")
        foreach(container1 ${container_list_out})
          cpp_type_to_name(container_name1 "${container1}")
          set(container_names "${container_name0}_${container_name1}")
          foreach(increment ${index_list})
            set(file_name
              "${func}_${executor}_${data}_${index}_${container_names}_${increment}.cpp")
            STRING(REGEX REPLACE "(\\*|<| |,|>)" "_" file_name ${file_name})
            STRING(REGEX REPLACE "(___|__)" "_" file_name ${file_name})
            add_custom_command(OUTPUT "${LOCATION}/${file_name}"
              COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary_special.py
                ${PROJECT_SOURCE_DIR}/external/
                ${SYCLBLAS_SRC_GENERATOR}/gen
                ${blas_level}
                ${func}
                ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                ${executor}
                ${cpp_data}
                ${index}
                ${increment}
                ${container0}
                ${container1}
                ${file_name}
              MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
              DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary_special.py
              WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
              VERBATIM
            )
            list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
            endforeach(increment)
        endforeach(container1)
      endforeach(container0)
    endforeach(index)
  endforeach(data)
endforeach(executor)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_binary_special_objects)



# blas ternary function for generating source code
function(generate_blas_ternary_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${BLAS_DATA_TYPES})
    to_cpp_type(cpp_data "${data}")
    set(container_list "BufferIterator<${cpp_data},codeplay_policy>")
    foreach(index ${index_list})
      foreach(container0 ${container_list})
        cpp_type_to_name(container_name0 "${container0}")
        foreach(container1 ${container_list})
          cpp_type_to_name(container_name1 "${container1}")
          foreach(container2 ${container_list})
            cpp_type_to_name(container_name2 "${container2}")
            set(container_names
              "${container_name0}_${container_name1}_${container_name2}")
            foreach(increment ${index_list})
              set(file_name
                "${func}_${executor}_${data}_${index}_${container_names}_${increment}.cpp")
              STRING(REGEX REPLACE "(\\*|<| |,|>)" "_" file_name ${file_name})
              STRING(REGEX REPLACE "(___|__)" "_" file_name ${file_name})
              add_custom_command(OUTPUT "${LOCATION}/${file_name}"
                COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_ternary.py
                  ${PROJECT_SOURCE_DIR}/external/
                  ${SYCLBLAS_SRC_GENERATOR}/gen
                  ${blas_level}
                  ${func}
                  ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                  ${executor}
                  ${cpp_data}
                  ${index}
                  ${increment}
                  ${container0}
                  ${container1}
                  ${container2}
                  ${file_name}
                MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_ternary.py
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                VERBATIM
              )
              list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
            endforeach(increment)
          endforeach(container2)
        endforeach(container1)
      endforeach(container0)
    endforeach(index)
  endforeach(data)
endforeach(executor)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_ternary_objects)


# blas gemm function for generating source code
function(generate_blas_gemm_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
      foreach(trans_a ${boolean_list})
        foreach(trans_b ${boolean_list})
          foreach(is_beta_zero ${boolean_list})
            foreach(executor ${executor_list})
                foreach(index ${index_list})
                  foreach(gemm_list ${gemm_configuration_lists})
                    list(GET ${gemm_list} 0 data)
                    list(GET ${gemm_list} 1 wg_size)
                    list(GET ${gemm_list} 2 double_buffer)
                    list(GET ${gemm_list} 3 conflict_a)
                    list(GET ${gemm_list} 4 conflict_b)
                    list(GET ${gemm_list} 5 cl_size)
                    list(GET ${gemm_list} 6 tir)
                    list(GET ${gemm_list} 7 tic)
                    list(GET ${gemm_list} 8 twr)
                    list(GET ${gemm_list} 9 twc)
                    list(GET ${gemm_list} 10 tlr)
                    list(GET ${gemm_list} 11 tlc)
                    list(GET ${gemm_list} 12 tib)
                    list(GET ${gemm_list} 13 twb)
                    list(GET ${gemm_list} 14 gemm_memory_type)
                    list(GET ${gemm_list} 15 gemm_shape_type)
                    list(GET ${gemm_list} 16 gemm_vectorize_type)
                    list(GET ${gemm_list} 17 vector_size)
                    list(GET ${gemm_list} 18 batch_type)
                    to_cpp_type(cpp_data "${data}")
                    set(file_name "${func}_${double_buffer}_${conflict_a}_"
                                  "${conflict_b}_${trans_a}_${trans_b}_"
                                  "${is_beta_zero}_${gemm_memory_type}_"
                                  "${gemm_shape_type}_${gemm_vectorize_type}_"
                                  "${vector_size}_${batch_type}_${executor}_"
                                  "${data}_${index}_${tir}_${tic}_${twr}_"
                                  "${twc}_${tlr}_${tlc}_${tib}_${twb}_"
                                  "${wg_size}_${cl_size}.cpp")
                    STRING(REGEX REPLACE "(\\*|<| |,|>)" "_" file_name ${file_name})
                    STRING(REGEX REPLACE "(___|__)" "_" file_name ${file_name})
                    add_custom_command(OUTPUT "${LOCATION}/${file_name}"
                      COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_gemm_launcher.py
                        ${PROJECT_SOURCE_DIR}/external/
                        ${SYCLBLAS_SRC_GENERATOR}/gen
                        ${blas_level}
                        ${func}
                        ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                        ${executor}
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
                        ${tlr}
                        ${tlc}
                        ${tib}
                        ${twb}
                        ${wg_size}
                        ${cl_size}
                        ${file_name}
                        ${gemm_vectorize_type}
                        ${vector_size}
                        ${batch_type}
                      MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                      DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_gemm_launcher.py
                      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                      VERBATIM
                    )
                    list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
                  endforeach(gemm_list)
                endforeach(index)
            endforeach(executor)
          endforeach(is_beta_zero)
        endforeach(trans_b)
      endforeach(trans_a)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
# The blas library depends on FindComputeCpp
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_gemm_objects)


function (build_library LIB_NAME)
add_library(${LIB_NAME}
                             $<TARGET_OBJECTS:sycl_policy>
                             $<TARGET_OBJECTS:axpy>
                             $<TARGET_OBJECTS:asum>
                             $<TARGET_OBJECTS:asum_return>
                             $<TARGET_OBJECTS:copy>
                             $<TARGET_OBJECTS:dot>
                             $<TARGET_OBJECTS:dot_return>
                             $<TARGET_OBJECTS:iamax>
                             $<TARGET_OBJECTS:iamax_return>
                             $<TARGET_OBJECTS:iamin>
                             $<TARGET_OBJECTS:iamin_return>
                             $<TARGET_OBJECTS:nrm2>
                             $<TARGET_OBJECTS:nrm2_return>
                             $<TARGET_OBJECTS:rot>
                             $<TARGET_OBJECTS:scal>
                             $<TARGET_OBJECTS:swap>
                             $<TARGET_OBJECTS:gemv>
                             $<TARGET_OBJECTS:ger>
                             $<TARGET_OBJECTS:symv>
                             $<TARGET_OBJECTS:syr>
                             $<TARGET_OBJECTS:syr2>
                             $<TARGET_OBJECTS:trmv>
                             $<TARGET_OBJECTS:gemm_launcher>
                             $<TARGET_OBJECTS:gemm>
                            )
endfunction(build_library)
