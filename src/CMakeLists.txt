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
# *  @filename CMakeLists.txt
# *
# **************************************************************************/

find_package(PythonInterp 3.3 REQUIRED)

add_library(sycl-blas)
add_library(codeplay::sycl-blas ALIAS sycl-blas)

function(syclblas_add_binary_function_instantiations)
    cmake_parse_arguments(ARG "" "TEMPLATE;MODEL;OUTPUT" "" ${ARGN})
    add_custom_command(OUTPUT ${ARG_OUTPUT}
        COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/python_generator/jinja.py
                --model=${ARG_MODEL} --template=${ARG_TEMPLATE}
                --output=${ARG_OUTPUT}
        MAIN_DEPENDENCY ${ARG_TEMPLATE}
        DEPENDS ${ARG_MODEL} ${PROJECT_SOURCE_DIR}/python_generator/jinja.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM
    )
endfunction()


function(generate_template_instantiations)
    cmake_parse_arguments(ARG "" "DIRECTORY" "FUNCTIONS" ${ARGN})
    foreach(func ${ARG_FUNCTIONS})
        syclblas_add_binary_function_instantiations(
            MODEL "${CMAKE_CURRENT_LIST_DIR}/interface/model.yaml"
            TEMPLATE ${ARG_DIRECTORY}/${func}.cpp.in
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${func}.cpp"
        )
        target_sources(sycl-blas PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/${func}.cpp")
    endforeach()
endfunction()

generate_template_instantiations(
    DIRECTORY interface/blas1
    FUNCTIONS axpy asum copy dot_return nrm2 rot nrm2_return swap iamin iamax iamax_return iamin_return scal
)

generate_template_instantiations(
    DIRECTORY interface/blas2
    FUNCTIONS gemv ger symv syr syr2 trmv
)

generate_template_instantiations(
    DIRECTORY interface/blas3
    FUNCTIONS gemm gemm_launcher
)

add_subdirectory(policy)

configure_file(config.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/config.hpp")
target_include_directories(sycl-blas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sycl-blas
    PUBLIC codeplay::sycl-blas-header-only
           codeplay::computecpp-sdk
)

get_property(sycl_blas_sources TARGET sycl-blas PROPERTY SOURCES)
add_sycl_to_target(TARGET sycl-blas SOURCES ${sycl_blas_sources})

