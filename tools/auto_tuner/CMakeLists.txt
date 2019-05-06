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

set(GEN_COMB ${CMAKE_CURRENT_BINARY_DIR}/generate_combinations.inc.hpp)

# The generator's configuration file - add any new jsons to this
set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/default.json)
if(${TARGET} STREQUAL "RCAR")
    set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/rcar.json)
endif()

add_custom_command(
    OUTPUT generate_combinations.inc.hpp
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen/generate_combinations.py ${GEN_CONFIG} ${GEN_COMB}
    COMMENT "Generating tuning configurations"
    VERBATIM
)

foreach(tuner tune_nn tune_nt tune_tn tune_tt tune_all)
    add_executable(${tuner} src/${tuner}.cpp ${CMAKE_CURRENT_BINARY_DIR}/generate_combinations.inc.hpp)
    add_sycl_to_target(TARGET ${tuner} SOURCES src/${tuner}.cpp)
    target_include_directories(${tuner} PRIVATE include ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${tuner} PRIVATE codeplay::sycl-blas)
endforeach()
