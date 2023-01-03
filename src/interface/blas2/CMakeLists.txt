
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
#blas2
generate_blas_ternary_objects(blas2 gbmv)
generate_blas_ternary_objects(blas2 gemv)
generate_blas_ternary_objects(blas2 ger)
generate_blas_ternary_objects(blas2 sbmv)
generate_blas_ternary_objects(blas2 symv)
generate_blas_ternary_objects(blas2 syr2)
generate_blas_binary_objects(blas2 syr)
generate_blas_binary_objects(blas2 trmv)

if(BLAS_ENABLE_CONST_INPUT)
    generate_blas_ternary_objects(blas2 gemv_const)
endif()
