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
# *  @filename CMakeList.txt
# *
# **************************************************************************/
#blas1 
generate_blas_objects(blas1 axpy)
generate_blas_objects(blas1 asum)
generate_blas_objects(blas1 copy)
generate_blas_objects(blas1 dot_return)
generate_blas_objects(blas1 sdsdot_return)
generate_blas_objects(blas1 nrm2)
generate_blas_objects(blas1 rot)
generate_blas_objects(blas1 nrm2_return)
generate_blas_objects(blas1 swap)

generate_blas_objects(blas1 asum_return)
generate_blas_objects(blas1 iamax_return)
generate_blas_objects(blas1 iamin_return)
generate_blas_objects(blas1 scal)

generate_blas_objects(blas1 dot)
generate_blas_objects(blas1 sdsdot)
generate_blas_objects(blas1 rotm)
generate_blas_objects(blas1 iamax)
generate_blas_objects(blas1 iamin)

generate_blas_rotg_objects(blas1 rotg)
generate_blas_rotg_objects(blas1 rotg_return)
generate_blas_rotg_objects(blas1 rotmg)
