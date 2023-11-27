/***************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename portblas.hpp
 *
 **************************************************************************/
#include "portblas.h"

#include "container/sycl_iterator.hpp"

#include "sb_handle/portblas_handle.hpp"

#include "sb_handle/kernel_constructor.hpp"

#include "interface/blas1_interface.hpp"

#include "interface/blas2_interface.hpp"

#include "interface/blas3_interface.hpp"

#include "interface/gemm_launcher.hpp"

#include "interface/extension_interface.hpp"

#include "operations/blas1_trees.hpp"

#include "operations/blas2_trees.hpp"

#include "operations/blas3_trees.hpp"

#include "operations/extension/reduction.hpp"

#include "operations/extension/transpose.hpp"

#include "operations/extension/matcopy_batch.hpp"

#include "operations/extension/axpy_batch.hpp"

#include "operations/blas_constants.hpp"

#include "operations/blas_operators.hpp"

#include "views/view_sycl.hpp"

#include "views/view.hpp"
