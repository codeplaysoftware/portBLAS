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
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename sycl_blas.h
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include "blas_meta.h"

#include "container/sycl_iterator.h"

#include "sb_handle/sycl_blas_handle.h"

#include "sb_handle/kernel_constructor.h"

#include "interface/blas1_interface.h"

#include "interface/blas2_interface.h"

#include "interface/blas3_interface.h"

#include "interface/gemm_launcher.h"

#include "interface/extension_interface.h"

#include "operations/blas1_trees.h"

#include "operations/blas2_trees.h"

#include "operations/blas3_trees.h"

#include "operations/extension/reduction.h"

#include "operations/blas_constants.h"

#include "operations/blas_operators.h"

#include "sycl_blas_helper.h"

#include "views/view.h"
