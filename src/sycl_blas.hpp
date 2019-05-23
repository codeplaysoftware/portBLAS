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
 *  @filename sycl_blas.hpp
 *
 **************************************************************************/

#include "container/sycl_iterator.hpp"
#include "executors/executor_sycl.hpp"
#include "executors/kernel_constructor.hpp"
#include "interface/blas1_interface.hpp"
#include "interface/blas2_interface.hpp"
#include "interface/blas3_interface.hpp"
#include "interface/gemm_launcher.hpp"
#include "operations/blas1_trees.hpp"
#include "operations/blas2_trees.hpp"
#include "operations/blas3_trees.hpp"
#include "operations/blas_constants.hpp"
#include "operations/blas_operators.hpp"
#include "policy/sycl_policy_handler.hpp"
#include "sycl_blas.h"
#include "views/view_sycl.hpp"
