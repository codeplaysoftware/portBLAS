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
 *  @filename blas_test_macros.hpp
 *
 **************************************************************************/

#ifndef VERBOSE_HPP
#define VERBOSE_HPP
#include "config.hpp"

#ifdef VERBOSE
#define DEBUG_PRINT(command) command
#else
#define DEBUG_PRINT(command)
#endif /* ifdef VERBOSE */

#ifndef SYCL_DEVICE
#define SYCL_DEVICE_SELECTOR cl::sycl::default_selector
#else
#define PASTER(x, y) x##y
#define EVALUATOR(x, y) PASTER(x, y)
#define SYCL_DEVICE_SELECTOR cl::sycl::EVALUATOR(SYCL_DEVICE, _selector)
#undef PASTER
#undef EVALUATOR
#endif /* ifndef SYCL_DEVICE */

#endif /* end of include guard: VERBOSE_HPP */
