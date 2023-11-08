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
 *  @filename tune.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_TOOLS_AUTO_TUNER_TUNE_HPP_
#define PORTBLAS_TOOLS_AUTO_TUNER_TUNE_HPP_

#include "tuner_types.hpp"

template <int VecSize, int Cls, typename Tile, bool DoubleBuffer, bool Nbca,
          bool Nbcb, typename Config, typename T>
TestResultEntry tune(portblas_handle_t &sb_handle, int r, GemmArgs<T> a);

#endif  // PORTBLAS_TOOLS_AUTO_TUNER_TUNE_HPP_
