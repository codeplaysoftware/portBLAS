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
 *  @filename blas1_iaminmax_common.hpp
 *
 **************************************************************************/

#ifndef BLAS_TEST_IAMINMAX_COMMON_HPP
#define BLAS_TEST_IAMINMAX_COMMON_HPP

#include "blas_test.hpp"

enum class generation_mode_t : char {
  Random = 'r',
  Limit = 'l',  // The largest (iamin) or smallest (iamax) legal value
  Incrementing = 'i',
  Decrementing = 'd'
};

template <typename scalar_t>
using combination_t = std::tuple<int, int, generation_mode_t>;

template <typename scalar_t>
void populate_data(generation_mode_t mode, scalar_t limit,
                   std::vector<scalar_t> &vec) {
  switch (mode) {
    case generation_mode_t::Random:
      fill_random(vec);
      break;
    case generation_mode_t::Limit:
      std::fill(vec.begin(), vec.end(), limit);
      break;
    case generation_mode_t::Incrementing:
      for (int i = 0; i < vec.size(); i++) {
        vec[i] = scalar_t(i);
      }
      break;
    case generation_mode_t::Decrementing:
      for (int i = 0; i < vec.size(); i++) {
        vec[i] = scalar_t(vec.size() - i - 1);
      }
      break;
  }
}

#ifdef STRESS_TESTING
const auto combi = ::testing::Combine(
    ::testing::Values(11, 65, 10000, 1002400),  // size
    ::testing::Values(1, 5),                    // incX
    ::testing::Values(generation_mode_t::Random, generation_mode_t::Limit,
                      generation_mode_t::Incrementing,
                      generation_mode_t::Decrementing));
#else
const auto combi = ::testing::Combine(
    ::testing::Values(11, 65, 1000000),  // size
    ::testing::Values(5),                // incX
    ::testing::Values(generation_mode_t::Random, generation_mode_t::Limit,
                      generation_mode_t::Incrementing,
                      generation_mode_t::Decrementing));
#endif

#endif
