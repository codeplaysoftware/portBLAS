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
 *  @filename blas1_scal_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, scalar_t, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  index_t size;
  scalar_t alpha;
  index_t incX;
  std::tie(size, alpha, incX) = combi;

  // Input/output vector
  std::vector<scalar_t> x_v(size * incX);
  std::vector<scalar_t> x_cpu_v(x_v);

  // Reference implementation
  reference_blas::scal(size, alpha, x_cpu_v.data(), incX);

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(x_v, size * incX);

  _scal(sb_handle, size, alpha, gpu_x_v, incX);
  auto event = blas::helper::copy_to_host(sb_handle.get_queue(), gpu_x_v,
                                          x_v.data(), size * incX);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(x_v, x_cpu_v);
  ASSERT_TRUE(isAlmostEqual);
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 1002, 1002400),    // size
                       ::testing::Values<scalar_t>(0.0, 1.0, 1.5),  // alpha
                       ::testing::Values(1, 4)                      // incX
    );
#else
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values(11, 1002),            // size
                       ::testing::Values<scalar_t>(0.0, 1.5),  // alpha
                       ::testing::Values(4)                    // incX
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  int size, incX;
  T alpha;
  BLAS_GENERATE_NAME(info.param, size, alpha, incX);
}

BLAS_REGISTER_TEST_ALL(Scal, combination_t, combi, generate_name);
