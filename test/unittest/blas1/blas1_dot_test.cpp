/***************************************************************************
 *
 *  @license
 *  Dotright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a dot of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a dot of the License has been included in this
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
 *  @filename blas1_dot_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<api_type, int, int, int>;

template <bool isUsm>
struct TestRunner {
  template <typename scalar_t>
  static void run_test(const combination_t<scalar_t> combi) {
    api_type api;
    index_t size;
    index_t incX;
    index_t incY;
    std::tie(api, size, incX, incY) = combi;

    // Input vectors
    std::vector<scalar_t> x_v(size * incX);
    fill_random(x_v);
    std::vector<scalar_t> y_v(size * incY);
    fill_random(y_v);

    // Output
    scalar_t out_s = 0;

    // Reference implementation
    auto out_cpu_s =
        reference_blas::dot(size, x_v.data(), incX, y_v.data(), incY);

    // SYCL implementation
    auto q = make_queue();
    blas::SB_Handle sb_handle(q);

    // Iterators
    auto gpu_x_v = blas::helper::allocate<isUsm, scalar_t>(size * incX, q);
    auto gpu_y_v = blas::helper::allocate<isUsm, scalar_t>(size * incY, q);

    auto copy_x =
        blas::helper::copy_to_device(q, x_v.data(), gpu_x_v, size * incX);
    auto copy_y =
        blas::helper::copy_to_device(q, y_v.data(), gpu_y_v, size * incY);
    sb_handle.wait({copy_x, copy_y});

    if (api == api_type::async) {
      auto gpu_out_s = blas::helper::allocate<isUsm, scalar_t>(1, q);
      auto copy_out =
          blas::helper::copy_to_device<scalar_t>(q, &out_s, gpu_out_s, 1);
      sb_handle.wait(copy_out);
      _dot(sb_handle, size, gpu_x_v, incX, gpu_y_v, incY, gpu_out_s);
      auto event = blas::helper::copy_to_host(sb_handle.get_queue(), gpu_out_s,
                                              &out_s, 1);
      sb_handle.wait(event);
    } else {
      out_s = _dot(sb_handle, size, gpu_x_v, incX, gpu_y_v, incY);
    }

    // Validate the result
    const bool isAlmostEqual = utils::almost_equal(out_s, out_cpu_s);
    ASSERT_TRUE(isAlmostEqual);
  }
};

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values(api_type::async,
                                                        api_type::sync),  // Api
                                      ::testing::Values(11, 65, 1002,
                                                        1002400),  // size
                                      ::testing::Values(1, 4),     // incX
                                      ::testing::Values(1, 3)      // incY
);
#else
template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values(api_type::async,
                                                        api_type::sync),  // Api
                                      ::testing::Values(11, 1002),  // size
                                      ::testing::Values(1, 4),      // incX
                                      ::testing::Values(1, 3)       // incY
);
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  api_type api;
  int size, incX, incY;
  BLAS_GENERATE_NAME(info.param, api, size, incX, incY);
}

BLAS_REGISTER_TEST_CUSTOM_NAME(DotUSM, DotUSM, TestRunner<true>::run_test,
                               combination_t, combi, generate_name);
BLAS_REGISTER_TEST_CUSTOM_NAME(DotBuffer, DotBuffer,
                               TestRunner<false>::run_test, combination_t,
                               combi, generate_name);
