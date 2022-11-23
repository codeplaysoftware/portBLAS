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
 *  @filename blas1_rotg_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<api_type, scalar_t, scalar_t>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  api_type api;
  scalar_t a_input;
  scalar_t b_input;

  std::tie(api, a_input, b_input) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  data_t c_ref;
  data_t s_ref;
  data_t a_ref = a_input;
  data_t b_ref = b_input;
  reference_blas::rotg(&a_ref, &b_ref, &c_ref, &s_ref);

  auto q = make_queue();
  test_executor_t ex(q);

  data_t c;
  data_t s;
  data_t a = a_input;
  data_t b = b_input;
  if (api == api_type::event) {
    auto device_a = utils::make_quantized_buffer<scalar_t>(ex, a);
    auto device_b = utils::make_quantized_buffer<scalar_t>(ex, b);
    auto device_c = utils::make_quantized_buffer<scalar_t>(ex, c);
    auto device_s = utils::make_quantized_buffer<scalar_t>(ex, s);
    auto event0 = _rotg(ex, device_a, device_b, device_c, device_s);
    ex.get_policy_handler().wait(event0);

    auto event1 = ex.get_policy_handler().copy_to_host(device_c, &c, 1);
    auto event2 = ex.get_policy_handler().copy_to_host(device_s, &s, 1);
    auto event3 = ex.get_policy_handler().copy_to_host(device_a, &a, 1);
    auto event4 = ex.get_policy_handler().copy_to_host(device_b, &b, 1);
    ex.get_policy_handler().wait(event1);
    ex.get_policy_handler().wait(event2);
    ex.get_policy_handler().wait(event3);
    ex.get_policy_handler().wait(event4);
  }
  else {
    _rotg(ex, a, b, c, s);
  }

  /* When there is an overflow in the calculation of the hypotenuse, results are
   * implementation defined but r should return inf like reference_blas does */
  if (std::isinf(a_ref)) {
    ASSERT_TRUE(std::isinf(a));
    return;
  }

  const bool isAlmostEqual = utils::almost_equal<data_t, scalar_t>(a, a_ref) &&
                             utils::almost_equal<data_t, scalar_t>(b, b_ref) &&
                             utils::almost_equal<data_t, scalar_t>(c, c_ref) &&
                             utils::almost_equal<data_t, scalar_t>(s, s_ref);
  ASSERT_TRUE(isAlmostEqual);
}

/* Using std::numeric_limits<float> to test overflows on float types. If the
 * test type is something else, then this will be implicitly cast. double should
 * not overflow */
const auto combi = ::testing::Combine(
    ::testing::Values(api_type::event, api_type::result),                  // Api
    ::testing::Values(0, 2.5, -7.3, std::numeric_limits<float>::max()),    // a
    ::testing::Values(0, 0.5, -4.3, std::numeric_limits<float>::lowest())  // b
);

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  api_type api;
  T a, b;
  BLAS_GENERATE_NAME(info.param, api, a, b);
}

BLAS_REGISTER_TEST_ALL(Rotg, combination_t, combi, generate_name);
