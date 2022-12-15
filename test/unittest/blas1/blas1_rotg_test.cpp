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

  scalar_t c_ref;
  scalar_t s_ref;
  scalar_t a_ref = a_input;
  scalar_t b_ref = b_input;
  reference_blas::rotg(&a_ref, &b_ref, &c_ref, &s_ref);

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  scalar_t c;
  scalar_t s;
  scalar_t a = a_input;
  scalar_t b = b_input;
  if (api == api_type::event) {
    auto device_a = blas::make_sycl_iterator_buffer<scalar_t>(&a_input, 1);
    auto device_b = blas::make_sycl_iterator_buffer<scalar_t>(&b_input, 1);
    auto device_c = blas::make_sycl_iterator_buffer<scalar_t>(1);
    auto device_s = blas::make_sycl_iterator_buffer<scalar_t>(1);
    auto event0 = _rotg(sb_handle, device_a, device_b, device_c, device_s);
    sb_handle.wait(event0);

    auto event1 =
        blas::helper::copy_to_host(sb_handle.get_queue(), device_c, &c, 1);
    auto event2 =
        blas::helper::copy_to_host(sb_handle.get_queue(), device_s, &s, 1);
    auto event3 =
        blas::helper::copy_to_host(sb_handle.get_queue(), device_a, &a, 1);
    auto event4 =
        blas::helper::copy_to_host(sb_handle.get_queue(), device_b, &b, 1);
    sb_handle.wait({event1, event2, event3, event4});
  } else {
    _rotg(sb_handle, a, b, c, s);
  }

  /* When there is an overflow in the calculation of the hypotenuse, results are
   * implementation defined but r should return inf like reference_blas does */
  if (std::isinf(a_ref)) {
    ASSERT_TRUE(std::isinf(a));
    return;
  }

  const bool isAlmostEqual =
      utils::almost_equal(a, a_ref) && utils::almost_equal(b, b_ref) &&
      utils::almost_equal(c, c_ref) && utils::almost_equal(s, s_ref);
  ASSERT_TRUE(isAlmostEqual);
}

template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values(api_type::event, api_type::result),  // Api
    ::testing::Values<scalar_t>(0, 2.5, -7.3,
                                std::numeric_limits<scalar_t>::max()),  // a
    ::testing::Values<scalar_t>(0, 0.5, -4.3,
                                std::numeric_limits<scalar_t>::lowest())  // b
);

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  api_type api;
  T a, b;
  BLAS_GENERATE_NAME(info.param, api, a, b);
}

BLAS_REGISTER_TEST_ALL(Rotg, combination_t, combi, generate_name);
