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
 *  @filename blas1_rotg_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<std::string, api_type, scalar_t, scalar_t>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  api_type api;
  scalar_t a_input;
  scalar_t b_input;
  scalar_t c_input = 1;
  scalar_t s_input = 1;

  std::tie(alloc, api, a_input, b_input) = combi;

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
  if (api == api_type::async) {
    auto device_a = helper::allocate<mem_alloc, scalar_t>(1, q);
    auto device_b = helper::allocate<mem_alloc, scalar_t>(1, q);
    auto device_c = helper::allocate<mem_alloc, scalar_t>(1, q);
    auto device_s = helper::allocate<mem_alloc, scalar_t>(1, q);

    auto copy_a = helper::copy_to_device(q, &a_input, device_a, 1);
    auto copy_b = helper::copy_to_device(q, &b_input, device_b, 1);
    auto set_c = helper::copy_to_device(q, &c_input, device_c, 1);
    auto set_s = helper::copy_to_device(q, &s_input, device_s, 1);

    auto rotg_event = _rotg(sb_handle, device_a, device_b, device_c, device_s,
                            {copy_a, copy_b, set_c, set_s});
    sb_handle.wait(rotg_event);

    auto event1 = helper::copy_to_host(sb_handle.get_queue(), device_c, &c, 1);
    auto event2 = helper::copy_to_host(sb_handle.get_queue(), device_s, &s, 1);
    auto event3 = helper::copy_to_host(sb_handle.get_queue(), device_a, &a, 1);
    auto event4 = helper::copy_to_host(sb_handle.get_queue(), device_b, &b, 1);
    sb_handle.wait({event1, event2, event3, event4});

    helper::deallocate<mem_alloc>(device_a, q);
    helper::deallocate<mem_alloc>(device_b, q);
    helper::deallocate<mem_alloc>(device_c, q);
    helper::deallocate<mem_alloc>(device_s, q);

  } else {
    _rotg(sb_handle, a, b, c, s);
  }

  /* When there is an overflow in the calculation of the hypotenuse, results
   * are implementation defined but r should return inf like reference_blas
   * does */
  if (std::isinf(a_ref)) {
    ASSERT_TRUE(std::isinf(a));
    return;
  }

  ASSERT_TRUE(utils::almost_equal(a, a_ref));
  ASSERT_TRUE(utils::almost_equal(b, b_ref));
  ASSERT_TRUE(utils::almost_equal(c, c_ref));
  ASSERT_TRUE(utils::almost_equal(s, s_ref));
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  api_type api;
  scalar_t a_input;
  scalar_t b_input;
  scalar_t c_input = 1;
  scalar_t s_input = 1;

  std::tie(alloc, api, a_input, b_input) = combi;

  if (alloc == "usm") {  // usm alloc
#ifdef SB_ENABLE_USM
    run_test<scalar_t, helper::AllocType::usm>(combi);
#else
    GTEST_SKIP();
#endif
  } else {  // buffer alloc
    run_test<scalar_t, helper::AllocType::buffer>(combi);
  }
}

template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values("usm", "buf"),                     // allocation type
    ::testing::Values(api_type::async, api_type::sync),  // Api
    ::testing::Values<scalar_t>(0, 2.5, -7.3,
                                std::numeric_limits<scalar_t>::max()),  // a
    ::testing::Values<scalar_t>(0, 0.5, -4.3,
                                std::numeric_limits<scalar_t>::lowest())  // b
);

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  api_type api;
  T a, b;
  BLAS_GENERATE_NAME(info.param, alloc, api, a, b);
}

BLAS_REGISTER_TEST_ALL(Rotg, combination_t, combi, generate_name);
