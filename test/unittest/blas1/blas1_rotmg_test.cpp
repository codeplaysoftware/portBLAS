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
 *  @filename blas1_rotmg_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>;

/* Check if rotmg can handle overflows and underflows without entering into
 * infinite loops. The aim is to test for with floats. If the test type is
 * something else, then this will be implicitly cast and might not overflow */
const float max_float = std::numeric_limits<float>::max();
const float min_float = std::numeric_limits<float>::min();

template <typename scalar_t>
struct RotmgTest {
  /* Magic numbers used by the rotmg algorithm */
  static constexpr scalar_t gamma = static_cast<scalar_t>(4096.0);
  static constexpr scalar_t gamma_sq = static_cast<scalar_t>(gamma * gamma);
  static constexpr scalar_t inv_gamma_sq =
      static_cast<scalar_t>(static_cast<scalar_t>(1) / gamma);
  static constexpr size_t param_size = 5;

  struct RotmgParameters {
    scalar_t d1{};
    scalar_t d2{};
    scalar_t x1{};
    scalar_t y1{};
    std::vector<scalar_t> param = std::vector<scalar_t>(param_size);

    RotmgParameters() = default;
    RotmgParameters(scalar_t d1, scalar_t d2, scalar_t x1, scalar_t y1)
        : d1{d1}, d2{d2}, x1{x1}, y1{y1} {}
  };

  using data_t = utils::data_storage_t<scalar_t>;

  const RotmgParameters input;
  RotmgParameters sycl_out;

  RotmgTest(scalar_t d1, scalar_t d2, scalar_t x1, scalar_t y1)
      : input{d1, d2, x1, y1} {}

  void run_sycl_blas_rotmg();
  void validate_with_reference();
  void validate_with_rotm();
  bool isOverflowTest();
};

template <typename scalar_t>
bool RotmgTest<scalar_t>::isOverflowTest() {
  if (input.d1 == static_cast<scalar_t>(max_float) ||
      input.d1 == static_cast<scalar_t>(min_float) ||
      input.d2 == static_cast<scalar_t>(max_float) ||
      input.d2 == static_cast<scalar_t>(min_float) ||
      input.x1 == static_cast<scalar_t>(max_float) ||
      input.x1 == static_cast<scalar_t>(min_float) ||
      input.y1 == static_cast<scalar_t>(max_float) ||
      input.y1 == static_cast<scalar_t>(min_float)) {
    return true;
  }

  return false;
}

template <typename scalar_t>
void RotmgTest<scalar_t>::run_sycl_blas_rotmg() {

  auto q = make_queue();
  test_executor_t ex(q);

  sycl_out = RotmgParameters{input.d1, input.d2, input.x1, input.y1};

  auto device_d1 = utils::make_quantized_buffer<scalar_t>(ex, sycl_out.d1);
  auto device_d2 = utils::make_quantized_buffer<scalar_t>(ex, sycl_out.d2);
  auto device_x1 = utils::make_quantized_buffer<scalar_t>(ex, sycl_out.x1);
  auto device_y1 = utils::make_quantized_buffer<scalar_t>(ex, sycl_out.y1);
  auto device_param =
      utils::make_quantized_buffer<scalar_t>(ex, sycl_out.param);
  auto event0 =
      _rotmg(ex, device_d1, device_d2, device_x1, device_y1, device_param);
  ex.get_policy_handler().wait(event0);

  auto event1 =
      ex.get_policy_handler().copy_to_host(device_d1, &sycl_out.d1, 1);
  auto event2 =
      ex.get_policy_handler().copy_to_host(device_d2, &sycl_out.d2, 1);
  auto event3 =
      ex.get_policy_handler().copy_to_host(device_x1, &sycl_out.x1, 1);
  auto event4 =
      ex.get_policy_handler().copy_to_host(device_y1, &sycl_out.y1, 1);
  auto event5 = ex.get_policy_handler().copy_to_host(
      device_param, sycl_out.param.data(), param_size);
  ex.get_policy_handler().wait(event1);
  ex.get_policy_handler().wait(event2);
  ex.get_policy_handler().wait(event3);
  ex.get_policy_handler().wait(event4);
  ex.get_policy_handler().wait(event5);
}

template <typename scalar_t>
void RotmgTest<scalar_t>::validate_with_reference() {
  data_t d1_ref = input.d1;
  data_t d2_ref = input.d2;
  data_t x1_ref = input.x1;
  data_t y1_ref = input.y1;
  std::vector<data_t> param_ref(param_size);

  /* Cannot test this scenario since cblas enters into an infinite loop */
  if (d2_ref < 0 && y1_ref == min_float) {
    return;
  }
  reference_blas::rotmg(&d1_ref, &d2_ref, &x1_ref, &y1_ref, param_ref.data());

  /* If d1 is less than 0 then the results are implementation defined.
   * Unfortunately, cblas is not consistent with the outputs */
  if (input.d1 < 0) {
    ASSERT_TRUE(sycl_out.param[0] == 2);
    ASSERT_TRUE(y1_ref == sycl_out.y1); /* y1 should still be unchanged */
    return;
  }

  /* cblas does not seem to be rescaling d1 and d2 in the (0, c) scenario */
  if (((input.d1 == 0 || input.x1 == 0) && input.y1 != 0) &&
      (d1_ref > gamma_sq || d1_ref < inv_gamma_sq || d2_ref > gamma_sq ||
       d2_ref < inv_gamma_sq)) {
    return;
  }

  const bool isAlmostEqual =
      utils::almost_equal<data_t, scalar_t>(sycl_out.d1, d1_ref) &&
      utils::almost_equal<data_t, scalar_t>(sycl_out.d2, d2_ref) &&
      utils::almost_equal<data_t, scalar_t>(sycl_out.x1, x1_ref) &&
      utils::almost_equal<data_t, scalar_t>(sycl_out.y1, y1_ref);
  ASSERT_TRUE(isAlmostEqual);

  /* Validate param */
  constexpr scalar_t unit_matrix = -2;
  constexpr scalar_t rescaled_matrix = -1;
  constexpr scalar_t sltc_matrix = 0;
  constexpr scalar_t clts_matrix = 1;
  constexpr scalar_t error_matrix = 2;

  scalar_t flag_sycl = sycl_out.param[0];
  scalar_t h11_sycl = sycl_out.param[1];
  scalar_t h12_sycl = sycl_out.param[3];
  scalar_t h21_sycl = sycl_out.param[2];
  scalar_t h22_sycl = sycl_out.param[4];

  scalar_t flag_ref = param_ref[0];
  scalar_t h11_ref = param_ref[1];
  scalar_t h12_ref = param_ref[3];
  scalar_t h21_ref = param_ref[2];
  scalar_t h22_ref = param_ref[4];

  if (flag_sycl != error_matrix && flag_ref != unit_matrix) {
    if (flag_ref == sltc_matrix) {
      ASSERT_TRUE((utils::almost_equal<scalar_t>(h12_sycl, h12_ref)));
      ASSERT_TRUE((utils::almost_equal<scalar_t>(h21_sycl, h21_ref)));
    } else if (flag_ref == clts_matrix) {
      ASSERT_TRUE((utils::almost_equal<scalar_t>(h11_sycl, h11_ref)));
      ASSERT_TRUE((utils::almost_equal<data_t, scalar_t>(h22_sycl, h22_ref)));
    } else {
      ASSERT_TRUE((utils::almost_equal<scalar_t>(h11_sycl, h11_ref)));
      ASSERT_TRUE((utils::almost_equal<scalar_t>(h12_sycl, h12_ref)));
      ASSERT_TRUE((utils::almost_equal<scalar_t>(h21_sycl, h21_ref)));
      ASSERT_TRUE((utils::almost_equal<scalar_t>(h22_sycl, h22_ref)));
    }
  }
}

/*
 * Rotm can be used to validate that Rotmg outputs will set y to 0. The right
 * side of the following formula is calculated by rotm.
 *
 * x1_output * sqrt(d1_output) = [ h11 h12 ] * [ x1_input]
 * 0.0       * sqrt(d2_output)   [h21  h22 ]   [ y1_input]
 */
template <typename scalar_t>
void RotmgTest<scalar_t>::validate_with_rotm() {
  if (sycl_out.param[0] == 2 || sycl_out.d2 < 0) {
    return;
  }

  index_t size = 1;
  index_t incX = 1;
  index_t incY = 1;

  std::vector<data_t> x_cpu_v{input.x1};
  std::vector<data_t> y_cpu_v{input.y1};

  reference_blas::rotm(size, x_cpu_v.data(), incX, y_cpu_v.data(), incY,
                       sycl_out.param.data());

  x_cpu_v[0] = x_cpu_v[0] * static_cast<scalar_t>(sqrt(sycl_out.d1));
  y_cpu_v[0] = y_cpu_v[0] * static_cast<scalar_t>(sqrt(sycl_out.d2));

  bool y1_becomes_zero = utils::almost_equal<data_t, scalar_t>(y_cpu_v[0], 0);
  ASSERT_TRUE(y1_becomes_zero);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  scalar_t d1_input;
  scalar_t d2_input;
  scalar_t x1_input;
  scalar_t y1_input;

  std::tie(d1_input, d2_input, x1_input, y1_input) = combi;

  RotmgTest<scalar_t> test{d1_input, d2_input, x1_input, y1_input};
  test.run_sycl_blas_rotmg();

  /* Do not test with things that might overflow or underflow. Results will not
   * make sense if that happens */
  if (!test.isOverflowTest()) {
    test.validate_with_reference();
    test.validate_with_rotm();
  }
}

const auto combi = ::testing::Combine(
    ::testing::Values(0, 15.5, -2.2, max_float, min_float),  // d1
    ::testing::Values(0, 3.0, -2.2, max_float, min_float),   // d2
    ::testing::Values(0, 12.1, -7.3, max_float, min_float),  // x1
    ::testing::Values(0, 0.5, -4.3, max_float, min_float)    // y1
);

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  T d1, d2, x1, y1;
  BLAS_GENERATE_NAME(info.param, d1, d2, x1, y1);
}

BLAS_REGISTER_TEST_ALL(Rotmg, combination_t, combi, generate_name);
