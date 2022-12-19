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
struct RotmgTest {
  /* Magic numbers used by the rotmg algorithm */
  static constexpr scalar_t gamma = static_cast<scalar_t>(4096.0);
  static constexpr scalar_t gamma_sq = static_cast<scalar_t>(gamma * gamma);
  static constexpr scalar_t inv_gamma_sq =
      static_cast<scalar_t>(static_cast<scalar_t>(1.0) / gamma_sq);
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

  const RotmgParameters input;
  RotmgParameters sycl_out;

  RotmgTest(scalar_t d1, scalar_t d2, scalar_t x1, scalar_t y1)
      : input{d1, d2, x1, y1} {}

  void run_sycl_blas_rotmg();
  void validate_with_reference();
  void validate_with_rotm();
};

template <typename scalar_t>
void RotmgTest<scalar_t>::run_sycl_blas_rotmg() {
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  sycl_out = RotmgParameters{input.d1, input.d2, input.x1, input.y1};

  auto device_d1 = blas::make_sycl_iterator_buffer<scalar_t>(&sycl_out.d1, 1);
  auto device_d2 = blas::make_sycl_iterator_buffer<scalar_t>(&sycl_out.d2, 1);
  auto device_x1 = blas::make_sycl_iterator_buffer<scalar_t>(&sycl_out.x1, 1);
  auto device_y1 = blas::make_sycl_iterator_buffer<scalar_t>(&sycl_out.y1, 1);
  auto device_param =
      blas::make_sycl_iterator_buffer<scalar_t>(sycl_out.param, param_size);
  auto event0 = _rotmg(sb_handle, device_d1, device_d2, device_x1, device_y1,
                       device_param);
  sb_handle.wait(event0);

  auto event1 = blas::helper::copy_to_host(sb_handle.get_queue(), device_d1,
                                           &sycl_out.d1, 1);
  auto event2 = blas::helper::copy_to_host(sb_handle.get_queue(), device_d2,
                                           &sycl_out.d2, 1);
  auto event3 = blas::helper::copy_to_host(sb_handle.get_queue(), device_x1,
                                           &sycl_out.x1, 1);
  auto event4 = blas::helper::copy_to_host(sb_handle.get_queue(), device_y1,
                                           &sycl_out.y1, 1);
  auto event5 = blas::helper::copy_to_host(sb_handle.get_queue(), device_param,
                                           sycl_out.param.data(), param_size);
  sb_handle.wait({event1, event2, event3, event4, event5});
}

template <typename scalar_t>
void RotmgTest<scalar_t>::validate_with_reference() {
  scalar_t d1_ref = input.d1;
  scalar_t d2_ref = input.d2;
  scalar_t x1_ref = input.x1;
  scalar_t y1_ref = input.y1;
  std::vector<scalar_t> param_ref(param_size);

  /* Cannot test this scenario since cblas enters into an infinite loop */
  if (d2_ref < 0 && y1_ref == std::numeric_limits<scalar_t>::min()) {
    return;
  }

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

  reference_blas::rotmg(&d1_ref, &d2_ref, &x1_ref, &y1_ref, param_ref.data());

  const bool isAlmostEqual = utils::almost_equal(sycl_out.d1, d1_ref) &&
                             utils::almost_equal(sycl_out.d2, d2_ref) &&
                             utils::almost_equal(sycl_out.x1, x1_ref) &&
                             utils::almost_equal(sycl_out.y1, y1_ref);
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
      ASSERT_TRUE((utils::almost_equal(h12_sycl, h12_ref)));
      ASSERT_TRUE((utils::almost_equal(h21_sycl, h21_ref)));
    } else if (flag_ref == clts_matrix) {
      ASSERT_TRUE((utils::almost_equal(h11_sycl, h11_ref)));
      ASSERT_TRUE((utils::almost_equal(h22_sycl, h22_ref)));
    } else {
      ASSERT_TRUE((utils::almost_equal(h11_sycl, h11_ref)));
      ASSERT_TRUE((utils::almost_equal(h12_sycl, h12_ref)));
      ASSERT_TRUE((utils::almost_equal(h21_sycl, h21_ref)));
      ASSERT_TRUE((utils::almost_equal(h22_sycl, h22_ref)));
    }
  }
}

/*
 * Rotm can be used to validate that Rotmg outputs will set y to 0. The right
 * side of the following formula is calculated by rotm.
 *
 * x1_output * sqrt(d1_output) = [ h11 h12 ] * [ x1_input]
 * 0.0       * sqrt(d2_output)   [ h21 h22 ]   [ y1_input]
 */
template <typename scalar_t>
void RotmgTest<scalar_t>::validate_with_rotm() {
  if (sycl_out.param[0] == 2 || sycl_out.d2 < 0) {
    return;
  }

  index_t size = 1;
  index_t incX = 1;
  index_t incY = 1;

  std::vector<scalar_t> x_cpu_v{input.x1};
  std::vector<scalar_t> y_cpu_v{input.y1};

  reference_blas::rotm(size, x_cpu_v.data(), incX, y_cpu_v.data(), incY,
                       sycl_out.param.data());

  x_cpu_v[0] = x_cpu_v[0] * static_cast<scalar_t>(sqrt(sycl_out.d1));
  y_cpu_v[0] = y_cpu_v[0] * static_cast<scalar_t>(sqrt(sycl_out.d2));

  bool y1_becomes_zero = utils::almost_equal<scalar_t>(y_cpu_v[0], 0);
  ASSERT_TRUE(y1_becomes_zero);
}

template <typename scalar_t>
using combination_t = std::tuple<scalar_t, scalar_t, scalar_t, scalar_t, bool>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  scalar_t d1_input;
  scalar_t d2_input;
  scalar_t x1_input;
  scalar_t y1_input;
  bool will_overflow;

  std::tie(d1_input, d2_input, x1_input, y1_input, will_overflow) = combi;

  RotmgTest<scalar_t> test{d1_input, d2_input, x1_input, y1_input};
  test.run_sycl_blas_rotmg();

  /* Do not test with things that might overflow or underflow. Results will not
   * make sense if that happens */
  if (!will_overflow) {
    test.validate_with_reference();
    test.validate_with_rotm();
  }
}

template <typename scalar_t>
constexpr scalar_t min_rng = 0.5;
template <typename scalar_t>
constexpr scalar_t max_rng = 10.0;

/* Positive number generator to generate values for d1 and d2 parameters */
template <typename scalar_t>
scalar_t p_gen() {
  return random_scalar<scalar_t>(min_rng<scalar_t>, max_rng<scalar_t>);
}

/* Real number generator to generate values for x1 and y1 parameters */
template <typename scalar_t>
scalar_t r_gen() {
  auto real_scalar =
      random_scalar<scalar_t>(-max_rng<scalar_t>, max_rng<scalar_t>);
  /* Limit to 2 decimal places so that the value does not underflow when
   * multiplied */
  return std::round(real_scalar * 100.0) / 100.0;
}

/* Generate large enough number so that rotmg will scale it down */
template <typename scalar_t>
scalar_t scale_down_gen() {
  return random_scalar<scalar_t>(RotmgTest<scalar_t>::gamma_sq,
                                 RotmgTest<scalar_t>::gamma_sq * 2);
}

/* Generate small enough number so that rotmg will scale it up */
template <typename scalar_t>
scalar_t scale_up_gen() {
  return random_scalar<scalar_t>(RotmgTest<scalar_t>::inv_gamma_sq / 2,
                                 RotmgTest<scalar_t>::inv_gamma_sq);
}

/* This tests try to cover every code path of the rotmg algorithm */
template <typename scalar_t>
const auto combi = ::testing::Values(
    /* d1 < 0 */
    std::make_tuple(-2.5, p_gen<scalar_t>(), r_gen<scalar_t>(),
                    r_gen<scalar_t>(), false),
    /* Input point (c, 0) */
    std::make_tuple(p_gen<scalar_t>(), p_gen<scalar_t>(), r_gen<scalar_t>(),
                    0.0, false),
    /* Input point (c, 0) && d2 == 0 */
    std::make_tuple(p_gen<scalar_t>(), 0.0, r_gen<scalar_t>(), 0.0, false),
    /* Input point (c, 0) && d2 == 0 */
    std::make_tuple(p_gen<scalar_t>(), 0.0, r_gen<scalar_t>(),
                    r_gen<scalar_t>(), false),
    /* Input point (c, 0) and big numbers (test that no rescaling happened) */
    std::make_tuple(scale_up_gen<scalar_t>(), scale_up_gen<scalar_t>(),
                    scale_up_gen<scalar_t>(), 0.0, false),
    std::make_tuple(scale_down_gen<scalar_t>(), scale_down_gen<scalar_t>(),
                    scale_down_gen<scalar_t>(), 0.0, false),
    /* Input point (0, c) */
    std::make_tuple(p_gen<scalar_t>(), p_gen<scalar_t>(), 0.0,
                    r_gen<scalar_t>(), false),
    /* Input point (0, c) && d1 == 0 */
    std::make_tuple(0.0, p_gen<scalar_t>(), 0.0, r_gen<scalar_t>(), false),
    /* Input point (0, c) && d2 == 0 */
    std::make_tuple(p_gen<scalar_t>(), 0.0, 0.0, r_gen<scalar_t>(), false),
    /* Input point (0, c) && d2 < 0 */
    std::make_tuple(p_gen<scalar_t>(), -3.4, 0.0, r_gen<scalar_t>(), false),
    /* Input point (0, c) && rescaling */
    std::make_tuple(p_gen<scalar_t>(), scale_up_gen<scalar_t>(), 0.0,
                    r_gen<scalar_t>(), false),
    std::make_tuple(p_gen<scalar_t>(), scale_down_gen<scalar_t>(), 0.0,
                    r_gen<scalar_t>(), false),
    std::make_tuple(scale_up_gen<scalar_t>(), p_gen<scalar_t>(), 0.0,
                    r_gen<scalar_t>(), false),
    std::make_tuple(scale_down_gen<scalar_t>(), p_gen<scalar_t>(), 0.0,
                    r_gen<scalar_t>(), false),
    /* d1 == 0 */
    std::make_tuple(0.0, p_gen<scalar_t>(), r_gen<scalar_t>(),
                    r_gen<scalar_t>(), false),
    /* d1 == 0 && d2 < 0 */
    std::make_tuple(0.0, -3.4, r_gen<scalar_t>(), r_gen<scalar_t>(), false),
    /* d1 * x1 > d2 * y1 (i.e. abs_c > abs_s) */
    std::make_tuple(4.0, 2.1, 3.4, 1.5, false),
    std::make_tuple(4.0, 1.5, -3.4, 2.1, false),
    std::make_tuple(4.0, -1.5, 3.4, 2.1, false),
    std::make_tuple(4.0, -1.5, 3.4, -2.1, false),
    std::make_tuple(4.0, -1.5, -3.4, -2.1, false),
    /* d1 * x1 > d2 * y1 (i.e. abs_c > abs_s) && rescaling */
    std::make_tuple(scale_down_gen<scalar_t>(), 2.1, 3.4, 1.5, false),
    std::make_tuple(scale_down_gen<scalar_t>(), 2.1, scale_down_gen<scalar_t>(),
                    1.5, false),
    std::make_tuple(scale_up_gen<scalar_t>(), 2.1, scale_down_gen<scalar_t>(),
                    1.5, false),
    std::make_tuple(scale_down_gen<scalar_t>(), 2.1, scale_up_gen<scalar_t>(),
                    1.5, false),
    /* d1 * x1 > d2 * y1 (i.e. abs_c > abs_s) && Underflow */
    std::make_tuple(0.01, 0.01, std::numeric_limits<scalar_t>::min(),
                    std::numeric_limits<scalar_t>::min(), true),
    /* d1 * x1 > d2 * y1 && Overflow */
    std::make_tuple(std::numeric_limits<scalar_t>::max(),
                    std::numeric_limits<scalar_t>::max(), 0.01, 0.01, true),
    /* d1 * x1 <= d2 * y1 (i.e. abs_c <= abs_s) */
    std::make_tuple(2.1, 4.0, 1.5, 3.4, false),
    std::make_tuple(2.1, 4.0, -1.5, 3.4, false),
    std::make_tuple(2.1, -4.0, 1.5, 3.4, false),
    std::make_tuple(2.1, -4.0, 1.5, -3.4, false),
    std::make_tuple(2.1, -4.0, -1.5, -3.4, false),
    /* d1 * x1 <= d2 * y1 (i.e. abs_c <= abs_s) && rescaling */
    std::make_tuple(2.1, scale_down_gen<scalar_t>(), 1.5, 3.4, false),
    std::make_tuple(2.1, scale_down_gen<scalar_t>(), 1.5,
                    scale_down_gen<scalar_t>(), false),
    std::make_tuple(2.1, scale_up_gen<scalar_t>(), 1.5,
                    scale_down_gen<scalar_t>(), false),
    std::make_tuple(2.1, scale_down_gen<scalar_t>(), 1.5,
                    scale_up_gen<scalar_t>(), false),
    /* d1 * x1 <= d2 * y1 (i.e. abs_c <= abs_s) && Underflow */
    std::make_tuple(std::numeric_limits<scalar_t>::min(),
                    std::numeric_limits<scalar_t>::min(), 0.01, 0.01, true),
    /* d1 * x1 <= d2 * y1 (i.e. abs_c <= abs_s) && Overflow */
    std::make_tuple(0.01, 0.01, std::numeric_limits<scalar_t>::max(),
                    std::numeric_limits<scalar_t>::max(), true),
    /* Overflow all */
    std::make_tuple(std::numeric_limits<scalar_t>::max(),
                    std::numeric_limits<scalar_t>::max(),
                    std::numeric_limits<scalar_t>::max(),
                    std::numeric_limits<scalar_t>::max(), true),
    /* Underflow all */
    std::make_tuple(std::numeric_limits<scalar_t>::min(),
                    std::numeric_limits<scalar_t>::min(),
                    std::numeric_limits<scalar_t>::min(),
                    std::numeric_limits<scalar_t>::min(), true),
    /* Numeric limits of one parameter */
    std::make_tuple(1.0, 1.0, 1.0, std::numeric_limits<scalar_t>::max(), false),
    std::make_tuple(1.0, 1.0, std::numeric_limits<scalar_t>::max(), 1.0, false),
    std::make_tuple(1.0, std::numeric_limits<scalar_t>::max(), 1.0, 1.0, false),
    std::make_tuple(std::numeric_limits<scalar_t>::max(), 1.0, 1.0, 1.0, false),
    /* Case that creates an infinite loop on cblas */
    std::make_tuple(std::numeric_limits<scalar_t>::min(), -2.2,
                    std::numeric_limits<scalar_t>::min(),
                    std::numeric_limits<scalar_t>::min(), true),
    /* Case that triggers underflow detection on abs_c <= abs_s && s >= 0 */
    std::make_tuple(15.5, -2.2, std::numeric_limits<scalar_t>::min(),
                    std::numeric_limits<scalar_t>::min(), false));

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  T d1, d2, x1, y1;
  bool will_overflow;
  BLAS_GENERATE_NAME(info.param, d1, d2, x1, y1, will_overflow);
}

BLAS_REGISTER_TEST_ALL(Rotmg, combination_t, combi, generate_name);
