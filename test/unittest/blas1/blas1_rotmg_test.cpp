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
 *  @filename blas1_rotmg_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t, helper::AllocType mem_alloc,
          bool is_pointer = false>
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

  void run_portblas_rotmg();
  void validate_with_reference();
  void validate_with_rotm();
};

template <typename scalar_t, helper::AllocType mem_alloc, bool is_pointer>
void RotmgTest<scalar_t, mem_alloc, is_pointer>::run_portblas_rotmg() {
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  sycl_out = RotmgParameters{input.d1, input.d2, input.x1, input.y1};

  auto device_d1 = helper::allocate<mem_alloc, scalar_t>(1, q);
  auto device_d2 = helper::allocate<mem_alloc, scalar_t>(1, q);
  auto device_x1 = helper::allocate<mem_alloc, scalar_t>(1, q);
  decltype(device_x1) device_y1;
  auto device_param = helper::allocate<mem_alloc, scalar_t>(param_size, q);

  auto copy_d1 = helper::copy_to_device(q, &sycl_out.d1, device_d1, 1);
  auto copy_d2 = helper::copy_to_device(q, &sycl_out.d2, device_d2, 1);
  auto copy_x1 = helper::copy_to_device(q, &sycl_out.x1, device_x1, 1);
  auto copy_params = helper::copy_to_device(q, sycl_out.param.data(),
                                            device_param, param_size);

  if constexpr (is_pointer) {
    device_y1 = helper::allocate<mem_alloc, scalar_t>(1, q);
    auto copy_y1 = helper::copy_to_device(q, &sycl_out.y1, device_y1, 1);

    auto rotmg_event =
        _rotmg(sb_handle, device_d1, device_d2, device_x1, device_y1,
               device_param, {copy_d1, copy_d2, copy_x1, copy_y1, copy_params});
    sb_handle.wait(rotmg_event);
  } else {
    auto rotmg_event =
        _rotmg(sb_handle, device_d1, device_d2, device_x1, sycl_out.y1,
               device_param, {copy_d1, copy_d2, copy_x1, copy_params});
    sb_handle.wait(rotmg_event);
  }

  auto event1 = helper::copy_to_host(q, device_d1, &sycl_out.d1, 1);
  auto event2 = helper::copy_to_host(q, device_d2, &sycl_out.d2, 1);
  auto event3 = helper::copy_to_host(q, device_x1, &sycl_out.x1, 1);
  auto event4 =
      helper::copy_to_host(q, device_param, sycl_out.param.data(), param_size);
  sb_handle.wait({event1, event2, event3, event4});

  helper::deallocate<mem_alloc>(device_d1, q);
  helper::deallocate<mem_alloc>(device_d2, q);
  helper::deallocate<mem_alloc>(device_x1, q);
  helper::deallocate<mem_alloc>(device_param, q);
  if constexpr (is_pointer) {
    helper::deallocate<mem_alloc>(device_y1, q);
  }
}

template <typename scalar_t, helper::AllocType mem_alloc, bool is_pointer>
void RotmgTest<scalar_t, mem_alloc, is_pointer>::validate_with_reference() {
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
    ASSERT_TRUE(sycl_out.param[0] == -1);
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
                             utils::almost_equal(sycl_out.x1, x1_ref);
  ASSERT_TRUE(isAlmostEqual);

  /* Validate param */
  constexpr scalar_t unit_matrix = -2;
  constexpr scalar_t rescaled_matrix = -1;
  constexpr scalar_t sltc_matrix = 0;
  constexpr scalar_t clts_matrix = 1;

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

  if (flag_ref != unit_matrix) {
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
  ASSERT_TRUE(flag_ref == flag_sycl);
}

/*
 * Rotm can be used to validate that Rotmg outputs will set y to 0. The right
 * side of the following formula is calculated by rotm.
 *
 * x1_output * sqrt(d1_output) = [ h11 h12 ] * [ x1_input]
 * 0.0       * sqrt(d2_output)   [ h21 h22 ]   [ y1_input]
 */
template <typename scalar_t, helper::AllocType mem_alloc, bool is_pointer>
void RotmgTest<scalar_t, mem_alloc, is_pointer>::validate_with_rotm() {
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
using combination_t =
    std::tuple<std::string, scalar_t, scalar_t, scalar_t, scalar_t, bool, bool>;

template <typename scalar_t, helper::AllocType mem_alloc, bool is_pointer>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  scalar_t d1_input;
  scalar_t d2_input;
  scalar_t x1_input;
  scalar_t y1_input;
  bool will_overflow;
  bool is_pointer_unused;

  std::tie(alloc, d1_input, d2_input, x1_input, y1_input, will_overflow,
           is_pointer_unused) = combi;

  RotmgTest<scalar_t, mem_alloc, is_pointer> test{d1_input, d2_input, x1_input,
                                                  y1_input};
  test.run_portblas_rotmg();

  /* Do not test with things that might overflow or underflow. Results will
   * not make sense if that happens */
  if (!will_overflow) {
    test.validate_with_reference();
    test.validate_with_rotm();
  }
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  scalar_t d1_input;
  scalar_t d2_input;
  scalar_t x1_input;
  scalar_t y1_input;
  bool will_overflow;
  bool is_pointer;

  std::tie(alloc, d1_input, d2_input, x1_input, y1_input, will_overflow,
           is_pointer) = combi;

  if (alloc == "usm") {  // usm alloc
#ifdef SB_ENABLE_USM
    if (is_pointer)
      run_test<scalar_t, helper::AllocType::usm, true>(combi);
    else
      run_test<scalar_t, helper::AllocType::usm, false>(combi);
#else
    GTEST_SKIP();
#endif
  } else {  // buffer alloc
    if (is_pointer)
      run_test<scalar_t, helper::AllocType::buffer, true>(combi);
    else
      run_test<scalar_t, helper::AllocType::buffer, false>(combi);
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
  /*Setting the mem_alloc parameter to helper::AllocType::usm, it should work if
   * set to false as well*/
  return random_scalar<scalar_t>(
      RotmgTest<scalar_t, helper::AllocType::usm>::gamma_sq,
      RotmgTest<scalar_t, helper::AllocType::usm>::gamma_sq * 2);
}

/* Generate small enough number so that rotmg will scale it up */
template <typename scalar_t>
scalar_t scale_up_gen() {
  /*Setting the mem_alloc parameter to helper::AllocType::usm, it should work if
   * set to false as well*/
  return random_scalar<scalar_t>(
      RotmgTest<scalar_t, helper::AllocType::usm>::inv_gamma_sq / 2,
      RotmgTest<scalar_t, helper::AllocType::usm>::inv_gamma_sq);
}

/* This tests try to cover every code path of the rotmg algorithm */
#define INSTANTIATE_ROTMG_TESTS(NAME, C, IS_POINTER)                           \
  template <typename scalar_t>                                                 \
  const auto NAME = ::testing::                                                \
      Values(/* d1 < 0 */                                                      \
             std::make_tuple(C, -2.5, p_gen<scalar_t>(), r_gen<scalar_t>(),    \
                             r_gen<scalar_t>(), false,                         \
                             IS_POINTER), /* Input point (c, 0) */             \
             std::make_tuple(C, p_gen<scalar_t>(), p_gen<scalar_t>(),          \
                             r_gen<scalar_t>(), 0.0, false,                    \
                             IS_POINTER), /* Input point (c, 0) && d2 == 0 */  \
             std::make_tuple(C, p_gen<scalar_t>(), 0.0, r_gen<scalar_t>(),     \
                             r_gen<scalar_t>(), false,                         \
                             IS_POINTER), /* Input point (c, 0) and big        \
                                  numbers (test that no rescaling happened) */ \
             std::make_tuple(                                                  \
                 C, scale_up_gen<scalar_t>(), scale_up_gen<scalar_t>(),        \
                 scale_up_gen<scalar_t>(), 0.0, false, IS_POINTER),            \
             std::make_tuple(C, scale_down_gen<scalar_t>(),                    \
                             scale_down_gen<scalar_t>(),                       \
                             scale_down_gen<scalar_t>(), 0.0, false,           \
                             IS_POINTER), /* Input point (0, c) */             \
             std::make_tuple(C, p_gen<scalar_t>(), p_gen<scalar_t>(), 0.0,     \
                             r_gen<scalar_t>(), false,                         \
                             IS_POINTER), /* Input point (0, c) && d1 == 0 */  \
             std::make_tuple(C, 0.0, p_gen<scalar_t>(), 0.0,                   \
                             r_gen<scalar_t>(), false,                         \
                             IS_POINTER), /* Input point (0, c) && d2 == 0 */  \
             std::make_tuple(C, p_gen<scalar_t>(), 0.0, 0.0,                   \
                             r_gen<scalar_t>(), false,                         \
                             IS_POINTER), /* Input point (0, c) && d2 < 0 */   \
             std::make_tuple(                                                  \
                 C, p_gen<scalar_t>(), -3.4, 0.0, r_gen<scalar_t>(), false,    \
                 IS_POINTER), /* Input point (0, c) && rescaling */            \
             std::make_tuple(C, p_gen<scalar_t>(), scale_up_gen<scalar_t>(),   \
                             0.0, r_gen<scalar_t>(), false, IS_POINTER),       \
             std::make_tuple(C, p_gen<scalar_t>(), scale_down_gen<scalar_t>(), \
                             0.0, r_gen<scalar_t>(), false, IS_POINTER),       \
             std::make_tuple(C, scale_up_gen<scalar_t>(), p_gen<scalar_t>(),   \
                             0.0, r_gen<scalar_t>(), false, IS_POINTER),       \
             std::make_tuple(C, scale_down_gen<scalar_t>(), p_gen<scalar_t>(), \
                             0.0, r_gen<scalar_t>(), false,                    \
                             IS_POINTER), /* d1 == 0 */                        \
             std::make_tuple(C, 0.0, p_gen<scalar_t>(), r_gen<scalar_t>(),     \
                             r_gen<scalar_t>(), false,                         \
                             IS_POINTER), /* d1 == 0 && d2 < 0 */              \
             std::make_tuple(                                                  \
                 C, 0.0, -3.4, r_gen<scalar_t>(), r_gen<scalar_t>(), false,    \
                 IS_POINTER), /* d1 * x1 > d2 * y1 (i.e. abs_c > abs_s) */     \
             std::make_tuple(C, 4.0, 2.1, 3.4, 1.5, false, IS_POINTER),        \
             std::make_tuple(C, 4.0, 1.5, -3.4, 2.1, false, IS_POINTER),       \
             std::make_tuple(C, 4.0, -1.5, 3.4, 2.1, false, IS_POINTER),       \
             std::make_tuple(C, 4.0, -1.5, 3.4, -2.1, false, IS_POINTER),      \
             std::make_tuple(                                                  \
                 C, 4.0, -1.5, -3.4, -2.1, false,                              \
                 IS_POINTER), /* d1 * x1 > d2 * y1 (i.e. abs_c > abs_s)        \
                     && rescaling */                                           \
             std::make_tuple(C, scale_down_gen<scalar_t>(), 2.1, 3.4, 1.5,     \
                             false, IS_POINTER),                               \
             std::make_tuple(C, scale_down_gen<scalar_t>(), 2.1,               \
                             scale_down_gen<scalar_t>(), 1.5, false,           \
                             IS_POINTER),                                      \
             std::make_tuple(C, scale_up_gen<scalar_t>(), 2.1,                 \
                             scale_down_gen<scalar_t>(), 1.5, false,           \
                             IS_POINTER),                                      \
             std::make_tuple(                                                  \
                 C, scale_down_gen<scalar_t>(), 2.1, scale_up_gen<scalar_t>(), \
                 1.5, false,                                                   \
                 IS_POINTER), /* d1 * x1 > d2 * y1 (i.e. abs_c > abs_s)        \
                     && Underflow */                                           \
             std::make_tuple(C, 0.01, 0.01,                                    \
                             std::numeric_limits<scalar_t>::min(),             \
                             std::numeric_limits<scalar_t>::min(), true,       \
                             IS_POINTER), /* d1 * x1 > d2 * y1 && Overflow */  \
             std::make_tuple(                                                  \
                 C, std::numeric_limits<scalar_t>::max(),                      \
                 std::numeric_limits<scalar_t>::max(), 0.01, 0.01, true,       \
                 IS_POINTER), /* d1 * x1 <= d2 * y1 (i.e. abs_c <= abs_s) */   \
             std::make_tuple(C, 2.1, 4.0, 1.5, 3.4, false, IS_POINTER),        \
             std::make_tuple(C, 2.1, 4.0, -1.5, 3.4, false, IS_POINTER),       \
             std::make_tuple(C, 2.1, -4.0, 1.5, 3.4, false, IS_POINTER),       \
             std::make_tuple(C, 2.1, -4.0, 1.5, -3.4, false, IS_POINTER),      \
             std::make_tuple(C, 2.1, -4.0, -1.5, -3.4, false,                  \
                             IS_POINTER), /* d1 * x1 <= d2 * y1 (i.e. abs_c <= \
                                 abs_s) && rescaling */                        \
             std::make_tuple(C, 2.1, scale_down_gen<scalar_t>(), 1.5, 3.4,     \
                             false, IS_POINTER),                               \
             std::make_tuple(C, 2.1, scale_down_gen<scalar_t>(), 1.5,          \
                             scale_down_gen<scalar_t>(), false, IS_POINTER),   \
             std::make_tuple(C, 2.1, scale_up_gen<scalar_t>(), 1.5,            \
                             scale_down_gen<scalar_t>(), false, IS_POINTER),   \
             std::make_tuple(C, 2.1, scale_down_gen<scalar_t>(), 1.5,          \
                             scale_up_gen<scalar_t>(), false,                  \
                             IS_POINTER), /* d1 * x1 <= d2 * y1 (i.e. abs_c <= \
                                 abs_s) && Underflow */                        \
             std::make_tuple(C, std::numeric_limits<scalar_t>::min(),          \
                             std::numeric_limits<scalar_t>::min(), 0.01, 0.01, \
                             true, IS_POINTER), /* d1 * x1 <= d2 * y1 (i.e.    \
                                       abs_c <= abs_s) && Overflow */          \
             std::make_tuple(C, 0.01, 0.01,                                    \
                             std::numeric_limits<scalar_t>::max(),             \
                             std::numeric_limits<scalar_t>::max(), true,       \
                             IS_POINTER), /* Overflow all */                   \
             std::make_tuple(C, std::numeric_limits<scalar_t>::max(),          \
                             std::numeric_limits<scalar_t>::max(),             \
                             std::numeric_limits<scalar_t>::max(),             \
                             std::numeric_limits<scalar_t>::max(), true,       \
                             IS_POINTER), /* Underflow all */                  \
             std::make_tuple(                                                  \
                 C, std::numeric_limits<scalar_t>::min(),                      \
                 std::numeric_limits<scalar_t>::min(),                         \
                 std::numeric_limits<scalar_t>::min(),                         \
                 std::numeric_limits<scalar_t>::min(), true,                   \
                 IS_POINTER), /* Numeric limits of one parameter */            \
             std::make_tuple(C, 1.0, 1.0, 1.0,                                 \
                             std::numeric_limits<scalar_t>::max(), false,      \
                             IS_POINTER),                                      \
             std::make_tuple(C, 1.0, 1.0,                                      \
                             std::numeric_limits<scalar_t>::max(), 1.0, false, \
                             IS_POINTER),                                      \
             std::make_tuple(C, 1.0, std::numeric_limits<scalar_t>::max(),     \
                             1.0, 1.0, false, IS_POINTER),                     \
             std::make_tuple(C, std::numeric_limits<scalar_t>::max(), 1.0,     \
                             1.0, 1.0, false,                                  \
                             IS_POINTER), /* Case that creates an infinite     \
                                             loop on cblas */                  \
             std::make_tuple(C, std::numeric_limits<scalar_t>::min(), -2.2,    \
                             std::numeric_limits<scalar_t>::min(),             \
                             std::numeric_limits<scalar_t>::min(), true,       \
                             IS_POINTER), /* Case that triggers underflow      \
                                 detection on abs_c <= abs_s && s >= 0 */      \
             std::make_tuple(C, 15.5, -2.2,                                    \
                             std::numeric_limits<scalar_t>::min(),             \
                             std::numeric_limits<scalar_t>::min(), false,      \
                             IS_POINTER), /* Test for previous errors */       \
             std::make_tuple(C, 0.0516274, -0.197215, -0.270436, -0.157621,    \
                             false, IS_POINTER))

#ifdef SB_ENABLE_USM
INSTANTIATE_ROTMG_TESTS(combi_usm, "usm", true);  // instantiate usm tests
INSTANTIATE_ROTMG_TESTS(combi_usm_scalar, "usm",
                        false);  // instantiate usm tests
#endif
INSTANTIATE_ROTMG_TESTS(combi_buffer, "buf", true);  // instantiate buffer tests
INSTANTIATE_ROTMG_TESTS(combi_buffer_scalar, "buf",
                        false);  // instantiate buffer tests

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  T d1, d2, x1, y1;
  bool will_overflow, is_pointer;
  BLAS_GENERATE_NAME(info.param, alloc, d1, d2, x1, y1, will_overflow,
                     is_pointer);
}

#ifdef SB_ENABLE_USM
BLAS_REGISTER_TEST_ALL(Rotmg_Usm, combination_t, combi_usm, generate_name);
BLAS_REGISTER_TEST_ALL(Rotmg_Usm_scalar, combination_t, combi_usm_scalar,
                       generate_name);
#endif
BLAS_REGISTER_TEST_ALL(Rotmg_Buffer, combination_t, combi_buffer,
                       generate_name);
BLAS_REGISTER_TEST_ALL(Rotmg_Buffer_scalar, combination_t, combi_buffer_scalar,
                       generate_name);

#undef INSTANTIATE_ROTMG_TESTS
