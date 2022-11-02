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

#include <type_traits>
#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, float, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {

  /* sdsdot is only valid when using floats */
  static_assert(std::is_same<scalar_t, float>::value);

  index_t size;
  float sb;
  index_t incX;
  index_t incY;
  std::tie(size, sb,incX, incY) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  // Input vectors
  std::vector<data_t> x_v(size * incX);
  fill_random(x_v);
  std::vector<data_t> y_v(size * incY);
  fill_random(y_v);

  std::cout << "x_v: [";
  for (float i: x_v) { std::cout << i << ' '; }
  std::cout << "]" << std::endl;

  std::cout << "y_v: [";
  for (float i: y_v) { std::cout << i << ' '; }
  std::cout << "]" << std::endl;

  // Output vector
  std::vector<data_t> out_s(1, 10.0);

  // Reference implementation
  auto out_cpu_s =
      reference_blas::sdsdot(size, sb, x_v.data(), incX, y_v.data(), incY);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
  auto gpu_x_v = utils::make_quantized_buffer<scalar_t>(ex, x_v);
  auto gpu_y_v = utils::make_quantized_buffer<scalar_t>(ex, y_v);
  auto gpu_out_s = utils::make_quantized_buffer<scalar_t>(ex, out_s);

  _sdsdot(ex, size, sb, gpu_x_v, incX, gpu_y_v, incY, gpu_out_s);
  auto event = utils::quantized_copy_to_host<scalar_t>(ex, gpu_out_s, out_s);
  ex.get_policy_handler().wait(event);

  // Validate the result
  const bool isAlmostEqual =
      utils::almost_equal<data_t, scalar_t>(out_s[0], out_cpu_s);
  std::cout << "SYCL-BLAS: " << out_s[0] << " REFERENCE: " << out_cpu_s << std::endl;
  ASSERT_TRUE(isAlmostEqual);


  ex.get_policy_handler().get_queue().wait();
  std::cout << "END OF TEST" << std::endl;
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 1002, 1002400),  // size
                       ::testing::Values(9.5f, 0.1f),             // sb
                       ::testing::Values(1, 4),                   // incX
                       ::testing::Values(1, 3)                    // incY
    );
#else
const auto combi = ::testing::Combine(::testing::Values(11, 1002),  // size
                                      ::testing::Values(9.5f, 54.54f),                           // sb
                                      ::testing::Values(1, 4),                                 // incX
                                      ::testing::Values(1, 3)                                  // incY
);
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  int size, incX, incY;
  float sb;
  BLAS_GENERATE_NAME(info.param, size, sb, incX, incY);
}

BLAS_REGISTER_TEST_FLOAT(Sdsdot, combination_t, combi, generate_name);
