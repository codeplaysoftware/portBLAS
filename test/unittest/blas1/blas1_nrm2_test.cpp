/***************************************************************************
 *
 *  @license
 *  Nrm2right (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a nrm2 of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a nrm2 of the License has been included in this
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
 *  @filename blas1_nrm2_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<api_type, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  api_type api;
  index_t size;
  index_t incX;
  std::tie(api, size, incX) = combi;

  // Input vectors
  std::vector<scalar_t> x_v(size * incX);
  fill_random(x_v);

  // Output scalar
  scalar_t out_s = 10.0;

  // Reference implementation
  auto out_cpu_s = reference_blas::nrm2(size, x_v.data(), incX);

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(x_v, size * incX);

  if (api == api_type::async) {
    auto gpu_out_s = blas::make_sycl_iterator_buffer<scalar_t>(&out_s, 1);
    _nrm2(sb_handle, size, gpu_x_v, incX, gpu_out_s);
    auto event =
        blas::helper::copy_to_host(sb_handle.get_queue(), gpu_out_s, &out_s, 1);
    sb_handle.wait(event);
  } else {
    out_s = _nrm2(sb_handle, size, gpu_x_v, incX);
  }

  // Validate the result
  const bool isAlmostEqual = utils::almost_equal(out_s, out_cpu_s);
  ASSERT_TRUE(isAlmostEqual);
}

template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values(api_type::async,
                                                        api_type::sync),  // Api
                                      ::testing::Values(11, 1002),  // size
                                      ::testing::Values(1, 4)       // incX
);

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  api_type api;
  int size, incX;
  BLAS_GENERATE_NAME(info.param, api, size, incX);
}

BLAS_REGISTER_TEST_ALL(Nrm2, combination_t, combi, generate_name);
