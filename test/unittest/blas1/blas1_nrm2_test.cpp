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
using combination_t = std::tuple<std::string, api_type, int, int, scalar_t>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  api_type api;
  index_t size;
  index_t incX;
  scalar_t unused;
  std::tie(alloc, api, size, incX, unused) = combi;

  auto vector_size = size * std::abs(incX);
  // Input vectors
  std::vector<scalar_t> x_v(vector_size);
  fill_random(x_v);

  // Output scalar
  scalar_t out_s = 0.0;
  scalar_t out_cpu_s = 0.0;

  // Reference implementation
  if (incX < 0) {
    // Some reference implementations of BLAS do not support negative
    // increments for nrm2. To simulate what is specified in the
    // oneAPI spec, invert the vector and use a positive increment.
    std::vector<scalar_t> x_v_inv(vector_size);
    std::reverse_copy(x_v.begin(), x_v.end() + (incX + 1), x_v_inv.begin());
    out_cpu_s = reference_blas::nrm2(size, x_v_inv.data(), -incX);
  } else {
    out_cpu_s = reference_blas::nrm2(size, x_v.data(), incX);
  }

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = blas::helper::allocate<mem_alloc, scalar_t>(vector_size, q);

  auto copy_x =
      blas::helper::copy_to_device(q, x_v.data(), gpu_x_v, vector_size);

  if (api == api_type::async) {
    auto gpu_out_s = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
    auto copy_out =
        blas::helper::copy_to_device<scalar_t>(q, &out_s, gpu_out_s, 1);
    auto nrm2_event =
        _nrm2(sb_handle, size, gpu_x_v, incX, gpu_out_s, {copy_x, copy_out});
    sb_handle.wait(nrm2_event);
    auto event =
        blas::helper::copy_to_host(sb_handle.get_queue(), gpu_out_s, &out_s, 1);
    sb_handle.wait(event);
    helper::deallocate<mem_alloc>(gpu_out_s, q);
  } else {
    out_s = _nrm2(sb_handle, size, gpu_x_v, incX, {copy_x});
  }

  // Validate the result
  const bool isAlmostEqual = utils::almost_equal(out_s, out_cpu_s);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(gpu_x_v, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  api_type api;
  index_t size;
  index_t incX;
  scalar_t unused;
  std::tie(alloc, api, size, incX, unused) = combi;

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
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),    // allocation type
                       ::testing::Values(api_type::async,
                                         api_type::sync),  // Api
                       ::testing::Values(11, 1002),        // size
                       ::testing::Values(1, 4, -3),        // incX
                       ::testing::Values(scalar_t{1}));

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  api_type api;
  int size, incX;
  T unused;
  BLAS_GENERATE_NAME(info.param, alloc, api, size, incX, unused);
}

BLAS_REGISTER_TEST_ALL(Nrm2, combination_t, combi, generate_name);
