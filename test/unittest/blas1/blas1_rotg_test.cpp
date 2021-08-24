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
using combination_t = std::tuple<int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int size;
  int incA;
  int incB;
  std::tie(size, incA, incB) = combi;

#ifdef SYCL_BLAS_USE_USM
  using data_t = scalar_t;
#else
  using data_t = utils::data_storage_t<scalar_t>;
#endif

  // Input vectors
  std::vector<data_t> a_v(size * incA);
  fill_random(a_v);
  std::vector<data_t> b_v(size * incB);
  fill_random(b_v);

  // Output vectors
  std::vector<data_t> out_s(1, 10.0);
  std::vector<data_t> a_cpu_v(a_v);
  std::vector<data_t> b_cpu_v(b_v);

  // Looks like we don't have a SYCL rotg implementation
  data_t c_d;
  data_t s_d;
  data_t sa = a_v[0];
  data_t sb = a_v[1];
  reference_blas::rotg(&sa, &sb, &c_d, &s_d);

  // Reference implementation
  std::vector<data_t> c_cpu_v(size * incA);
  std::vector<data_t> s_cpu_v(size * incB);
  reference_blas::rot(size, a_cpu_v.data(), incA, b_cpu_v.data(), incB, c_d,
                      s_d);
  auto out_cpu_s =
      reference_blas::dot(size, a_cpu_v.data(), incA, b_cpu_v.data(), incB);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
#ifdef SYCL_BLAS_USE_USM
  data_t* gpu_a_v = cl::sycl::malloc_device<data_t>(size * incA, q);
  data_t* gpu_b_v = cl::sycl::malloc_device<data_t>(size * incB, q);
  data_t* gpu_out_s = cl::sycl::malloc_device<data_t>(1, q);

  q.memcpy(gpu_a_v, a_v.data(), sizeof(data_t) * size * incA).wait();
  q.memcpy(gpu_b_v, b_v.data(), sizeof(data_t) * size * incB).wait();
  q.memcpy(gpu_out_s, out_s.data(), sizeof(data_t)).wait();
#else
  auto gpu_a_v = utils::make_quantized_buffer<scalar_t>(ex, a_v);
  auto gpu_b_v = utils::make_quantized_buffer<scalar_t>(ex, b_v);
  auto gpu_out_s = utils::make_quantized_buffer<scalar_t>(ex, out_s);
#endif

  auto c = static_cast<scalar_t>(c_d);
  auto s = static_cast<scalar_t>(s_d);

  auto ev = _rot(ex, size, gpu_a_v, incA, gpu_b_v, incB, c, s);
#ifdef SYCL_BLAS_USE_USM
  ex.get_policy_handler().wait(ev);
#endif
  ev = _dot(ex, size, gpu_a_v, incA, gpu_b_v, incB, gpu_out_s);
#ifdef SYCL_BLAS_USE_USM
  ex.get_policy_handler().wait(ev);
#endif

  auto event = 
#ifdef SYCL_BLAS_USE_USM
  q.memcpy(out_s.data(), gpu_out_s, sizeof(data_t));
#else 
  utils::quantized_copy_to_host<scalar_t>(ex, gpu_out_s, out_s);
#endif
  ex.get_policy_handler().wait({event});

  // Validate the result
  const bool isAlmostEqual =
      utils::almost_equal<data_t, scalar_t>(out_s[0], out_cpu_s);
  ASSERT_TRUE(isAlmostEqual);

#ifdef SYCL_BLAS_USE_USM
  cl::sycl::free(gpu_a_v, q);
  cl::sycl::free(gpu_b_v, q);
  cl::sycl::free(gpu_out_s, q);
#endif
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 1002, 1002400),  // size
                       ::testing::Values(1, 4),                   // incX
                       ::testing::Values(1, 3)                    // incY
    );
#else
const auto combi = ::testing::Combine(::testing::Values(11, 1002),  // size
                                      ::testing::Values(4),         // incA
                                      ::testing::Values(3)          // incB
);
#endif

BLAS_REGISTER_TEST(Rotg, combination_t, combi);
