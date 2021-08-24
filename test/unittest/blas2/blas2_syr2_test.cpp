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
 *  @filename blas2_syr2_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<char, int, scalar_t, int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int n;
  int lda_mul;
  int incX;
  int incY;
  char uplo;
  scalar_t alpha;
  std::tie(uplo, n, alpha, incX, incY, lda_mul) = combi;
  int lda = n * lda_mul;

#ifdef SYCL_BLAS_USE_USM
  using data_t = scalar_t;
#else
  using data_t = utils::data_storage_t<scalar_t>;
#endif

  // Input vector
  std::vector<data_t> x_v(n * incX);
  std::vector<data_t> y_v(n * incY);
  fill_random(x_v);
  fill_random(y_v);

  // Output matrix
  std::vector<data_t> a_m(n * lda, 7.0);
  std::vector<data_t> a_cpu_m(n * lda, 7.0);

  // SYSTEM SYR
  reference_blas::syr2(&uplo, n, static_cast<data_t>(alpha), x_v.data(), incX,
                       y_v.data(), incY, a_cpu_m.data(), lda);

  auto q = make_queue();
  test_executor_t ex(q);

#ifdef SYCL_BLAS_USE_USM
  data_t* x_v_gpu = cl::sycl::malloc_device<data_t>(n * incX, q);
  data_t* y_v_gpu = cl::sycl::malloc_device<data_t>(n * incY, q);
  data_t* a_m_gpu = cl::sycl::malloc_device<data_t>(n * lda, q);

  q.memcpy(x_v_gpu, x_v.data(), sizeof(data_t) * n * incX).wait();
  q.memcpy(y_v_gpu, y_v.data(), sizeof(data_t) * n * incY).wait();
  q.memcpy(a_m_gpu, a_m.data(), sizeof(data_t) * n * lda).wait();
#else
  auto x_v_gpu = utils::make_quantized_buffer<scalar_t>(ex, x_v);
  auto y_v_gpu = utils::make_quantized_buffer<scalar_t>(ex, y_v);
  auto a_m_gpu = utils::make_quantized_buffer<scalar_t>(ex, a_m);
#endif

  // SYCLsyr2
  auto ev = _syr2(ex, uplo, n, alpha, x_v_gpu, incX, y_v_gpu, incY, a_m_gpu, lda);
#ifdef SYCL_BLAS_USE_USM
  ex.get_policy_handler().wait(ev);
#endif

  auto event = 
#ifdef SYCL_BLAS_USE_USM
      q.memcpy(a_m.data(), a_m_gpu, sizeof(data_t) * n * lda);
#else
      utils::quantized_copy_to_host<scalar_t>(ex, a_m_gpu, a_m);
#endif
  ex.get_policy_handler().wait({event});

  const bool isAlmostEqual =
      utils::compare_vectors<data_t, scalar_t>(a_m, a_cpu_m);
  ASSERT_TRUE(isAlmostEqual);

#ifdef SYCL_BLAS_USE_USM
  cl::sycl::free(x_v_gpu, q);
  cl::sycl::free(y_v_gpu, q);
  cl::sycl::free(a_m_gpu, q);
#endif
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values('u', 'l'),                 // UPLO
                       ::testing::Values(14, 63, 257, 1010, 2025),  // n
                       ::testing::Values(0.0, 1.0, 1.5),            // alpha
                       ::testing::Values(1, 2),                     // incX
                       ::testing::Values(1, 2),                     // incY
                       ::testing::Values(1, 2)                      // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
const auto combi = ::testing::Combine(::testing::Values('u', 'l'),  // UPLO
                                      ::testing::Values(2025),      // n
                                      ::testing::Values(0.0, 1.5),  // alpha
                                      ::testing::Values(2),         // incX
                                      ::testing::Values(2),         // incY
                                      ::testing::Values(2)          // lda_mul
);
#endif

BLAS_REGISTER_TEST(Syr2, combination_t, combi);
