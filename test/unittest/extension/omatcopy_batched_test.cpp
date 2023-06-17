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
 *  @filename omatcopy_batched_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

using index_t = int;

template <typename scalar_t>
using combination_t = std::tuple<char, index_t, index_t, scalar_t, index_t,
                                 index_t, index_t, index_t, index_t>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  char trans;
  index_t m, n, ld_in_m, ld_out_m, stride_in_m, stride_out_m, batch_size;
  scalar_t alpha;

  std::tie(trans, m, n, alpha, ld_in_m, ld_out_m, stride_in_m, stride_out_m,
           batch_size) = combi;

  // Compute leading dimensions using second_dim-ld multipliers
  index_t ld_in = ld_in_m * m;
  index_t ld_out = ld_out_m * (trans == 't' ? n : m);

  index_t size_a = ld_in * n;
  index_t size_b = ld_out * (trans == 't' ? m : n);

  // Compute Strides using size-stride multipliers
  index_t stride_in = stride_in_m * size_a;
  index_t stride_out = stride_out_m * size_b;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  std::vector<scalar_t> A(stride_in * batch_size);
  std::vector<scalar_t> B(stride_out * batch_size, 0);

  fill_random(A);

  std::vector<scalar_t> A_ref = A;
  std::vector<scalar_t> B_ref = B;

  // Reference implementation
  for (auto b = 0; b < batch_size; b++) {
    reference_blas::omatcopy(trans, m, n, alpha, A_ref.data() + b * stride_in,
                             ld_in, B_ref.data() + b * stride_out, ld_out);
  }

  auto matrix_in =
      blas::make_sycl_iterator_buffer<scalar_t>(A, stride_in * batch_size);
  auto matrix_out =
      blas::make_sycl_iterator_buffer<scalar_t>(B, stride_out * batch_size);

  blas::extension::_omatcopy_batch(sb_handle, trans, m, n, alpha, matrix_in,
                                   ld_in, stride_in, matrix_out, ld_out,
                                   stride_out, batch_size);

  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), matrix_out, B.data(), stride_out * batch_size);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(B, B_ref);
  ASSERT_TRUE(isAlmostEqual);
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values<char>('n', 't'),              // trans
                       ::testing::Values<index_t>(1024, 4050, 16380),  // m
                       ::testing::Values<index_t>(1024, 4050, 16380),  // n
                       ::testing::Values<scalar_t>(0, 1.05, -20.01),   // alpha
                       ::testing::Values<index_t>(3, 5),     // ld_in_m
                       ::testing::Values<index_t>(3, 5),     // ld_out_m
                       ::testing::Values<index_t>(5, 10),    // stride_in_m
                       ::testing::Values<index_t>(5, 10),    // stride_out_m
                       ::testing::Values<index_t>(10, 21));  // batch_size
#else
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values<char>('t'),              // trans
                       ::testing::Values<index_t>(64, 129, 255),  // m
                       ::testing::Values<index_t>(64, 129, 255),  // n
                       ::testing::Values<scalar_t>(0, 2),         // alpha
                       ::testing::Values<index_t>(1, 2, 3),       // ld_in_m
                       ::testing::Values<index_t>(1, 2, 3),       // ld_out_m
                       ::testing::Values<index_t>(1, 3),          // stride_in_m
                       ::testing::Values<index_t>(1, 3),      // stride_out_m
                       ::testing::Values<index_t>(1, 2, 5));  // batch_size
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  char trans;
  index_t m, n, ld_in_m, ld_out_m, stride_in_m, stride_out_m, batch_size;
  T alpha;
  BLAS_GENERATE_NAME(info.param, trans, m, n, alpha, ld_in_m, ld_out_m,
                     stride_in_m, stride_out_m, batch_size);
}

BLAS_REGISTER_TEST_ALL(OmatCopyBatched, combination_t, combi, generate_name);
