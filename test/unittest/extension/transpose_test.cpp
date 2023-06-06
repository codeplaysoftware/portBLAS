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
 *  @filename transpose.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

using index_t = int;

enum trans_type : int { Inplace = 0, Outplace = 1 };

template <typename scalar_t>
using combination_t =
    std::tuple<trans_type, index_t, index_t, index_t, index_t, scalar_t>;

template <>
inline void dump_arg<trans_type>(std::ostream& ss, trans_type op) {
  ss << (int)op;
}

namespace reference_blas {
/**
 * @brief Reference out-of-place transpose host-implementation.
 *
 * @param in Input matrix pointer
 * @param ld_in Input matrix leading dimension
 * @param out Output matrix pointer
 * @param ld_in Output matrix leading dimension
 * @param M Number of rows in input matrix (Columns in output )
 * @param N Number of columns in input matrix (Rows in out output)
 */

template <typename T>
void Transpose(const T* in, const index_t& ld_in, T* out, const index_t& ld_out,
               const index_t& M, const index_t& N) {
  for (index_t i = 0; i < M; i++) {
    for (index_t j = 0; j < N; j++) {
      out[j + i * ld_out] = in[i + j * ld_in];
    }
  }
}

}  // namespace reference_blas

template <typename scalar_t>
void run_test(const combination_t<scalar_t>& combi) {
  trans_type tr_type;
  index_t m, n, ld_in_m, ld_out_m;
  scalar_t unused; /* Work around dpcpp compiler bug
                      (https://github.com/intel/llvm/issues/7075) */
  std::tie(tr_type, m, n, ld_in_m, ld_out_m, unused) = combi;

  // Compute leading dimensions using ld multipliers
  index_t ld_in = ld_in_m * m;
  index_t ld_out = ld_out_m * n;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  const index_t size_a = ld_in * n;
  const index_t size_b = ld_out * m;

  std::vector<scalar_t> A(size_a);
  std::vector<scalar_t> B(size_b);

  fill_random(A);

  std::vector<scalar_t> A_ref = A;
  std::vector<scalar_t> B_ref = B;

  // Reference implementation
  reference_blas::Transpose<scalar_t>(A.data(), ld_in, B_ref.data(), ld_out, m,
                                      n);

  if (tr_type == trans_type::Outplace) {
    auto matrix_in = blas::make_sycl_iterator_buffer<scalar_t>(A, size_a);
    auto matrix_out = blas::make_sycl_iterator_buffer<scalar_t>(B, size_b);

    blas::extension::_transpose(sb_handle, m, n, matrix_in, ld_in, matrix_out,
                                ld_out);

    auto event = blas::helper::copy_to_host<scalar_t>(
        sb_handle.get_queue(), matrix_out, B.data(), size_b);
    sb_handle.wait(event);

    // Validate the result
    const bool isAlmostEqual = utils::compare_vectors(B, B_ref);
    ASSERT_TRUE(isAlmostEqual);

  } else {
    // Inplace Transpose: TODO
  }
}

template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values(trans_type::Inplace,
                      trans_type::Outplace),   // Inplace | Outplace
    ::testing::Values<index_t>(64, 129, 255),  // m
    ::testing::Values<index_t>(64, 129, 255),  // n
    ::testing::Values<index_t>(1, 2, 3),       // ld_in_m
    ::testing::Values<index_t>(1, 2, 3),       // ld_in_n
    ::testing::Values<scalar_t>(0));           // scalar_t unused

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  index_t m, n, ld_in_m, ld_out_m;
  T unused;
  trans_type tr_type;
  BLAS_GENERATE_NAME(info.param, tr_type, m, n, ld_in_m, ld_out_m, unused);
}

BLAS_REGISTER_TEST_ALL(TransposeTest, combination_t, combi, generate_name);
