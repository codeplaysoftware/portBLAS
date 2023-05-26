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
 *  @filename omatcopy2_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<char, index_t, index_t, scalar_t, index_t,
                                 index_t, index_t, index_t>;

namespace reference_blas {
template <bool col_major, typename scalar_t, typename index_t>
std::enable_if_t<col_major, std::vector<scalar_t>> omatcopy2(
    const char t, const index_t rows, const index_t cols, const scalar_t alpha,
    std::vector<scalar_t>& in_matrix, const index_t ld_in,
    const index_t in_stride, std::vector<scalar_t>& out_matrix,
    const index_t ld_out, const index_t out_stride) {
  if (t == 't') {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0, c = 0; j < cols; ++j, ++c) {
        {
          out_matrix[j * out_stride + i * ld_out] =
              alpha * in_matrix[i * in_stride + j * ld_in];
        }
      }
    }
  } else {
    for (int i = 0; i < cols; ++i) {
      for (int j = 0, c = 0; j < rows; ++j, ++c) {
        {
          out_matrix[j * out_stride + i * ld_out] =
              alpha * in_matrix[j * in_stride + i * ld_in];
        }
      }
    }
  }
  return out_matrix;
}
}  // namespace reference_blas

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  char trans;
  index_t m, n, stride_in, ld_in_m, stride_out, ld_out_m;
  scalar_t alpha;

  std::tie(trans, m, n, alpha, stride_in, ld_in_m, stride_out, ld_out_m) =
      combi;

  // Leading dimensions are computed as multiples of the minimum value specified
  // in the oneMKL documentation at :
  // https://spec.oneapi.io/versions/latest/elements/oneMKL/source/domains/blas/omatcopy2.html#onemkl-blas-omatcopy2
  index_t ld_in = (stride_in * (m - 1) + 1) * ld_in_m;
  index_t ld_out =
      ((trans == 't') ? stride_out * (n - 1) + 1 : stride_out * (m - 1) + 1) *
      ld_out_m;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  index_t m_a_size = ld_in * n;
  index_t m_b_size = ld_out * (trans == 't' ? m : n);

  std::vector<scalar_t> A(m_a_size);
  std::vector<scalar_t> B(m_b_size);

  fill_random(A);
  fill_random(B);

  std::vector<scalar_t> A_ref = A;
  std::vector<scalar_t> B_ref = B;

  // Reference implementation
  // TODO: There isn't a reference implemantion from any library. So we compare
  // the results with a basic host implementation above. Working on a better
  // comparison.
  reference_blas::omatcopy2<true>(trans, m, n, alpha, A_ref, ld_in, stride_in,
                                  B_ref, ld_out, stride_out);

  auto matrix_in = blas::make_sycl_iterator_buffer<scalar_t>(A, m_a_size);
  auto matrix_out = blas::make_sycl_iterator_buffer<scalar_t>(B, m_b_size);

  blas::extension::_omatcopy2(sb_handle, trans, m, n, alpha, matrix_in, ld_in,
                              stride_in, matrix_out, ld_out, stride_out);

  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), matrix_out, B.data(), m_b_size);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(B, B_ref);
  ASSERT_TRUE(isAlmostEqual);
}

template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values<char>('n', 't'), ::testing::Values<index_t>(64, 129, 255),
    ::testing::Values<index_t>(64, 129, 255), ::testing::Values<scalar_t>(0, 2),
    ::testing::Values<index_t>(1, 2, 7), ::testing::Values<index_t>(1, 2, 3),
    ::testing::Values<index_t>(1, 2, 7), ::testing::Values<index_t>(1, 2, 3));

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  char trans;
  index_t m, n, stride_in, ld_in_m, stride_out, ld_out_m;
  T alpha;
  BLAS_GENERATE_NAME(info.param, trans, m, n, alpha, stride_in, ld_in_m,
                     stride_out, ld_out_m);
}

BLAS_REGISTER_TEST_ALL(OmatCopy2, combination_t, combi, generate_name);
