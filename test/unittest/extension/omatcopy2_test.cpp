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
/*!
 * @brief Host-baseline implementation of omatcopy2 used as reference.
 */
template <bool col_major, typename scalar_t, typename index_t>
std::enable_if_t<col_major, std::vector<scalar_t>> omatcopy2(
    const char& t, const index_t& m, const index_t& n, const scalar_t& alpha,
    std::vector<scalar_t>& in_matrix, const index_t& ld_in,
    const index_t& inc_in, std::vector<scalar_t>& out_matrix,
    const index_t& ld_out, const index_t inc_out) {
  if (t == 't') {
    for (int i = 0; i < m; ++i) {
      for (int j = 0, c = 0; j < n; ++j, ++c) {
        {
          out_matrix[j * inc_out + i * ld_out] =
              alpha * in_matrix[i * inc_in + j * ld_in];
        }
      }
    }
  } else {
    for (int i = 0; i < n; ++i) {
      for (int j = 0, c = 0; j < m; ++j, ++c) {
        {
          out_matrix[j * inc_out + i * ld_out] =
              alpha * in_matrix[j * inc_in + i * ld_in];
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
  index_t m, n, inc_in, ld_in_m, inc_out, ld_out_m;
  scalar_t alpha;

  std::tie(trans, m, n, alpha, inc_in, ld_in_m, inc_out, ld_out_m) = combi;

  // Leading dimensions are computed as multiples of the minimum value specified
  // in the oneMKL documentation at :
  // https://spec.oneapi.io/versions/latest/elements/oneMKL/source/domains/blas/omatcopy2.html#onemkl-blas-omatcopy2
  index_t ld_in = (inc_in * (m - 1) + 1) * ld_in_m;
  index_t ld_out =
      ((trans == 't') ? inc_out * (n - 1) + 1 : inc_out * (m - 1) + 1) *
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
  // TODO: There isn't a reference implementation from any library. So we compare
  // the results with a basic host implementation above. Working on a better
  // comparison.
  reference_blas::omatcopy2<true>(trans, m, n, alpha, A_ref, ld_in, inc_in,
                                  B_ref, ld_out, inc_out);

  auto matrix_in = blas::make_sycl_iterator_buffer<scalar_t>(A, m_a_size);
  auto matrix_out = blas::make_sycl_iterator_buffer<scalar_t>(B, m_b_size);

  blas::extension::_omatcopy2(sb_handle, trans, m, n, alpha, matrix_in, ld_in,
                              inc_in, matrix_out, ld_out, inc_out);

  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), matrix_out, B.data(), m_b_size);
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
                       ::testing::Values<scalar_t>(0, 2.5),            // alpha
                       ::testing::Values<index_t>(1, 2, 3),            // inc_in
                       ::testing::Values<index_t>(1, 3),     // ld_in_m
                       ::testing::Values<index_t>(1, 2, 3),  // inc_out
                       ::testing::Values<index_t>(1, 3));    // ld_out_m
#else
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values<char>('n', 't'),         // trans
                       ::testing::Values<index_t>(64, 129, 255),  // m
                       ::testing::Values<index_t>(64, 129, 255),  // n
                       ::testing::Values<scalar_t>(0, 2),         // alpha
                       ::testing::Values<index_t>(1, 7),          // inc_in
                       ::testing::Values<index_t>(1, 3),          // ld_in_m
                       ::testing::Values<index_t>(1, 7),          // inc_out
                       ::testing::Values<index_t>(1, 3));         // ld_out_m
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  char trans;
  index_t m, n, inc_in, ld_in_m, inc_out, ld_out_m;
  T alpha;
  BLAS_GENERATE_NAME(info.param, trans, m, n, alpha, inc_in, ld_in_m, inc_out,
                     ld_out_m);
}

BLAS_REGISTER_TEST_ALL(OmatCopy2, combination_t, combi, generate_name);
