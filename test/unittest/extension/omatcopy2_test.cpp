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
 *  @filename omatcopy2_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include "extension_reference.hpp"

template <typename scalar_t>
using combination_t = std::tuple<std::string, char, index_t, index_t, scalar_t,
                                 index_t, index_t, index_t, index_t>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  char trans;
  index_t m, n, inc_in, ld_in_m, inc_out, ld_out_m;
  scalar_t alpha;

  std::tie(alloc, trans, m, n, alpha, inc_in, ld_in_m, inc_out, ld_out_m) =
      combi;

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
  // TODO: There isn't a reference implementation from any library. So we
  // compare the results with a basic host implementation. Working on a
  // better comparison.
  reference_blas::ext_omatcopy2(trans, m, n, alpha, A_ref.data(), ld_in, inc_in,
                                B_ref.data(), ld_out, inc_out);

  auto matrix_in = helper::allocate<mem_alloc, scalar_t>(m_a_size, q);
  auto matrix_out = helper::allocate<mem_alloc, scalar_t>(m_b_size, q);

  auto copy_in =
      helper::copy_to_device<scalar_t>(q, A.data(), matrix_in, m_a_size);
  auto copy_out =
      helper::copy_to_device<scalar_t>(q, B.data(), matrix_out, m_b_size);

  auto omatcopy2_event =
      blas::_omatcopy2(sb_handle, trans, m, n, alpha, matrix_in, ld_in, inc_in,
                       matrix_out, ld_out, inc_out, {copy_in, copy_out});

  sb_handle.wait(omatcopy2_event);

  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), matrix_out, B.data(), m_b_size);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(B, B_ref);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(matrix_in, q);
  helper::deallocate<mem_alloc>(matrix_out, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  char trans;
  index_t m, n, inc_in, ld_in_m, inc_out, ld_out_m;
  scalar_t alpha;

  std::tie(alloc, trans, m, n, alpha, inc_in, ld_in_m, inc_out, ld_out_m) =
      combi;

  if (alloc == "usm") {
#ifdef SB_ENABLE_USM
    run_test<scalar_t, helper::AllocType::usm>(combi);
#else
    GTEST_SKIP();
#endif
  } else {
    run_test<scalar_t, helper::AllocType::buffer>(combi);
  }
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),
                       ::testing::Values<char>('n', 't'),              // trans
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
    ::testing::Combine(::testing::Values("usm", "buf"),    // allocation type
                       ::testing::Values<char>('n', 't'),  // trans
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
  std::string alloc;
  char trans;
  index_t m, n, inc_in, ld_in_m, inc_out, ld_out_m;
  T alpha;
  BLAS_GENERATE_NAME(info.param, alloc, trans, m, n, alpha, inc_in, ld_in_m,
                     inc_out, ld_out_m);
}

BLAS_REGISTER_TEST_ALL(OmatCopy2, combination_t, combi, generate_name);
