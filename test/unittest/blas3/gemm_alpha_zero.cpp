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
 *  @filename gemm_alpha_zero.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include "utils/float_comparison.hpp"

#include "container/sycl_iterator.h"
#include "interface/blas3_interface.h"

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

using DataType = float;
using IndexType = int;

struct Arguments {
  char trans_a;
  char trans_b;
  DataType alpha;
  DataType beta;
  IndexType m;
  IndexType k;
  IndexType n;
  IndexType lda;
  IndexType ldb;
  IndexType ldc;
};

void check_alpha0_gemm_results(Arguments const& args,
                               std::vector<DataType> const& expected) {
  IndexType size_a = args.lda * args.k;
  IndexType size_b = args.ldb * args.n;
  IndexType size_c = args.ldc * args.n;

  std::vector<DataType> host_a(size_a);
  std::iota(begin(host_a), end(host_a), 1);
  std::vector<DataType> host_b(size_b);
  std::iota(begin(host_b), end(host_b), 1);
  std::vector<DataType> host_c(size_c);
  std::iota(begin(host_c), end(host_c), 1);

  auto q = make_queue();
  test_executor_t ex(q);
  auto policy_handler = ex.get_policy_handler();
  {
    auto device_a =
        blas::make_sycl_iterator_buffer(host_a.data(), host_a.size());
    auto device_b =
        blas::make_sycl_iterator_buffer(host_b.data(), host_b.size());
    auto device_c =
        blas::make_sycl_iterator_buffer(host_c.data(), host_c.size());

    auto event_list = ::blas::_gemm(
        ex, args.trans_a, args.trans_b, args.m, args.n, args.k, args.alpha,
        device_a, args.lda, device_b, args.ldb, args.beta, device_c, args.ldc);

    policy_handler.wait(event_list);
  }

  ASSERT_TRUE(utils::compare_vectors(host_c, expected));
  policy_handler.wait();
}

TEST(GemmAlphaZero, AlphaAndBetaZero) {
  Arguments args;
  args.alpha = 0;
  args.beta = 0;
  args.trans_a = 'n';
  args.trans_b = 'n';

  args.m = 4;
  args.k = 4;
  args.n = 4;

  args.lda = args.m;
  args.ldb = args.k;
  args.ldc = args.m;

  IndexType size_c = args.ldc * args.n;

  // When beta = 0 the output should all be zero
  std::vector<DataType> expected(size_c, DataType{0});

  check_alpha0_gemm_results(args, expected);
}

TEST(GemmAlphaZero, BetaOneLeadingDimsMatchSize) {
  Arguments args;
  args.alpha = 0;
  args.beta = 1;
  args.trans_a = 'n';
  args.trans_b = 'n';

  args.m = 4;
  args.k = 2;
  args.n = 4;

  args.lda = args.m;
  args.ldb = args.k;
  args.ldc = args.m;

  // With beta = 1 the output is the same as the input
  std::vector<DataType> expected = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,
                                    7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                    13.0, 14.0, 15.0, 16.0};

  check_alpha0_gemm_results(args, expected);
}

TEST(GemmAlphaZero, BetaTwoLeadingDimsMatchSize) {
  Arguments args;
  args.alpha = 0;
  args.beta = 2;
  args.trans_a = 'n';
  args.trans_b = 'n';

  args.m = 4;
  args.k = 2;
  args.n = 4;

  args.lda = args.m;
  args.ldb = args.k;
  args.ldc = args.m;

  // With beta = 2 all entries are doubled
  std::vector<DataType> expected = {2.0,  4.0,  6.0,  8.0,  10.0, 12.0,
                                    14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
                                    26.0, 28.0, 30.0, 32.0};

  check_alpha0_gemm_results(args, expected);
}

TEST(GemmAlphaZero, BetaOneLeadingDimLargerThanM) {
  Arguments args;
  args.alpha = 0;
  args.beta = 1;
  args.trans_a = 'n';
  args.trans_b = 'n';

  args.m = 4;
  args.k = 2;
  args.n = 4;

  args.lda = args.m;
  args.ldb = args.k;
  args.ldc = args.m + 2;

  // With beta = 1 the output is the same as the input
  std::vector<DataType> expected = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

  check_alpha0_gemm_results(args, expected);
}

TEST(GemmAlphaZero, BetaTwoLeadingDimLargerThanM) {
  Arguments args;
  args.alpha = 0;
  args.beta = 2;
  args.trans_a = 'n';
  args.trans_b = 'n';

  args.m = 4;
  args.k = 2;
  args.n = 4;

  args.lda = args.m;
  args.ldb = args.k;
  args.ldc = args.m + 2;

  // With LDC larger than M, C is a submatrix of a larger matrix, so not all
  // entries are doubled.
  std::vector<DataType> expected = {
      2.0,  4.0,  6.0,  8.0,  5.0,  6.0,  14.0, 16.0, 18.0, 20.0, 11.0, 12.0,
      26.0, 28.0, 30.0, 32.0, 17.0, 18.0, 38.0, 40.0, 42.0, 44.0, 23.0, 24.0};

  check_alpha0_gemm_results(args, expected);
}
