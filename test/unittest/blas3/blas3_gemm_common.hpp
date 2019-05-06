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
 *  @filename blas3_gemm_values.hpp
 *
 **************************************************************************/

#include "blas_test.hpp"

#ifdef STRESS_TESTING
// This takes about ~6 hours on a decent CPU. Watch out!
const auto combi = ::testing::Combine(
    ::testing::Values(1, 5),  // batch_size
    ::testing::Values(1, 11, 13, 39, 64, 127, 255, 512, 1023, 1025),  // m
    ::testing::Values(14, 63, 257, 2, 31, 65, 255, 1023, 1025),       // n
    ::testing::Values(2, 67, 253, 65, 11, 39, 127, 511, 1023, 1025),  // k
    ::testing::Values('n', 't', 'c'),                                 // transa
    ::testing::Values('n', 't', 'c'),                                 // transb
    ::testing::Values(0.0, 1.0, 1.5),                                 // alpha
    ::testing::Values(0.0, 1.0, 1.5),                                 // beta
    ::testing::Values(1, 2),                                          // lda_mul
    ::testing::Values(1, 3),                                          // ldb_mul
    ::testing::Values(1, 2)                                           // ldc_mul
);
#else
// For the purpose of travis and other slower platforms, we need a faster test
const auto combi = ::testing::Combine(::testing::Values(5),        // batch_size
                                      ::testing::Values(11, 512),  // m
                                      ::testing::Values(14, 49),   // n
                                      ::testing::Values(21),       // k
                                      ::testing::Values('n', 't'),  // transa
                                      ::testing::Values('n', 't'),  // transb
                                      ::testing::Values(1.5),       // alpha
                                      ::testing::Values(0.0, 1.5),  // beta
                                      ::testing::Values(2),         // lda_mul
                                      ::testing::Values(3),         // ldb_mul
                                      ::testing::Values(2)          // ldc_mul
);
#endif

template <typename T>
using combination_t =
    std::tuple<int, int, int, int, char, char, T, T, int, int, int>;
