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
 *  @filename blas3_gemm_tall_skinny_test.cpp
 *
 **************************************************************************/

#include "blas3_gemm_common.hpp"
#include "blas_test.hpp"

const auto BetaNonZeroLDMatch = ::testing::Combine(
    ::testing::Values(0),                          // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(7, 65),                      // m
    ::testing::Values(9, 126),                     // n
    ::testing::Values(2049),                       // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values(1.5),                        // alpha
    ::testing::Values(0.5),                        // beta
    ::testing::Values(1),                          // lda_mul
    ::testing::Values(1),                          // ldb_mul
    ::testing::Values(1),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(TallSkinnyGemm, BetaNonZeroLDMatch);

const auto BetaNonZeroLDMultiplied = ::testing::Combine(
    ::testing::Values(0),                          // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(7, 65),                      // m
    ::testing::Values(9, 126),                     // n
    ::testing::Values(2049),                       // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values(1.5),                        // alpha
    ::testing::Values(0.5),                        // beta
    ::testing::Values(2),                          // lda_mul
    ::testing::Values(3),                          // ldb_mul
    ::testing::Values(4),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(TallSkinnyGemm, BetaNonZeroLDMultiplied);

const auto BetaZero = ::testing::Combine(
    ::testing::Values(0),                          // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(7),                          // m
    ::testing::Values(9),                          // n
    ::testing::Values(1026),                       // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values(1.5),                        // alpha
    ::testing::Values(0.0),                        // beta
    ::testing::Values(1),                          // lda_mul
    ::testing::Values(1),                          // ldb_mul
    ::testing::Values(1),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(TallSkinnyGemm, BetaZero);

const auto OffsetNonZero = ::testing::Combine(
    ::testing::Values(10),                         // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(7),                          // m
    ::testing::Values(9),                          // n
    ::testing::Values(1026),                       // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values(1.5),                        // alpha
    ::testing::Values(0.5),                        // beta
    ::testing::Values(1),                          // lda_mul
    ::testing::Values(1),                          // ldb_mul
    ::testing::Values(1),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(TallSkinnyGemm, OffsetNonZero);
