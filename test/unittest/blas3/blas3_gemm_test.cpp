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
 *  @filename blas3_gemm_test.cpp
 *
 **************************************************************************/

#include "blas3_gemm_common.hpp"
#include "blas_test.hpp"

template <typename scalar_t>
const auto SmallBetaNonZeroLDMatch = ::testing::Combine(
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(0),                          // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(11, 16, 32),                 // m
    ::testing::Values(11, 16, 32),                 // n
    ::testing::Values(16, 17),                     // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values<scalar_t>(1.5),              // alpha
    ::testing::Values<scalar_t>(1.5),              // beta
    ::testing::Values(1),                          // lda_mul
    ::testing::Values(1),                          // ldb_mul
    ::testing::Values(1),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(Gemm, SmallBetaNonZeroLDMatch);

template <typename scalar_t>
const auto SmallBetaZeroLDMatch = ::testing::Combine(
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(0),                          // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(11, 32),                     // m
    ::testing::Values(11, 32),                     // n
    ::testing::Values(17),                         // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values<scalar_t>(1.5),              // alpha
    ::testing::Values<scalar_t>(0.0),              // beta
    ::testing::Values(1),                          // lda_mul
    ::testing::Values(1),                          // ldb_mul
    ::testing::Values(1),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(Gemm, SmallBetaZeroLDMatch);

template <typename scalar_t>
const auto SmallBetaZeroLDMultiplied = ::testing::Combine(
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(0),                          // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(11, 32),                     // m
    ::testing::Values(11, 32),                     // n
    ::testing::Values(17),                         // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values<scalar_t>(1.5),              // alpha
    ::testing::Values<scalar_t>(0.0),              // beta
    ::testing::Values(2),                          // lda_mul
    ::testing::Values(3),                          // ldb_mul
    ::testing::Values(4),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(Gemm, SmallBetaZeroLDMultiplied);

template <typename scalar_t>
const auto AlphaZero = ::testing::Combine(
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(0, 10),                      // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(16),                         // m
    ::testing::Values(16),                         // n
    ::testing::Values(17),                         // k
    ::testing::Values('n'),                        // transa
    ::testing::Values('n'),                        // transb
    ::testing::Values<scalar_t>(0.0),              // alpha
    ::testing::Values<scalar_t>(0.0, 1.0),         // beta
    ::testing::Values(1, 2),                       // lda_mul
    ::testing::Values(1, 2),                       // ldb_mul
    ::testing::Values(1, 2),                       // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(Gemm, AlphaZero);

template <typename scalar_t>
const auto OffsetNonZero = ::testing::Combine(
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(1, 10),                      // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(16, 63),                     // m
    ::testing::Values(16, 63),                     // n
    ::testing::Values(17, 63),                     // k
    ::testing::Values('n'),                        // transa
    ::testing::Values('n'),                        // transb
    ::testing::Values<scalar_t>(1.0),              // alpha
    ::testing::Values<scalar_t>(1.0),              // beta
    ::testing::Values(1, 2),                       // lda_mul
    ::testing::Values(1, 2),                       // ldb_mul
    ::testing::Values(1, 2),                       // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(Gemm, OffsetNonZero);

template <typename scalar_t>
const auto LargeBetaNonZeroLDMatch = ::testing::Combine(
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(0),                          // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(253, 511, 1024, 2048, 2200), // m
    ::testing::Values(257, 511),                   // n
    ::testing::Values(253, 511),                   // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values<scalar_t>(1.0),              // alpha
    ::testing::Values<scalar_t>(1.0),              // beta
    ::testing::Values(1),                          // lda_mul
    ::testing::Values(1),                          // ldb_mul
    ::testing::Values(1),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(Gemm, LargeBetaNonZeroLDMatch);
