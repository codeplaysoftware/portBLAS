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
 *  @filename blas3_gemm_batched_test.cpp
 *
 **************************************************************************/

#include "blas3_gemm_common.hpp"
#include "blas_test.hpp"

template <typename scalar_t>
const auto BetaNonZeroLDMatch = ::testing::Combine(
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(0),                          // offset
    ::testing::Values(5),                          // batch
    ::testing::Values(63, 128),                    // m
    ::testing::Values(63, 128),                    // n
    ::testing::Values(63, 128),                    // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values<scalar_t>(3.0),              // alpha
    ::testing::Values<scalar_t>(7.0),              // beta
    ::testing::Values(1),                          // lda_mul
    ::testing::Values(1),                          // ldb_mul
    ::testing::Values(1),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(BatchGemm, BetaNonZeroLDMatch);

template <typename scalar_t>
const auto BetaNonZeroLDMultiplied = ::testing::Combine(
    ::testing::Values("usm", "buf"),   // allocation type
    ::testing::Values(0),              // offset
    ::testing::Values(1, 5),           // batch
    ::testing::Values(63, 128, 129),   // m
    ::testing::Values(63, 128, 129),   // n
    ::testing::Values(63, 128, 129),   // k
    ::testing::Values('n', 't'),       // transa
    ::testing::Values('n', 't'),       // transb
    ::testing::Values<scalar_t>(3.0),  // alpha
    ::testing::Values<scalar_t>(7.0),  // beta
    ::testing::Values(2),              // lda_mul
    ::testing::Values(3),              // ldb_mul
    ::testing::Values(4),              // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided,
                      gemm_batch_type_t::interleaved)  // batch_type
);
GENERATE_GEMM_TEST(BatchGemm, BetaNonZeroLDMultiplied);

template <typename scalar_t>
const auto BetaNonZeroLDMatchAlpha0 = ::testing::Combine(
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(0),                          // offset
    ::testing::Values(5),                          // batch
    ::testing::Values(128),                        // m
    ::testing::Values(128),                        // n
    ::testing::Values(128),                        // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values<scalar_t>(0.0),              // alpha
    ::testing::Values<scalar_t>(7.0),              // beta
    ::testing::Values(1),                          // lda_mul
    ::testing::Values(1),                          // ldb_mul
    ::testing::Values(1),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(BatchGemm, BetaNonZeroLDMatchAlpha0);

template <typename scalar_t>
const auto BetaNonZeroLDMultipliedAlpha0 = ::testing::Combine(
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(0),                          // offset
    ::testing::Values(5),                          // batch
    ::testing::Values(63),                         // m
    ::testing::Values(63),                         // n
    ::testing::Values(63),                         // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values<scalar_t>(0.0),              // alpha
    ::testing::Values<scalar_t>(7.0),              // beta
    ::testing::Values(2),                          // lda_mul
    ::testing::Values(3),                          // ldb_mul
    ::testing::Values(4),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_GEMM_TEST(BatchGemm, BetaNonZeroLDMultipliedAlpha0);

// GEMM STRIDED BATCHED tests
template <typename scalar_t>
const auto DefaultGemmAndGemmBatched =
    ::testing::Combine(::testing::Values("usm", "buf"),   // allocation type
                       ::testing::Values(0),              // offset
                       ::testing::Values(1, 5),           // batch
                       ::testing::Values(63, 128),        // m
                       ::testing::Values(63, 128),        // n
                       ::testing::Values(63, 128),        // k
                       ::testing::Values('n', 't'),       // transa
                       ::testing::Values('n', 't'),       // transb
                       ::testing::Values<scalar_t>(3.0),  // alpha
                       ::testing::Values<scalar_t>(7.0),  // beta
                       ::testing::Values(1),              // lda_mul
                       ::testing::Values(1),              // ldb_mul
                       ::testing::Values(1),              // ldc_mul
                       ::testing::Values(1),              // stride_a_mul
                       ::testing::Values(1),              // stride_b_mul
                       ::testing::Values(1)               // stride_c_mul
    );
GENERATE_GEMM_STRIDED_BATCHED_TEST(BatchStridedGemm, DefaultGemmAndGemmBatched);

template <typename scalar_t>
const auto AllStridedBatched =
    ::testing::Combine(::testing::Values("usm", "buf"),   // allocation type
                       ::testing::Values(0),              // offset
                       ::testing::Values(5),              // batch
                       ::testing::Values(128),            // m
                       ::testing::Values(128),            // n
                       ::testing::Values(128),            // k
                       ::testing::Values('n', 't'),       // transa
                       ::testing::Values('n', 't'),       // transb
                       ::testing::Values<scalar_t>(3.0),  // alpha
                       ::testing::Values<scalar_t>(7.0),  // beta
                       ::testing::Values(2),              // lda_mul
                       ::testing::Values(3),              // ldb_mul
                       ::testing::Values(4),              // ldc_mul
                       ::testing::Values(0, 1, 2),        // stride_a_mul
                       ::testing::Values(0, 1, 2),        // stride_b_mul
                       ::testing::Values(1, 2, 3)         // stride_c_mul
    );
GENERATE_GEMM_STRIDED_BATCHED_TEST(BatchStridedGemm, AllStridedBatched);

#ifdef BLAS_ENABLE_COMPLEX
template <typename scalar_t>
const auto CplxBetaNonZeroLDMatch = ::testing::Combine(
    ::testing::Values("usm", "buf"),                        // allocation type
    ::testing::Values(0),                                   // offset
    ::testing::Values(5),                                   // batch
    ::testing::Values(63, 128),                             // m
    ::testing::Values(63, 128),                             // n
    ::testing::Values(63, 128),                             // k
    ::testing::Values('n', 't'),                            // transa
    ::testing::Values('n', 't'),                            // transb
    ::testing::Values<std::complex<scalar_t>>({1.5, 1.0}),  // alpha
    ::testing::Values<std::complex<scalar_t>>({1.5, 3.0}),  // beta
    ::testing::Values(1),                                   // lda_mul
    ::testing::Values(1),                                   // ldb_mul
    ::testing::Values(1),                                   // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)           // batch_type
);
GENERATE_CPLX_GEMM_TEST(BatchGemm, CplxBetaNonZeroLDMatch);

template <typename scalar_t>
const auto CplxDefaultGemmAndGemmBatched = ::testing::Combine(
    ::testing::Values("usm", "buf"),                        // allocation type
    ::testing::Values(0),                                   // offset
    ::testing::Values(1, 5),                                // batch
    ::testing::Values(63, 128),                             // m
    ::testing::Values(63, 128),                             // n
    ::testing::Values(63, 128),                             // k
    ::testing::Values('n', 't'),                            // transa
    ::testing::Values('n', 't'),                            // transb
    ::testing::Values<std::complex<scalar_t>>({2.5, 1.0}),  // alpha
    ::testing::Values<std::complex<scalar_t>>({1.5, 3.0}),  // beta
    ::testing::Values(1),                                   // lda_mul
    ::testing::Values(1),                                   // ldb_mul
    ::testing::Values(1),                                   // ldc_mul
    ::testing::Values(1),                                   // stride_a_mul
    ::testing::Values(1),                                   // stride_b_mul
    ::testing::Values(1)                                    // stride_c_mul
);
GENERATE_CPLXGEMM_STRIDED_BATCHED_TEST(BatchStridedGemm,
                                       CplxDefaultGemmAndGemmBatched);

template <typename scalar_t>
const auto CplxAllStridedBatched = ::testing::Combine(
    ::testing::Values("usm", "buf"),                        // allocation type
    ::testing::Values(0),                                   // offset
    ::testing::Values(5),                                   // batch
    ::testing::Values(128),                                 // m
    ::testing::Values(128),                                 // n
    ::testing::Values(128),                                 // k
    ::testing::Values('n', 't'),                            // transa
    ::testing::Values('n', 't'),                            // transb
    ::testing::Values<std::complex<scalar_t>>({2.5, 1.0}),  // alpha
    ::testing::Values<std::complex<scalar_t>>({1.5, 3.0}),  // beta
    ::testing::Values(2),                                   // lda_mul
    ::testing::Values(3),                                   // ldb_mul
    ::testing::Values(4),                                   // ldc_mul
    ::testing::Values(0, 1, 2),                             // stride_a_mul
    ::testing::Values(0, 1, 2),                             // stride_b_mul
    ::testing::Values(1, 2, 3)                              // stride_c_mul
);
GENERATE_CPLXGEMM_STRIDED_BATCHED_TEST(BatchStridedGemm, CplxAllStridedBatched);
#endif
