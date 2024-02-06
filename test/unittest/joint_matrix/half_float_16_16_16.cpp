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
 *  @filename half_float_16_16_16.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include "joint_matrix_common.hpp"

template <typename scalar_t>
const auto SmallMatricesHalfFloat161616 = ::testing::Combine(
    ::testing::Values("half"),                     // input type
    ::testing::Values("float"),                    // output type
    ::testing::Values(16),                         // jm_m
    ::testing::Values(16),                         // jm_n
    ::testing::Values(16),                         // jm_n
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(0, 33),                      // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(11, 16, 32, 63),             // m
    ::testing::Values(11, 16, 32, 63),             // n
    ::testing::Values(17, 33, 64),                 // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values<scalar_t>(1.5),              // alpha
    ::testing::Values<scalar_t>(0, 1.5),           // beta
    ::testing::Values(1),                          // lda_mul
    ::testing::Values(1),                          // ldb_mul
    ::testing::Values(1),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_JOINTMATRIX_TEST(JointMatrix, SmallMatricesHalfFloat161616);

template <typename scalar_t>
const auto MediumMatricesHalfFloat161616 = ::testing::Combine(
    ::testing::Values("half"),                     // input type
    ::testing::Values("float"),                    // output type
    ::testing::Values(16),                         // jm_m
    ::testing::Values(16),                         // jm_n
    ::testing::Values(16),                         // jm_n
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(0, 33),                      // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(65, 127, 234, 511, 768),     // m
    ::testing::Values(65, 127, 234, 511, 768),     // n
    ::testing::Values(65, 127),                    // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values<scalar_t>(1.5),              // alpha
    ::testing::Values<scalar_t>(0, 1.5),           // beta
    ::testing::Values(1),                          // lda_mul
    ::testing::Values(1),                          // ldb_mul
    ::testing::Values(1),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_JOINTMATRIX_TEST(JointMatrix, MediumMatricesHalfFloat161616);

template <typename scalar_t>
const auto LargeMatricesHalfFloat161616 = ::testing::Combine(
    ::testing::Values("half"),                     // input type
    ::testing::Values("float"),                    // output type
    ::testing::Values(16),                         // jm_m
    ::testing::Values(16),                         // jm_n
    ::testing::Values(16),                         // jm_n
    ::testing::Values("usm", "buf"),               // allocation type
    ::testing::Values(0, 33),                      // offset
    ::testing::Values(1),                          // batch
    ::testing::Values(1024, 1535, 2024),           // m
    ::testing::Values(1024, 1535, 2024),           // n
    ::testing::Values(1536, 2049),                 // k
    ::testing::Values('n', 't'),                   // transa
    ::testing::Values('n', 't'),                   // transb
    ::testing::Values<scalar_t>(1.5),              // alpha
    ::testing::Values<scalar_t>(0, 1.5),           // beta
    ::testing::Values(1),                          // lda_mul
    ::testing::Values(1),                          // ldb_mul
    ::testing::Values(1),                          // ldc_mul
    ::testing::Values(gemm_batch_type_t::strided)  // batch_type
);
GENERATE_JOINTMATRIX_TEST(JointMatrix, LargeMatricesHalfFloat161616);
