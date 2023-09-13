/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2018 Codeplay Software Limited
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
 *  @filename reference_gemm.hpp
 *
 **************************************************************************/

namespace reference_gemm {
#define ENABLE_SYSTEM_GEMM(_type, _system_name)                               \
  extern "C" void _system_name(                                               \
      const char *, const char *, const int *, const int *, const int *,      \
      const _type *, const _type *, const int *, const _type *, const int *,  \
      const _type *, _type *, const int *);                                   \
  void gemm(const char *transA, const char *transB, int m, int n, int k,      \
            _type alpha, const _type a[], int lda, const _type b[], int ldb,  \
            _type beta, _type c[], int ldc) {                                 \
    _system_name(transA, transB, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, \
                 c, &ldc);                                                    \
  }

ENABLE_SYSTEM_GEMM(float, sgemm_)
ENABLE_SYSTEM_GEMM(double, dgemm_)
#undef ENABLE_SYSTEM_GEMM
}  // namespace reference_gemm
