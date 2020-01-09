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
 *  @filename tune_impl.hpp
 *
 **************************************************************************/

#ifndef SYCLBLAS_TOOLS_AUTO_TUNER_TUNE_IMPL_HPP_
#define SYCLBLAS_TOOLS_AUTO_TUNER_TUNE_IMPL_HPP_

#include "tuner_types.hpp"
#include "utils.hpp"

#include "sycl_blas.hpp"

template <int Cls, typename Tile, bool DoubleBuffer, bool Nbca, bool Nbcb,
          typename Config, typename T>
TestResultEntry tune(int r, GemmArgs<T> a) {
  using Gemm =
      ::blas::Gemm<MatrixContainer<T>, MatrixContainer<T>, DoubleBuffer, Nbca,
                   Nbcb, Cls, Tile, Config::TransA, Config::TransB, T, false,
                   static_cast<int>(Config::MemoryMode),
                   static_cast<int>(Config::ShapeMode), 1>;

  TestResultEntry result(Gemm::get_type_string());
  auto ex = get_sycl_executor();
  {
    {
      auto event_list = ex.get_policy_handler().copy_to_device(
          a.init_c.data(), a.c, a.init_c.size());
      event_list.back().wait_and_throw();
    }

    auto accA =
        ::blas::make_matrix_view<::blas::col_major>(ex, a.a, a.m, a.k, a.lda);
    auto accB =
        ::blas::make_matrix_view<::blas::col_major>(ex, a.b, a.k, a.n, a.ldb);
    auto accC =
        ::blas::make_matrix_view<::blas::col_major>(ex, a.c, a.m, a.n, a.ldc);
    auto gemm = Gemm(accA, accB, accC, a.alpha, a.beta, a.batch_size);
    const double flop_count = 2.0 * a.m * a.n * a.k * a.batch_size;
    run_tune(r, flop_count, result, [&] {
      auto event_list = ex.execute(gemm);
      for (auto &event : event_list) {
        event.wait_and_throw();
      }
    });
    {
      auto event_list = ex.get_policy_handler().copy_to_host(
          a.c, a.output_c.data(), a.output_c.size());
      event_list.back().wait_and_throw();
    }
  }
  result.error = relative_diff(a.expected_c, a.output_c);
  return result;
}

#endif  // SYCLBLAS_TOOLS_AUTO_TUNER_TUNE_IMPL_HPP_
