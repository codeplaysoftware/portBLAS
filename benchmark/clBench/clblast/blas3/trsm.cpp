/**************************************************************************
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
 *  @filename trsm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(char side, char triangle, char transpose, char diagonal,
                     index_t m, index_t n) {
  std::ostringstream str{};
  str << "BM_Trsm<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << side << "/" << triangle << "/" << transpose << "/" << diagonal << "/"
      << m << "/" << n;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, char side,
         char triangle, char transpose, char diagonal, index_t m, index_t n,
         scalar_t alpha, bool* success) {
  // Standard test setup.
  index_t lda = side == 'l' ? m : n;
  index_t ldb = m;
  index_t k = side == 'l' ? m : n;

  using data_t = scalar_t;

  const int sizeA = k * lda;
  const int sizeB = n * ldb;

  blas_benchmark::utils::init_level_3_counters<
      blas_benchmark::utils::Level3Op::trsm, scalar_t>(state, 0, m, n, 0, 1,
                                                       side);

  // Matrices
  std::vector<data_t> a(sizeA);
  std::vector<data_t> b = blas_benchmark::utils::random_data<data_t>(sizeB);

  const scalar_t diagValue =
      diagonal == 'u'
          ? data_t{1}
          : blas_benchmark::utils::random_scalar<data_t>(data_t{1}, data_t{10});

  blas_benchmark::utils::fill_trsm_matrix(a, k, lda, triangle, diagValue,
                                          data_t{0});

  clblast::Transpose transA =
      blas_benchmark::utils::translate_transposition(&transpose);
  clblast::Side sideA = blas_benchmark::utils::translate_side(&side);
  clblast::Triangle triangleA =
      blas_benchmark::utils::translate_triangle(&triangle);
  clblast::Diagonal diagA =
      blas_benchmark::utils::translate_diagonal(&diagonal);

  MemBuffer<scalar_t> a_gpu(executorPtr, a.data(), static_cast<size_t>(sizeA));
  MemBuffer<scalar_t> b_gpu(executorPtr, b.data(), static_cast<size_t>(sizeB));

#ifdef BLAS_VERIFY_BENCHMARK
  // Run once verifying the results against the reference blas implementation.
  std::vector<data_t> x_ref = b;
  std::vector<data_t> b_temp = b;

  reference_blas::trsm(&side, &triangle, &transpose, &diagonal, m, n,
                       static_cast<data_t>(alpha), a.data(), lda, x_ref.data(),
                       ldb);

  {
    MemBuffer<scalar_t> b_temp_gpu(executorPtr, b_temp.data(),
                                   static_cast<size_t>(sizeB));
    cl_event event;
    clblast::Trsm(clblast::Layout::kColMajor, sideA, triangleA, transA, diagA,
                  m, n, alpha, a_gpu.dev(), 0, lda, b_temp_gpu.dev(), 0, ldb,
                  executorPtr->_queue(), &event);
    CLEventHandler::wait(event);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(b_temp, x_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl_event> {
    cl_event event;
    clblast::StatusCode ret = clblast::Trsm<scalar_t>(
        clblast::Layout::kColMajor, sideA, triangleA, transA, diagA, m, n,
        alpha, a_gpu.dev(), 0, lda, b_gpu.dev(), 0, ldb, executorPtr->_queue(),
        &event);
    if (ret != clblast::StatusCode::kSuccess) {
      *success = false;
      state.SkipWithError("Failed");
      return {};
    } else {
      CLEventHandler::wait(event);
      return {event};
    }
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        blas_benchmark::utils::timef(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                        bool* success) {
  auto trsm_params = blas_benchmark::utils::get_trsm_params<scalar_t>(args);

  for (auto p : trsm_params) {
    char side, triangle, transpose, diagonal;
    index_t m, n;
    scalar_t alpha;
    std::tie(side, triangle, transpose, diagonal, m, n, alpha) = p;

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr, char side,
                         char triangle, char transpose, char diagonal,
                         index_t m, index_t n, scalar_t alpha, bool* success) {
      run<scalar_t>(st, exPtr, side, triangle, transpose, diagonal, m, n, alpha,
                    success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(side, triangle, transpose, diagonal, m, n).c_str(),
        BM_lambda, exPtr, side, triangle, transpose, diagonal, m, n, alpha,
        success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, exPtr, success);
}
}  // namespace blas_benchmark
