/**************************************************************************
 *
 *  @license
 *  Copyright (C) 2021 Codeplay Software Limited
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

  // Matrices
  std::vector<data_t> a(sizeA);
  std::vector<data_t> b = blas_benchmark::utils::random_data<data_t>(sizeB);

  const data_t diagValue =
      diagonal == 'u'
          ? data_t{1}
          : blas_benchmark::utils::random_scalar<data_t>(data_t{1}, data_t{10});

  blas_benchmark::utils::fill_trsm_matrix(a, k, lda, triangle, diagValue,
                                          data_t{0});

  clblasTranspose transA =
      blas_benchmark::utils::translate_transposition(&transpose);
  clblasSide sideA = blas_benchmark::utils::translate_side(&side);
  clblasUplo triangleA = blas_benchmark::utils::translate_triangle(&triangle);
  clblasDiag diagA = blas_benchmark::utils::translate_diagonal(&diagonal);

  if (clblasSetup() != CL_SUCCESS) {
    state.SkipWithError("error initiazing clblas");
    *success = false;
    return;
  }

  cl_int err = CL_SUCCESS;
  cl_mem a_gpu = clCreateBuffer(executorPtr->ctx(), CL_MEM_READ_ONLY,
                                sizeA * sizeof(scalar_t), nullptr, &err);
  cl_mem b_gpu = clCreateBuffer(executorPtr->ctx(), CL_MEM_READ_WRITE,
                                sizeB * sizeof(scalar_t), nullptr, &err);

  if (err != CL_SUCCESS) {
    state.SkipWithError("error creating opencl buffers");
    *success = false;
    return;
  }

  err = clEnqueueWriteBuffer(executorPtr->queue(), a_gpu, CL_TRUE, 0,
                             sizeA * sizeof(scalar_t), a.data(), 0, nullptr,
                             nullptr);
  err = clEnqueueWriteBuffer(executorPtr->queue(), b_gpu, CL_TRUE, 0,
                             sizeB * sizeof(scalar_t), b.data(), 0, nullptr,
                             nullptr);

  if (err != CL_SUCCESS) {
    state.SkipWithError("error writing opencl buffers");
    *success = false;
    return;
  }

  cl_command_queue clQueue = executorPtr->queue();

#ifdef BLAS_VERIFY_BENCHMARK
  // Run once verifying the results against the reference blas implementation.
  std::vector<data_t> x_ref = b;
  std::vector<data_t> b_temp = b;

  reference_blas::trsm(&side, &triangle, &transpose, &diagonal, m, n,
                       static_cast<data_t>(alpha), a.data(), lda, x_ref.data(),
                       ldb);

  cl_mem b_gpu_temp = clCreateBuffer(executorPtr->ctx(), CL_MEM_READ_WRITE,
                                     sizeB * sizeof(scalar_t), nullptr, &err);
  err = clEnqueueWriteBuffer(executorPtr->queue(), b_gpu_temp, CL_TRUE, 0,
                             sizeB * sizeof(scalar_t), b.data(), 0, nullptr,
                             nullptr);

  if (err != CL_SUCCESS) {
    state.SkipWithError("error writing tempb");
    *success = false;
    return;
  }

  cl_event event;
  err = clblasStrsm(clblasColumnMajor, sideA, triangleA, transA, diagA, m, n,
                    alpha, a_gpu, 0, lda, b_gpu_temp, 0, ldb, 1, &clQueue, 0,
                    nullptr, &event);
  err = clWaitForEvents(1, &event);

  if (err != CL_SUCCESS) {
    state.SkipWithError("error running clblasStrsm");
    *success = false;
    return;
  }

  err = clEnqueueReadBuffer(clQueue, b_gpu_temp, CL_TRUE, 0,
                            sizeB * sizeof(scalar_t), b_temp.data(), 0, nullptr,
                            nullptr);

  std::ostringstream err_stream;
  if (!utils::compare_vectors(b_temp, x_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl_event> {
    cl_event event;
    clblasStatus ret = clblasStrsm(clblasColumnMajor, sideA, triangleA, transA,
                                   diagA, m, n, alpha, a_gpu, 0, lda, b_gpu, 0,
                                   ldb, 1, &clQueue, 0, nullptr, &event);

    if (ret != clblasSuccess) {
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

  {
    // The counters are double. We convert m, n and k to double to avoid
    // integer overflows for n_fl_ops and bytes_processed
    double m_d = static_cast<double>(m);
    double n_d = static_cast<double>(n);
    double k_d = static_cast<double>(k);

    state.counters["m"] = m_d;
    state.counters["k"] = k_d;
    state.counters["n"] = n_d;

    double mem_read = k_d * (k_d + 1) / 2;
    double mem_write = m_d * n_d;

    double total_mem = (mem_read * mem_write) * sizeof(scalar_t);
    state.counters["bytes_processed"] = total_mem;
    state.SetBytesProcessed(state.iterations() * total_mem);

    double nflops_AtimesB = 2 * k_d * (k_d + 1) / 2 * (side == 'l' ? n_d : m_d);
    double nflops_timesAlpha = m_d * n_d;
    double nflops = nflops_AtimesB + nflops_timesAlpha;
    state.counters["n_fl_ops"] = nflops;
    state.SetItemsProcessed(state.iterations() * nflops);
  }

  blas_benchmark::utils::calc_avg_counters(state);
}

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
