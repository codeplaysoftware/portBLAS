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
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename trsv.cpp
 *
 **************************************************************************/

// This does not work due to a clBLAS bug, see
// https://github.com/clMathLibraries/clBLAS/issues/341
//
// error: variables in the local address space can only be declared in the
// outermost scope of a kernel function

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string uplo, std::string t, std::string diag, int n) {
  std::ostringstream str{};
  str << "BM_Trsv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << uplo << "/" << t << "/" << diag << "/" << n;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, std::string uplo,
         std::string t, std::string diag, index_t n, bool* success) {
  // Standard test setup.
  const char* uplo_str = uplo.c_str();
  const char* t_str = t.c_str();
  const char* diag_str = diag.c_str();

  index_t xlen = n;
  index_t lda = n;
  index_t incX = 1;

  // The counters are double. We convert n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double n_d = static_cast<double>(n);

  state.counters["n"] = n_d;

  // Compute the number of A non-zero elements.
  const double A_validVal = .5 * n_d * (n_d + 1);

  {
    double nflops = n_d * n_d;
    state.counters["n_fl_ops"] = nflops;
  }

  {
    double mem_readA = A_validVal;
    double mem_readX = A_validVal;
    double mem_writeX = A_validVal;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readX + mem_writeX) * sizeof(scalar_t);
  }

  ExecutorType& ex = *executorPtr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a(lda * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);

  // Populate the main diagonal with larger values.
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      m_a[(i * lda) + i] = (i == j) ? blas_benchmark::utils::random_scalar(
                                          scalar_t{9}, scalar_t{11})
                                    : blas_benchmark::utils::random_scalar(
                                          scalar_t{-0.1}, scalar_t{0.1});

  // Specify the layout.
  auto layout = clblasColumnMajor;

  // Specify the triangle.
  clblasUplo a_uplo = blas_benchmark::utils::translate_triangle(uplo_str);

  // Specify the transposition.
  clblasTranspose a_tr = blas_benchmark::utils::translate_transposition(t_str);

  // Specify the unit-diagonal.
  clblasDiag a_diag = blas_benchmark::utils::translate_diagonal(diag_str);

  if (clblasSetup() != CL_SUCCESS) {
    state.SkipWithError("error initiazing clblas");
    *success = false;
    return;
  }

  cl_int err = CL_SUCCESS;
  cl_mem m_a_gpu = clCreateBuffer(executorPtr->ctx(), CL_MEM_READ_ONLY,
                                  lda * n * sizeof(scalar_t), nullptr, &err);
  cl_mem v_x_gpu = clCreateBuffer(executorPtr->ctx(), CL_MEM_READ_WRITE,
                                  xlen * sizeof(scalar_t), nullptr, &err);
  cl_mem v_x_temp_gpu = clCreateBuffer(executorPtr->ctx(), CL_MEM_READ_WRITE,
                                       xlen * sizeof(scalar_t), nullptr, &err);

  if (err != CL_SUCCESS) {
    state.SkipWithError("error creating opencl buffers");
    *success = false;
    return;
  }

  err = clEnqueueWriteBuffer(executorPtr->queue(), m_a_gpu, CL_TRUE, 0,
                             lda * n * sizeof(scalar_t), m_a.data(), 0, nullptr,
                             nullptr);
  err = clEnqueueWriteBuffer(executorPtr->queue(), v_x_gpu, CL_TRUE, 0,
                             xlen * sizeof(scalar_t), v_x.data(), 0, nullptr,
                             nullptr);

  if (err != CL_SUCCESS) {
    state.SkipWithError("error writing opencl buffers");
    *success = false;
    return;
  }

  cl_command_queue clQueue = executorPtr->queue();

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_x_ref = v_x;
  reference_blas::trsv(uplo_str, t_str, diag_str, n, m_a.data(), lda,
                       v_x_ref.data(), incX);
  std::vector<scalar_t> v_x_temp = v_x;
  {
    err = clEnqueueWriteBuffer(executorPtr->queue(), v_x_temp_gpu, CL_TRUE, 0,
                               xlen * sizeof(scalar_t), v_x_temp.data(), 0,
                               nullptr, nullptr);

    cl_event event;
    err = clblasStrsv(layout, a_uplo, a_tr, a_diag, n, m_a_gpu, 0, lda,
                      v_x_temp_gpu, 0, incX, 1, &clQueue, 0, nullptr, &event);
    err = clWaitForEvents(1, &event);

    if (err != CL_SUCCESS) {
      state.SkipWithError("error running clblasStrsm");
      *success = false;
      return;
    }

    err = clEnqueueReadBuffer(clQueue, v_x_temp_gpu, CL_TRUE, 0,
                              xlen * sizeof(scalar_t), v_x_temp.data(), 0,
                              nullptr, nullptr);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_x_temp, v_x_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl_event> {
    cl_event event;
    clblasStatus ret =
        clblasStrsv(layout, a_uplo, a_tr, a_diag, n, m_a_gpu, 0, lda,
                    v_x_temp_gpu, 0, incX, 1, &clQueue, 0, nullptr, &event);

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

  blas_benchmark::utils::calc_avg_counters(state);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                        bool* success) {
  auto tbmv_params = blas_benchmark::utils::get_tbmv_params(args);

  for (auto p : tbmv_params) {
    std::string uplos;
    std::string ts;
    std::string diags;
    index_t n;
    index_t k;
    std::tie(uplos, ts, diags, n, k) = p;

    // Repurpose tbmv_params.
    if (k != 1) continue;

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr,
                         std::string uplos, std::string ts, std::string diags,
                         index_t n, bool* success) {
      run<scalar_t>(st, exPtr, uplos, ts, diags, n, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(uplos, ts, diags, n).c_str(), BM_lambda, exPtr,
        uplos, ts, diags, n, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, exPtr, success);
}
}  // namespace blas_benchmark
