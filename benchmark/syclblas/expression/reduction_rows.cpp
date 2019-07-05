/**************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename reduction_rows.cpp
 *
 **************************************************************************/

#include "sycl_blas.hpp"
#include "utils.hpp"

using namespace blas;

template <typename scalar_t>
std::string get_name(int rows, int cols) {
  std::ostringstream str{};
  str << "BM_RedRows<" << blas_benchmark::utils::get_type_name<scalar_t>()
      << ">/" << rows << "/" << cols;
  return str.str();
}

template <typename operator_t, typename scalar_t, typename executor_t,
          typename input_t, typename output_t>
std::vector<cl::sycl::event> launch_reduction(executor_t& ex, input_t buffer_in, output_t buffer_out,
                      index_t rows, index_t cols) {
  blas::Reduction<operator_t, input_t, output_t, 64, 64, 4, scalar_t,
                  static_cast<int>(Reduction_t::partial_rows)>
      reduction(buffer_in, buffer_out, rows, cols);
  return ex.execute(reduction);
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, index_t rows,
         index_t cols, bool* success) {
  // The counters are double. We convert m, n and k to double to avoid integer
  // overflows for n_fl_ops and bytes_processed
  double rows_d = static_cast<double>(rows);
  double cols_d = static_cast<double>(cols);

  state.counters["rows"] = rows_d;
  state.counters["cols"] = cols_d;

  state.counters["n_fl_ops"] = rows_d * cols_d;
  state.counters["bytes_processed"] = (rows_d * cols_d) * sizeof(scalar_t);

  ExecutorType& ex = *executorPtr;

  // Matrix
  std::vector<scalar_t> mat = blas_benchmark::utils::random_data<scalar_t>(rows * cols);
  auto mat_buffer = blas::make_sycl_iterator_buffer<scalar_t>(mat, rows * cols);
  auto mat_gpu = make_matrix_view<col_major>(ex, mat_buffer, rows, cols, rows);

  // Output vector
  std::vector<scalar_t> vec = blas_benchmark::utils::random_data<scalar_t>(rows);
  auto vec_buffer = blas::make_sycl_iterator_buffer<scalar_t>(vec, rows);
  auto vec_gpu = make_vector_view(ex, vec_buffer, 1, rows);

/* If enabled, run a first time with a verification of the results */
#ifdef BLAS_VERIFY_BENCHMARK
  std::vector<scalar_t> vec_ref = vec;
  /* Reduce the reference by hand on CPU */
  for (index_t i = 0; i < rows; i++) {
    vec_ref[i] = 0;
    for (index_t j = 0; j < cols; j++) {
      vec_ref[i] += mat[rows * j + i];
    }
  }
  std::vector<scalar_t> vec_temp = vec;
  {
    auto vec_temp_buffer = blas::make_sycl_iterator_buffer<scalar_t>(vec_temp, rows);
    auto vec_temp_gpu = make_vector_view(ex, vec_temp_buffer, 1, rows);
    auto event = launch_reduction<AddOperator, scalar_t>(ex, mat_gpu, vec_temp_gpu, rows, cols);
    ex.get_policy_handler().wait(event);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors<scalar_t>(vec_temp, vec_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = launch_reduction<AddOperator, scalar_t>(ex, mat_gpu, vec_gpu, rows, cols);
    ex.get_policy_handler().wait(event);
    return event;
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);
  ex.get_policy_handler().wait();

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
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                        bool* success) {
  auto red_params = blas_benchmark::utils::get_reduction_params<scalar_t>(args);

  for (auto p : red_params) {
    index_t rows, cols;
    std::tie(rows, cols) = p;

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr,
                         index_t rows, index_t cols, bool* success) {
      run<scalar_t>(st, exPtr, rows, cols, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(rows, cols).c_str(),
                                 BM_lambda, exPtr, rows, cols, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  register_benchmark<float>(args, exPtr, success);
#ifdef DOUBLE_SUPPORT
  register_benchmark<double>(args, exPtr, success);
#endif
}
}  // namespace blas_benchmark
