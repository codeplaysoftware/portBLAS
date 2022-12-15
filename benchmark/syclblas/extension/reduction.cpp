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
 *  @filename reduction.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

using namespace blas;

template <typename scalar_t>
std::string get_name(int rows, int cols, reduction_dim_t reduction_dim) {
  std::ostringstream str{};
  str << "BM_Reduction<" << blas_benchmark::utils::get_type_name<scalar_t>()
      << ">/" << rows << "/" << cols << "/"
      << (reduction_dim == reduction_dim_t::inner ? "inner" : "outer");
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, index_t rows,
         index_t cols, reduction_dim_t dim, bool* success) {
  // The counters are double. We convert m, n and k to double to avoid integer
  // overflows for n_fl_ops and bytes_processed
  double rows_d = static_cast<double>(rows);
  double cols_d = static_cast<double>(cols);

  state.counters["rows"] = rows_d;
  state.counters["cols"] = cols_d;

  state.counters["n_fl_ops"] = rows_d * cols_d;
  state.counters["bytes_processed"] = (rows_d * cols_d) * sizeof(scalar_t);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  // Matrix
  std::vector<scalar_t> mat =
      blas_benchmark::utils::random_data<scalar_t>(rows * cols);
  auto mat_buffer = blas::make_sycl_iterator_buffer<scalar_t>(mat, rows * cols);

  // Output vector
  std::vector<scalar_t> vec = blas_benchmark::utils::random_data<scalar_t>(
      (dim == reduction_dim_t::outer) ? rows : cols);
  auto vec_buffer = blas::make_sycl_iterator_buffer<scalar_t>(vec, vec.size());

/* If enabled, run a first time with a verification of the results */
#ifdef BLAS_VERIFY_BENCHMARK
  std::vector<scalar_t> vec_ref = vec;
  /* Reduce the reference by hand on CPU */
  if (dim == reduction_dim_t::outer) {
    for (index_t i = 0; i < rows; i++) {
      vec_ref[i] = 0;
      for (index_t j = 0; j < cols; j++) {
        vec_ref[i] += mat[rows * j + i];
      }
    }
  } else if (dim == reduction_dim_t::inner) {
    for (index_t i = 0; i < cols; i++) {
      vec_ref[i] = 0;
      for (index_t j = 0; j < rows; j++) {
        vec_ref[i] += mat[rows * i + j];
      }
    }
  }
  std::vector<scalar_t> vec_temp = vec;
  {
    auto vec_temp_buffer = blas::make_sycl_iterator_buffer<scalar_t>(
        vec_temp.data(), vec_temp.size());

    extension::_reduction<AddOperator, scalar_t>(
        sb_handle, mat_buffer, rows, vec_temp_buffer, rows, cols, dim);
    auto event =
        blas::helper::copy_to_host(sb_handle.get_queue(), vec_temp_buffer,
                                   vec_temp.data(), vec_temp.size());
    sb_handle.wait(event);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(vec_temp, vec_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = extension::_reduction<AddOperator, scalar_t>(
        sb_handle, mat_buffer, rows, vec_buffer, rows, cols, dim);
    sb_handle.wait(event);
    return event;
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);
  sb_handle.wait();

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
void register_benchmark(blas_benchmark::Args& args, blas::SB_Handle* sb_handle_ptr,
                        bool* success) {
  auto red_params = blas_benchmark::utils::get_reduction_params<scalar_t>(args);

  for (auto p : red_params) {
    index_t rows, cols;
    std::tie(rows, cols) = p;

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         index_t rows, index_t cols, reduction_dim_t dim,
                         bool* success) {
      run<scalar_t>(st, sb_handle_ptr, rows, cols, dim, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(rows, cols, reduction_dim_t::inner).c_str(),
        BM_lambda, sb_handle_ptr, rows, cols, reduction_dim_t::inner, success);
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(rows, cols, reduction_dim_t::outer).c_str(),
        BM_lambda, sb_handle_ptr, rows, cols, reduction_dim_t::outer, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, blas::SB_Handle* sb_handle_ptr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
